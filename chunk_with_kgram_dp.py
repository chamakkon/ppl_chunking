import math
import os
import re
from typing import List, Tuple, Dict

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def split_sentences(text: str) -> List[str]:
    if not text or not text.strip():
        return []
    normalized = re.sub(r"\s+", " ", text.strip())
    pattern = r"[^。．\.!?！？]+[。．\.!?！？]?"
    sentences = [s.strip() for s in re.findall(pattern, normalized) if s and s.strip()]
    sentences = [s for s in sentences if len(s) > 1]
    return sentences


def load_wikitext_articles(max_articles: int = 5, min_sentences: int = 6) -> List[List[str]]:
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    articles: List[List[str]] = []
    buffer_lines: List[str] = []
    for rec in ds:
        line = rec["text"]
        if line.strip() == "":
            if buffer_lines:
                article_text = " ".join(buffer_lines)
                sents = split_sentences(article_text)
                if len(sents) >= min_sentences:
                    articles.append(sents)
                buffer_lines = []
                if len(articles) >= max_articles:
                    break
        else:
            buffer_lines.append(line)
    if len(articles) < max_articles and buffer_lines:
        article_text = " ".join(buffer_lines)
        sents = split_sentences(article_text)
        if len(sents) >= min_sentences:
            articles.append(sents)
    return articles[:max_articles]


def load_wikipedia_ja_articles(dataset_slice: str, max_articles: int, min_sentences: int) -> List[List[str]]:
    ds = load_dataset("wikimedia/wikipedia", "20231101.ja", split=dataset_slice)
    articles: List[List[str]] = []
    for rec in ds:
        text = (rec.get("text") or "").strip()
        if not text:
            continue
        sents = split_sentences(text)
        if len(sents) >= min_sentences:
            articles.append(sents)
            if len(articles) >= max_articles:
                break
    return articles


def load_model(model_name: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token  # パディング用にEOSを使用
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    model.eval()
    return tokenizer, model


def compute_nll_sum_batch(
    tokenizer,
    model,
    contexts: List[str],
    targets: List[str],
    device: str,
) -> Tuple[List[float], List[int]]:
    """
    複数 (context, target) をまとめて計算し、各サンプルのNLL合計とターゲットトークン数を返す。
    """
    batch_size = len(contexts)
    input_ids_list: List[List[int]] = []
    labels_list: List[List[int]] = []
    for ctx, tgt in zip(contexts, targets):
        ctx_ids = tokenizer.encode(ctx, add_special_tokens=True)
        tgt_ids = tokenizer.encode(tgt, add_special_tokens=False)
        seq = ctx_ids + tgt_ids
        input_ids_list.append(seq)
        labels = [-100] * len(ctx_ids) + tgt_ids[:]  # コンテキストは無視
        labels_list.append(labels)

    # パディング
    max_len = max(len(x) for x in input_ids_list) if input_ids_list else 0
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    padded_input = []
    padded_labels = []
    attn = []
    for ids, lbl in zip(input_ids_list, labels_list):
        pad_len = max_len - len(ids)
        padded_input.append(ids + [pad_id] * pad_len)
        padded_labels.append(lbl + [-100] * pad_len)
        attn.append([1] * len(ids) + [0] * pad_len)

    if max_len == 0:
        return [0.0] * batch_size, [0] * batch_size

    input_ids = torch.tensor(padded_input, device=device)
    attention_mask = torch.tensor(attn, device=device)
    labels = torch.tensor(padded_labels, device=device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [B, T, V]

    # 次トークン予測（shift）
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    shift_attn = (shift_labels != -100).to(shift_logits.dtype)

    # 安全のため -100 を 0 に置換して gather
    safe_labels = shift_labels.clone()
    safe_labels[safe_labels == -100] = 0
    log_probs = F.log_softmax(shift_logits, dim=-1)
    gathered = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)  # [B, T-1]

    # サンプルごと NLL 合計
    nll_per_token = -gathered * shift_attn  # 無視位置は0
    sum_nll_per_sample = nll_per_token.sum(dim=1)  # [B]
    num_tokens_per_sample = shift_attn.sum(dim=1).to(torch.int64)  # [B]

    return sum_nll_per_sample.tolist(), num_tokens_per_sample.tolist()


def precompute_base_and_gains(
    sentences: List[str],
    tokenizer,
    model,
    device: str,
    k_gram: int,
    batch_size: int,
) -> Tuple[List[float], List[List[float]]]:
    """
    前処理で
      - B[i]: 文iのbase NLL（BOS文脈）合計
      - g[i][m]: 文iの局所coherence gain（直前m文文脈, 0..K）を前計算
    を返す。1-basedインデックスで返却（先頭にダミーを持つ）。
    """
    n = len(sentences)
    # 1) base NLL（B_i）
    B = [0.0] * (n + 1)
    # バッチで計算
    for start in range(1, n + 1, batch_size):
        end = min(n, start + batch_size - 1)
        ctxs = [""] * (end - start + 1)
        tgts = [sentences[i - 1] for i in range(start, end + 1)]
        sums, _ = compute_nll_sum_batch(tokenizer, model, ctxs, tgts, device)
        for offset, i in enumerate(range(start, end + 1)):
            B[i] = sums[offset]

    # 2) g_i^{(m)} の前計算（m=0..K）
    g: List[List[float]] = [[0.0] * (k_gram + 1) for _ in range(n + 1)]
    # m=0 は常に 0（チャンク先頭）
    # m>=1 は O(KN) 件をバッチにまとめて計算
    pairs: List[Tuple[int, int]] = []  # (i, m)
    ctx_texts: List[str] = []
    tgt_texts: List[str] = []
    for i in range(1, n + 1):
        max_m = min(k_gram, i - 1)
        for m in range(1, max_m + 1):
            ctx_text = "".join(sentences[i - m - 1:i - 1])
            pairs.append((i, m))
            ctx_texts.append(ctx_text)
            tgt_texts.append(sentences[i - 1])
    # バッチ評価
    for start in range(0, len(pairs), batch_size):
        end = min(len(pairs), start + batch_size)
        sums, _ = compute_nll_sum_batch(tokenizer, model, ctx_texts[start:end], tgt_texts[start:end], device)
        for offset, (i, m) in enumerate(pairs[start:end]):
            g[i][m] = B[i] - sums[offset]
    return B, g


def dp_kgram_segmentation(
    sentences: List[str],
    g: List[List[float]],
    k_gram: int,
    alpha: float,
) -> Tuple[List[Tuple[int, int]], float]:
    """
    k-gram DP（Viterbi型）
    DP[i][m]:
      文1..iまで処理し、文iでチャンクを継続しており、そのチャンク内の過去文数がm（0..K）の最大スコア
    遷移：
      - 新チャンク開始: DP[i][0] = max_m DP[i-1][m] - alpha
      - 継続: DP[i][m] = DP[i-1][m-1] + g[i][m]（m=1..K）ただし m=K のとき max(DP[i-1][K-1], DP[i-1][K]) + g[i][K]
    復元は boundary 配列と prev_m で行う。
    """
    n = len(sentences)
    # DPと復元用
    DP = [[float("-inf")] * (k_gram + 1) for _ in range(n + 1)]
    prev_m = [[-1] * (k_gram + 1) for _ in range(n + 1)]
    boundary = [[False] * (k_gram + 1) for _ in range(n + 1)]

    # 初期化（文1は新チャンク開始、gain=0）
    DP[0][0] = 0.0
    for m in range(1, k_gram + 1):
        DP[0][m] = float("-inf")
    DP[1][0] = 0.0
    boundary[1][0] = True
    prev_m[1][0] = 0
    for m in range(1, k_gram + 1):
        DP[1][m] = float("-inf")

    for i in range(2, n + 1):
        # 新チャンク開始（m=0）
        best_prev = float("-inf")
        best_prev_m_state = 0
        for m in range(0, k_gram + 1):
            if DP[i - 1][m] > best_prev:
                best_prev = DP[i - 1][m]
                best_prev_m_state = m
        DP[i][0] = best_prev - alpha
        boundary[i][0] = True
        prev_m[i][0] = best_prev_m_state

        # 継続（m=1..K-1）
        for m in range(1, k_gram):
            if DP[i - 1][m - 1] != float("-inf"):
                DP[i][m] = DP[i - 1][m - 1] + g[i][m]
                prev_m[i][m] = m - 1
                boundary[i][m] = False
            else:
                DP[i][m] = float("-inf")
                prev_m[i][m] = -1
                boundary[i][m] = False

        # 継続（m=K）
        cand1 = DP[i - 1][k_gram - 1] + g[i][k_gram] if DP[i - 1][k_gram - 1] != float("-inf") else float("-inf")
        cand2 = DP[i - 1][k_gram] + g[i][k_gram] if DP[i - 1][k_gram] != float("-inf") else float("-inf")
        if cand1 >= cand2:
            DP[i][k_gram] = cand1
            prev_m[i][k_gram] = k_gram - 1
        else:
            DP[i][k_gram] = cand2
            prev_m[i][k_gram] = k_gram
        boundary[i][k_gram] = False

    # 終端選択
    last_i = n
    last_m = max(range(k_gram + 1), key=lambda m: DP[last_i][m])
    total_score = DP[last_i][last_m]

    # 復元
    chunks: List[Tuple[int, int]] = []
    current_end = n
    i = n
    m = last_m
    while i > 0:
        if boundary[i][m]:
            # i がチャンク先頭
            start = i
            chunks.append((start, current_end))
            current_end = i - 1
        pm = prev_m[i][m]
        i -= 1
        m = pm if pm is not None else 0
    chunks.reverse()
    return chunks, total_score


def render_article_chunks(article_id: int, sentences: List[str], chunks: List[Tuple[int, int]], score: float) -> str:
    lines: List[str] = []
    lines.append(f"=== Article #{article_id} ===")
    lines.append(f"Sentences: {len(sentences)}")
    lines.append(f"Score (TotalGain - alpha*#boundaries): {score:.4f}")
    lines.append("")
    lines.append("Original sentences (indexed):")
    for i, s in enumerate(sentences, 1):
        lines.append(f"  [{i}] {s}")
    lines.append("")
    lines.append("Chunks:")
    for idx, (s, t) in enumerate(chunks, 1):
        text = "".join(sentences[s - 1:t])
        lines.append(f"  ({idx}) [{s}-{t}]")
        lines.append(f"    {text}")
    lines.append("")
    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="k-gram Coherence × Viterbi DP（高速版）")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-0.5B")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dataset", type=str, default="wikitext", choices=["wikitext", "wikipedia_ja"])
    parser.add_argument("--dataset_slice", type=str, default="train[:1%]", help="wikipedia_ja用のスライス")
    parser.add_argument("--max_articles", type=int, default=3)
    parser.add_argument("--min_sentences", type=int, default=6)
    parser.add_argument("--k_gram", type=int, default=2, help="文脈の最大文数K")
    parser.add_argument("--alpha", type=float, default=0.5, help="境界コスト")
    parser.add_argument("--batch_size", type=int, default=16, help="LM前計算のバッチサイズ")
    parser.add_argument("--output", type=str, default="outputs/kgram_dp_chunks.txt")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # データ読み込み
    if args.dataset == "wikitext":
        articles = load_wikitext_articles(max_articles=args.max_articles, min_sentences=args.min_sentences)
    else:
        articles = load_wikipedia_ja_articles(args.dataset_slice, args.max_articles, args.min_sentences)
    if len(articles) == 0:
        print("条件を満たす記事が見つかりませんでした。")
        return

    # モデル
    tokenizer, model = load_model(args.model, args.device)

    outputs: List[str] = []
    for idx, sents in enumerate(articles, 1):
        print(f"[{idx}/{len(articles)}] 前計算中 ...")
        _, g = precompute_base_and_gains(sents, tokenizer, model, args.device, args.k_gram, args.batch_size)
        print(f"[{idx}/{len(articles)}] DP最適化中 ...")
        chunks, score = dp_kgram_segmentation(sents, g, args.k_gram, args.alpha)
        outputs.append(render_article_chunks(idx, sents, chunks, score))

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(outputs))
    print(f"出力しました: {args.output}")


if __name__ == "__main__":
    main()


