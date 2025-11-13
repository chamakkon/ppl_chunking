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


def precompute_B_and_C(
    sentences: List[str],
    tokenizer,
    model,
    device: str,
    k_gram: int,
    batch_size: int,
    sep: str,
) -> Tuple[List[float], List[List[float]], List[float], List[float]]:
    """
    前処理で以下を返す（1-based配列、先頭にダミー）:
      - B[i]: 文iのbase NLL（BOS文脈）合計
      - C[i][m]: 文iの文脈m(0..K, ただしm<=i-1)でのNLL合計（C[i][0] = B[i]）
      - PB[j]: Bのprefix和
      - PCk[j]: C[i][min(K, i-1)] のprefix和
    """
    n = len(sentences)
    # 1) B_i をバッチで計算
    B: List[float] = [0.0] * (n + 1)
    for start in range(1, n + 1, batch_size):
        end = min(n, start + batch_size - 1)
        ctxs = [""] * (end - start + 1)
        tgts = [sentences[i - 1] for i in range(start, end + 1)]
        sums, _ = compute_nll_sum_batch(tokenizer, model, ctxs, tgts, device)
        for offset, i in enumerate(range(start, end + 1)):
            B[i] = sums[offset]

    # 2) C_i^{(m)} を前計算（m=0..K, m<=i-1）
    C: List[List[float]] = [[0.0] * (k_gram + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        C[i][0] = B[i]
    pairs: List[Tuple[int, int]] = []  # (i, m)
    ctx_texts: List[str] = []
    tgt_texts: List[str] = []
    for i in range(1, n + 1):
        max_m = min(k_gram, i - 1)
        for m in range(1, max_m + 1):
            ctx_text = sep.join(sentences[i - m - 1:i - 1])
            pairs.append((i, m))
            ctx_texts.append(ctx_text)
            tgt_texts.append(sentences[i - 1])
    # バッチ評価して C を埋める
    for start in range(0, len(pairs), batch_size):
        end = min(len(pairs), start + batch_size)
        sums, _ = compute_nll_sum_batch(tokenizer, model, ctx_texts[start:end], tgt_texts[start:end], device)
        for offset, (i, m) in enumerate(pairs[start:end]):
            C[i][m] = sums[offset]

    # 3) prefix 和
    PB: List[float] = [0.0] * (n + 1)
    PCk: List[float] = [0.0] * (n + 1)
    for i in range(1, n + 1):
        PB[i] = PB[i - 1] + B[i]
        m_eff = min(k_gram, i - 1) if i - 1 >= 0 else 0
        PCk[i] = PCk[i - 1] + (C[i][m_eff] if m_eff >= 0 else 0.0)

    return B, C, PB, PCk


def dp_k_constrained_interval_segmentation(
    sentences: List[str],
    B: List[float],
    C: List[List[float]],
    PB: List[float],
    PCk: List[float],
    k_gram: int,
    alpha: float,
    tokenizer=None,
    model=None,
    device: str = "cpu",
    sep: str = "",
) -> Tuple[List[Tuple[int, int]], float]:
    """
    目的関数は従来の coherence gain:
      G_k(s..t) = sum_{i=s}^t B_i - NLL_k(s..t)
    遷移は区間DP:
      DP[j] = max_s DP[s-1] + G_k(s..j) - (0 if s==1 else alpha)
    """
    n = len(sentences)
    # DPと復元用
    DP = [float("-inf")] * (n + 1)
    prev_idx = [-1] * (n + 1)
    DP[0] = 0.0
    # 直接チャンクNLLのキャッシュ（必要時のみ使用）
    chunk_nll_cache: Dict[Tuple[int, int, int, int], float] = {}

    def nll_k_with_prev_tail(s: int, t: int, L_prev: int) -> float:
        """
        k-sent制約に加え、前チャンク末尾の min(k, L_prev) 文も文脈として利用可能とする。
        文 i の有効文脈 m(i) = min(k, L_prev + (i - s)), ただし m(i) <= i-1 を満たす。
        これにより k が十分大きい場合、制約なし版の文脈（前チャンク全体 + 同一チャンク内）に近づく。
        """
        if s > t:
            return 0.0
        # しきい値 i >= s + max(0, k - L_prev) で m(i) は k に飽和
        thresh = s + max(0, k_gram - L_prev)
        head_end = min(t, thresh - 1)
        head = 0.0
        # 頭側は高々 (k - L_prev) 項（L_prev>=kなら0項）
        for i in range(s, head_end + 1):
            d = i - s
            m_eff = L_prev + d
            # 安全側: m_eff は i-1 を超えない（理論上超えないが念のためmin）
            if m_eff > i - 1:
                m_eff = i - 1
            if m_eff > k_gram:
                m_eff = k_gram
            head += C[i][m_eff]
        tail = 0.0
        # テールは m(i) = k 固定。i-1 >= k が保証される領域のみ。
        if thresh <= t:
            tail = PCk[t] - PCk[thresh - 1]
        return head + tail

    for j in range(1, n + 1):
        best = float("-inf")
        best_s = -1
        for s in range(1, j + 1):
            # 前チャンク長 L_prev を最適prefix (s-1) の最後の区間から算出
            if s == 1:
                L_prev = 0
            else:
                last_start = prev_idx[s - 1]
                # 万一未定義なら直前文のみを前チャンクとするフォールバック
                if last_start == -1:
                    last_start = s - 1
                L_prev = (s - 1) - last_start + 1
                if L_prev < 0:
                    L_prev = 0
            # k が十分大きい場合は、厳密に「前チャンク全文脈 + 現チャンク丸ごと」のNLLを直接計算して一致性を担保
            use_direct = (k_gram >= (j - 1)) and (tokenizer is not None) and (model is not None)
            if use_direct:
                if s == 1:
                    ctx_s, ctx_t = 0, 0
                else:
                    ctx_s, ctx_t = last_start, s - 1
                key = (ctx_s, ctx_t, s, j)
                if key in chunk_nll_cache:
                    nll_val = chunk_nll_cache[key]
                else:
                    # 前チャンク: [ctx_s .. ctx_t] を包含するように、終端は +1 する
                    ctx_text = "" if ctx_s == 0 else sep.join(sentences[ctx_s - 1:ctx_t + 1])
                    # ターゲット: [s .. j]（終端は既に排他的仕様で j を指定）
                    tgt_text = sep.join(sentences[s - 1:j])
                    sums, _ = compute_nll_sum_batch(tokenizer, model, [ctx_text], [tgt_text], device)
                    nll_val = sums[0]
                    chunk_nll_cache[key] = nll_val
            else:
                nll_val = nll_k_with_prev_tail(s, j, L_prev)
            gain = (PB[j] - PB[s - 1]) - nll_val
            boundary_cost = 0.0 if s == 1 else alpha
            cand = DP[s - 1] + gain - boundary_cost
            if cand > best:
                best = cand
                best_s = s
        DP[j] = best
        prev_idx[j] = best_s

    total_score = DP[n]

    # 復元（区間）
    chunks: List[Tuple[int, int]] = []
    j = n
    while j > 0:
        s = prev_idx[j]
        chunks.append((s, j))
        j = s - 1
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
    parser = argparse.ArgumentParser(description="k-sent制約付き Coherence Gain × 区間DP（高速前計算）")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-0.5B")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dataset", type=str, default="wikitext", choices=["wikitext", "wikipedia_ja"])
    parser.add_argument("--dataset_slice", type=str, default="train[:1%]", help="wikipedia_ja用のスライス")
    parser.add_argument("--max_articles", type=int, default=3)
    parser.add_argument("--min_sentences", type=int, default=6)
    parser.add_argument("--k_gram", type=int, default=2, help="文脈の最大文数k（遷移コストの制約）")
    parser.add_argument("--alpha", type=float, default=0.5, help="境界コスト（先頭チャンクは未適用）")
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
        # 制約なし実装に合わせる: どのデータセットでも文結合は空文字
        sep = ""
        B, C, PB, PCk = precompute_B_and_C(sents, tokenizer, model, args.device, args.k_gram, args.batch_size, sep)
        print(f"[{idx}/{len(articles)}] DP最適化中 ...")
        chunks, score = dp_k_constrained_interval_segmentation(
            sents, B, C, PB, PCk, args.k_gram, args.alpha,
            tokenizer=tokenizer, model=model, device=args.device, sep=sep
        )
        outputs.append(render_article_chunks(idx, sents, chunks, score))

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(outputs))
    print(f"出力しました: {args.output}")


if __name__ == "__main__":
    main()


