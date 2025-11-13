import math
import os
import re
from typing import List, Tuple, Dict, Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def split_sentences(text: str) -> List[str]:
    """
    簡易な日英対応の文分割。
    """
    if not text or not text.strip():
        return []
    normalized = re.sub(r"\s+", " ", text.strip())
    pattern = r"[^。．\.!?！？]+[。．\.!?！？]?"
    sentences = [s.strip() for s in re.findall(pattern, normalized) if s and s.strip()]
    sentences = [s for s in sentences if len(s) > 1]
    return sentences


def load_wikitext_articles(max_articles: int = 5, min_sentences: int = 6) -> List[List[str]]:
    """
    wikitext-2-raw-v1 から記事を構築（空行で区切り）し、文分割して返す。
    """
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


def load_model(model_name: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    model.eval()
    return tokenizer, model


def compute_nll_sum(
    tokenizer,
    model,
    context_text: str,
    target_text: str,
    device: str
) -> Tuple[float, int]:
    """
    teacher forcing によるターゲットのNLL（トークン合計, nats）を返す。
    戻り値: (sum_nll, num_target_tokens)
    """
    context_ids = tokenizer.encode(context_text, add_special_tokens=True)
    target_ids = tokenizer.encode(target_text, add_special_tokens=False)
    num_target = len(target_ids)
    if num_target == 0:
        return 0.0, 0
    input_ids = torch.tensor([context_ids + target_ids], device=device)
    labels = input_ids.clone()
    labels[:, :len(context_ids)] = -100
    if tokenizer.pad_token_id is not None:
        labels[labels == tokenizer.pad_token_id] = -100
    with torch.no_grad():
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss.item()  # 平均CE（未マスクトークン上）
    sum_nll = loss * num_target
    return float(sum_nll), num_target


def precompute_base_nll(
    sentences: List[str],
    tokenizer,
    model,
    device: str
) -> Tuple[List[float], List[float]]:
    """
    各文の base NLL（コンテキストなし/BOSのみ）を計算。
    返り値:
      - base_nll[i] = 文i(1-based)のNLL合計
      - base_prefix[j] = 文1..jのbase NLL合計（prefix-sum）
    """
    n = len(sentences)
    base_nll = [0.0] * (n + 1)  # 1-based
    for i in range(1, n + 1):
        sum_nll, num_tok = compute_nll_sum(tokenizer, model, context_text="", target_text=sentences[i - 1], device=device)
        base_nll[i] = sum_nll
    base_prefix = [0.0] * (n + 1)
    for j in range(1, n + 1):
        base_prefix[j] = base_prefix[j - 1] + base_nll[j]
    return base_nll, base_prefix


def chunk_text(sentences: List[str], s: int, t: int) -> str:
    # 1-based inclusive
    return "".join(sentences[s - 1:t])


def dp_coherence_gain_segmentation(
    sentences: List[str],
    tokenizer,
    model,
    device: str
) -> Tuple[List[Tuple[int, int]], float, List[float]]:
    """
    Coherence Gain を最大化するViterbi型DP。
    返り値:
      - チャンク区間のリスト（1-based, 両端含む）
      - DP_score[N]（最大TotalGain）
      - チャンクごとの gain 値（復元順）
    """
    n = len(sentences)
    if n == 0:
        return [], 0.0, []

    # 前計算：各文のbase NLLとprefix-sum
    base_nll, base_prefix = precompute_base_nll(sentences, tokenizer, model, device)

    # チャンクNLLのキャッシュ: key = (ctx_s, ctx_t, s, j) with 0表示BOS
    chunk_nll_cache: Dict[Tuple[int, int, int, int], float] = {}

    def chunk_nll_with_context(ctx_s: int, ctx_t: int, s: int, j: int) -> float:
        key = (ctx_s, ctx_t, s, j)
        if key in chunk_nll_cache:
            return chunk_nll_cache[key]
        context_text = "" if ctx_s == 0 else chunk_text(sentences, ctx_s, ctx_t)
        target_text = chunk_text(sentences, s, j)
        sum_nll, num_tok = compute_nll_sum(tokenizer, model, context_text=context_text, target_text=target_text, device=device)
        chunk_nll_cache[key] = sum_nll
        return sum_nll

    # DP配列
    dp_score = [float("-inf")] * (n + 1)
    prev_idx = [-1] * (n + 1)  # jを終端とする最適チャンクの開始位置 s
    # 「prefixの最後のチャンク開始位置」を保持して文脈復元を可能にする
    last_chunk_start_for_prefix = [-1] * (n + 1)
    last_chunk_end_for_prefix = [-1] * (n + 1)

    dp_score[0] = 0.0
    prev_idx[0] = -1
    last_chunk_start_for_prefix[0] = 0  # BOS
    last_chunk_end_for_prefix[0] = 0

    for j in range(1, n + 1):
        best_score = float("-inf")
        best_s = 1
        best_last_chunk_s = -1
        best_last_chunk_t = -1
        for s in range(1, j + 1):
            base_sum = base_prefix[j] - base_prefix[s - 1]
            if s == 1:
                ctx_s, ctx_t = 0, 0  # BOS
            else:
                ctx_s = last_chunk_start_for_prefix[s - 1]
                ctx_t = last_chunk_end_for_prefix[s - 1]
                if ctx_s == -1:
                    # まだ定義されていない場合は直前文のみを簡略文脈に
                    ctx_s, ctx_t = s - 1, s - 1
            nll_chunk = chunk_nll_with_context(ctx_s, ctx_t, s, j)
            gain = base_sum - nll_chunk
            cand = dp_score[s - 1] + gain
            if cand > best_score:
                best_score = cand
                best_s = s
                best_last_chunk_s = s
                best_last_chunk_t = j
        dp_score[j] = best_score
        prev_idx[j] = best_s
        last_chunk_start_for_prefix[j] = best_last_chunk_s
        last_chunk_end_for_prefix[j] = best_last_chunk_t

    # 復元
    chunks: List[Tuple[int, int]] = []
    gains: List[float] = []
    j = n
    while j > 0:
        s = prev_idx[j]
        chunks.append((s, j))
        # gainの再計算（ログ出力用）
        base_sum = base_prefix[j] - base_prefix[s - 1]
        if s == 1:
            ctx_s, ctx_t = 0, 0
        else:
            # prefix s-1 の最後のチャンクを復元
            # 一段戻って取得
            ps = prev_idx[s - 1]
            if ps == -1 and s - 1 == 0:
                ctx_s, ctx_t = 0, 0
            else:
                # ps..(s-1) が直前チャンク
                ctx_s, ctx_t = ps, s - 1
        nll_chunk = chunk_nll_with_context(ctx_s, ctx_t, s, j)
        gains.append(base_sum - nll_chunk)
        j = s - 1
    chunks.reverse()
    gains.reverse()
    return chunks, dp_score[n], gains


def render_article_chunks(
    article_id: int,
    sentences: List[str],
    chunks: List[Tuple[int, int]],
    total_gain: float,
    gains: List[float]
) -> str:
    lines: List[str] = []
    lines.append(f"=== Article #{article_id} ===")
    lines.append(f"Sentences: {len(sentences)}")
    lines.append(f"TotalGain: {total_gain:.4f}")
    lines.append("")
    lines.append("Original sentences (indexed):")
    for i, s in enumerate(sentences, 1):
        lines.append(f"  [{i}] {s}")
    lines.append("")
    lines.append("Chunks (with coherence gain):")
    for idx, ((s, t), g) in enumerate(zip(chunks, gains), 1):
        text = "".join(sentences[s - 1:t])
        lines.append(f"  ({idx}) [{s}-{t}]  Gain={g:.4f}")
        lines.append(f"    {text}")
    lines.append("")
    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Coherence Gain × Viterbi による文チャンク分割")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-0.5B", help="Hugging Face モデル名")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_articles", type=int, default=3, help="処理する記事数（小さめ推奨）")
    parser.add_argument("--min_sentences", type=int, default=6, help="記事採用の最小文数")
    parser.add_argument("--output", type=str, default="outputs/coherence_gain_chunks.txt", help="出力TXTパス")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print("記事読み込み中...")
    articles = load_wikitext_articles(max_articles=args.max_articles, min_sentences=args.min_sentences)
    if len(articles) == 0:
        print("条件を満たす記事が見つかりません。パラメータを調整してください。")
        return
    print(f"記事数: {len(articles)}")

    print(f"モデルをロード中: {args.model}")
    tokenizer, model = load_model(args.model, args.device)

    outputs: List[str] = []
    for idx, sents in enumerate(articles, 1):
        print(f"記事 {idx}/{len(articles)} をチャンク分割中...")
        chunks, total_gain, gains = dp_coherence_gain_segmentation(sents, tokenizer, model, args.device)
        outputs.append(render_article_chunks(idx, sents, chunks, total_gain, gains))

    result_text = "\n".join(outputs)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(result_text)
    print(f"出力しました: {args.output}")


if __name__ == "__main__":
    main()


