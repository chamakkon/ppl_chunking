import math
import os
import re
from typing import List, Tuple, Dict

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def split_sentences_ja(text: str) -> List[str]:
    """
    日本語向け簡易文分割（句点終端ベース）。
    強すぎる分割を避けるため、空白正規化後に句点・感嘆・疑問で文を切る。
    """
    if not text or not text.strip():
        return []
    normalized = re.sub(r"\s+", " ", text.strip())
    pattern = r"[^。．！？!?]+[。．！？!?]?"
    sents = [s.strip() for s in re.findall(pattern, normalized) if s and s.strip()]
    # 1文字などの破片は除外
    sents = [s for s in sents if len(s) > 1]
    return sents


def load_wikipedia_ja_articles(
    dataset_slice: str = "train[:1%]",
    max_articles: int = 5,
    min_sentences: int = 6
) -> List[List[str]]:
    """
    Wikipedia（JA）から小規模サンプルを取得し、文分割して記事ごとの文列を返す。
    dataset_slice 例: 'train[:1%]', 'train[:2000]'
    """
    # 新方式のデータセット（旧 'wikipedia' スクリプトは非対応）
    ds = load_dataset("wikimedia/wikipedia", "20231101.ja", split=dataset_slice)
    articles: List[List[str]] = []
    for rec in ds:
        text = rec.get("text") or ""
        sents = split_sentences_ja(text)
        if len(sents) >= min_sentences:
            articles.append(sents)
            if len(articles) >= max_articles:
                break
    return articles


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
    teacher forcing によるターゲットNLL（合計, nats）を返す。
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
        loss = outputs.loss.item()  # 平均CE（未マスク部）
    return float(loss * num_target), num_target


def precompute_base_nll(sentences: List[str], tokenizer, model, device: str) -> Tuple[List[float], List[float]]:
    """
    各文の base NLL（BOSのみ文脈）を前計算し、prefix-sumも返す。
    """
    n = len(sentences)
    base_nll = [0.0] * (n + 1)
    for i in range(1, n + 1):
        sum_nll, _ = compute_nll_sum(tokenizer, model, context_text="", target_text=sentences[i - 1], device=device)
        base_nll[i] = sum_nll
    base_prefix = [0.0] * (n + 1)
    for j in range(1, n + 1):
        base_prefix[j] = base_prefix[j - 1] + base_nll[j]
    return base_nll, base_prefix


def chunk_text(sents: List[str], s: int, t: int) -> str:
    return "".join(sents[s - 1:t])


def dp_coherence_gain_segmentation(
    sentences: List[str],
    tokenizer,
    model,
    device: str,
    length_penalty_gamma: float = 0.0,
    max_tokens_per_chunk: int = 0
) -> Tuple[List[Tuple[int, int]], float, List[float]]:
    """
    Coherence Gain を最大化するViterbi DP（日本語向けデータ前提）。
    追加のソフト制約（任意）:
      - length_penalty_gamma: チャンクが長すぎる場合の二乗ペナルティ係数
      - max_tokens_per_chunk: 0以外なら、この上限超過分に対してペナルティ
    """
    n = len(sentences)
    if n == 0:
        return [], 0.0, []

    base_nll, base_prefix = precompute_base_nll(sentences, tokenizer, model, device)

    chunk_nll_cache: Dict[Tuple[int, int, int, int], float] = {}
    toklen_cache: Dict[Tuple[int, int], int] = {}

    def chunk_nll(ctx_s: int, ctx_t: int, s: int, j: int) -> float:
        key = (ctx_s, ctx_t, s, j)
        if key in chunk_nll_cache:
            return chunk_nll_cache[key]
        context_text = "" if ctx_s == 0 else chunk_text(sentences, ctx_s, ctx_t)
        target_text = chunk_text(sentences, s, j)
        val, _ = compute_nll_sum(tokenizer, model, context_text, target_text, device)
        chunk_nll_cache[key] = val
        return val

    def chunk_toklen(s: int, j: int) -> int:
        key = (s, j)
        if key in toklen_cache:
            return toklen_cache[key]
        text = chunk_text(sentences, s, j)
        tl = len(tokenizer.encode(text, add_special_tokens=False))
        toklen_cache[key] = tl
        return tl

    dp_score = [float("-inf")] * (n + 1)
    prev_idx = [-1] * (n + 1)
    last_chunk_start_for_prefix = [-1] * (n + 1)
    last_chunk_end_for_prefix = [-1] * (n + 1)

    dp_score[0] = 0.0
    last_chunk_start_for_prefix[0] = 0
    last_chunk_end_for_prefix[0] = 0

    for j in range(1, n + 1):
        best_score = float("-inf")
        best_s = 1
        best_lc_s = -1
        best_lc_t = -1
        for s in range(1, j + 1):
            base_sum = base_prefix[j] - base_prefix[s - 1]
            if s == 1:
                ctx_s, ctx_t = 0, 0
            else:
                ctx_s = last_chunk_start_for_prefix[s - 1]
                ctx_t = last_chunk_end_for_prefix[s - 1]
                if ctx_s == -1:
                    ctx_s, ctx_t = s - 1, s - 1
            nll = chunk_nll(ctx_s, ctx_t, s, j)
            gain = base_sum - nll
            # ソフト長さペナルティ
            if length_penalty_gamma > 0.0 and max_tokens_per_chunk > 0:
                tl = chunk_toklen(s, j)
                over = max(0, tl - max_tokens_per_chunk)
                if over > 0:
                    gain -= length_penalty_gamma * (over ** 2)
            cand = dp_score[s - 1] + gain
            if cand > best_score:
                best_score = cand
                best_s = s
                best_lc_s = s
                best_lc_t = j
        dp_score[j] = best_score
        prev_idx[j] = best_s
        last_chunk_start_for_prefix[j] = best_lc_s
        last_chunk_end_for_prefix[j] = best_lc_t

    chunks: List[Tuple[int, int]] = []
    gains: List[float] = []
    j = n
    while j > 0:
        s = prev_idx[j]
        chunks.append((s, j))
        base_sum = base_prefix[j] - base_prefix[s - 1]
        if s == 1:
            ctx_s, ctx_t = 0, 0
        else:
            ps = prev_idx[s - 1]
            if ps == -1 and s - 1 == 0:
                ctx_s, ctx_t = 0, 0
            else:
                ctx_s, ctx_t = ps, s - 1
        nll = chunk_nll(ctx_s, ctx_t, s, j)
        g = base_sum - nll
        gains.append(g)
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
    lines.append(f"=== 記事 #{article_id} ===")
    lines.append(f"文数: {len(sentences)}")
    lines.append(f"TotalGain: {total_gain:.4f}")
    lines.append("")
    lines.append("文一覧（インデックス付き）:")
    for i, s in enumerate(sentences, 1):
        lines.append(f"  [{i}] {s}")
    lines.append("")
    lines.append("チャンク一覧（coherence gain付き）:")
    for idx, ((s, t), g) in enumerate(zip(chunks, gains), 1):
        text = "".join(sentences[s - 1:t])
        lines.append(f"  ({idx}) [{s}-{t}]  Gain={g:.4f}")
        lines.append(f"    {text}")
    lines.append("")
    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="日本語Wikipedia × Coherence Gain × Viterbi チャンク分割")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-0.5B", help="Hugging Face モデル名")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dataset_slice", type=str, default="train[:1%]", help="wikipedia(ja)の読み込みスライス")
    parser.add_argument("--max_articles", type=int, default=3, help="処理する記事数")
    parser.add_argument("--min_sentences", type=int, default=6, help="記事採用の最小文数")
    parser.add_argument("--output", type=str, default="outputs/coherence_gain_chunks_ja.txt", help="出力TXTパス")
    parser.add_argument("--length_penalty_gamma", type=float, default=0.0, help="長さペナルティ係数（0で無効）")
    parser.add_argument("--max_tokens_per_chunk", type=int, default=0, help="チャンクトークン上限（0で無効）")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print("日本語Wikipedia記事の読み込み中...")
    articles = load_wikipedia_ja_articles(
        dataset_slice=args.dataset_slice,
        max_articles=args.max_articles,
        min_sentences=args.min_sentences
    )
    if len(articles) == 0:
        print("条件を満たす記事が見つかりません。パラメータを調整してください。")
        return
    print(f"記事数: {len(articles)}")

    print(f"モデルをロード中: {args.model}")
    tokenizer, model = load_model(args.model, args.device)

    outputs: List[str] = []
    for idx, sents in enumerate(articles, 1):
        print(f"記事 {idx}/{len(articles)} をチャンク分割中...")
        chunks, total_gain, gains = dp_coherence_gain_segmentation(
            sents, tokenizer, model, args.device,
            length_penalty_gamma=args.length_penalty_gamma,
            max_tokens_per_chunk=args.max_tokens_per_chunk
        )
        outputs.append(render_article_chunks(idx, sents, chunks, total_gain, gains))

    result_text = "\n".join(outputs)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(result_text)
    print(f"出力しました: {args.output}")


if __name__ == "__main__":
    main()


