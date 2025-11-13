import math
import os
import re
from typing import List, Tuple, Dict

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


def compute_ce(tokenizer, model, context: str, target: str, device: str) -> float:
    """
    teacher forcing によるターゲットの平均クロスエントロピー（nats）を返す。
    """
    context_ids = tokenizer.encode(context, add_special_tokens=True)
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    if len(target_ids) == 0:
        return float("nan")
    input_ids = torch.tensor([context_ids + target_ids], device=device)
    labels = input_ids.clone()
    labels[:, :len(context_ids)] = -100
    if tokenizer.pad_token_id is not None:
        labels[labels == tokenizer.pad_token_id] = -100
    with torch.no_grad():
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss.item()
    return loss


def nse_from_ce(ce: float, vocab_size: int) -> float:
    """
    NSE = 1 - CE / ln(|V|)
    0〜1にクリップ。数値安定化のため下限を小さく確保。
    """
    if not math.isfinite(ce):
        return 1e-8
    h_max = math.log(vocab_size)
    if h_max <= 0:
        return 1e-8
    nse = 1.0 - (ce / h_max)
    nse = max(1e-8, min(1.0 - 1e-8, nse))
    return nse


def viterbi_chunk(
    sentences: List[str],
    tokenizer,
    model,
    device: str,
) -> Tuple[List[Tuple[int, int]], float]:
    """
    thoughts.txt の仕様に基づく Viterbi チャンク分割。
    返り値:
      - チャンクのインデックス区間リスト（1-based, 両端含む）
      - 最終評価値（1チャンクあたり平均logスコア）
    """
    n = len(sentences)
    if n == 0:
        return [], float("-inf")

    vocab_size = len(tokenizer)

    # 退化回避のための調整項
    # - chunk_bias: チャンク1つごとに与える正のボーナス（分割を促す）
    # - len_penalty_per_token: チャンクのトークン長に比例した負のペナルティ（長大チャンクを抑制）
    chunk_bias = 0.05
    len_penalty_per_token = 0.001

    # メモ化（再計算コスト削減）
    # φ(C) 用: 文区間テキスト -> log NSE
    node_log_cache: Dict[Tuple[int, int], float] = {}
    # ψ(C | C_prev) 用: (prev_s, prev_t, s, t) -> log NSE
    trans_log_cache: Dict[Tuple[int, int, int, int], float] = {}
    # チャンクのトークン長キャッシュ
    chunk_toklen_cache: Dict[Tuple[int, int], int] = {}

    def chunk_text(si: int, tj: int) -> str:
        # 1-based inclusive indices
        return "".join(sentences[si - 1:tj])

    def chunk_toklen(si: int, tj: int) -> int:
        key = (si, tj)
        if key in chunk_toklen_cache:
            return chunk_toklen_cache[key]
        text = chunk_text(si, tj)
        toklen = len(tokenizer.encode(text, add_special_tokens=False))
        chunk_toklen_cache[key] = toklen
        return toklen

    def phi(si: int, tj: int) -> float:
        key = (si, tj)
        if key in node_log_cache:
            return node_log_cache[key]
        ce = compute_ce(tokenizer, model, context="", target=chunk_text(si, tj), device=device)
        nse = nse_from_ce(ce, vocab_size)
        # log(NSE) に調整項を加える
        val = math.log(nse)
        # チャンク数ボーナス（分割を促進）
        val += chunk_bias
        # 長さペナルティ（長大チャンク抑制）
        val -= len_penalty_per_token * float(chunk_toklen(si, tj))
        node_log_cache[key] = val
        return val

    def psi(prev_s: int, prev_t: int, si: int, tj: int) -> float:
        key = (prev_s, prev_t, si, tj)
        if key in trans_log_cache:
            return trans_log_cache[key]
        context = chunk_text(prev_s, prev_t)
        target = chunk_text(si, tj)
        ce = compute_ce(tokenizer, model, context=context, target=target, device=device)
        nse = nse_from_ce(ce, vocab_size)
        val = math.log(nse)
        trans_log_cache[key] = val
        return val

    # DP 配列
    dp_score = [float("-inf")] * (n + 1)
    dp_chunks = [0] * (n + 1)
    prev_idx = [-1] * (n + 1)

    dp_score[0] = 0.0
    dp_chunks[0] = 0
    prev_idx[0] = -1

    for j in range(1, n + 1):
        best_score = float("-inf")
        best_s = 1
        best_chunks = 0
        for s in range(1, j + 1):
            # チャンク C = S_{s..j}
            score_c = phi(s, j)
            if s == 1:
                trans = 0.0  # BOS
            else:
                # 直前チャンクは prev_idx[s-1]..(s-1)
                prev_s = prev_idx[s - 1]
                if prev_s == -1 and s - 1 == 0:
                    # 文1..(s-1)=空 → BOS扱い
                    trans = 0.0
                else:
                    prev_t = s - 1
                    trans = psi(prev_s, prev_t, s, j)
            cand_score = dp_score[s - 1] + score_c + trans
            cand_chunks = dp_chunks[s - 1] + 1
            if cand_score > best_score:
                best_score = cand_score
                best_s = s
                best_chunks = cand_chunks
        dp_score[j] = best_score
        dp_chunks[j] = best_chunks
        prev_idx[j] = best_s

    # 復元
    chunks: List[Tuple[int, int]] = []
    j = n
    while j > 0:
        s = prev_idx[j]
        chunks.append((s, j))
        j = s - 1
    chunks.reverse()

    final_score = dp_score[n] / max(1, dp_chunks[n])
    return chunks, final_score


def render_article_chunks(
    article_id: int,
    sentences: List[str],
    chunks: List[Tuple[int, int]],
    final_score: float
) -> str:
    lines: List[str] = []
    lines.append(f"=== Article #{article_id} ===")
    lines.append(f"Sentences: {len(sentences)}")
    lines.append(f"FinalScore (avg log): {final_score:.4f}")
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

    parser = argparse.ArgumentParser(description="NSE×Viterbi による文チャンク分割（サンプル）")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-0.5B", help="Hugging Face モデル名")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_articles", type=int, default=3, help="出力用に処理する記事数（小さめ推奨）")
    parser.add_argument("--min_sentences", type=int, default=6, help="記事採用の最小文数")
    parser.add_argument("--output", type=str, default="outputs/nse_viterbi_chunks.txt", help="出力TXTパス")
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
        chunks, final_score = viterbi_chunk(sents, tokenizer, model, args.device)
        outputs.append(render_article_chunks(idx, sents, chunks, final_score))

    result_text = "\n".join(outputs)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(result_text)
    print(f"出力しました: {args.output}")


if __name__ == "__main__":
    main()


