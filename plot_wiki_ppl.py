import math
import re
from typing import List, Tuple, Dict

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt


def split_sentences(text: str) -> List[str]:
    """
    言語非依存の簡易文分割。
    - 日本語句読点: 。！？？
    - 英語: .!? （連続句読点や空白も考慮）
    """
    if not text or not text.strip():
        return []
    # 改行はスペース化し、余分な空白を潰す
    normalized = re.sub(r"\s+", " ", text.strip())
    # 末尾の区切りまで1文として貪欲に取る
    pattern = r"[^。．\.!?！？]+[。．\.!?！？]?"
    sentences = [s.strip() for s in re.findall(pattern, normalized) if s and s.strip()]
    # 短すぎる破片は除外
    sentences = [s for s in sentences if len(s) > 1]
    return sentences


def load_wikitext_articles(max_articles: int = 10, min_sentences: int = 6) -> List[List[str]]:
    """
    小規模wikipedia相当データとして wikitext-2-raw-v1 を使用。
    行テキストから空行で記事を区切り、文分割して返す。
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

    # 最後の残りを処理
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


def compute_loss_and_ppl(
    tokenizer,
    model,
    context: str,
    target: str,
    device: str
) -> Tuple[float, float]:
    """
    teacher forcing によるターゲット文の平均CEとPPLを返す。
    """
    context_ids = tokenizer.encode(context, add_special_tokens=True)
    target_ids = tokenizer.encode(target, add_special_tokens=False)

    if len(target_ids) == 0:
        return float("nan"), float("nan")

    input_ids = torch.tensor([context_ids + target_ids], device=device)
    labels = input_ids.clone()
    labels[:, :len(context_ids)] = -100
    if tokenizer.pad_token_id is not None:
        labels[labels == tokenizer.pad_token_id] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss.item()  # 平均CE（ターゲット部）
    ppl = math.exp(loss)
    return loss, ppl


def aggregate_over_articles(
    articles: List[List[str]],
    tokenizer,
    model,
    device: str,
    max_context_sentences: int = 10
) -> Tuple[List[int], List[float], List[float], List[float], List[float]]:
    """
    記事ごとに、コンテキスト文数を1..kと増やしながら
    - 平均PPL（mean）と標準偏差（std）
    - 正規化エントロピー（loss / ln(vocab)）のmean,std
    を集計する。
    返り値: xs, ppl_mean, ppl_std, nent_mean, nent_std
    """
    vocab_size = len(tokenizer)
    log_vocab = math.log(vocab_size)

    # i番目（1-based）のコンテキスト長に対する全記事結果を蓄積
    ppl_buckets: Dict[int, List[float]] = {i: [] for i in range(1, max_context_sentences + 1)}
    nent_buckets: Dict[int, List[float]] = {i: [] for i in range(1, max_context_sentences + 1)}

    for sents in articles:
        # その記事で扱う最大コンテキスト長
        k = min(max_context_sentences, max(1, len(sents) - 1))
        # 1..k の各iについて、最初のi文をコンテキストに、次の1文をターゲットに
        for i in range(1, k + 1):
            context = "".join(sents[:i])
            target = sents[i]
            loss, ppl = compute_loss_and_ppl(tokenizer, model, context, target, device)
            if not math.isnan(loss) and not math.isnan(ppl) and math.isfinite(loss) and math.isfinite(ppl):
                ppl_buckets[i].append(ppl)
                nent_buckets[i].append(loss / log_vocab)

    xs = []
    ppl_mean, ppl_std = [], []
    nent_mean, nent_std = [], []

    for i in range(1, max_context_sentences + 1):
        vals_ppl = ppl_buckets[i]
        vals_nent = nent_buckets[i]
        if len(vals_ppl) == 0:
            continue
        xs.append(i)
        # 平均と不偏標準偏差
        m_ppl = sum(vals_ppl) / len(vals_ppl)
        m_nent = sum(vals_nent) / len(vals_nent)
        s_ppl = (sum((v - m_ppl) ** 2 for v in vals_ppl) / (len(vals_ppl) - 1)) ** 0.5 if len(vals_ppl) > 1 else 0.0
        s_nent = (sum((v - m_nent) ** 2 for v in vals_nent) / (len(vals_nent) - 1)) ** 0.5 if len(vals_nent) > 1 else 0.0
        ppl_mean.append(m_ppl)
        ppl_std.append(s_ppl)
        nent_mean.append(m_nent)
        nent_std.append(s_nent)

    return xs, ppl_mean, ppl_std, nent_mean, nent_std


def main():
    import argparse

    parser = argparse.ArgumentParser(description="コンテキスト長を増やしたときのPPLと正規化エントロピー推移をプロット")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-0.5B", help="Hugging Face モデル名")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_articles", type=int, default=10, help="使用する記事数（最大）")
    parser.add_argument("--min_sentences", type=int, default=6, help="記事として採用する最小文数")
    parser.add_argument("--max_context_sentences", type=int, default=10, help="コンテキストに使う最大文数")
    parser.add_argument("--output", type=str, default="wiki_ppl_plot.png", help="出力PNGファイル名")
    args = parser.parse_args()

    print("記事の読み込み中（wikitext-2-raw-v1）...")
    articles = load_wikitext_articles(max_articles=args.max_articles, min_sentences=args.min_sentences)
    if len(articles) == 0:
        print("条件を満たす記事が見つかりませんでした。パラメータを下げて再実行してください。")
        return
    print(f"記事数: {len(articles)}")

    print(f"モデルをロード中: {args.model}")
    tokenizer, model = load_model(args.model, args.device)

    print("計測中...")
    xs, ppl_mean, ppl_std, nent_mean, nent_std = aggregate_over_articles(
        articles, tokenizer, model, args.device, max_context_sentences=args.max_context_sentences
    )

    if len(xs) == 0:
        print("十分なデータ点が得られませんでした。")
        return

    # プロット
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), dpi=150)

    # PPL
    axs[0].plot(xs, ppl_mean, label="mean PPL", color="#1f77b4")
    axs[0].fill_between(
        xs,
        [m - s for m, s in zip(ppl_mean, ppl_std)],
        [m + s for m, s in zip(ppl_mean, ppl_std)],
        color="#1f77b4",
        alpha=0.2,
        label="±1 std"
    )
    axs[0].set_xlabel("コンテキスト文数")
    axs[0].set_ylabel("Perplexity")
    axs[0].set_title("コンテキスト増加に伴うPPLの推移")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()

    # 正規化エントロピー
    axs[1].plot(xs, nent_mean, label="mean normalized entropy", color="#ff7f0e")
    axs[1].fill_between(
        xs,
        [m - s for m, s in zip(nent_mean, nent_std)],
        [m + s for m, s in zip(nent_mean, nent_std)],
        color="#ff7f0e",
        alpha=0.2,
        label="±1 std"
    )
    axs[1].set_xlabel("コンテキスト文数")
    axs[1].set_ylabel("Normalized entropy (CE / ln|V|)")
    axs[1].set_ylim(0.0, 1.0)
    axs[1].set_title("コンテキスト増加に伴う正規化エントロピーの推移")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(args.output)
    print(f"図を保存しました: {args.output}")


if __name__ == "__main__":
    main()


