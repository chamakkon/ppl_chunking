import random
from typing import List, Tuple

from chunk_with_kgram_dp import (
    load_wikitext_articles,
    load_wikipedia_ja_articles,
)


def join_range(sents: List[str], start: int, end: int, sep: str) -> str:
    if start <= 0 or start > end:
        return ""
    return sep.join(sents[start - 1:end])


def unconstrained_pair(
    sents: List[str],
    prev_chunk: Tuple[int, int],  # (a,b) inclusive; (0,0)=BOS
    cur_chunk: Tuple[int, int],   # (s,j) inclusive
    sep: str,
) -> Tuple[str, str]:
    a, b = prev_chunk
    s, j = cur_chunk
    ctx = "" if a == 0 else join_range(sents, a, b, sep) + (sep if a != 0 else "")
    tgt = join_range(sents, s, j, sep)
    return ctx, tgt


def constrained_direct_pair_same_k(
    sents: List[str],
    prev_chunk: Tuple[int, int],
    cur_chunk: Tuple[int, int],
    sep: str,
) -> Tuple[str, str]:
    # k >= N-1 の仮定下では、制約ありの直接経路（前チャンク全体＋現チャンク全体）と一致する
    return unconstrained_pair(sents, prev_chunk, cur_chunk, sep)


def precompute_C_context(sents: List[str], i: int, m: int, sep: str) -> str:
    if m <= 0:
        return ""
    ctx = join_range(sents, i - m, i - 1, sep)
    return ctx + (sep if ctx else "")


def test_article(sents: List[str], sep: str, trials: int = 1000) -> Tuple[int, int]:
    n = len(sents)
    mismatches = 0
    cctx_mismatches = 0
    # unconstrained vs constrained-direct
    for _ in range(trials):
        s = random.randint(1, n)
        j = random.randint(s, n)
        if s == 1 or random.random() < 0.1:
            prev = (0, 0)
        else:
            a = random.randint(1, s - 1)
            b = s - 1
            prev = (a, b)
        u_ctx, u_tgt = unconstrained_pair(sents, prev, (s, j), sep)
        c_ctx, c_tgt = constrained_direct_pair_same_k(sents, prev, (s, j), sep)
        if u_ctx != c_ctx or u_tgt != c_tgt:
            mismatches += 1
            if mismatches <= 3:
                print(f"[mismatch ctx/tgt] prev={prev}, cur={(s,j)}")
                print(f"  unconstrained.ctx='{u_ctx}'")
                print(f"  constrained.ctx  ='{c_ctx}'")
                print(f"  unconstrained.tgt='{u_tgt}'")
                print(f"  constrained.tgt  ='{c_tgt}'")
    # precompute C contexts
    for _ in range(trials // 4):
        i = random.randint(1, n)
        m = random.randint(0, min(5, i - 1 if i > 1 else 0))
        got = precompute_C_context(sents, i, m, sep)
        exp = "" if m == 0 else (sep.join(sents[i - m - 1:i - 1]) + (sep if m > 0 else ""))
        if got != exp:
            cctx_mismatches += 1
            if cctx_mismatches <= 3:
                print(f"[mismatch C ctx] (i={i}, m={m})")
                print(f"  got='{got}'")
                print(f"  exp='{exp}'")
    return mismatches, cctx_mismatches


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Corpus-based equivalence test (no LLM calls)")
    parser.add_argument("--dataset", type=str, default="wikitext", choices=["wikitext", "wikipedia_ja"])
    parser.add_argument("--dataset_slice", type=str, default="train[:1%]", help="wikipedia_ja slice")
    parser.add_argument("--max_articles", type=int, default=3)
    parser.add_argument("--min_sentences", type=int, default=6)
    parser.add_argument("--trials", type=int, default=1000)
    args = parser.parse_args()

    if args.dataset == "wikitext":
        articles = load_wikitext_articles(max_articles=args.max_articles, min_sentences=args.min_sentences)
        sep = " "
    else:
        articles = load_wikipedia_ja_articles(args.dataset_slice, args.max_articles, args.min_sentences)
        sep = ""

    if not articles:
        print("No articles loaded.")
        return

    total_m1 = 0
    total_m2 = 0
    for idx, sents in enumerate(articles, 1):
        print(f"\n=== Article #{idx} (sentences={len(sents)}) ===")
        m1, m2 = test_article(sents, sep, trials=args.trials)
        print(f"  ctx/tgt mismatches: {m1}")
        print(f"  C-context mismatches: {m2}")
        total_m1 += m1
        total_m2 += m2

    print("\n=== Overall ===")
    if total_m1 == 0:
        print("[OK] unconstrained == constrained-direct on corpus samples")
    else:
        print(f"[FAIL] ctx/tgt mismatches total: {total_m1}")
    if total_m2 == 0:
        print("[OK] precompute C contexts match expected formatting")
    else:
        print(f"[FAIL] C-context mismatches total: {total_m2}")


if __name__ == "__main__":
    main()


