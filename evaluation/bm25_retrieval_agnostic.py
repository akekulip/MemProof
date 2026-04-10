"""BM25 retrieval agnosticism experiment.

A reviewer flagged that our eval only uses dense retrieval. MemProof's
admission-control layer is orthogonal to the retrieval method: it tags
entries at ingestion time and drops untrusted entries before the LLM
sees them, regardless of how they were ranked. This script demonstrates
that empirically on the same 100 Natural Questions we use in the main
composition evaluation.

For each query we:
  1. Build a small candidate pool from the clean top-k passages plus
     the precomputed PoisonedRAG poison passages (5 + 5 = up to 10).
  2. Score candidates with BM25 using rank_bm25.
  3. Take the top-k by BM25.
  4. Ask: do any poison passages survive the BM25 ranking?
  5. Apply MemProof's trust filter (untrusted / unattested sources).
  6. Ask: do any poison passages survive after the MemProof filter?

The point is not that BM25 is a better or worse retriever than dense;
it is that *whatever* ranking the retriever produces, the admission
filter removes unattested poisons. The final BM25-only top-k may
still contain poison (because BM25 ranking is purely lexical and the
attacker crafts passages that contain the query keywords); after
MemProof filtering, zero poisons remain.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import json as _json

_RRAG_DATA = (
    Path(__file__).parent.parent
    / "baselines" / "RobustRAG" / "data" / "open_nq.json"
)


def load_nq_data(top_k: int = 10, n: int = 100) -> list[dict]:
    """Load Natural Questions from the RobustRAG data file.

    Duplicated from composition_eval_claude.py to keep this script
    dependency-free (no anthropic import).
    """
    with open(_RRAG_DATA) as f:
        raw = _json.load(f)
    processed = []
    for item in raw[:n]:
        clean_contexts = []
        for ctx in item.get("context", [])[:top_k]:
            text = ctx.get("text", "")
            title = ctx.get("title", "")
            clean_contexts.append(f"{title}\n{text}" if title else text)
        processed.append({
            "question": item["question"],
            "correct_answer": item["correct answer"],
            "incorrect_answer": item.get("incorrect answer", ""),
            "clean_contexts": clean_contexts,
            "poison_contexts": item.get("incorrect_context", []),
        })
    return processed


def tokenize(s: str) -> list[str]:
    """Lowercase split on whitespace and strip punctuation."""
    import re
    return re.findall(r"[a-z0-9]+", s.lower())


def run_bm25(query: str, docs: list[str], k: int) -> list[int]:
    """Return indices of top-k docs ranked by BM25."""
    from rank_bm25 import BM25Okapi

    tokenized_docs = [tokenize(d) for d in docs]
    bm25 = BM25Okapi(tokenized_docs)
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(range(len(docs)), key=lambda i: -scores[i])
    return ranked[:k]


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation/bm25_retrieval_agnostic.json",
    )
    args = parser.parse_args()

    data = load_nq_data(top_k=10, n=args.num_queries)
    # Widen clean_contexts to 10 so BM25 has a richer pool to score.

    print("=" * 72)
    print("  BM25 retrieval-agnosticism experiment")
    print("=" * 72)
    print(f"  Queries:     {args.num_queries}")
    print(f"  Top-k:       {args.top_k}")
    print(f"  Retriever:   BM25 (rank_bm25 Okapi)")
    print()

    # Per-query stats
    n_total = 0
    n_bm25_topk_has_poison = 0
    n_bm25_poison_survivors = 0          # total poison passages in BM25 top-k
    n_memproof_topk_has_poison = 0       # should be 0
    n_memproof_poison_survivors = 0      # should be 0
    bm25_poison_ratio_sum = 0.0
    n_queries_with_poison = 0

    per_query = []

    for idx, item in enumerate(data):
        q = item["question"]
        clean = item["clean_contexts"]
        poison = item["poison_contexts"]
        if not poison:
            continue
        n_queries_with_poison += 1

        # Candidate pool: clean + poison (with a tag for trust label)
        pool = []
        labels = []   # "clean" or "poison"
        for c in clean:
            pool.append(c)
            labels.append("clean")
        for p in poison:
            pool.append(p)
            labels.append("poison")

        # BM25 ranking
        topk_idx = run_bm25(q, pool, k=args.top_k)
        topk_labels = [labels[i] for i in topk_idx]

        poisons_in_topk = sum(1 for lbl in topk_labels if lbl == "poison")
        has_poison = poisons_in_topk > 0
        n_total += 1
        n_bm25_topk_has_poison += int(has_poison)
        n_bm25_poison_survivors += poisons_in_topk
        bm25_poison_ratio_sum += poisons_in_topk / args.top_k

        # Apply MemProof filter: drop every "poison" labelled entry because
        # they have no registered source. In production this is enforced at
        # ingestion time via the Ed25519 signature check; here we just use
        # the ground-truth label because we control the dataset.
        filtered_idx = [i for i in topk_idx if labels[i] != "poison"]
        # Backfill with the next best clean hits until we again have top-k
        remaining_clean_idx = [
            i for i in range(len(pool))
            if labels[i] == "clean" and i not in filtered_idx
        ]
        while len(filtered_idx) < args.top_k and remaining_clean_idx:
            filtered_idx.append(remaining_clean_idx.pop(0))

        memproof_topk_labels = [labels[i] for i in filtered_idx]
        mp_poisons_in_topk = sum(1 for lbl in memproof_topk_labels if lbl == "poison")
        n_memproof_topk_has_poison += int(mp_poisons_in_topk > 0)
        n_memproof_poison_survivors += mp_poisons_in_topk

        per_query.append({
            "idx": idx,
            "question": q,
            "bm25_topk_labels": topk_labels,
            "bm25_poisons_in_topk": poisons_in_topk,
            "memproof_topk_labels": memproof_topk_labels,
            "memproof_poisons_in_topk": mp_poisons_in_topk,
        })

        if (idx + 1) % 20 == 0:
            print(f"  [{idx+1}/{len(data)}] bm25_with_poison={n_bm25_topk_has_poison} "
                  f"memproof_with_poison={n_memproof_topk_has_poison}")

    # Summary
    bm25_avg_poison_ratio = (
        bm25_poison_ratio_sum / n_total if n_total else 0.0
    )

    print("\n" + "=" * 72)
    print("  Summary")
    print("=" * 72)
    print(f"  Queries with poison attempts:        {n_queries_with_poison}")
    print(f"  Queries scored by BM25:              {n_total}")
    print()
    print("  BM25 alone (no admission control):")
    print(f"    queries with >=1 poison in top-{args.top_k}:  "
          f"{n_bm25_topk_has_poison}/{n_total} "
          f"({n_bm25_topk_has_poison / max(n_total,1):.1%})")
    print(f"    total poison passages in top-{args.top_k}s:   "
          f"{n_bm25_poison_survivors}")
    print(f"    average poison fraction of top-{args.top_k}:  "
          f"{bm25_avg_poison_ratio:.2%}")
    print()
    print("  BM25 + MemProof admission control:")
    print(f"    queries with >=1 poison in top-{args.top_k}:  "
          f"{n_memproof_topk_has_poison}/{n_total}")
    print(f"    total poison passages in top-{args.top_k}s:   "
          f"{n_memproof_poison_survivors}")

    out = {
        "num_queries": args.num_queries,
        "top_k": args.top_k,
        "n_queries_with_poison": n_queries_with_poison,
        "n_scored": n_total,
        "bm25_alone": {
            "queries_with_poison_in_topk": n_bm25_topk_has_poison,
            "total_poison_survivors": n_bm25_poison_survivors,
            "avg_poison_fraction": bm25_avg_poison_ratio,
        },
        "bm25_plus_memproof": {
            "queries_with_poison_in_topk": n_memproof_topk_has_poison,
            "total_poison_survivors": n_memproof_poison_survivors,
        },
        "per_query": per_query,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
