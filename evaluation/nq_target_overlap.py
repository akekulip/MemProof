"""Count NQ queries where the PoisonedRAG target answer string appears
naturally in the clean retrieval context.

Reviewers questioned the paper's claim that string-match ASR has a
built-in false-positive floor on Natural Questions. This script makes
that claim auditable by counting, over the first 100 open_nq queries
from RobustRAG's data file, how many have the 'incorrect answer'
(PoisonedRAG target) string appearing in any of the top-10 clean
context passages that come with the dataset.

Writes evaluation/nq_target_overlap.json with the count and a list of
matching (index, question, target) tuples.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "baselines" / "RobustRAG" / "data" / "open_nq.json"
OUT_PATH = Path(__file__).parent / "nq_target_overlap.json"


def main() -> None:
    raw = json.loads(DATA_PATH.read_text())
    hits: list[dict] = []
    for i, item in enumerate(raw[:100]):
        incorrect = item.get("incorrect answer", "").strip().lower()
        if not incorrect:
            continue
        clean_contexts = item.get("context", [])[:10]
        for ctx in clean_contexts:
            text = (ctx.get("text", "") + " " + ctx.get("title", "")).lower()
            if incorrect in text:
                hits.append({
                    "idx": i,
                    "question": item["question"],
                    "target_string": item.get("incorrect answer", ""),
                    "correct_answer": item.get("correct answer", []),
                })
                break

    total = sum(1 for item in raw[:100] if item.get("incorrect answer", "").strip())
    print(f"Examined {total} queries with non-empty target answer")
    print(f"Queries where target string appears in clean context: {len(hits)}/{total}")
    print()
    print("First 10 matches:")
    for h in hits[:10]:
        print(f"  idx={h['idx']:3d}  target={h['target_string']!r:<25}  q={h['question'][:60]!r}")

    out = {
        "total_examined": total,
        "matches": len(hits),
        "hits": hits,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
