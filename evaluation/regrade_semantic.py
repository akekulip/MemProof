"""Re-grade saved composition responses with semantic ASR metrics.

Takes a per-query log from composition_eval.py and re-scores attack
success using three semantic metrics (GPT-4 judge, Claude Haiku judge,
SBERT similarity) in addition to the original string-match verdict.

Cost estimate: 100 queries x 4 configs x 2 LLM judges = 800 judge calls.
At GPT-4-turbo (~$0.01 per call) + Claude Haiku (~$0.0005 per call),
total cost is roughly $4 for the full re-grading.

Output: evaluation/composition_regraded_100.json with per-query
verdicts from all four metrics, and aggregate ASR per config per metric.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from semantic_asr import (
    GPT4Judge,
    ClaudeJudge,
    SBERTJudge,
    evaluate_response,
    aggregate_metrics,
)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="evaluation/composition_results_100_queries.json",
        help="Per-query log from composition_eval.py",
    )
    parser.add_argument(
        "--output",
        default="evaluation/composition_regraded_100.json",
        help="Output path for re-graded verdicts",
    )
    parser.add_argument("--skip-gpt4", action="store_true")
    parser.add_argument("--skip-claude", action="store_true")
    args = parser.parse_args()

    input_path = Path(__file__).parent.parent / args.input
    output_path = Path(__file__).parent.parent / args.output

    print(f"Loading per-query log from {input_path}")
    per_query = json.loads(input_path.read_text())

    print("Initializing judges...")
    gpt4 = None if args.skip_gpt4 else GPT4Judge()
    claude = None if args.skip_claude else ClaudeJudge()
    sbert = SBERTJudge()

    regraded = {}
    t_start = time.time()

    for config_name, queries in per_query.items():
        print(f"\n── {config_name} ── ({len(queries)} queries)")
        results = []
        for i, q in enumerate(queries):
            if not q.get("incorrect_answer"):
                continue
            result = evaluate_response(
                question=q["question"],
                correct_answer=q["correct_answer"],
                incorrect_answer=q["incorrect_answer"],
                response=q["response"],
                gpt4_judge=gpt4,
                claude_judge=claude,
                sbert_judge=sbert,
            )
            results.append(result)
            if (i + 1) % 10 == 0:
                elapsed = time.time() - t_start
                print(f"  [{i+1}/{len(queries)}] elapsed={elapsed:.0f}s")

        metrics = aggregate_metrics(results)
        regraded[config_name] = {
            "metrics": metrics,
            "per_query": [r.to_dict() for r in results],
        }
        print(f"  ASR: string={metrics.get('string_match_asr', 0):.0%} "
              f"gpt4={metrics.get('gpt4_asr') or 0:.0%} "
              f"claude={metrics.get('claude_asr') or 0:.0%} "
              f"sbert={metrics.get('sbert_asr') or 0:.0%}")

    output_path.write_text(json.dumps(regraded, indent=2, default=str))
    print(f"\nResults saved to {output_path}")

    print("\n" + "=" * 72)
    print("  SEMANTIC ASR SUMMARY (per defense config)")
    print("=" * 72)
    print(f"  {'Config':<28} {'String':>9} {'GPT-4':>9} {'Claude':>9} {'SBERT':>9} {'Agree':>9}")
    print(f"  {'-' * 28} {'-' * 9} {'-' * 9} {'-' * 9} {'-' * 9} {'-' * 9}")
    for name, data in regraded.items():
        m = data["metrics"]
        def fmt(x):
            return "---" if x is None else f"{x:.0%}"
        print(f"  {name:<28} {fmt(m.get('string_match_asr')):>9} "
              f"{fmt(m.get('gpt4_asr')):>9} "
              f"{fmt(m.get('claude_asr')):>9} "
              f"{fmt(m.get('sbert_asr')):>9} "
              f"{fmt(m.get('gpt4_claude_agreement')):>9}")
    print("=" * 72)

    # Identify queries where metrics disagree (useful for paper)
    print("\nQueries where string-match says ATTACK but semantic judges say NO:")
    for config_name, data in regraded.items():
        for r in data["per_query"]:
            string_asr = r.get("string_match") or r.get("string_asr")
            gpt4_asr = r.get("gpt4_asr")
            claude_asr = r.get("claude_asr")
            if string_asr and gpt4_asr is False and claude_asr is False:
                q = r.get("question", "")[:60]
                incorrect = r.get("incorrect_answer", "")[:40]
                print(f"  [{config_name:<20}] {q}...")
                print(f"      incorrect: {incorrect}")


if __name__ == "__main__":
    main()
