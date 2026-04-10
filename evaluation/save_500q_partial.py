"""Parse the partial 500-query run log into a JSON snapshot.

The 500-query NQ rerun (composition_eval.py --num-queries 500) was
killed mid-Config-4 because OpenAI rate-limited the 5x RobustRAG
isolation calls and the remaining wall time was 1-2 hours. This
script extracts every aggregate result that did finish, plus the
last observed partial counters for the in-flight config, and writes
them to evaluation/composition_results_500_partial.json so the
state is durable across sessions.

Run this once before killing the process. Tomorrow's resume is a
fresh full rerun (this is just evidence + a starting position).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

LOG = Path(__file__).parent / "_500q_run.log"
OUT = Path(__file__).parent / "composition_results_500_partial.json"


def main() -> None:
    text = LOG.read_text()
    lines = text.splitlines()

    completed: dict[str, dict] = {}
    current_cfg: str | None = None
    last_partial = None

    cfg_re = re.compile(r"Config:\s*(.+)$")
    result_re = re.compile(
        r"RESULT:\s*Accuracy=([\d.]+)%\s*\((\d+)/(\d+)\),\s*ASR=([\d.]+)%\s*\((\d+)/(\d+)\)"
    )
    progress_re = re.compile(r"\[(\d+)/(\d+)\]\s*Acc=(\d+)/(\d+)\s*ASR=(\d+)/(\d+)")

    for line in lines:
        m = cfg_re.search(line)
        if m:
            current_cfg = m.group(1).strip()
            continue
        m = result_re.search(line)
        if m and current_cfg:
            acc_pct, acc_n, total, asr_pct, asr_n, _ = m.groups()
            completed[current_cfg] = {
                "n": int(total),
                "accuracy_count": int(acc_n),
                "asr_count": int(asr_n),
                "accuracy": float(acc_pct) / 100,
                "asr": float(asr_pct) / 100,
                "status": "complete",
            }
            current_cfg = None  # next "Config:" header starts the next one
            continue
        m = progress_re.search(line)
        if m and current_cfg:
            pos, total, acc_n, _, asr_n, _ = m.groups()
            last_partial = {
                "config": current_cfg,
                "position": int(pos),
                "total_planned": int(total),
                "accuracy_count": int(acc_n),
                "asr_count": int(asr_n),
                "accuracy": int(acc_n) / int(pos),
                "asr": int(asr_n) / int(pos),
                "status": "partial (killed: rate-limited)",
            }

    snapshot = {
        "source": "evaluation/_500q_run.log",
        "intent": "5x scale-up of NQ composition eval (real Ed25519 path)",
        "killed_reason": "OpenAI rate limit on 5-prompt RobustRAG isolation; ~2h wall time remaining",
        "completed_configs": completed,
        "in_flight_config": last_partial,
    }
    OUT.write_text(json.dumps(snapshot, indent=2))
    print(f"Saved snapshot to {OUT}")
    print()
    print("Completed configs at n=500:")
    for cfg, r in completed.items():
        print(f"  {cfg:<25} acc={r['accuracy']:.1%}  asr={r['asr']:.1%}  "
              f"({r['accuracy_count']}/{r['n']} acc, {r['asr_count']}/{r['n']} asr)")
    if last_partial:
        c = last_partial
        print(f"\nIn-flight (killed at {c['position']}/{c['total_planned']}):")
        print(f"  {c['config']:<25} acc={c['accuracy']:.1%}  asr={c['asr']:.1%}  "
              f"({c['accuracy_count']}/{c['position']} acc, {c['asr_count']}/{c['position']} asr)")


if __name__ == "__main__":
    main()
