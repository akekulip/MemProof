"""Final integrity audit of paper/memproof_full.tex.

Cross-checks claims in the paper against committed JSON files in
evaluation/ and reports any mismatches, dangling references, or
broken label/ref pairs. Run before building the submission bundle.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper" / "memproof_full.tex"
EVAL = ROOT / "evaluation"
BIBS = [ROOT / "paper" / "references.bib", ROOT / "paper" / "references_full.bib"]
FIGS = ROOT / "paper" / "figures"


def load_json(name: str):
    return json.loads((EVAL / name).read_text())


def sec(title: str) -> None:
    print()
    print(f"=== {title} ===")


def check(label: str, expected, actual, tol: float = 0) -> bool:
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        ok = abs(expected - actual) <= tol
    else:
        ok = expected == actual
    status = "OK  " if ok else "FAIL"
    print(f"  [{status}] {label}: paper={expected!r}  json={actual!r}")
    return ok


def main() -> int:
    paper = PAPER.read_text()
    fails = 0

    sec("Composition main-results (NQ GPT-3.5) vs composition_results_100_real.json")
    r = load_json("composition_results_100_real.json")
    pct = lambda k, f: int(round(r[k][f] * 100))
    # Paper table 2 row values
    table_pattern = re.compile(
        r"No Defense\s+&\s*(\d+)\s*&\s*(\d+).*?"
        r"RobustRAG Only\s+&\s*(\d+)\s*&\s*(\d+).*?"
        r"MemProof Only\s+&\s*\\textbf\{(\d+)\}\s*&\s*\\textbf\{(\d+)\}",
        re.DOTALL,
    )
    m = table_pattern.search(paper)
    if m:
        nd_a, nd_s, rr_a, rr_s, mp_a, mp_s = [int(x) for x in m.groups()]
        if not check("NQ No Defense acc", nd_a, pct("No Defense", "accuracy")): fails += 1
        if not check("NQ No Defense asr", nd_s, pct("No Defense", "asr")): fails += 1
        if not check("NQ RobustRAG Only acc", rr_a, pct("RobustRAG Only", "accuracy")): fails += 1
        if not check("NQ RobustRAG Only asr", rr_s, pct("RobustRAG Only", "asr")): fails += 1
        if not check("NQ MemProof Only acc", mp_a, pct("MemProof Only", "accuracy")): fails += 1
        if not check("NQ MemProof Only asr", mp_s, pct("MemProof Only", "asr")): fails += 1
    else:
        print("  [FAIL] could not parse main-results table")
        fails += 1

    sec("Insider threat vs insider_threat_100_real.json")
    r = load_json("insider_threat_100_real.json")
    if not check("insider MemProof Only acc",
                 51, pct("MemProof Only (insider)", "accuracy")): fails += 1
    if not check("insider MemProof Only asr",
                 17, pct("MemProof Only (insider)", "asr")): fails += 1
    if not check("insider MP+RobustRAG acc",
                 14, pct("MemProof + RobustRAG (insider)", "accuracy")): fails += 1
    if not check("insider MP+RobustRAG asr",
                 2, pct("MemProof + RobustRAG (insider)", "asr")): fails += 1

    sec("Ed25519 latency vs ed25519_latency.json")
    r = load_json("ed25519_latency.json")
    total = r["ingest_crypto_total_ms"]
    if not check("crypto total ms (~0.077)", 0.077, total, tol=0.005): fails += 1
    if not check("crypto overhead percent (~1.5)", 1.5, r["crypto_overhead_percent"], tol=0.2):
        fails += 1

    sec("Scale / tamper at 10K vs tamper_at_scale_10k.json")
    r = load_json("tamper_at_scale_10k.json")
    if not check("10K scan per entry (ms)", 0.113, r["scan_per_entry_ms"], tol=0.01): fails += 1
    if not check("10K ingest docs/s", 10864,
                 int(round(r["ingest_throughput_docs_per_s"])), tol=200): fails += 1

    sec("Scale / tamper at 100K vs tamper_at_scale_100k.json")
    r = load_json("tamper_at_scale_100k.json")
    if not check("100K scan per entry (ms)", 0.115, r["scan_per_entry_ms"], tol=0.01): fails += 1
    if not check("100K ingest docs/s", 9835,
                 int(round(r["ingest_throughput_docs_per_s"])), tol=200): fails += 1

    sec("Tail latency vs tail_latency.json / tail_latency_100k.json")
    r10 = load_json("tail_latency.json")["verification_latency_ms"]
    r100 = load_json("tail_latency_100k.json")["verification_latency_ms"]
    if not check("tail 10K mean (ms)", 0.126, r10["mean"], tol=0.003): fails += 1
    if not check("tail 10K p99 (ms)", 0.137, r10["p99"], tol=0.003): fails += 1
    if not check("tail 100K mean (ms)", 0.124, r100["mean"], tol=0.003): fails += 1
    if not check("tail 100K p99 (ms)", 0.134, r100["p99"], tol=0.003): fails += 1

    sec("Storage overhead vs storage_overhead.json")
    r = load_json("storage_overhead.json")
    if not check("1K JSON overhead %", 43.4,
                 round(r["n_1000"]["json"]["overhead_percent"], 1), tol=0.5): fails += 1
    if not check("10K JSON overhead %", 51.2,
                 round(r["n_10000"]["json"]["overhead_percent"], 1), tol=0.5): fails += 1
    if not check("100K JSON overhead %", 57.0,
                 round(r["n_100000"]["json"]["overhead_percent"], 1), tol=0.5): fails += 1
    if not check("1K binary overhead %", 32.1,
                 round(r["n_1000"]["binary"]["overhead_percent"], 1), tol=0.5): fails += 1
    if not check("10K binary overhead %", 39.9,
                 round(r["n_10000"]["binary"]["overhead_percent"], 1), tol=0.5): fails += 1
    if not check("100K binary overhead %", 45.7,
                 round(r["n_100000"]["binary"]["overhead_percent"], 1), tol=0.5): fails += 1

    sec("Concurrency vs concurrency_eval.json")
    r = load_json("concurrency_eval.json")
    if not check("1 worker throughput", 10096,
                 int(round(r["workers_1"]["throughput_docs_per_s"])), tol=100): fails += 1
    if not check("16 worker throughput", 4246,
                 int(round(r["workers_16"]["throughput_docs_per_s"])), tol=100): fails += 1

    sec("BM25 agnosticism vs bm25_retrieval_agnostic.json")
    r = load_json("bm25_retrieval_agnostic.json")
    if not check("BM25 alone queries with poison",
                 84, r["bm25_alone"]["queries_with_poison_in_topk"]): fails += 1
    if not check("BM25 + MemProof queries with poison",
                 0, r["bm25_plus_memproof"]["queries_with_poison_in_topk"]): fails += 1

    sec("NQ target overlap vs nq_target_overlap.json")
    r = load_json("nq_target_overlap.json")
    if not check("NQ target-string overlap count", 19, r["matches"]): fails += 1

    sec("Multi-embedder vs multi_embedder_latency.json")
    r = load_json("multi_embedder_latency.json")
    mini = r["embedders"]["all-MiniLM-L6-v2"]
    if not check("MiniLM crypto total (ms)", 0.078,
                 round(mini["crypto_total_ms"], 3), tol=0.015): fails += 1

    sec("Dangling LaTeX refs")
    labels = set(re.findall(r"\\label\{([^}]+)\}", paper))
    refs = set(re.findall(r"\\ref\{([^}]+)\}", paper))
    missing = refs - labels
    if missing:
        for m in sorted(missing):
            print(f"  [FAIL] \\ref{{{m}}} has no \\label")
            fails += 1
    else:
        print(f"  [OK] {len(refs)} refs all resolve to labels")

    sec("Citation keys vs bib files")
    cites = set()
    for m in re.finditer(r"\\cite\{([^}]+)\}", paper):
        for k in m.group(1).split(","):
            cites.add(k.strip())
    bib_keys = set()
    for bib in BIBS:
        if bib.exists():
            bib_keys |= set(re.findall(r"^@\w+\{([^,\s]+),", bib.read_text(), re.M))
    missing_cites = cites - bib_keys
    if missing_cites:
        for c in sorted(missing_cites):
            print(f"  [FAIL] \\cite{{{c}}} has no bib entry")
            fails += 1
    else:
        print(f"  [OK] {len(cites)} citations all resolve to bib entries")

    sec("Figure files")
    fig_refs = re.findall(r"\\includegraphics[^{]*\{figures/([^}]+)\}", paper)
    for f in fig_refs:
        base = f.replace(".pdf", "").replace(".png", "")
        pdf = FIGS / f"{base}.pdf"
        png = FIGS / f"{base}.png"
        if not (pdf.exists() or png.exists()):
            print(f"  [FAIL] figure {f} not found in paper/figures/")
            fails += 1
        else:
            print(f"  [OK] {f}")

    sec('"our" instances referring to MemProof (should be 0)')
    bad_patterns = [
        r"\bour protocol\b",
        r"\bour design\b",
        r"\bour approach\b",
        r"\bour proposal\b",
        r"\bour four layers\b",
        r"\bour architecture\b",
        r"\bour four-layer\b",
        r"\bour filter\b",
        r"\bOur approach\b",
        r"\bOur protocol\b",
    ]
    bad_hits = 0
    for pat in bad_patterns:
        for m in re.finditer(pat, paper):
            line_num = paper[:m.start()].count("\n") + 1
            print(f"  [FAIL] line {line_num}: {m.group()}")
            bad_hits += 1
    if bad_hits == 0:
        print("  [OK] no offending 'our <proposal>' matches")
    else:
        fails += bad_hits

    sec("AI writing patterns (informational only)")
    ai_patterns = [
        r"\bMoreover\b", r"\bFurthermore\b", r"\bparamount\b",
        r"\bseamlessly\b", r"\bcrucially\b", r"\btapestry\b",
        r"it is worth noting", r"it should be noted",
    ]
    for pat in ai_patterns:
        n = len(re.findall(pat, paper))
        if n:
            print(f"  [INFO] {pat}: {n} hits")

    sec("Summary")
    if fails == 0:
        print("  CLEAR TO BUILD ✓  (no blocking issues found)")
        return 0
    else:
        print(f"  {fails} BLOCKING ISSUES — do not build submission yet")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
