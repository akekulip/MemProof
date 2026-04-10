# MemProof: Cryptographic Admission Control Against External Corpus Poisoning in RAG

Reference implementation, evaluation scripts, and JSON outputs for the
MemProof paper. Code release for ACM TAISAP submission.

Repository: <https://github.com/akekulip/MemProof>

## What MemProof actually does

MemProof is an admission-control protocol for one specific subproblem:
**external corpus poisoning by sources the operator never authorized**.
It does not try to solve every RAG-poisoning threat. The contribution
is the framing plus a cryptographic enforcement layer that, on the
external-poisoning subproblem, matches the attack-success reduction of
generation-side defenses without paying the generation-side accuracy
cost.

The protocol has four layers, but the empirically load-bearing one
against PoisonedRAG-style external poisoning is Layer 1:

1. **Layer 1 — Provenance-verified ingestion.** Every document must
   carry an Ed25519 signature from a source registered in the
   SourceRegistry. Unsigned documents never reach retrieval. This is
   the layer that drives the headline ASR result in Table 2 of the
   paper.
2. **Layer 2 — Embedding integrity.** HMAC-SHA256 commitment binding
   the embedding to the document. Defense-in-depth against a
   storage-layer attacker who can swap embeddings without touching
   content.
3. **Layer 3 — Temporal audit.** Hash-chained, monotonic-timestamp
   audit log of every ingest, update, delete, and query operation.
4. **Layer 4 — Verified retrieval.** Merkle membership proof attached
   to each retrieved entry; supports public verification by a
   third-party auditor given only the registry and a Merkle root.

The paper is explicit: a compromised trusted source bypasses Layer 1,
and the insider-threat experiment in Section 6.5 shows MemProof
degrading to No Defense in that regime. The right tool for that
threat is RobustRAG or another content-level defense.

## Repository layout

```
src/memproof/
  crypto/
    attestation.py     Ed25519 source attestation, JSON + binary serialization
    commitment.py      HMAC-SHA256 embedding commitment
    merkle.py          Incremental O(log n) Merkle tree
    source_registry.py SourceRegistry: source -> public key + trust level
  audit/
    log.py             Hash-chained audit log with monotonicity check
  ingestion/
    pipeline.py        Layer 1 + 2 + 3 ingestion path
  retrieval/
    verified.py        Layer 4 query-time verification
  backends/
    chroma.py          ChromaDB-backed vector store wrapper
  store.py             High-level MemProofStore convenience API

tests/
  test_memproof.py     35 unit tests; round-trip, signature, monotonicity,
                       binary encoding regression tests

evaluation/
  composition_eval.py            Main 4-way comparison on NQ (real Ed25519 path)
  composition_eval_realtimeqa.py Same comparison on RealtimeQA
  composition_eval_claude.py     Same on NQ with Claude Haiku 4.5 generator
  insider_threat_eval.py         Insider-threat experiment (signed poison)
  poisonedrag_eval.py            PoisonedRAG attack reproduction
  regrade_semantic.py            GPT-4 + Claude + SBERT re-grading of responses
  semantic_asr.py                Judge implementations
  nq_target_overlap.py           Measures the 19/100 string-match false-positive count
  tamper_at_scale.py             10K and 100K corpus tamper detection
  tail_latency.py                p50/p95/p99 verification latency at scale
  ed25519_latency.py             Per-operation latency benchmark
  concurrency_eval.py            Lock contention under N writers
  storage_overhead.py            Per-entry storage, JSON + binary attestations
  multi_embedder_latency.py      Crypto overhead across MiniLM, mpnet, BGE
  bm25_retrieval_agnostic.py     BM25 retrieval + admission filter
  ann_rebuild_compat.py          ANN index rebuild from canonical entries
  meaningful_benchmark.py        Tamper + bounded-influence ablation
  generate_real_figures.py       Publication figures from JSON outputs
  generate_architecture.py       Architecture diagram (Figure 1)
  final_audit.py                 Self-checking audit: every paper number vs JSON

paper/
  memproof_full.tex     Full manuscript
  references_full.bib   Bibliography
  figures/              All figures referenced by the paper

submission/
  cover_letter.tex      TAISAP cover letter
  title_page.tex        Title page

build_submission.sh     Builds memproof_overleaf.zip + memproof_code.zip
                        gated on final_audit.py + pytest passing
```

## Quick start

```bash
# Python 3.10+, uv recommended
uv run --with openai python -m pytest tests/test_memproof.py -q

# Smoke test the protocol
uv run python -c "
from memproof.store import MemProofStore
from memproof.crypto.attestation import TrustLevel

def embed(text): return [hash(text) % 1000 / 1000.0] * 384
store = MemProofStore(embed_fn=embed)
store.register_source('wiki', TrustLevel.HIGH)
store.ingest('Paris is the capital of France', source_id='wiki')
print('ingested:', store.entry_count, 'entries; root hex:', store.merkle_root.hex()[:16])
"
```

## Reproducing the paper

Every numeric claim in the paper traces to a JSON file in
`evaluation/`. The mapping is in **Appendix A** of the manuscript and
in `evaluation/final_audit.py`. Random seeds are fixed at 42
throughout.

```bash
# Cross-check every paper number against the committed JSONs
uv run python evaluation/final_audit.py
# expected: "CLEAR TO BUILD" (30 numbers, 47 refs, 61 citations)

# Run the unit tests
uv run --with pytest python -m pytest tests/test_memproof.py -q
# expected: 35 passed
```

To rerun an end-to-end experiment, set the relevant API key (OpenAI
for GPT-3.5, Anthropic for Claude Haiku) and run the script. For
example:

```bash
export OPENAI_API_KEY=sk-...
uv run --with openai python evaluation/composition_eval.py \
    --num-queries 100 --output evaluation/composition_results_100_real.json
```

## Headline numbers

All measured end-to-end on the real Ed25519 verify pipeline,
seeded at 42. NQ and RealtimeQA use 100 queries each. Storage is
measured at three corpus sizes from 1,000 to 100,000 synthetic
entries.

| Configuration | NQ Acc | NQ ASR | RTQA Acc | RTQA ASR |
|---|---|---|---|---|
| No Defense | 52% | 17% | 41% | 43% |
| RobustRAG Only | 14% | 2% | 22% | 1% |
| **MemProof Only** | **55%** | **2%** | **59%** | **0%** |
| MemProof + RobustRAG | 15% | 1% | 24% | 0% |

Cross-family validation on Claude Haiku 4.5 (NQ, 100 queries):
MemProof 62% / 2% vs RobustRAG 8% / 1%. Same Pareto pattern.

| Overhead | JSON attestation | Binary attestation |
|---|---|---|
| Crypto / ingest | 0.077 ms (1.5% of embedding) | (same) |
| Storage @ 1K | 43.4% | 32.1% |
| Storage @ 10K | 51.2% | 39.9% |
| Storage @ 100K | 57.0% | 45.7% |

## Honest limitations

1. **Trust labels are the unsolved problem.** MemProof's guarantees
   are conditional on correct trust labels at ingestion time. The
   paper does not solve trust-policy management.
2. **Insider threat is out of scope by design.** A trusted source
   that publishes adversarial content bypasses Layer 1. Section 6.5
   measures this and reports MemProof degrading to No Defense.
3. **Bounded-influence theorem retracted.** The earlier draft proposed
   a theorem that does not hold against embeddings optimized for
   target queries; the 48× empirical gap is reported in
   Section 4.2.
4. **Evaluation scale.** 100 queries per dataset, single LLM
   per-row in Table 2 (with one cross-family Claude column),
   500–1000 passages per query context drawn from the
   RobustRAG-preprocessed subset rather than a 2.6M-passage live
   corpus. Section 6.1 explains the choice.

## Citation

```
@article{memproof2026,
  title  = {MemProof: Cryptographic Admission Control Against External Corpus Poisoning in Retrieval-Augmented Generation},
  author = {Akekudaga, Philip},
  journal = {Submission to ACM TAISAP},
  year   = {2026},
  url    = {https://github.com/akekulip/MemProof}
}
```

## License

Apache License 2.0. See `LICENSE`.
