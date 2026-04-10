"""Latency benchmark for Ed25519-based MemProof crypto components.

Measures per-operation latency for:
  - Embedding (all-MiniLM-L6-v2)
  - Ed25519 signing (document attestation)
  - Ed25519 verification
  - HMAC-SHA256 embedding commitment (unchanged from old benchmark)
  - Merkle tree insert
  - Merkle proof generation
  - Merkle proof verification
  - Audit log append

Writes results to evaluation/ed25519_latency.json and updates
the latency breakdown figure.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from memproof.audit.log import AuditLog, OperationType
from memproof.crypto.attestation import (
    TrustLevel,
    attest_document,
    generate_source_keypair,
)
from memproof.crypto.commitment import commit_embedding, verify_embedding
from memproof.crypto.merkle import MerkleTree

N_ITERS = 1000
FIGURES_DIR = Path(__file__).parent.parent / "paper" / "figures"
RESULTS_PATH = Path(__file__).parent / "ed25519_latency.json"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
})
COL_W = 3.45


def bench(name: str, fn, iters: int = N_ITERS) -> float:
    """Run fn `iters` times, return mean latency in milliseconds."""
    # Warmup
    for _ in range(10):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    elapsed_ms = (time.perf_counter() - t0) * 1000 / iters
    print(f"  {name:<30} {elapsed_ms:.4f} ms/op ({iters} iters)")
    return elapsed_ms


def main() -> None:
    print("=" * 60)
    print("  Ed25519 MemProof Latency Benchmark")
    print("=" * 60)

    # Setup
    print("\nLoading embedding model...", flush=True)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")

    sample_text = "This is a sample document for latency benchmarking."
    sample_doc_hash = hashlib.sha256(sample_text.encode()).digest()

    # Embedding latency
    print("\n── Embedding ──")
    def _embed():
        return model.encode(sample_text, normalize_embeddings=True)
    embedding_ms = bench("Embedding (MiniLM-L6-v2)", _embed)
    sample_embedding = model.encode(sample_text, normalize_embeddings=True).tolist()

    # Ed25519 attestation
    print("\n── Ed25519 Attestation ──")
    priv, pub = generate_source_keypair()

    def _attest():
        return attest_document(
            content=sample_text,
            source_id="bench-src",
            trust_level=TrustLevel.HIGH,
            private_key=priv,
        )
    attest_ms = bench("Ed25519 sign (attestation)", _attest)

    cached_att = _attest()
    def _verify_att():
        return cached_att.verify(pub)
    verify_att_ms = bench("Ed25519 verify", _verify_att)

    # Commitment (HMAC-SHA256, unchanged)
    print("\n── Embedding Commitment (HMAC-SHA256) ──")
    hmac_key = os.urandom(32)
    def _commit():
        return commit_embedding(sample_doc_hash, sample_embedding, hmac_key)
    commit_ms = bench("HMAC commit", _commit)

    cached_commit = _commit()
    def _verify_commit():
        return verify_embedding(cached_commit, sample_embedding, hmac_key)
    verify_commit_ms = bench("HMAC verify", _verify_commit)

    # Merkle tree
    print("\n── Merkle Tree ──")
    tree = MerkleTree()
    for i in range(100):
        tree.insert(
            doc_hash=hashlib.sha256(f"seed-{i}".encode()).digest(),
            embedding_hash=hashlib.sha256(f"emb-{i}".encode()).digest(),
            provenance=f"prov-{i}".encode(),
        )

    counter = {"i": 100}
    def _merkle_insert():
        tree.insert(
            doc_hash=hashlib.sha256(f"bench-{counter['i']}".encode()).digest(),
            embedding_hash=hashlib.sha256(f"emb-{counter['i']}".encode()).digest(),
            provenance=f"prov-{counter['i']}".encode(),
        )
        counter["i"] += 1
    merkle_insert_ms = bench("Merkle insert", _merkle_insert, iters=500)

    def _merkle_prove():
        return tree.prove(50)
    merkle_prove_ms = bench("Merkle prove", _merkle_prove)

    proof = tree.prove(50)
    def _merkle_verify():
        return proof.verify()
    merkle_verify_ms = bench("Merkle verify", _merkle_verify)

    # Audit log
    print("\n── Audit Log ──")
    log = AuditLog()
    def _audit_append():
        log.append(OperationType.INGEST, sample_doc_hash)
    audit_ms = bench("Audit log append", _audit_append)

    # Summary
    results = {
        "iterations": N_ITERS,
        "embedding_ms": embedding_ms,
        "ed25519_sign_ms": attest_ms,
        "ed25519_verify_ms": verify_att_ms,
        "hmac_commit_ms": commit_ms,
        "hmac_verify_ms": verify_commit_ms,
        "merkle_insert_ms": merkle_insert_ms,
        "merkle_prove_ms": merkle_prove_ms,
        "merkle_verify_ms": merkle_verify_ms,
        "audit_append_ms": audit_ms,
    }

    # Per-entry totals (ingest path: embed + sign + commit + merkle + audit)
    ingest_crypto_ms = attest_ms + commit_ms + merkle_insert_ms + audit_ms
    ingest_total_ms = embedding_ms + ingest_crypto_ms
    query_crypto_ms = verify_att_ms + verify_commit_ms + merkle_prove_ms + merkle_verify_ms

    results["ingest_crypto_total_ms"] = ingest_crypto_ms
    results["ingest_total_ms"] = ingest_total_ms
    results["query_crypto_total_ms"] = query_crypto_ms
    results["crypto_overhead_percent"] = (ingest_crypto_ms / embedding_ms) * 100

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  Embedding (dominant):           {embedding_ms:.4f} ms")
    print(f"  Ingest crypto total:            {ingest_crypto_ms:.4f} ms")
    print(f"    Ed25519 sign:                 {attest_ms:.4f} ms")
    print(f"    HMAC commit:                  {commit_ms:.4f} ms")
    print(f"    Merkle insert:                {merkle_insert_ms:.4f} ms")
    print(f"    Audit append:                 {audit_ms:.4f} ms")
    print(f"  Crypto overhead vs embedding:   {results['crypto_overhead_percent']:.2f}%")
    print(f"  Query crypto total:             {query_crypto_ms:.4f} ms")
    print(f"    Ed25519 verify:               {verify_att_ms:.4f} ms")
    print(f"    HMAC verify:                  {verify_commit_ms:.4f} ms")
    print(f"    Merkle prove + verify:        {(merkle_prove_ms + merkle_verify_ms):.4f} ms")

    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {RESULTS_PATH}")

    # Figure: horizontal bar chart of ingest-path components
    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    components = ["Audit\nAppend", "Merkle\nInsert", "HMAC\nCommit", "Ed25519\nSign", "Embedding"]
    times = [audit_ms, merkle_insert_ms, commit_ms, attest_ms, embedding_ms]
    colors = ["#F59E0B", "#FFBB78", "#16A34A", "#8B5CF6", "#4878CF"]
    bars = ax.barh(components, times, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Latency (ms, log scale)")
    ax.set_xscale("log")
    ax.set_xlim(0.001, max(times) * 2)
    for bar, t in zip(bars, times):
        ax.text(bar.get_width() * 1.15, bar.get_y() + bar.get_height() / 2,
                f"{t:.4f}", va="center", fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig_path = FIGURES_DIR / "fig_real_latency.pdf"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path)
    plt.savefig(FIGURES_DIR / "fig_real_latency.png")
    plt.close()
    print(f"Figure saved to {fig_path}")


if __name__ == "__main__":
    main()
