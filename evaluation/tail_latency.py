"""Tail latency measurement.

Populate a store with 10K entries, then issue 1000 verification queries
(Layer 2 + Layer 4 checks) and measure p50, p95, p99, max.

This tests the query-path verification cost without going through an
LLM, isolating MemProof's own overhead.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from memproof.crypto.attestation import (
    TrustLevel,
    attest_document,
    generate_source_keypair,
)
from memproof.crypto.commitment import commit_embedding, verify_embedding
from memproof.crypto.merkle import MerkleTree
from memproof.crypto.source_registry import SourceRegistry


def _synth_embed(text: str, dim: int = 384) -> list[float]:
    h = hashlib.sha256(text.encode()).digest()
    raw = (h * (dim // 32 + 1))[: dim]
    return [(b / 255.0) * 2.0 - 1.0 for b in raw]


def pct(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = int(len(sorted_vals) * p)
    idx = min(max(idx, 0), len(sorted_vals) - 1)
    return sorted_vals[idx]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--output", type=str, default="evaluation/tail_latency.json")
    args = parser.parse_args()

    print(f"=== Tail latency measurement (n={args.n}, queries={args.queries}) ===")

    # Setup
    registry = SourceRegistry()
    priv, pub = registry.register_source("corpus", TrustLevel.HIGH)
    commit_key = os.urandom(32)
    tree = MerkleTree()
    store: list[dict] = []

    # Populate
    print(f"\n[1/2] Populating {args.n}-entry store...")
    t0 = time.perf_counter()
    for i in range(args.n):
        doc = f"Document {i} with synthetic content {i % 13}"
        doc_hash = hashlib.sha256(doc.encode()).digest()
        emb = _synth_embed(doc)
        attestation = attest_document(
            content=doc,
            source_id="corpus",
            trust_level=TrustLevel.HIGH,
            private_key=priv,
        )
        commitment = commit_embedding(doc_hash, emb, commit_key)
        leaf_idx = tree.insert(
            doc_hash=doc_hash,
            embedding_hash=commitment.embedding_digest,
            provenance=attestation.to_bytes(),
        )
        store.append({
            "doc": doc,
            "emb": emb,
            "attestation": attestation,
            "commitment": commitment,
            "leaf_idx": leaf_idx,
        })
    populate_s = time.perf_counter() - t0
    print(f"  Populated in {populate_s:.2f}s")

    # Verify timing: randomly pick entries and do full verification
    # (document hash check, Ed25519 signature, HMAC commitment, Merkle proof)
    import random
    rng = random.Random(42)

    print(f"\n[2/2] Running {args.queries} verification queries...")
    latencies_ms = []
    component_times = {
        "doc_hash": [],
        "ed25519_verify": [],
        "hmac_verify": [],
        "merkle_prove": [],
        "merkle_verify": [],
    }

    for q in range(args.queries):
        idx = rng.randint(0, args.n - 1)
        entry = store[idx]

        t_start = time.perf_counter_ns()

        # 1. Document hash check
        t0 = time.perf_counter_ns()
        computed_hash = hashlib.sha256(entry["doc"].encode()).digest()
        doc_ok = computed_hash == entry["attestation"].doc_hash
        component_times["doc_hash"].append(time.perf_counter_ns() - t0)

        # 2. Ed25519 signature verify
        t0 = time.perf_counter_ns()
        att_ok = entry["attestation"].verify(pub)
        component_times["ed25519_verify"].append(time.perf_counter_ns() - t0)

        # 3. HMAC commitment verify
        t0 = time.perf_counter_ns()
        emb_ok = verify_embedding(entry["commitment"], entry["emb"], commit_key)
        component_times["hmac_verify"].append(time.perf_counter_ns() - t0)

        # 4. Merkle membership proof
        t0 = time.perf_counter_ns()
        proof = tree.prove(entry["leaf_idx"])
        component_times["merkle_prove"].append(time.perf_counter_ns() - t0)

        t0 = time.perf_counter_ns()
        merkle_ok = tree.verify(proof)
        component_times["merkle_verify"].append(time.perf_counter_ns() - t0)

        total_ns = time.perf_counter_ns() - t_start
        latencies_ms.append(total_ns / 1e6)

        assert doc_ok and att_ok and emb_ok and merkle_ok

    latencies_ms.sort()
    p50 = pct(latencies_ms, 0.50)
    p95 = pct(latencies_ms, 0.95)
    p99 = pct(latencies_ms, 0.99)
    max_lat = latencies_ms[-1]
    mean = sum(latencies_ms) / len(latencies_ms)

    print(f"\n  Verification latency (ms) over {args.queries} queries:")
    print(f"    mean  = {mean:.4f}")
    print(f"    p50   = {p50:.4f}")
    print(f"    p95   = {p95:.4f}")
    print(f"    p99   = {p99:.4f}")
    print(f"    max   = {max_lat:.4f}")

    print("\n  Component means (us):")
    for k, vs in component_times.items():
        mean_us = (sum(vs) / len(vs)) / 1000
        print(f"    {k:<18} {mean_us:.2f} us")

    results = {
        "n": args.n,
        "queries": args.queries,
        "verification_latency_ms": {
            "mean": mean, "p50": p50, "p95": p95, "p99": p99, "max": max_lat,
        },
        "component_mean_us": {
            k: (sum(vs) / len(vs)) / 1000 for k, vs in component_times.items()
        },
    }
    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
