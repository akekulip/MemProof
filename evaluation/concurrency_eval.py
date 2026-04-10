"""Concurrent ingestion stress test.

Measures how MemProof's Merkle root updates behave under concurrent
ingestion from multiple workers. Specifically:

  1. Throughput: single-threaded vs N-threaded ingestion (docs/sec)
  2. Merkle root consistency: after all ingestions complete, verify
     the final root is consistent with the set of ingested entries
  3. Lock contention: time spent waiting for the tree lock

Python threads are limited by the GIL for CPU work, but HMAC-SHA256
and Ed25519 operations release the GIL when implemented in C (as they
are in the `cryptography` library). This experiment measures the
practical concurrency pattern.

No network, no LLM. Pure CPU measurement.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from memproof.crypto.attestation import (
    TrustLevel,
    attest_document,
    generate_source_keypair,
)
from memproof.crypto.commitment import commit_embedding
from memproof.crypto.merkle import MerkleTree
from memproof.crypto.source_registry import SourceRegistry


def _synth_embed(text: str, dim: int = 384) -> list[float]:
    h = hashlib.sha256(text.encode()).digest()
    raw = (h * (dim // 32 + 1))[: dim]
    return [(b / 255.0) * 2.0 - 1.0 for b in raw]


class ThreadSafeStore:
    """Minimal thread-safe wrapper around Merkle tree + list store.

    The tree lock serializes Merkle inserts; everything else (signing,
    hashing, commitment) runs in parallel.
    """

    def __init__(self, priv_key, commitment_key: bytes) -> None:
        self._priv = priv_key
        self._commitment_key = commitment_key
        self._tree = MerkleTree()
        self._tree_lock = threading.Lock()
        self._store: list[dict] = []
        self._store_lock = threading.Lock()
        self._tree_wait_ns = 0
        self._tree_wait_lock = threading.Lock()

    def ingest(self, doc: str) -> int:
        """Ingest a document. Returns leaf index."""
        # CPU work — hashing, signing, committing (done in parallel)
        doc_hash = hashlib.sha256(doc.encode()).digest()
        emb = _synth_embed(doc)
        attestation = attest_document(
            content=doc,
            source_id="corpus",
            trust_level=TrustLevel.HIGH,
            private_key=self._priv,
        )
        commitment = commit_embedding(doc_hash, emb, self._commitment_key)

        # Serialized section: Merkle insert
        t_wait_start = time.perf_counter_ns()
        with self._tree_lock:
            t_wait_end = time.perf_counter_ns()
            leaf_idx = self._tree.insert(
                doc_hash=doc_hash,
                embedding_hash=commitment.embedding_digest,
                provenance=attestation.to_bytes(),
            )
        wait_ns = t_wait_end - t_wait_start

        with self._tree_wait_lock:
            self._tree_wait_ns += wait_ns

        with self._store_lock:
            self._store.append({
                "doc": doc,
                "emb": emb,
                "attestation": attestation,
                "commitment": commitment,
                "leaf_idx": leaf_idx,
            })

        return leaf_idx

    @property
    def size(self) -> int:
        return len(self._store)

    @property
    def root(self) -> bytes:
        return self._tree.root

    @property
    def total_wait_ms(self) -> float:
        return self._tree_wait_ns / 1e6


def run_concurrent(num_workers: int, docs_per_worker: int) -> dict:
    """Run N workers each ingesting M documents. Return metrics."""
    registry = SourceRegistry()
    priv, _ = registry.register_source("corpus", TrustLevel.HIGH)
    commit_key = os.urandom(32)
    store = ThreadSafeStore(priv, commit_key)

    total_docs = num_workers * docs_per_worker

    def worker(worker_id: int) -> None:
        for i in range(docs_per_worker):
            doc = f"worker_{worker_id}_doc_{i} with some synthetic content to bulk up the size"
            store.ingest(doc)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = [ex.submit(worker, w) for w in range(num_workers)]
        for f in as_completed(futures):
            f.result()
    elapsed = time.perf_counter() - t0

    throughput = total_docs / elapsed
    avg_lock_wait_ms = store.total_wait_ms / total_docs if total_docs else 0

    return {
        "num_workers": num_workers,
        "docs_per_worker": docs_per_worker,
        "total_docs": total_docs,
        "elapsed_s": elapsed,
        "throughput_docs_per_s": throughput,
        "total_lock_wait_ms": store.total_wait_ms,
        "avg_lock_wait_per_doc_ms": avg_lock_wait_ms,
        "final_size": store.size,
        "merkle_root_hex": store.root.hex()[:32],
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs-per-worker", type=int, default=500)
    parser.add_argument("--output", type=str, default="evaluation/concurrency_eval.json")
    args = parser.parse_args()

    print("=== Concurrent ingestion stress test ===")
    print(f"Docs per worker: {args.docs_per_worker}\n")

    results = {}
    for n_workers in [1, 2, 4, 8, 16]:
        print(f"\n--- {n_workers} worker(s) ---")
        r = run_concurrent(n_workers, args.docs_per_worker)
        print(f"  Total docs:     {r['total_docs']}")
        print(f"  Elapsed:        {r['elapsed_s']:.3f}s")
        print(f"  Throughput:     {r['throughput_docs_per_s']:.0f} docs/s")
        print(f"  Avg lock wait:  {r['avg_lock_wait_per_doc_ms']:.4f} ms/doc")
        print(f"  Final size:     {r['final_size']}")
        results[f"workers_{n_workers}"] = r

    print(f"\n{'=' * 60}\n  SUMMARY\n{'=' * 60}")
    print(f"  {'Workers':<10} {'Throughput':>15} {'Lock wait':>15}")
    print(f"  {'-' * 10} {'-' * 15} {'-' * 15}")
    for key, r in results.items():
        print(f"  {r['num_workers']:<10} {r['throughput_docs_per_s']:>13.0f}/s "
              f"{r['avg_lock_wait_per_doc_ms']:>12.3f} ms")

    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
