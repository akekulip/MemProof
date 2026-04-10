"""Multi-embedder latency benchmark.

A reviewer concern was that our latency numbers report only one embedder
(all-MiniLM-L6-v2, 384-d). MemProof is embedder-agnostic: it hashes
whatever vector the retrieval system hands it, so the crypto path is
independent of the embedding model. This script demonstrates that
empirically by running the ingest crypto path under three embedders of
increasing cost:

  * all-MiniLM-L6-v2  (384-d, fast)   -- the model used in the main paper
  * all-mpnet-base-v2 (768-d, medium)
  * BAAI/bge-base-en-v1.5 (768-d, slower)

Crypto operations (Ed25519 sign, HMAC commit, Merkle insert, audit
append) are measured on their own and also folded in after the
embedding call, so the reader can see that the crypto *absolute* cost
stays flat while the *percentage* overhead shrinks as the embedder
gets more expensive.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from memproof.audit.log import AuditLog, OperationType
from memproof.crypto.attestation import (
    TrustLevel,
    attest_document,
    generate_source_keypair,
)
from memproof.crypto.commitment import commit_embedding
from memproof.crypto.merkle import MerkleTree


N_ITERS = 500
SAMPLE_TEXT = "This is a sample document for multi-embedder latency benchmarking."

EMBEDDERS = [
    ("all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2", 384),
    ("all-mpnet-base-v2", "sentence-transformers/all-mpnet-base-v2", 768),
    ("bge-base-en-v1.5", "BAAI/bge-base-en-v1.5", 768),
]


def bench(fn, iters: int = N_ITERS) -> float:
    for _ in range(10):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) * 1000 / iters


def bench_crypto_once(priv, pub, hmac_key: bytes, tree: MerkleTree,
                      log: AuditLog, doc_hash: bytes,
                      embedding: list[float]) -> dict:
    """Measure each crypto op in isolation for this (doc_hash, embedding)."""
    # Ed25519 sign
    def _sign():
        return attest_document(
            content=SAMPLE_TEXT,
            source_id="bench",
            trust_level=TrustLevel.HIGH,
            private_key=priv,
        )
    sign_ms = bench(_sign, N_ITERS)

    # HMAC commit
    def _commit():
        return commit_embedding(doc_hash, embedding, hmac_key)
    commit_ms = bench(_commit, N_ITERS)

    # Merkle insert -- use a fresh counter so indices stay unique
    counter = {"i": 0}
    def _insert():
        tree.insert(
            doc_hash=hashlib.sha256(f"mi-{counter['i']}".encode()).digest(),
            embedding_hash=hashlib.sha256(f"emb-{counter['i']}".encode()).digest(),
            provenance=b"p",
        )
        counter["i"] += 1
    insert_ms = bench(_insert, N_ITERS)

    # Audit append
    def _audit():
        log.append(OperationType.INGEST, doc_hash)
    audit_ms = bench(_audit, N_ITERS)

    return {
        "ed25519_sign_ms": sign_ms,
        "hmac_commit_ms": commit_ms,
        "merkle_insert_ms": insert_ms,
        "audit_append_ms": audit_ms,
        "crypto_total_ms": sign_ms + commit_ms + insert_ms + audit_ms,
    }


def main() -> None:
    print("=" * 72)
    print("  Multi-embedder latency benchmark")
    print("=" * 72)
    print(f"  Iterations per op: {N_ITERS}")
    print(f"  Sample text length: {len(SAMPLE_TEXT)} chars")

    priv, pub = generate_source_keypair()
    hmac_key = os.urandom(32)

    from sentence_transformers import SentenceTransformer

    out: dict = {"iterations": N_ITERS, "embedders": {}}
    for short, hf_id, dim in EMBEDDERS:
        print(f"\n── {short} ({hf_id}, dim={dim}) ──")
        print(f"  loading model...", flush=True)
        try:
            model = SentenceTransformer(hf_id)
        except Exception as e:
            print(f"  SKIP ({e})")
            out["embedders"][short] = {"error": str(e)}
            continue

        def _embed():
            return model.encode(SAMPLE_TEXT, normalize_embeddings=True)
        embed_ms = bench(_embed, iters=100)
        print(f"  embedding:         {embed_ms:.4f} ms")

        sample_vec = model.encode(SAMPLE_TEXT, normalize_embeddings=True).tolist()
        doc_hash = hashlib.sha256(SAMPLE_TEXT.encode()).digest()

        # Fresh crypto state per embedder so the benchmarks do not
        # cross-contaminate and tree growth is comparable
        tree = MerkleTree()
        for i in range(100):
            tree.insert(
                doc_hash=hashlib.sha256(f"seed-{short}-{i}".encode()).digest(),
                embedding_hash=hashlib.sha256(f"seed-emb-{i}".encode()).digest(),
                provenance=b"seed",
            )
        log = AuditLog()

        crypto = bench_crypto_once(
            priv, pub, hmac_key, tree, log, doc_hash, sample_vec,
        )
        print(f"  Ed25519 sign:      {crypto['ed25519_sign_ms']:.4f} ms")
        print(f"  HMAC commit:       {crypto['hmac_commit_ms']:.4f} ms")
        print(f"  Merkle insert:     {crypto['merkle_insert_ms']:.4f} ms")
        print(f"  Audit append:      {crypto['audit_append_ms']:.4f} ms")
        print(f"  Crypto total:      {crypto['crypto_total_ms']:.4f} ms")
        overhead_pct = (crypto["crypto_total_ms"] / embed_ms) * 100
        total_ms = embed_ms + crypto["crypto_total_ms"]
        print(f"  Embed + crypto:    {total_ms:.4f} ms ({overhead_pct:.2f}% overhead)")

        out["embedders"][short] = {
            "hf_id": hf_id,
            "dim": dim,
            "embedding_ms": embed_ms,
            **crypto,
            "embed_plus_crypto_ms": total_ms,
            "crypto_overhead_percent": overhead_pct,
        }

    print("\n" + "=" * 72)
    print("  Summary")
    print("=" * 72)
    print(f"  {'Embedder':<22} {'Embed (ms)':>12} {'Crypto (ms)':>13} "
          f"{'Overhead %':>12}")
    print(f"  {'-' * 22} {'-' * 12} {'-' * 13} {'-' * 12}")
    for short, _, _ in EMBEDDERS:
        e = out["embedders"].get(short, {})
        if "error" in e:
            print(f"  {short:<22} {'skipped':>12}")
            continue
        print(
            f"  {short:<22} "
            f"{e['embedding_ms']:>12.4f} "
            f"{e['crypto_total_ms']:>13.4f} "
            f"{e['crypto_overhead_percent']:>11.2f}%"
        )

    out_path = Path(__file__).parent / "multi_embedder_latency.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
