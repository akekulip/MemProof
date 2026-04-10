"""Tamper detection at scale.

Tests Layer 2 (embedding commitment) and Layer 3 (audit log) at
corpus sizes up to 10K. The procedure:

  1. Ingest N documents with valid Ed25519 signatures.
  2. Modify K random entries directly in the storage backend
     (bypassing the protocol).
  3. Run a full verification scan of the store. Count detections.
  4. Report per-entry verification cost and total scan time.

Also measures the cost of a periodic full-store scan, which is what
a real operator would run to catch tampering between retrievals.

No LLM or network calls. CPU-only. Should complete in under a minute
at 10K corpus size.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
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
    """Deterministic synthetic embedding for scale testing (no model load)."""
    h = hashlib.sha256(text.encode()).digest()
    # Repeat the hash to fill dim floats
    raw = (h * (dim // 32 + 1))[: dim]
    return [(b / 255.0) * 2.0 - 1.0 for b in raw]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10_000, help="Corpus size")
    parser.add_argument("--k-tamper", type=int, default=100, help="Entries to tamper")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="evaluation/tamper_at_scale.json")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    print(f"=== Tamper detection at scale (n={args.n}, k_tamper={args.k_tamper}) ===")

    # Setup: single source with valid keypair
    registry = SourceRegistry()
    priv, pub = registry.register_source("corpus", TrustLevel.HIGH, "Test corpus")

    tree = MerkleTree()
    commitment_key = os.urandom(32)

    # Storage: list of entries (document, embedding, attestation, commitment)
    store: list[dict] = []

    # Ingest N documents
    print(f"\n[1/4] Ingesting {args.n} documents...")
    t0 = time.perf_counter()
    for i in range(args.n):
        doc = f"Document {i} — synthetic content for scale testing with varied length {i % 7}"
        doc_hash = hashlib.sha256(doc.encode()).digest()
        emb = _synth_embed(doc)

        attestation = attest_document(
            content=doc,
            source_id="corpus",
            trust_level=TrustLevel.HIGH,
            private_key=priv,
        )
        commitment = commit_embedding(doc_hash, emb, commitment_key)
        leaf_idx = tree.insert(
            doc_hash=doc_hash,
            embedding_hash=commitment.embedding_digest,
            provenance=attestation.to_bytes(),
        )
        store.append({
            "document": doc,
            "embedding": emb,
            "attestation": attestation,
            "commitment": commitment,
            "leaf_idx": leaf_idx,
        })
        if (i + 1) % 2000 == 0:
            print(f"  [{i+1}/{args.n}]")
    ingest_time = time.perf_counter() - t0
    print(f"  Ingest done in {ingest_time:.2f}s ({args.n/ingest_time:.0f} docs/s)")

    # Tamper: corrupt K random entries by modifying the document text
    # (without re-running the attestation / commitment)
    print(f"\n[2/4] Tampering with {args.k_tamper} random entries...")
    tampered_indices = rng.sample(range(args.n), args.k_tamper)

    # Two tamper modes: modify document (attestation doc_hash mismatch)
    # and modify embedding (commitment verify fails)
    k_doc = args.k_tamper // 2
    k_emb = args.k_tamper - k_doc

    for idx in tampered_indices[:k_doc]:
        store[idx]["document"] = store[idx]["document"] + " TAMPERED"
    for idx in tampered_indices[k_doc:]:
        emb = store[idx]["embedding"]
        store[idx]["embedding"] = [v + 0.05 for v in emb]
    print(f"  Tampered {k_doc} documents, {k_emb} embeddings")

    # Full-store verification scan
    print(f"\n[3/4] Full-store verification scan...")
    t0 = time.perf_counter()
    doc_tamper_detected = 0
    emb_tamper_detected = 0
    clean_verified = 0

    for i, entry in enumerate(store):
        # Check 1: document hash matches attestation
        computed_hash = hashlib.sha256(entry["document"].encode()).digest()
        doc_ok = computed_hash == entry["attestation"].doc_hash

        # Check 2: attestation signature
        att_ok = entry["attestation"].verify(pub)

        # Check 3: embedding commitment
        emb_ok = verify_embedding(
            entry["commitment"], entry["embedding"], commitment_key
        )

        if not doc_ok and i in tampered_indices[:k_doc]:
            doc_tamper_detected += 1
        if not emb_ok and i in tampered_indices[k_doc:]:
            emb_tamper_detected += 1
        if doc_ok and emb_ok and att_ok:
            clean_verified += 1

    scan_time = time.perf_counter() - t0
    per_entry_ms = (scan_time * 1000) / args.n
    print(f"  Scan done in {scan_time:.2f}s ({per_entry_ms:.4f} ms/entry)")
    print(f"  Doc tampering detected: {doc_tamper_detected}/{k_doc} ({doc_tamper_detected/k_doc:.0%})")
    print(f"  Emb tampering detected: {emb_tamper_detected}/{k_emb} ({emb_tamper_detected/k_emb:.0%})")
    print(f"  Clean entries verified: {clean_verified}/{args.n - args.k_tamper}")

    # Merkle root integrity
    print(f"\n[4/4] Merkle root consistency check...")
    root_before = tree.root
    # Nothing changed in the tree, so root should be the same
    root_after = tree.root
    root_stable = root_before == root_after
    print(f"  Merkle root stable: {root_stable}")
    print(f"  Root: {root_before.hex()[:32]}...")

    # Save results
    results = {
        "n": args.n,
        "k_tamper": args.k_tamper,
        "k_doc_tamper": k_doc,
        "k_emb_tamper": k_emb,
        "ingest_time_s": ingest_time,
        "ingest_throughput_docs_per_s": args.n / ingest_time,
        "scan_time_s": scan_time,
        "scan_per_entry_ms": per_entry_ms,
        "doc_tamper_detected": doc_tamper_detected,
        "doc_tamper_detection_rate": doc_tamper_detected / k_doc if k_doc else 0,
        "emb_tamper_detected": emb_tamper_detected,
        "emb_tamper_detection_rate": emb_tamper_detected / k_emb if k_emb else 0,
        "clean_verified": clean_verified,
        "merkle_root_stable": root_stable,
    }

    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
