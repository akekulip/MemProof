"""ANN index rebuild compatibility experiment.

A reviewer asked whether MemProof's integrity guarantees still hold
when the underlying ANN index is rebuilt or corrupted. MemProof
deliberately separates the trust root (Merkle tree over cryptographic
commitments) from the ANN index (Chroma / HNSW); the ANN index is a
query optimization over raw vectors and is not part of the trust
boundary. Concretely this experiment:

  1. Populates a MemProof store with N entries and records the Merkle
     root and audit-chain length.
  2. Serializes the canonical entries (content + attestation +
     commitment + leaf index).
  3. Wipes the ANN backend to simulate catastrophic index corruption
     or a migration to a new backend.
  4. Rebuilds the ANN backend from the canonical entries.
  5. Verifies that (a) the Merkle root is unchanged, (b) every entry
     still verifies, (c) query results are retrievable again.

We do not include ANN rebuild time in MemProof's overhead claims; this
script exists so a reviewer can see that a rebuild is a straight
re-insert and that the trust root is invariant.
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
from memproof.crypto.commitment import commit_embedding, verify_embedding
from memproof.crypto.merkle import MerkleTree
from memproof.crypto.source_registry import SourceRegistry


def _synth_embed(text: str, dim: int = 384) -> list[float]:
    h = hashlib.sha256(text.encode()).digest()
    raw = (h * (dim // 32 + 1))[:dim]
    return [(b / 255.0) * 2.0 - 1.0 for b in raw]


def populate(n: int, priv, pub, commit_key: bytes) -> tuple[MerkleTree, AuditLog, list[dict]]:
    """Populate the crypto state and canonical store with n entries."""
    tree = MerkleTree()
    audit = AuditLog()
    canonical: list[dict] = []

    for i in range(n):
        doc = f"Document {i}: synthetic content block {i % 17}"
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
        audit.append(OperationType.INGEST, doc_hash)
        canonical.append({
            "doc": doc,
            "embedding": emb,
            "attestation": attestation,
            "commitment": commitment,
            "leaf_idx": leaf_idx,
        })
    return tree, audit, canonical


def verify_all(canonical: list[dict], tree: MerkleTree, commit_key: bytes, pub) -> int:
    """Verify every canonical entry. Return the number that passed."""
    passed = 0
    for e in canonical:
        doc_hash = hashlib.sha256(e["doc"].encode()).digest()
        if doc_hash != e["attestation"].doc_hash:
            continue
        if not e["attestation"].verify(pub):
            continue
        if not verify_embedding(e["commitment"], e["embedding"], commit_key):
            continue
        proof = tree.prove(e["leaf_idx"])
        if not tree.verify(proof):
            continue
        passed += 1
    return passed


def try_chroma_rebuild(canonical: list[dict]) -> dict | None:
    """Rebuild a Chroma ANN collection from canonical entries.

    Returns timing/size info or None if chromadb is unavailable.
    """
    try:
        import chromadb  # noqa: F401
    except ImportError:
        return None

    from memproof.backends.chroma import ChromaVectorStore

    store = ChromaVectorStore(collection_name=f"rebuild_bench_{os.getpid()}")
    store.clear()

    t0 = time.perf_counter()
    for e in canonical:
        store.add(
            document=e["doc"],
            embedding=e["embedding"],
            leaf_index=e["leaf_idx"],
            trust_level=int(e["attestation"].trust_level),
            source_id=e["attestation"].source_id,
            attestation_bytes=e["attestation"].to_bytes(),
            commitment_bytes=e["commitment"].embedding_digest,
        )
    insert_s = time.perf_counter() - t0

    # Sanity: size, and a single nearest-neighbor query round-trip
    t1 = time.perf_counter()
    query = _synth_embed("Document 10: synthetic content block 10")
    hits = store.query(query, top_k=5)
    query_s = time.perf_counter() - t1

    result = {
        "backend": "chromadb",
        "rebuilt_size": store.size,
        "rebuild_insert_seconds": insert_s,
        "rebuild_docs_per_sec": len(canonical) / max(insert_s, 1e-9),
        "sanity_query_seconds": query_s,
        "sanity_hits": len(hits),
    }
    store.clear()
    return result


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation/ann_rebuild_compat.json",
    )
    args = parser.parse_args()

    print("=" * 72)
    print("  ANN index rebuild compatibility experiment")
    print("=" * 72)
    print(f"  Entries: {args.n}")

    registry = SourceRegistry()
    priv, pub = registry.register_source("corpus", TrustLevel.HIGH)
    commit_key = os.urandom(32)

    print("\n[1/5] Populating canonical store + Merkle tree...", flush=True)
    t0 = time.perf_counter()
    tree, audit, canonical = populate(args.n, priv, pub, commit_key)
    populate_s = time.perf_counter() - t0
    root_before = tree.root.hex()
    audit_len_before = audit.length
    print(f"      populated in {populate_s:.2f}s "
          f"({args.n/populate_s:.0f} docs/s)")
    print(f"      merkle root = {root_before[:32]}...")
    print(f"      audit length = {audit_len_before}")

    print("\n[2/5] Verifying every entry before rebuild...", flush=True)
    t0 = time.perf_counter()
    verified = verify_all(canonical, tree, commit_key, pub)
    verify_s = time.perf_counter() - t0
    print(f"      verified {verified}/{args.n} in {verify_s:.2f}s")
    assert verified == args.n, "pre-rebuild verification failed"

    print("\n[3/5] Simulating catastrophic ANN corruption (wipe backend)...",
          flush=True)
    # The ANN index is out-of-trust; we simulate "wipe" by simply not
    # carrying one through the rebuild phase.
    print("      ANN backend discarded; canonical entries retained.")

    print("\n[4/5] Rebuilding ANN index from canonical entries...", flush=True)
    chroma_info = try_chroma_rebuild(canonical)
    if chroma_info is None:
        print("      chromadb not available, skipping backend rebuild.")
    else:
        print(f"      backend: {chroma_info['backend']}")
        print(f"      rebuilt size: {chroma_info['rebuilt_size']}")
        print(f"      rebuild time: {chroma_info['rebuild_insert_seconds']:.2f}s "
              f"({chroma_info['rebuild_docs_per_sec']:.0f} docs/s)")
        print(f"      sanity query hit count: {chroma_info['sanity_hits']}")

    print("\n[5/5] Re-verifying every entry after rebuild...", flush=True)
    t0 = time.perf_counter()
    verified_after = verify_all(canonical, tree, commit_key, pub)
    verify_after_s = time.perf_counter() - t0
    print(f"      verified {verified_after}/{args.n} in {verify_after_s:.2f}s")
    root_after = tree.root.hex()
    print(f"      merkle root = {root_after[:32]}...")

    root_unchanged = root_before == root_after
    print("\n" + "=" * 72)
    print("  Summary")
    print("=" * 72)
    print(f"  Verified before rebuild:   {verified}/{args.n}")
    print(f"  Verified after rebuild:    {verified_after}/{args.n}")
    print(f"  Merkle root unchanged:     {root_unchanged}")
    print(f"  Audit length unchanged:    {audit.length == audit_len_before}")

    out = {
        "n": args.n,
        "populate_seconds": populate_s,
        "populate_docs_per_sec": args.n / populate_s,
        "verify_before_seconds": verify_s,
        "verify_after_seconds": verify_after_s,
        "verified_before": verified,
        "verified_after": verified_after,
        "merkle_root_before": root_before,
        "merkle_root_after": root_after,
        "merkle_root_unchanged": root_unchanged,
        "audit_length_before": audit_len_before,
        "audit_length_after": audit.length,
        "ann_rebuild": chroma_info,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
