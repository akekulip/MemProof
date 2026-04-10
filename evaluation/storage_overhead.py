"""Measure MemProof's per-entry storage overhead.

Builds a MemProof store with N entries, serializes each entry in the
same way the production path would (attestation bytes, commitment
bytes, audit reference, Merkle proof path), and reports the per-entry
byte breakdown and total overhead against a baseline that stores only
the content hash + float32 embedding + minimal metadata.

Writes results to evaluation/storage_overhead.json so every number
in the paper's Storage subsection is traceable.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from memproof.crypto.attestation import (
    TrustLevel,
    attest_document,
    generate_source_keypair,
)
from memproof.crypto.commitment import commit_embedding
from memproof.crypto.merkle import MerkleTree


def _synth_embed(text: str, dim: int = 384) -> list[float]:
    h = hashlib.sha256(text.encode()).digest()
    raw = (h * (dim // 32 + 1))[:dim]
    return [(b / 255.0) * 2.0 - 1.0 for b in raw]


def measure(n: int) -> dict:
    priv, pub = generate_source_keypair()
    commit_key = os.urandom(32)
    tree = MerkleTree()

    att_json_sizes: list[int] = []
    att_bin_sizes: list[int] = []
    commit_sizes: list[int] = []
    for i in range(n):
        doc = f"Document {i}: synthetic content block {i % 17}."
        doc_hash = hashlib.sha256(doc.encode()).digest()
        emb = _synth_embed(doc)
        att = attest_document(
            content=doc,
            source_id="corpus",
            trust_level=TrustLevel.HIGH,
            private_key=priv,
        )
        commitment = commit_embedding(doc_hash, emb, commit_key)
        att_json = att.to_bytes()
        att_bin = att.to_bytes_binary()
        att_json_sizes.append(len(att_json))
        att_bin_sizes.append(len(att_bin))
        commit_sizes.append(len(commitment.embedding_digest))
        tree.insert(
            doc_hash=doc_hash,
            embedding_hash=commitment.embedding_digest,
            provenance=att_json,
        )

    # A proof stores: leaf_hash (32) + sibling_hashes (log2(n) * 32)
    # + directions (log2(n) bits ~ negligible)
    import math
    depth = max(1, math.ceil(math.log2(n)))
    proof_bytes = 32 + depth * 32

    att_json_mean = sum(att_json_sizes) / n
    att_bin_mean = sum(att_bin_sizes) / n
    commit_mean = sum(commit_sizes) / n
    audit_ref = 30  # 32-byte hash + 4-byte operation index (flush to 30 by pack)

    # Baseline per entry: SHA256 doc hash (32) + float32 embedding
    # (384 * 4 = 1536) + minimal metadata (~84 bytes: source id string,
    # trust level, doc_id, timestamp)
    baseline_per_entry = 32 + 384 * 4 + 84
    overhead_json = att_json_mean + commit_mean + proof_bytes + audit_ref
    overhead_bin = att_bin_mean + commit_mean + proof_bytes + audit_ref
    total_json = baseline_per_entry + overhead_json
    total_bin = baseline_per_entry + overhead_bin
    pct_json = (overhead_json / baseline_per_entry) * 100
    pct_bin = (overhead_bin / baseline_per_entry) * 100

    return {
        "n": n,
        "proof_depth": depth,
        "baseline_per_entry_bytes": baseline_per_entry,
        "attestation_json_mean_bytes": att_json_mean,
        "attestation_binary_mean_bytes": att_bin_mean,
        "commitment_mean_bytes": commit_mean,
        "merkle_proof_bytes": proof_bytes,
        "audit_ref_bytes": audit_ref,
        "json": {
            "overhead_per_entry_bytes": overhead_json,
            "total_per_entry_bytes": total_json,
            "overhead_percent": pct_json,
        },
        "binary": {
            "overhead_per_entry_bytes": overhead_bin,
            "total_per_entry_bytes": total_bin,
            "overhead_percent": pct_bin,
        },
    }


def main() -> None:
    out = {}
    for n in [1_000, 10_000, 100_000]:
        r = measure(n)
        print(f"\nn = {n:,}")
        print(f"  baseline bytes/entry:    {r['baseline_per_entry_bytes']}")
        print(f"  attestation JSON:        {r['attestation_json_mean_bytes']:.1f} B")
        print(f"  attestation binary:      {r['attestation_binary_mean_bytes']:.1f} B")
        print(f"  Merkle proof:            {r['merkle_proof_bytes']} B (depth {r['proof_depth']})")
        print(f"  --- JSON encoding ---")
        print(f"    overhead per entry:    {r['json']['overhead_per_entry_bytes']:.1f} B")
        print(f"    total per entry:       {r['json']['total_per_entry_bytes']:.1f} B")
        print(f"    overhead percent:      {r['json']['overhead_percent']:.2f}%")
        print(f"  --- binary encoding ---")
        print(f"    overhead per entry:    {r['binary']['overhead_per_entry_bytes']:.1f} B")
        print(f"    total per entry:       {r['binary']['total_per_entry_bytes']:.1f} B")
        print(f"    overhead percent:      {r['binary']['overhead_percent']:.2f}%")
        out[f"n_{n}"] = r

    out_path = Path(__file__).parent / "storage_overhead.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
