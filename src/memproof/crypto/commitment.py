"""Embedding commitment scheme.

Binds embedding vectors to their source documents cryptographically.
If an embedding is tampered with (adversarial perturbation), the
commitment verification fails.

Uses HMAC-SHA256 over quantized embedding coordinates as a practical
instantiation. For production, this could be replaced with vector
commitment schemes (KZG over groups of coordinates).
"""

from __future__ import annotations

import hashlib
import hmac
import struct
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class EmbeddingCommitment:
    """Commitment binding an embedding vector to its source document."""

    doc_hash: bytes
    embedding_digest: bytes
    commitment: bytes

    def verify(self, embedding: Sequence[float], key: bytes) -> bool:
        """Verify that the embedding matches the commitment."""
        computed_digest = _hash_embedding(embedding)
        if computed_digest != self.embedding_digest:
            return False
        expected = _compute_commitment(self.doc_hash, computed_digest, key)
        return hmac.compare_digest(expected, self.commitment)


def commit_embedding(
    doc_hash: bytes,
    embedding: Sequence[float],
    key: bytes,
) -> EmbeddingCommitment:
    """Create a commitment binding an embedding to its source document.

    Args:
        doc_hash: SHA-256 hash of the source document.
        embedding: The embedding vector (float sequence).
        key: HMAC key (held by the trusted ingestion pipeline).

    Returns:
        An EmbeddingCommitment that can later verify integrity.
    """
    embedding_digest = _hash_embedding(embedding)
    commitment = _compute_commitment(doc_hash, embedding_digest, key)
    return EmbeddingCommitment(
        doc_hash=doc_hash,
        embedding_digest=embedding_digest,
        commitment=commitment,
    )


def verify_embedding(
    commitment: EmbeddingCommitment,
    embedding: Sequence[float],
    key: bytes,
) -> bool:
    """Verify an embedding against its commitment."""
    return commitment.verify(embedding, key)


def _hash_embedding(embedding: Sequence[float]) -> bytes:
    """Deterministic hash of an embedding vector.

    Quantizes to float32 for reproducibility across platforms.
    """
    h = hashlib.sha256()
    h.update(struct.pack("<I", len(embedding)))
    for val in embedding:
        h.update(struct.pack("<f", float(val)))
    return h.digest()


def _compute_commitment(doc_hash: bytes, embedding_digest: bytes, key: bytes) -> bytes:
    """HMAC-SHA256 commitment over (doc_hash || embedding_digest)."""
    msg = doc_hash + embedding_digest
    return hmac.new(key, msg, hashlib.sha256).digest()
