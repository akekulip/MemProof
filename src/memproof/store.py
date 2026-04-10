"""MemProofStore: Main entry point tying all four layers together.

Usage:
    from memproof.store import MemProofStore
    from memproof.crypto.attestation import TrustLevel

    store = MemProofStore(embed_fn=my_embedder)

    # Register a source (generates an Ed25519 keypair). Keep the private
    # key with the source; the public key stays in MemProofStore's registry.
    priv_key, _ = store.register_source("wikipedia", TrustLevel.HIGH)

    # Ingest a document, signed by the source
    store.ingest(
        content="The capital of France is Paris",
        source_id="wikipedia",
        source_private_key=priv_key,
    )

    # Query with cryptographic verification
    results = store.query("What is the capital of France?", top_k=5)
    for r in results:
        assert r.fully_verified

    valid, broken_at = store.verify_audit_chain()
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from memproof.audit.log import AuditLog
from memproof.crypto.attestation import TrustLevel
from memproof.crypto.merkle import MerkleTree
from memproof.crypto.source_registry import SourceRegistry
from memproof.ingestion.pipeline import EmbeddingFunction, IngestPipeline, IngestResult
from memproof.retrieval.verified import VerifiedResult, VerifiedRetriever


@dataclass
class MemoryEntry:
    """In-memory representation of a stored entry."""

    document: str
    embedding: list[float]
    leaf_index: int
    trust_level: TrustLevel
    source_id: str
    attestation: Any
    commitment: Any


class SimpleVectorStore:
    """Minimal in-memory vector store for prototyping.

    Replace with ChromaDB / FAISS for production evaluation.
    """

    def __init__(self) -> None:
        self._entries: list[MemoryEntry] = []

    def add(self, entry: MemoryEntry) -> None:
        self._entries.append(entry)

    def query(self, embedding: list[float], top_k: int) -> list[dict[str, Any]]:
        """Cosine similarity search."""
        scored: list[tuple[float, MemoryEntry]] = []
        for entry in self._entries:
            score = _cosine_similarity(embedding, entry.embedding)
            scored.append((score, entry))
        scored.sort(key=lambda x: x[0], reverse=True)

        results: list[dict[str, Any]] = []
        for score, entry in scored[:top_k]:
            results.append({
                "document": entry.document,
                "embedding": entry.embedding,
                "score": score,
                "leaf_index": entry.leaf_index,
                "trust_level": int(entry.trust_level),
                "source_id": entry.source_id,
                "attestation": entry.attestation,
                "commitment": entry.commitment,
            })
        return results

    @property
    def size(self) -> int:
        return len(self._entries)


class MemProofStore:
    """MemProof-protected memory store.

    Integrates all four layers:
      Layer 1: Provenance-verified ingestion (Ed25519 signatures)
      Layer 2: Embedding commitment integrity (HMAC-SHA256)
      Layer 3: Temporal audit log (hash-chained)
      Layer 4: Verified retrieval with Merkle membership proofs

    The store manages a SourceRegistry internally. For simple use cases,
    call `register_source()` to create and register a source in one step,
    keeping the returned private key with the source.

    For deployments that already have a key management system, inject an
    external SourceRegistry via the constructor and use `add_source()`
    with pre-generated public keys.
    """

    def __init__(
        self,
        embed_fn: EmbeddingFunction,
        commitment_key: bytes | None = None,
        audit_path: Path | None = None,
        min_trust: TrustLevel = TrustLevel.LOW,
        source_registry: SourceRegistry | None = None,
    ) -> None:
        self._commitment_key = commitment_key or os.urandom(32)
        self._tree = MerkleTree()
        self._audit = AuditLog(log_path=audit_path)
        self._vstore = SimpleVectorStore()
        self._registry = source_registry or SourceRegistry()
        # Map source_id -> private key (held locally for convenience APIs;
        # in production the private key lives with the source, not the store).
        self._source_keys: dict[str, Ed25519PrivateKey] = {}

        self._pipeline = IngestPipeline(
            embed_fn=embed_fn,
            commitment_key=self._commitment_key,
            merkle_tree=self._tree,
            audit_log=self._audit,
            source_registry=self._registry,
        )

        self._retriever = VerifiedRetriever(
            vector_store=self._vstore,
            merkle_tree=self._tree,
            audit_log=self._audit,
            commitment_key=self._commitment_key,
            source_registry=self._registry,
            min_trust=min_trust,
        )

        self._embed_fn = embed_fn

    # ---- Source management ----

    def register_source(
        self,
        source_id: str,
        default_trust: TrustLevel = TrustLevel.MEDIUM,
        description: str = "",
    ) -> tuple[Ed25519PrivateKey, Ed25519PublicKey]:
        """Generate a new Ed25519 keypair for a source and register it.

        The store also caches the private key locally so that subsequent
        calls to `ingest()` can sign automatically. In production this
        cache would not exist; the source holds the private key.

        Returns:
            (private_key, public_key) tuple.
        """
        priv, pub = self._registry.register_source(
            source_id=source_id,
            default_trust=default_trust,
            description=description,
        )
        self._source_keys[source_id] = priv
        return priv, pub

    def add_external_source(
        self,
        source_id: str,
        public_key: Ed25519PublicKey | bytes,
        default_trust: TrustLevel = TrustLevel.MEDIUM,
        description: str = "",
    ) -> None:
        """Register a source with a pre-existing public key.

        Use this when the source holds its own key and only publishes the
        public key to the store. Ingestion requires the source to sign
        attestations itself; use `ingest_signed()` in that case.
        """
        self._registry.add_source(
            source_id=source_id,
            public_key=public_key,
            default_trust=default_trust,
            description=description,
        )

    @property
    def registry(self) -> SourceRegistry:
        """The source registry (read access for inspection)."""
        return self._registry

    # ---- Ingestion ----

    def ingest(
        self,
        content: str,
        source_id: str,
        trust_level: TrustLevel | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> IngestResult:
        """Ingest a document signed by the registered source.

        The source must have been registered via `register_source()` so
        that the store holds its private key. For external sources that
        sign attestations themselves, use `ingest_signed()` instead.
        """
        if source_id not in self._source_keys:
            raise ValueError(
                f"Source {source_id!r} is not registered with a private key. "
                f"Call register_source() first, or use ingest_signed()."
            )

        priv_key = self._source_keys[source_id]
        return self._ingest_internal(content, source_id, priv_key, trust_level, metadata)

    def ingest_signed(
        self,
        content: str,
        source_id: str,
        source_private_key: Ed25519PrivateKey,
        trust_level: TrustLevel | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> IngestResult:
        """Ingest a document with an externally-held source private key.

        The source must already be registered (via `add_external_source()`
        or `register_source()`) so the public key is available for later
        verification.
        """
        return self._ingest_internal(
            content, source_id, source_private_key, trust_level, metadata,
        )

    def _ingest_internal(
        self,
        content: str,
        source_id: str,
        priv_key: Ed25519PrivateKey,
        trust_level: TrustLevel | None,
        metadata: dict[str, Any] | None,
    ) -> IngestResult:
        result = self._pipeline.ingest(
            content=content,
            source_id=source_id,
            source_private_key=priv_key,
            trust_level=trust_level,
            metadata=metadata,
        )

        # Resolve the effective trust level (pipeline uses registry default
        # if trust_level is None)
        effective_trust = trust_level
        if effective_trust is None:
            record = self._registry.get(source_id)
            effective_trust = record.default_trust if record else TrustLevel.UNTRUSTED

        self._vstore.add(MemoryEntry(
            document=content,
            embedding=result.embedding,
            leaf_index=result.leaf_index,
            trust_level=effective_trust,
            source_id=source_id,
            attestation=result.attestation,
            commitment=result.commitment,
        ))

        return result

    # ---- Query ----

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        verify: bool = True,
        trust_filter: bool = True,
    ) -> list[VerifiedResult]:
        """Query memory with cryptographic verification."""
        query_embedding = self._embed_fn(query_text)
        return self._retriever.query(
            query_embedding=query_embedding,
            top_k=top_k,
            verify=verify,
            trust_filter=trust_filter,
        )

    # ---- Audit and introspection ----

    def verify_audit_chain(self) -> tuple[bool, int]:
        """Verify the integrity of the audit log chain."""
        return self._audit.verify_chain()

    def detect_drift(self, window_size: int = 100) -> list[dict[str, Any]]:
        """Check for anomalous patterns in recent operations."""
        return self._audit.detect_drift(window_size)

    @property
    def merkle_root(self) -> bytes:
        """Current Merkle root over all memory entries."""
        return self._tree.root

    @property
    def entry_count(self) -> int:
        """Number of entries in memory."""
        return self._vstore.size

    @property
    def audit_length(self) -> int:
        """Number of audit log entries."""
        return self._audit.length


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
