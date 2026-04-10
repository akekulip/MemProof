"""Provenance-verified ingestion pipeline.

Implements Layer 1 of MemProof:
  1. Hash the document content
  2. Create an Ed25519-signed provenance attestation
     (source is looked up in the SourceRegistry; the source's private
     key signs the attestation)
  3. Compute embedding via the embedding function
  4. Create an embedding commitment binding embedding to document
  5. Insert into Merkle tree
  6. Store (document, embedding, attestation, commitment) in vector store
  7. Log the operation to the audit log
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Protocol

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from memproof.audit.log import AuditLog, OperationType
from memproof.crypto.attestation import (
    DocumentAttestation,
    TrustLevel,
    attest_document,
)
from memproof.crypto.commitment import EmbeddingCommitment, commit_embedding
from memproof.crypto.merkle import MerkleTree
from memproof.crypto.source_registry import SourceRegistry


class EmbeddingFunction(Protocol):
    """Protocol for embedding functions."""

    def __call__(self, text: str) -> list[float]: ...


@dataclass(frozen=True)
class IngestResult:
    """Result of ingesting a document into MemProof-protected memory."""

    leaf_index: int
    doc_hash: bytes
    attestation: DocumentAttestation
    commitment: EmbeddingCommitment
    embedding: list[float]
    merkle_root: bytes


class IngestPipeline:
    """Provenance-verified ingestion pipeline.

    Orchestrates Layer 1 (Ed25519 attestation) and Layer 2 (HMAC
    commitment) during document ingestion. The source registry is
    consulted to verify the source is known and to look up the
    default trust level.

    The commitment key is a system-level secret used only to bind
    embeddings to documents. It is distinct from source signing keys.
    """

    def __init__(
        self,
        embed_fn: EmbeddingFunction,
        commitment_key: bytes,
        merkle_tree: MerkleTree,
        audit_log: AuditLog,
        source_registry: SourceRegistry,
    ) -> None:
        self._embed_fn = embed_fn
        self._commitment_key = commitment_key
        self._tree = merkle_tree
        self._audit = audit_log
        self._registry = source_registry

    def ingest(
        self,
        content: str,
        source_id: str,
        source_private_key: Ed25519PrivateKey,
        trust_level: TrustLevel | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> IngestResult:
        """Ingest a document with Ed25519-signed provenance.

        Args:
            content: Document text content.
            source_id: Source identifier (must be registered).
            source_private_key: The source's Ed25519 private key for signing.
            trust_level: Trust level. Defaults to the source's registered default.
            metadata: Additional metadata.

        Returns:
            IngestResult with all cryptographic artifacts.

        Raises:
            ValueError: If source_id is not in the registry.
        """
        # Step 0: Verify source is registered
        record = self._registry.get(source_id)
        if record is None:
            raise ValueError(
                f"Source {source_id!r} is not registered. Call "
                f"SourceRegistry.register_source() or add_source() first."
            )

        # Default trust level from registry if not specified
        if trust_level is None:
            trust_level = record.default_trust

        # Step 1: Hash document
        doc_hash = hashlib.sha256(content.encode("utf-8")).digest()

        # Step 2: Sign attestation with source's Ed25519 private key
        attestation = attest_document(
            content=content,
            source_id=source_id,
            trust_level=trust_level,
            private_key=source_private_key,
            metadata=metadata,
        )

        # Step 3: Compute embedding
        embedding = self._embed_fn(content)

        # Step 4: Create HMAC embedding commitment (system key, not source key)
        commitment = commit_embedding(
            doc_hash=doc_hash,
            embedding=embedding,
            key=self._commitment_key,
        )

        # Step 5: Insert into Merkle tree
        leaf_index = self._tree.insert(
            doc_hash=doc_hash,
            embedding_hash=commitment.embedding_digest,
            provenance=attestation.to_bytes(),
        )

        # Step 6: Log to audit
        self._audit.append(
            operation=OperationType.INGEST,
            doc_hash=doc_hash,
            metadata={
                "source_id": source_id,
                "trust_level": int(trust_level),
                "leaf_index": leaf_index,
                **(metadata or {}),
            },
        )

        return IngestResult(
            leaf_index=leaf_index,
            doc_hash=doc_hash,
            attestation=attestation,
            commitment=commitment,
            embedding=embedding,
            merkle_root=self._tree.root,
        )
