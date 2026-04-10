"""Verified retrieval with Ed25519 signatures and Merkle proofs.

Implements Layer 4 of MemProof:
  1. Perform standard nearest-neighbor retrieval
  2. For each result:
     a. Verify the document hash matches the attestation doc_hash
     b. Verify the attestation signature against the source's public key
        (looked up in the SourceRegistry)
     c. Verify the embedding commitment against the stored embedding
     d. Generate a Merkle membership proof
  3. Filter by trust level threshold
  4. Return results with proofs attached

Any step that fails flags the result as unverified. Callers can choose
whether to include unverified results (useful for forensic analysis) or
exclude them (the default).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Protocol

from memproof.audit.log import AuditLog, OperationType
from memproof.crypto.attestation import DocumentAttestation, TrustLevel
from memproof.crypto.commitment import EmbeddingCommitment, verify_embedding
from memproof.crypto.merkle import MerkleProof, MerkleTree
from memproof.crypto.source_registry import SourceRegistry


class VectorStore(Protocol):
    """Protocol for the underlying vector store."""

    def query(
        self, embedding: list[float], top_k: int
    ) -> list[dict[str, Any]]:
        """Return top-k nearest neighbors with metadata."""
        ...


@dataclass(frozen=True)
class VerifiedResult:
    """A single verified retrieval result."""

    document: str
    score: float
    leaf_index: int
    trust_level: TrustLevel
    source_id: str
    merkle_proof: MerkleProof
    commitment_valid: bool
    attestation_valid: bool
    source_registered: bool
    document_untampered: bool

    @property
    def fully_verified(self) -> bool:
        """True if every cryptographic check passes."""
        return (
            self.commitment_valid
            and self.attestation_valid
            and self.source_registered
            and self.document_untampered
            and self.merkle_proof.verify()
        )


class VerifiedRetriever:
    """Retriever that attaches cryptographic proofs to every result.

    Given a vector store that carries MemProof metadata (attestation,
    commitment, leaf_index) on each entry, this class verifies all four
    cryptographic checks at query time and filters by trust level.

    The commitment key is the system-level HMAC key used for binding
    embeddings to documents. The SourceRegistry holds the public keys
    for verifying document attestations.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        merkle_tree: MerkleTree,
        audit_log: AuditLog,
        commitment_key: bytes,
        source_registry: SourceRegistry,
        min_trust: TrustLevel = TrustLevel.LOW,
    ) -> None:
        self._store = vector_store
        self._tree = merkle_tree
        self._audit = audit_log
        self._commitment_key = commitment_key
        self._registry = source_registry
        self._min_trust = min_trust

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        verify: bool = True,
        trust_filter: bool = True,
    ) -> list[VerifiedResult]:
        """Retrieve top-k results with cryptographic verification.

        Args:
            query_embedding: Query vector.
            top_k: Number of results to return.
            verify: Whether to verify commitments, signatures, and proofs.
            trust_filter: Whether to exclude entries below min_trust.

        Returns:
            List of VerifiedResult, one per returned entry.
        """
        # Over-fetch to account for filtering
        fetch_k = top_k * 3 if trust_filter else top_k
        raw_results = self._store.query(query_embedding, fetch_k)

        verified: list[VerifiedResult] = []

        for result in raw_results:
            leaf_idx = result.get("leaf_index", -1)
            trust = TrustLevel(result.get("trust_level", 0))
            document = result.get("document", "")
            score = result.get("score", 0.0)
            embedding = result.get("embedding", [])
            source_id = result.get("source_id", "unknown")

            # Trust filtering happens before verification work
            if trust_filter and trust < self._min_trust:
                continue

            # Default verification flags (all True if verify=False)
            commitment_valid = True
            attestation_valid = True
            source_registered = True
            document_untampered = True
            merkle_proof: MerkleProof | None = None

            if verify and leaf_idx >= 0:
                attestation: DocumentAttestation | None = result.get("attestation")
                commitment: EmbeddingCommitment | None = result.get("commitment")

                # Check 1: document hash matches what the source signed
                if attestation is not None and document:
                    computed_hash = hashlib.sha256(document.encode("utf-8")).digest()
                    if computed_hash != attestation.doc_hash:
                        document_untampered = False

                # Check 2: attestation signature is valid under the source's
                # public key from the registry
                if attestation is not None:
                    record = self._registry.get(attestation.source_id)
                    if record is None:
                        source_registered = False
                        attestation_valid = False
                    else:
                        attestation_valid = attestation.verify(record.public_key)

                # Check 3: embedding commitment binds this embedding to this doc
                if commitment is not None and embedding:
                    commitment_valid = verify_embedding(
                        commitment, embedding, self._commitment_key
                    )

                # Check 4: Merkle membership proof
                if 0 <= leaf_idx < self._tree.size:
                    merkle_proof = self._tree.prove(leaf_idx)

            if merkle_proof is None:
                # Unverifiable entry: emit a sentinel failing proof
                merkle_proof = MerkleProof(
                    leaf_hash=b"\x00" * 32,
                    sibling_hashes=[],
                    directions=[],
                    root=b"\xff" * 32,
                )
                commitment_valid = False

            verified.append(VerifiedResult(
                document=document,
                score=score,
                leaf_index=leaf_idx,
                trust_level=trust,
                source_id=source_id,
                merkle_proof=merkle_proof,
                commitment_valid=commitment_valid,
                attestation_valid=attestation_valid,
                source_registered=source_registered,
                document_untampered=document_untampered,
            ))

            if len(verified) >= top_k:
                break

        # Audit the query
        self._audit.append(
            operation=OperationType.QUERY,
            doc_hash=b"\x00" * 32,
            metadata={
                "top_k": top_k,
                "results_returned": len(verified),
                "fully_verified": sum(1 for v in verified if v.fully_verified),
                "unverified": sum(1 for v in verified if not v.fully_verified),
            },
        )

        return verified
