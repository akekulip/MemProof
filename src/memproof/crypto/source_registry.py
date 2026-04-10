"""SourceRegistry: public key distribution for MemProof attestations.

Maps each source identifier to an Ed25519 public key and a default trust
level. The registry is the trust root: a document attestation is only
accepted if the claimed source_id is in the registry and the attestation
verifies against the registered public key.

Deployment model
----------------
The registry is a simple mapping (source_id -> public_key, trust_level)
maintained by the RAG system operator. Adding a new source requires
out-of-band verification of the public key (e.g., via TLS, physical
media, or an existing PKI). Once registered, the source's signatures
are unforgeable under Ed25519 EUF-CMA.

Compared to a full PKI:
  - No certificate chains
  - No revocation lists (revocation is explicit deletion)
  - No hierarchical authorities
  - Simple enough to implement and audit in a single-tenant deployment

For multi-organization deployments, the registry can be composed with
a transparency log (e.g., Sigstore Rekor) for auditability of key
additions and removals.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from memproof.crypto.attestation import (
    DocumentAttestation,
    TrustLevel,
    generate_source_keypair,
    private_key_bytes,
    private_key_from_bytes,
    public_key_bytes,
    public_key_from_bytes,
)


@dataclass
class SourceRecord:
    """A registered source in the SourceRegistry."""

    source_id: str
    public_key: Ed25519PublicKey
    default_trust: TrustLevel
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "public_key": public_key_bytes(self.public_key).hex(),
            "default_trust": int(self.default_trust),
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SourceRecord":
        return cls(
            source_id=data["source_id"],
            public_key=public_key_from_bytes(bytes.fromhex(data["public_key"])),
            default_trust=TrustLevel(data["default_trust"]),
            description=data.get("description", ""),
        )


class SourceRegistry:
    """In-memory source registry with JSON persistence.

    Usage:
        registry = SourceRegistry()
        priv, pub = registry.register_source("wikipedia", TrustLevel.HIGH)
        # priv stays with the source, pub is in the registry

        # Later, when verifying an attestation:
        record = registry.get("wikipedia")
        if attestation.verify(record.public_key):
            ...
    """

    def __init__(self) -> None:
        self._sources: dict[str, SourceRecord] = {}

    def register_source(
        self,
        source_id: str,
        default_trust: TrustLevel = TrustLevel.MEDIUM,
        description: str = "",
    ) -> tuple[Ed25519PrivateKey, Ed25519PublicKey]:
        """Register a new source and return its generated keypair.

        The private key is returned to the caller, who is responsible for
        distributing it to the actual source. The public key is stored in
        the registry for later verification.

        Args:
            source_id: Unique identifier for the source.
            default_trust: Default trust level for documents from this source.
            description: Human-readable description.

        Returns:
            (private_key, public_key) tuple.

        Raises:
            ValueError: If source_id already exists.
        """
        if source_id in self._sources:
            raise ValueError(f"Source {source_id!r} already registered")

        private_key, public_key = generate_source_keypair()
        self._sources[source_id] = SourceRecord(
            source_id=source_id,
            public_key=public_key,
            default_trust=default_trust,
            description=description,
        )
        return private_key, public_key

    def add_source(
        self,
        source_id: str,
        public_key: Ed25519PublicKey | bytes,
        default_trust: TrustLevel = TrustLevel.MEDIUM,
        description: str = "",
    ) -> None:
        """Add a source with an externally-generated public key.

        Used when the source already has a keypair (e.g., a pre-existing
        Ed25519 key from another system).
        """
        if source_id in self._sources:
            raise ValueError(f"Source {source_id!r} already registered")

        if isinstance(public_key, bytes):
            public_key = public_key_from_bytes(public_key)

        self._sources[source_id] = SourceRecord(
            source_id=source_id,
            public_key=public_key,
            default_trust=default_trust,
            description=description,
        )

    def get(self, source_id: str) -> SourceRecord | None:
        """Look up a source by ID. Returns None if not registered."""
        return self._sources.get(source_id)

    def remove_source(self, source_id: str) -> bool:
        """Remove a source from the registry. Returns True if removed."""
        return self._sources.pop(source_id, None) is not None

    def verify_attestation(self, attestation: DocumentAttestation) -> bool:
        """Verify an attestation against the registered public key for its source.

        Returns False if:
          - The source is not registered
          - The signature does not verify against the registered public key
        """
        record = self._sources.get(attestation.source_id)
        if record is None:
            return False
        return attestation.verify(record.public_key)

    def __contains__(self, source_id: str) -> bool:
        return source_id in self._sources

    def __len__(self) -> int:
        return len(self._sources)

    def __iter__(self) -> Iterator[SourceRecord]:
        return iter(self._sources.values())

    def save(self, path: Path | str) -> None:
        """Persist the registry to a JSON file. Only stores public keys."""
        data = {"sources": [r.to_dict() for r in self._sources.values()]}
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path | str) -> "SourceRegistry":
        """Load a registry from a JSON file."""
        data = json.loads(Path(path).read_text())
        registry = cls()
        for record_data in data.get("sources", []):
            record = SourceRecord.from_dict(record_data)
            registry._sources[record.source_id] = record
        return registry


@dataclass
class SourceKeyPair:
    """Convenience holder for a source's keypair outside the registry.

    This is what the source itself holds. The private key is used to
    sign attestations; the registry only holds the public key.
    """

    source_id: str
    private_key: Ed25519PrivateKey
    public_key: Ed25519PublicKey

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "private_key": private_key_bytes(self.private_key).hex(),
            "public_key": public_key_bytes(self.public_key).hex(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SourceKeyPair":
        return cls(
            source_id=data["source_id"],
            private_key=private_key_from_bytes(bytes.fromhex(data["private_key"])),
            public_key=public_key_from_bytes(bytes.fromhex(data["public_key"])),
        )

    def save(self, path: Path | str) -> None:
        """Persist the keypair to a JSON file. WARNING: includes private key."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path | str) -> "SourceKeyPair":
        return cls.from_dict(json.loads(Path(path).read_text()))
