"""Document attestation using Ed25519 digital signatures.

Each source holds an Ed25519 keypair. The private key signs attestations;
the public key (registered in a SourceRegistry) verifies them. Unlike a
shared-secret MAC, this provides:

  - Public verifiability: anyone with the source's public key can verify
  - Non-repudiation: the source cannot later deny a signature
  - EUF-CMA security under the Ed25519 signature scheme

A document attestation binds:
  - Content hash (SHA-256)
  - Source identifier
  - Timestamp
  - Trust level
  - Arbitrary metadata

Inspired by C2PA content credentials and SLSA provenance attestations.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PrivateFormat,
    PublicFormat,
    NoEncryption,
)


class TrustLevel(IntEnum):
    """Trust levels for document provenance."""

    UNTRUSTED = 0       # Unknown source
    LOW = 1             # User-provided, unverified
    MEDIUM = 2          # Known source, baseline trust
    HIGH = 3            # Verified pipeline source
    AUTHORITATIVE = 4   # First-party authoritative source


@dataclass(frozen=True)
class DocumentAttestation:
    """Provenance attestation for a document entering memory.

    The signature is an Ed25519 signature over the canonical serialization
    of (doc_hash, source_id, timestamp, trust_level, metadata). Verification
    requires the source's public key, typically looked up from a
    SourceRegistry.
    """

    doc_hash: bytes
    source_id: str
    timestamp: float
    trust_level: TrustLevel
    metadata: dict[str, Any]
    signature: bytes

    def verify(self, public_key: Ed25519PublicKey | bytes) -> bool:
        """Verify the attestation signature with the source's public key.

        Args:
            public_key: Ed25519PublicKey object or 32-byte raw public key.

        Returns:
            True if the signature is valid, False otherwise.
        """
        if isinstance(public_key, bytes):
            public_key = Ed25519PublicKey.from_public_bytes(public_key)

        message = _canonical_message(
            self.doc_hash,
            self.source_id,
            self.timestamp,
            self.trust_level,
            self.metadata,
        )

        try:
            public_key.verify(self.signature, message)
            return True
        except InvalidSignature:
            return False

    def to_bytes(self) -> bytes:
        """Serialize attestation for Merkle tree inclusion (JSON form).

        This is the human-readable serialization: canonical JSON with
        sorted keys and no whitespace. It is used by the reference
        Merkle leaf construction and by the external source-registry
        persistence path. For storage-sensitive deployments, use
        :meth:`to_bytes_binary` instead; the binary form is roughly
        55% smaller at the cost of not being diff-friendly.
        """
        data = {
            "doc_hash": self.doc_hash.hex(),
            "source_id": self.source_id,
            "timestamp": self.timestamp,
            "trust_level": int(self.trust_level),
            "metadata": self.metadata,
            "signature": self.signature.hex(),
        }
        return json.dumps(data, sort_keys=True, separators=(",", ":")).encode()

    def to_bytes_binary(self) -> bytes:
        """Packed binary serialization for compact storage.

        Layout (all integers big-endian):

            signature        64 B  fixed
            doc_hash         32 B  fixed
            timestamp         8 B  IEEE 754 double
            trust_level       1 B  uint8
            src_id_len        2 B  uint16 (length-prefixed utf-8)
            src_id          var B
            meta_len          2 B  uint16 (length-prefixed canonical JSON)
            meta_json       var B

        The signature, document hash, and trust level dominate the
        total. For a typical source_id of 6-10 ASCII characters and
        empty metadata, the encoded length is ~117 bytes. The inverse
        function :meth:`from_bytes_binary` parses this encoding back
        into a DocumentAttestation so deployments can round-trip
        through the compact form.
        """
        import struct

        src_bytes = self.source_id.encode("utf-8")
        meta_json = json.dumps(
            self.metadata, sort_keys=True, separators=(",", ":"),
        ).encode()

        if len(self.signature) != 64:
            raise ValueError(
                f"Ed25519 signatures must be 64 bytes, got {len(self.signature)}",
            )
        if len(self.doc_hash) != 32:
            raise ValueError(
                f"SHA-256 hashes must be 32 bytes, got {len(self.doc_hash)}",
            )
        if len(src_bytes) > 0xFFFF:
            raise ValueError("source_id too long for uint16 length prefix")
        if len(meta_json) > 0xFFFF:
            raise ValueError("metadata JSON too long for uint16 length prefix")

        return (
            self.signature
            + self.doc_hash
            + struct.pack("!d", float(self.timestamp))
            + struct.pack("!B", int(self.trust_level))
            + struct.pack("!H", len(src_bytes))
            + src_bytes
            + struct.pack("!H", len(meta_json))
            + meta_json
        )

    @classmethod
    def from_bytes_binary(cls, data: bytes) -> "DocumentAttestation":
        """Parse a binary-encoded attestation produced by to_bytes_binary."""
        import struct

        if len(data) < 64 + 32 + 8 + 1 + 2 + 2:
            raise ValueError("binary attestation truncated")

        offset = 0
        signature = data[offset:offset + 64]
        offset += 64
        doc_hash = data[offset:offset + 32]
        offset += 32
        (timestamp,) = struct.unpack("!d", data[offset:offset + 8])
        offset += 8
        (trust_level_int,) = struct.unpack("!B", data[offset:offset + 1])
        offset += 1
        (src_id_len,) = struct.unpack("!H", data[offset:offset + 2])
        offset += 2
        src_id = data[offset:offset + src_id_len].decode("utf-8")
        offset += src_id_len
        (meta_len,) = struct.unpack("!H", data[offset:offset + 2])
        offset += 2
        meta_json = data[offset:offset + meta_len]
        offset += meta_len

        if offset != len(data):
            raise ValueError(
                f"trailing bytes in binary attestation: "
                f"parsed {offset} of {len(data)}",
            )

        metadata = json.loads(meta_json.decode()) if meta_json else {}
        return cls(
            doc_hash=doc_hash,
            source_id=src_id,
            timestamp=timestamp,
            trust_level=TrustLevel(trust_level_int),
            metadata=metadata,
            signature=signature,
        )


def generate_source_keypair() -> tuple[Ed25519PrivateKey, Ed25519PublicKey]:
    """Generate a new Ed25519 keypair for a source.

    Returns:
        Tuple of (private_key, public_key).
    """
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    return private_key, public_key


def private_key_bytes(key: Ed25519PrivateKey) -> bytes:
    """Serialize an Ed25519 private key to raw 32-byte form."""
    return key.private_bytes(
        encoding=Encoding.Raw,
        format=PrivateFormat.Raw,
        encryption_algorithm=NoEncryption(),
    )


def public_key_bytes(key: Ed25519PublicKey) -> bytes:
    """Serialize an Ed25519 public key to raw 32-byte form."""
    return key.public_bytes(
        encoding=Encoding.Raw,
        format=PublicFormat.Raw,
    )


def private_key_from_bytes(data: bytes) -> Ed25519PrivateKey:
    """Load an Ed25519 private key from raw 32-byte form."""
    return Ed25519PrivateKey.from_private_bytes(data)


def public_key_from_bytes(data: bytes) -> Ed25519PublicKey:
    """Load an Ed25519 public key from raw 32-byte form."""
    return Ed25519PublicKey.from_public_bytes(data)


def attest_document(
    content: bytes | str,
    source_id: str,
    trust_level: TrustLevel,
    private_key: Ed25519PrivateKey,
    metadata: dict[str, Any] | None = None,
) -> DocumentAttestation:
    """Create a signed provenance attestation for a document.

    Args:
        content: Raw document content.
        source_id: Identifier for the source (URI, author, pipeline name).
        trust_level: Trust level assigned to this source.
        private_key: The source's Ed25519 private signing key.
        metadata: Additional provenance metadata.

    Returns:
        A DocumentAttestation signed by the source.
    """
    if isinstance(content, str):
        content = content.encode("utf-8")

    doc_hash = hashlib.sha256(content).digest()
    ts = time.time()
    meta = metadata or {}

    message = _canonical_message(doc_hash, source_id, ts, trust_level, meta)
    signature = private_key.sign(message)

    return DocumentAttestation(
        doc_hash=doc_hash,
        source_id=source_id,
        timestamp=ts,
        trust_level=trust_level,
        metadata=meta,
        signature=signature,
    )


def _canonical_message(
    doc_hash: bytes,
    source_id: str,
    timestamp: float,
    trust_level: TrustLevel,
    metadata: dict[str, Any],
) -> bytes:
    """Produce the canonical byte string that gets signed.

    Uses explicit length-prefixed TLV encoding so the mapping from
    (doc_hash, source_id, timestamp, trust_level, metadata) to bytes
    is injective. Each field is prefixed with a 4-byte big-endian
    unsigned length. This prevents ambiguity between, for example,
    source_id="ab|cd" and source_id="ab" with extra data concatenated
    by the legacy "|" separator.

    Layout:
        doc_hash_len  (4B BE uint)  doc_hash (variable)
        src_id_len    (4B BE uint)  src_id_utf8 (variable)
        ts_packed     (8B BE double, IEEE 754)
        trust_level   (4B BE int)
        meta_len      (4B BE uint)  meta_json (variable)
    """
    import struct

    src_bytes = source_id.encode("utf-8")
    meta_json = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    return (
        struct.pack("!I", len(doc_hash)) + doc_hash
        + struct.pack("!I", len(src_bytes)) + src_bytes
        + struct.pack("!d", float(timestamp))
        + struct.pack("!i", int(trust_level))
        + struct.pack("!I", len(meta_json)) + meta_json
    )
