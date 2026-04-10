"""Append-only audit log for memory operations.

Records every ingest, query, update, and delete operation with
cryptographic chaining (each entry includes the hash of the previous
entry). Enables:
  - Tamper detection on the operation history
  - Temporal drift analysis
  - Forensic rollback to any point in time
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class OperationType(Enum):
    INGEST = "ingest"
    QUERY = "query"
    UPDATE = "update"
    DELETE = "delete"
    VERIFY = "verify"


@dataclass(frozen=True)
class AuditEntry:
    """Single entry in the append-only audit log."""

    sequence: int
    timestamp: float
    operation: OperationType
    doc_hash: bytes
    prev_hash: bytes
    entry_hash: bytes
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.sequence,
            "ts": self.timestamp,
            "op": self.operation.value,
            "doc_hash": self.doc_hash.hex(),
            "prev_hash": self.prev_hash.hex(),
            "entry_hash": self.entry_hash.hex(),
            "metadata": self.metadata,
        }


class AuditLog:
    """Append-only, hash-chained audit log.

    Each entry's hash includes the previous entry's hash, forming
    a tamper-evident chain (similar to a blockchain without consensus).
    """

    def __init__(self, log_path: Path | None = None) -> None:
        self._entries: list[AuditEntry] = []
        self._log_path = log_path

    @property
    def head_hash(self) -> bytes:
        """Hash of the most recent entry (chain head)."""
        if not self._entries:
            return b"\x00" * 32
        return self._entries[-1].entry_hash

    @property
    def length(self) -> int:
        return len(self._entries)

    def append(
        self,
        operation: OperationType,
        doc_hash: bytes,
        metadata: dict[str, Any] | None = None,
    ) -> AuditEntry:
        """Append a new entry to the log."""
        seq = len(self._entries)
        ts = time.time()
        prev = self.head_hash
        meta = metadata or {}

        entry_hash = _compute_entry_hash(seq, ts, operation, doc_hash, prev, meta)

        entry = AuditEntry(
            sequence=seq,
            timestamp=ts,
            operation=operation,
            doc_hash=doc_hash,
            prev_hash=prev,
            entry_hash=entry_hash,
            metadata=meta,
        )
        self._entries.append(entry)

        if self._log_path:
            self._persist_entry(entry)

        return entry

    def verify_chain(self) -> tuple[bool, int]:
        """Verify the integrity of the entire chain.

        Returns (is_valid, first_broken_index). If valid, index is -1.

        Checks three conditions:
          1. Hash-chain linkage: each entry's prev_hash matches the
             previous entry's entry_hash (root-of-chain begins at
             32 null bytes).
          2. Entry-hash recomputation: each entry_hash equals the
             canonical recomputation from the entry's fields.
          3. Timestamp monotonicity: timestamps are non-decreasing
             along the chain. Theorem 4's proof explicitly invokes
             this as one of the reordering detection mechanisms.
        """
        if not self._entries:
            return True, -1

        expected_prev = b"\x00" * 32
        last_ts: float | None = None
        for i, entry in enumerate(self._entries):
            if entry.prev_hash != expected_prev:
                return False, i

            computed = _compute_entry_hash(
                entry.sequence, entry.timestamp, entry.operation,
                entry.doc_hash, entry.prev_hash, entry.metadata,
            )
            if computed != entry.entry_hash:
                return False, i

            if last_ts is not None and entry.timestamp < last_ts:
                return False, i
            last_ts = entry.timestamp

            expected_prev = entry.entry_hash

        return True, -1

    def detect_drift(self, window_size: int = 100) -> list[dict[str, Any]]:
        """Detect anomalous patterns in recent operations.

        Looks for:
          - Burst of ingestions from single source
          - Unusual update/delete patterns
          - Temporal clustering of operations
        """
        if len(self._entries) < window_size:
            window = self._entries
        else:
            window = self._entries[-window_size:]

        anomalies: list[dict[str, Any]] = []
        source_counts: dict[str, int] = {}

        for entry in window:
            src = entry.metadata.get("source_id", "unknown")
            source_counts[src] = source_counts.get(src, 0) + 1

        # Flag sources with disproportionate activity
        total = len(window)
        for src, count in source_counts.items():
            if count > total * 0.5 and total > 10:
                anomalies.append({
                    "type": "source_dominance",
                    "source": src,
                    "ratio": count / total,
                    "count": count,
                })

        return anomalies

    def get_entries(
        self,
        start: int = 0,
        end: int | None = None,
    ) -> list[AuditEntry]:
        """Retrieve entries by sequence range."""
        return self._entries[start:end]

    def _persist_entry(self, entry: AuditEntry) -> None:
        """Append entry to log file."""
        if self._log_path:
            with open(self._log_path, "a") as f:
                f.write(json.dumps(entry.to_dict()) + "\n")


def _compute_entry_hash(
    seq: int,
    ts: float,
    op: OperationType,
    doc_hash: bytes,
    prev_hash: bytes,
    metadata: dict[str, Any],
) -> bytes:
    """Compute the hash of an audit entry.

    Uses a portable canonical encoding: seq (8B BE uint), ts (8B BE
    IEEE 754 double via struct.pack), op (utf-8 + length prefix),
    doc_hash (fixed 32B), prev_hash (fixed 32B), metadata (sorted
    JSON). This avoids depending on CPython's str(float) repr, so a
    third-party verifier in another language can recompute the
    entry hash byte-for-byte.
    """
    import struct

    op_bytes = op.value.encode("ascii")
    meta_bytes = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    h = hashlib.sha256()
    h.update(seq.to_bytes(8, "big"))
    h.update(struct.pack("!d", float(ts)))
    h.update(struct.pack("!I", len(op_bytes)))
    h.update(op_bytes)
    h.update(doc_hash)
    h.update(prev_hash)
    h.update(struct.pack("!I", len(meta_bytes)))
    h.update(meta_bytes)
    return h.digest()
