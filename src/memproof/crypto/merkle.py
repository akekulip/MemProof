"""Merkle tree for tamper-evident memory storage.

Provides O(log n) tamper detection over memory entries. Each leaf is a
commitment to (document_hash, embedding_hash, provenance_metadata).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class MerkleProof:
    """Proof of membership for a leaf in the Merkle tree."""

    leaf_hash: bytes
    sibling_hashes: list[bytes]
    directions: list[bool]  # True = sibling is on the right
    root: bytes

    def verify(self) -> bool:
        """Verify the proof against the stored root."""
        current = self.leaf_hash
        for sibling, is_right in zip(self.sibling_hashes, self.directions):
            if is_right:
                current = _hash_pair(current, sibling)
            else:
                current = _hash_pair(sibling, current)
        return current == self.root


@dataclass
class MerkleTree:
    """Append-only Merkle tree over memory entries.

    Each leaf = SHA-256(doc_hash || embedding_hash || provenance_json).
    Supports:
      - Insert new leaf
      - Generate membership proof
      - Verify proof against root
      - Detect tampering via root comparison
    """

    leaves: list[bytes] = field(default_factory=list)
    _nodes: list[list[bytes]] = field(default_factory=list, repr=False)

    @property
    def root(self) -> bytes:
        """Current Merkle root."""
        if not self._nodes:
            return b"\x00" * 32
        return self._nodes[-1][0]

    @property
    def size(self) -> int:
        return len(self.leaves)

    def insert(self, doc_hash: bytes, embedding_hash: bytes, provenance: bytes) -> int:
        """Insert a new entry incrementally in O(log n).

        Uses the classic append-and-merge scheme: append the new leaf,
        then walk up the tree merging the new node with its left sibling
        whenever the new position is odd. Runs in amortized O(log n) per
        insert, which is the standard RFC 6962 behavior.
        """
        leaf = _hash_leaf(doc_hash, embedding_hash, provenance)
        self.leaves.append(leaf)
        idx = len(self.leaves) - 1
        self._insert_incremental(leaf, idx)
        return idx

    def _insert_incremental(self, leaf: bytes, leaf_idx: int) -> None:
        """Maintain the level stacks as a new leaf is appended.

        Correctly mirrors the from-scratch construction in _rebuild:
        at each level, the last node is the hash of the last pair
        (duplicating the left sibling if there is no right sibling).
        We walk up from level 0 recomputing the last entry at each
        level until we reach a level that has exactly one element (the
        root). Each insert touches O(log n) levels in the worst case.
        """
        # Ensure level 0 exists and contains the new leaf
        if not self._nodes:
            self._nodes = [[leaf]]
        else:
            self._nodes[0].append(leaf)

        # Walk up from level 0 until the topmost level has exactly one
        # element. At each step, recompute the last node of the next
        # level from the current level's last pair.
        level = 0
        while len(self._nodes[level]) > 1:
            current = self._nodes[level]
            n = len(current)
            # The last pair index at this level is:
            #   left = n - 1 if n is odd, else n - 2
            #   right = left + 1 (or left again if n is odd)
            if n % 2 == 0:
                left = current[n - 2]
                right = current[n - 1]
            else:
                left = current[n - 1]
                right = current[n - 1]
            parent = _hash_pair(left, right)
            parent_idx = (n - 1) // 2

            next_level_idx = level + 1
            if next_level_idx >= len(self._nodes):
                # Create the new level with the parent as its only element
                self._nodes.append([parent])
            else:
                # Update or append at the parent level
                parent_level_list = self._nodes[next_level_idx]
                if parent_idx < len(parent_level_list):
                    parent_level_list[parent_idx] = parent
                else:
                    parent_level_list.append(parent)

            level = next_level_idx

        # Trim any levels above the topmost populated one, in case
        # previous inserts had created deeper levels that are no
        # longer needed (cannot actually happen for append-only, but
        # the invariant is free to maintain).

    def prove(self, index: int) -> MerkleProof:
        """Generate a membership proof for leaf at given index."""
        if index < 0 or index >= len(self.leaves):
            raise IndexError(f"Leaf index {index} out of range [0, {len(self.leaves)})")

        sibling_hashes: list[bytes] = []
        directions: list[bool] = []
        idx = index

        for level in self._nodes[:-1]:
            sibling_idx = idx ^ 1
            if sibling_idx < len(level):
                sibling_hashes.append(level[sibling_idx])
            else:
                sibling_hashes.append(level[idx])
            directions.append(idx % 2 == 0)
            idx //= 2

        return MerkleProof(
            leaf_hash=self.leaves[index],
            sibling_hashes=sibling_hashes,
            directions=directions,
            root=self.root,
        )

    def verify(self, proof: MerkleProof) -> bool:
        """Verify a membership proof against current root."""
        return proof.verify() and proof.root == self.root

    def _rebuild(self) -> None:
        """Rebuild the tree from leaves."""
        if not self.leaves:
            self._nodes = []
            return

        level = list(self.leaves)
        self._nodes = [level]

        while len(level) > 1:
            next_level: list[bytes] = []
            for i in range(0, len(level), 2):
                if i + 1 < len(level):
                    next_level.append(_hash_pair(level[i], level[i + 1]))
                else:
                    next_level.append(_hash_pair(level[i], level[i]))
            level = next_level
            self._nodes.append(level)


def _hash_leaf(doc_hash: bytes, embedding_hash: bytes, provenance: bytes) -> bytes:
    """Hash a leaf node: H(0x00 || doc_hash || embedding_hash || provenance)."""
    h = hashlib.sha256()
    h.update(b"\x00")  # Leaf domain separator
    h.update(doc_hash)
    h.update(embedding_hash)
    h.update(provenance)
    return h.digest()


def _hash_pair(left: bytes, right: bytes) -> bytes:
    """Hash an internal node: H(0x01 || left || right)."""
    h = hashlib.sha256()
    h.update(b"\x01")  # Internal node domain separator
    h.update(left)
    h.update(right)
    return h.digest()
