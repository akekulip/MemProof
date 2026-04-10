"""ChromaDB-backed vector store for MemProof evaluation.

Wraps ChromaDB to store MemProof metadata (attestation, commitment,
trust level, leaf index) alongside embeddings, enabling realistic
latency and scalability measurements.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """ChromaDB-backed vector store compatible with MemProof.

    Stores documents with embeddings and MemProof metadata. Supports
    the VectorStore protocol expected by VerifiedRetriever.
    """

    def __init__(
        self,
        collection_name: str = "memproof",
        persist_dir: str | None = None,
    ) -> None:
        if persist_dir:
            self._client = chromadb.PersistentClient(path=persist_dir)
        else:
            self._client = chromadb.Client()

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._next_id = self._collection.count()

    def add(
        self,
        document: str,
        embedding: list[float],
        leaf_index: int,
        trust_level: int,
        source_id: str,
        attestation_bytes: bytes | None = None,
        commitment_bytes: bytes | None = None,
    ) -> str:
        """Add a document with MemProof metadata.

        Returns:
            The document ID in ChromaDB.
        """
        doc_id = f"mp_{self._next_id}"
        self._next_id += 1

        metadata: dict[str, Any] = {
            "leaf_index": leaf_index,
            "trust_level": trust_level,
            "source_id": source_id,
        }

        if attestation_bytes:
            metadata["attestation_hex"] = attestation_bytes.hex()
        if commitment_bytes:
            metadata["commitment_hex"] = commitment_bytes.hex()

        self._collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[document],
            metadatas=[metadata],
        )

        return doc_id

    def query(
        self,
        embedding: list[float],
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Query for nearest neighbors.

        Returns results in the format expected by VerifiedRetriever.
        """
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, self._collection.count()),
            include=["documents", "embeddings", "metadatas", "distances"],
        )

        output: list[dict[str, Any]] = []

        if not results["ids"] or not results["ids"][0]:
            return output

        for i, doc_id in enumerate(results["ids"][0]):
            doc = results["documents"][0][i] if results["documents"] else ""
            emb_raw = (
                results["embeddings"][0][i]
                if results["embeddings"] is not None
                else None
            )
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            dist = results["distances"][0][i] if results["distances"] else 1.0

            # ChromaDB returns distances; convert to similarity score
            score = 1.0 - dist

            emb_list = list(emb_raw) if emb_raw is not None else []

            output.append({
                "document": doc,
                "embedding": emb_list,
                "score": score,
                "leaf_index": meta.get("leaf_index", -1),
                "trust_level": meta.get("trust_level", 0),
                "source_id": meta.get("source_id", "unknown"),
            })

        return output

    @property
    def size(self) -> int:
        return self._collection.count()

    def clear(self) -> None:
        """Remove all entries."""
        ids = self._collection.get()["ids"]
        if ids:
            self._collection.delete(ids=ids)
        self._next_id = 0
