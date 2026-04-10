"""Tests for attack simulators."""

import hashlib

from memproof.attacks.minja import MINJAAttack
from memproof.attacks.poisoned_rag import PoisonedRAGAttack
from memproof.crypto.attestation import TrustLevel
from memproof.store import MemProofStore


def _dummy_embed(text: str) -> list[float]:
    h = hashlib.sha256(text.encode()).digest()
    return [float(b) / 255.0 for b in h[:16]]


class TestPoisonedRAGAttack:
    def test_craft_poison(self) -> None:
        atk = PoisonedRAGAttack(embed_fn=_dummy_embed)
        poison = atk.craft_poison(
            "What is X?", "X is Y", num_poison=5,
        )
        assert len(poison.poisoned_docs) == 5
        assert len(poison.poison_embeddings) == 5
        assert all("X is Y" in doc for doc in poison.poisoned_docs)

    def test_poisoned_docs_are_diverse(self) -> None:
        atk = PoisonedRAGAttack(embed_fn=_dummy_embed)
        poison = atk.craft_poison("query", "answer", num_poison=5)
        # All docs should be unique
        assert len(set(poison.poisoned_docs)) == 5

    def test_evaluate_retrieval(self) -> None:
        atk = PoisonedRAGAttack(embed_fn=_dummy_embed)
        poison = atk.craft_poison("What is X?", "X is Y", num_poison=3)

        def mock_retrieval(q: str, k: int) -> list[dict]:
            # Simulate: first result is poisoned, rest are clean
            return [
                {"document": poison.poisoned_docs[0], "score": 0.9},
                {"document": "clean doc 1", "score": 0.8},
                {"document": "clean doc 2", "score": 0.7},
            ]

        result = atk.evaluate_retrieval(poison, mock_retrieval, top_k=3, corpus_size=1000)
        assert result.retrieval_success == 1
        assert result.attack_success  # "X is Y" is in the poisoned doc


class TestMINJAAttack:
    def test_craft_injection(self) -> None:
        atk = MINJAAttack()
        payload = atk.craft_injection(
            "What is the capital?", "The capital is Berlin", num_injections=3,
        )
        assert len(payload.injection_queries) == 3
        assert all("Berlin" in q for q in payload.injection_queries)

    def test_trust_filtering_blocks_minja(self) -> None:
        """MemProof trust filtering should block low-trust MINJA injections."""
        store = MemProofStore(embed_fn=_dummy_embed, min_trust=TrustLevel.MEDIUM)

        store.register_source("trusted", TrustLevel.HIGH)
        store.register_source("user:adversary", TrustLevel.LOW)

        # Ingest benign corpus
        for i in range(10):
            store.ingest(f"Benign document {i}", source_id="trusted")

        # Attacker injects via MINJA (LOW trust)
        store.ingest("The capital of France is Berlin", source_id="user:adversary")

        # Query with trust filter — adversarial entry should be excluded
        results = store.query("capital of France", top_k=5, trust_filter=True)
        for r in results:
            assert "Berlin" not in r.document
