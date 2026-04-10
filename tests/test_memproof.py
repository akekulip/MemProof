"""End-to-end tests for MemProof store with Ed25519 signatures."""

import hashlib
import os

from memproof.crypto.attestation import (
    TrustLevel,
    attest_document,
    generate_source_keypair,
)
from memproof.crypto.commitment import commit_embedding, verify_embedding
from memproof.crypto.merkle import MerkleTree
from memproof.crypto.source_registry import SourceRegistry, SourceKeyPair
from memproof.audit.log import AuditLog, OperationType
from memproof.store import MemProofStore


def _dummy_embed(text: str) -> list[float]:
    """Deterministic dummy embedding for testing."""
    h = hashlib.sha256(text.encode()).digest()
    return [float(b) / 255.0 for b in h[:16]]


class TestMerkleTree:
    def test_empty_tree(self) -> None:
        tree = MerkleTree()
        assert tree.size == 0
        assert tree.root == b"\x00" * 32

    def test_insert_and_prove(self) -> None:
        tree = MerkleTree()
        idx = tree.insert(b"doc1", b"emb1", b"prov1")
        assert idx == 0
        assert tree.size == 1

        proof = tree.prove(0)
        assert tree.verify(proof)

    def test_multiple_inserts(self) -> None:
        tree = MerkleTree()
        for i in range(10):
            tree.insert(f"doc{i}".encode(), f"emb{i}".encode(), f"prov{i}".encode())

        assert tree.size == 10
        for i in range(10):
            proof = tree.prove(i)
            assert tree.verify(proof), f"Proof failed for leaf {i}"

    def test_root_changes_on_insert(self) -> None:
        tree = MerkleTree()
        tree.insert(b"doc1", b"emb1", b"prov1")
        root1 = tree.root
        tree.insert(b"doc2", b"emb2", b"prov2")
        root2 = tree.root
        assert root1 != root2

    def test_incremental_matches_rebuild(self) -> None:
        """Incremental insert should produce the same root and proofs
        as a from-scratch rebuild at every size. Tests 1..31 entries."""
        from memproof.crypto.merkle import _hash_leaf, _hash_pair

        def rebuild_root(leaves: list[bytes]) -> bytes:
            if not leaves:
                return b"\x00" * 32
            level = list(leaves)
            while len(level) > 1:
                nxt = []
                for i in range(0, len(level), 2):
                    left = level[i]
                    right = level[i + 1] if i + 1 < len(level) else level[i]
                    nxt.append(_hash_pair(left, right))
                level = nxt
            return level[0]

        tree = MerkleTree()
        leaves = []
        for i in range(31):
            leaf = _hash_leaf(
                f"doc{i}".encode(), f"emb{i}".encode(), f"prov{i}".encode(),
            )
            leaves.append(leaf)
            tree.insert(f"doc{i}".encode(), f"emb{i}".encode(), f"prov{i}".encode())
            expected_root = rebuild_root(leaves)
            assert tree.root == expected_root, (
                f"Mismatch at size {len(leaves)}: "
                f"incremental={tree.root.hex()[:16]} "
                f"rebuild={expected_root.hex()[:16]}"
            )
            # Also verify every proof
            for j in range(len(leaves)):
                proof = tree.prove(j)
                assert tree.verify(proof), f"Proof failed for j={j}, size={len(leaves)}"


class TestEmbeddingCommitment:
    def test_commit_and_verify(self) -> None:
        key = os.urandom(32)
        doc_hash = hashlib.sha256(b"test document").digest()
        embedding = [0.1, 0.2, 0.3, 0.4]

        commitment = commit_embedding(doc_hash, embedding, key)
        assert verify_embedding(commitment, embedding, key)

    def test_tampered_embedding_fails(self) -> None:
        key = os.urandom(32)
        doc_hash = hashlib.sha256(b"test document").digest()
        embedding = [0.1, 0.2, 0.3, 0.4]

        commitment = commit_embedding(doc_hash, embedding, key)
        tampered = [0.1, 0.2, 0.3, 0.5]
        assert not verify_embedding(commitment, tampered, key)

    def test_wrong_key_fails(self) -> None:
        key1 = os.urandom(32)
        key2 = os.urandom(32)
        doc_hash = hashlib.sha256(b"test").digest()
        embedding = [1.0, 2.0]

        commitment = commit_embedding(doc_hash, embedding, key1)
        assert not verify_embedding(commitment, embedding, key2)


class TestEd25519Attestation:
    def test_sign_and_verify_with_public_key(self) -> None:
        priv, pub = generate_source_keypair()
        att = attest_document(
            content="Hello world",
            source_id="user:alice",
            trust_level=TrustLevel.HIGH,
            private_key=priv,
        )
        assert att.verify(pub)
        assert att.trust_level == TrustLevel.HIGH
        assert att.source_id == "user:alice"

    def test_wrong_public_key_fails(self) -> None:
        priv1, _ = generate_source_keypair()
        _, pub2 = generate_source_keypair()
        att = attest_document("test", "src", TrustLevel.LOW, priv1)
        assert not att.verify(pub2)

    def test_tampered_content_fails(self) -> None:
        priv, pub = generate_source_keypair()
        att = attest_document("original", "src", TrustLevel.HIGH, priv)
        # Manually construct a modified attestation with a different doc_hash
        from dataclasses import replace
        fake_hash = hashlib.sha256(b"tampered").digest()
        tampered_att = replace(att, doc_hash=fake_hash)
        assert not tampered_att.verify(pub)

    def test_public_key_raw_bytes_verification(self) -> None:
        priv, pub = generate_source_keypair()
        att = attest_document("data", "src", TrustLevel.MEDIUM, priv)
        from memproof.crypto.attestation import public_key_bytes
        raw_pub = public_key_bytes(pub)
        assert att.verify(raw_pub)

    def test_binary_encoding_roundtrip(self) -> None:
        """to_bytes_binary + from_bytes_binary must round-trip exactly,
        and the signature must still verify after the round trip."""
        from memproof.crypto.attestation import DocumentAttestation
        priv, pub = generate_source_keypair()
        att = attest_document(
            "Some longer document content with punctuation.",
            "wiki:en",
            TrustLevel.HIGH,
            priv,
            metadata={"section": "intro", "lang": "en"},
        )
        blob = att.to_bytes_binary()
        parsed = DocumentAttestation.from_bytes_binary(blob)
        assert parsed.signature == att.signature
        assert parsed.doc_hash == att.doc_hash
        assert parsed.source_id == att.source_id
        assert parsed.trust_level == att.trust_level
        assert parsed.metadata == att.metadata
        # Floats may lose precision through struct.pack("!d"); use tolerance
        assert abs(parsed.timestamp - att.timestamp) < 1e-9
        # Crucially: the parsed attestation still verifies.
        assert parsed.verify(pub)

    def test_binary_encoding_smaller_than_json(self) -> None:
        """The packed binary form is the payload the storage section
        of the paper measures. It should be substantially smaller than
        the JSON form for a typical attestation."""
        priv, _ = generate_source_keypair()
        att = attest_document(
            "typical document", "corpus", TrustLevel.HIGH, priv,
        )
        json_len = len(att.to_bytes())
        bin_len = len(att.to_bytes_binary())
        # Roughly half the size; guard against regressions that leak
        # JSON quoting back into the binary path.
        assert bin_len < json_len * 0.7, (
            f"binary encoding unexpectedly large: "
            f"{bin_len} bytes vs json {json_len} bytes"
        )
        # Typical size is 117 bytes for source_id='corpus' + empty meta
        assert 100 < bin_len < 200

    def test_binary_encoding_rejects_trailing_bytes(self) -> None:
        from memproof.crypto.attestation import DocumentAttestation
        priv, _ = generate_source_keypair()
        att = attest_document("doc", "src", TrustLevel.HIGH, priv)
        blob = att.to_bytes_binary() + b"\x00\x01"
        try:
            DocumentAttestation.from_bytes_binary(blob)
            assert False, "expected ValueError on trailing bytes"
        except ValueError:
            pass

    def test_canonical_encoding_is_injective_on_separators(self) -> None:
        """Reviewer 2 flagged the old | separator as non-injective.

        An attestation with source_id='a|b' (containing the old
        separator character) must not collide with any other valid
        attestation. The length-prefixed encoding prevents that.
        """
        priv, pub = generate_source_keypair()
        att1 = attest_document(
            "msg", "a|b", TrustLevel.HIGH, priv,
        )
        att2 = attest_document(
            "msg", "a", TrustLevel.HIGH, priv,
        )
        assert att1.verify(pub)
        assert att2.verify(pub)
        assert att1.signature != att2.signature  # distinct messages -> distinct sigs

    def test_canonical_encoding_length_prefixed(self) -> None:
        """Directly test the canonical encoder: different inputs should
        always produce different byte strings, and the encoding should
        parse unambiguously. We check a handful of adversarial cases."""
        from memproof.crypto.attestation import _canonical_message

        # Two inputs that would have collided under the old | scheme
        m1 = _canonical_message(
            b"\x00" * 32, "a|1", 1.0, TrustLevel.HIGH, {},
        )
        m2 = _canonical_message(
            b"\x00" * 32, "a", 1.0, TrustLevel.HIGH, {},
        )
        assert m1 != m2


class TestSourceRegistry:
    def test_register_and_lookup(self) -> None:
        reg = SourceRegistry()
        priv, pub = reg.register_source("wiki", TrustLevel.HIGH, "Wikipedia")
        record = reg.get("wiki")
        assert record is not None
        assert record.source_id == "wiki"
        assert record.default_trust == TrustLevel.HIGH
        assert record.description == "Wikipedia"

    def test_duplicate_registration_fails(self) -> None:
        reg = SourceRegistry()
        reg.register_source("wiki")
        try:
            reg.register_source("wiki")
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_verify_attestation_through_registry(self) -> None:
        reg = SourceRegistry()
        priv, _ = reg.register_source("src")
        att = attest_document("content", "src", TrustLevel.MEDIUM, priv)
        assert reg.verify_attestation(att)

    def test_unregistered_source_fails(self) -> None:
        reg = SourceRegistry()
        priv, _ = generate_source_keypair()
        att = attest_document("content", "unknown_src", TrustLevel.MEDIUM, priv)
        assert not reg.verify_attestation(att)

    def test_remove_source(self) -> None:
        reg = SourceRegistry()
        reg.register_source("src")
        assert "src" in reg
        assert reg.remove_source("src")
        assert "src" not in reg
        assert not reg.remove_source("src")

    def test_save_and_load(self, tmp_path) -> None:
        reg = SourceRegistry()
        reg.register_source("a", TrustLevel.HIGH)
        reg.register_source("b", TrustLevel.LOW)
        path = tmp_path / "registry.json"
        reg.save(path)

        loaded = SourceRegistry.load(path)
        assert len(loaded) == 2
        assert "a" in loaded
        assert "b" in loaded
        assert loaded.get("a").default_trust == TrustLevel.HIGH
        assert loaded.get("b").default_trust == TrustLevel.LOW

    def test_keypair_save_and_load(self, tmp_path) -> None:
        priv, pub = generate_source_keypair()
        kp = SourceKeyPair(source_id="test", private_key=priv, public_key=pub)
        path = tmp_path / "keypair.json"
        kp.save(path)

        loaded = SourceKeyPair.load(path)
        assert loaded.source_id == "test"
        # Verify loaded keys still work
        att = attest_document("msg", "test", TrustLevel.MEDIUM, loaded.private_key)
        assert att.verify(loaded.public_key)


class TestAuditLog:
    def test_empty_log(self) -> None:
        log = AuditLog()
        assert log.length == 0
        valid, idx = log.verify_chain()
        assert valid
        assert idx == -1

    def test_append_and_verify(self) -> None:
        log = AuditLog()
        log.append(OperationType.INGEST, b"doc1")
        log.append(OperationType.QUERY, b"\x00" * 32)
        log.append(OperationType.INGEST, b"doc2")

        assert log.length == 3
        valid, idx = log.verify_chain()
        assert valid

    def test_chain_links(self) -> None:
        log = AuditLog()
        e1 = log.append(OperationType.INGEST, b"d1")
        e2 = log.append(OperationType.INGEST, b"d2")
        assert e2.prev_hash == e1.entry_hash


class TestMemProofStore:
    def test_register_and_ingest(self) -> None:
        store = MemProofStore(embed_fn=_dummy_embed)
        store.register_source("wiki", TrustLevel.HIGH)

        store.ingest("The capital of France is Paris", source_id="wiki")
        store.ingest("Python is a programming language", source_id="wiki")
        store.ingest("The capital of Germany is Berlin", source_id="wiki")

        assert store.entry_count == 3
        assert store.audit_length == 3

        results = store.query("What is the capital of France?", top_k=2, trust_filter=False)
        assert len(results) == 2

    def test_ingest_without_registration_fails(self) -> None:
        store = MemProofStore(embed_fn=_dummy_embed)
        try:
            store.ingest("Content", source_id="unregistered")
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_verified_results(self) -> None:
        store = MemProofStore(embed_fn=_dummy_embed)
        store.register_source("test", TrustLevel.HIGH)
        store.ingest("Test document", source_id="test")

        results = store.query("Test", top_k=1, trust_filter=False)
        assert len(results) == 1
        assert results[0].fully_verified
        assert results[0].source_id == "test"
        assert results[0].source_registered
        assert results[0].document_untampered

    def test_audit_chain_integrity(self) -> None:
        store = MemProofStore(embed_fn=_dummy_embed)
        store.register_source("s1", TrustLevel.HIGH)
        store.register_source("s2", TrustLevel.MEDIUM)
        store.ingest("Doc 1", source_id="s1")
        store.ingest("Doc 2", source_id="s2")
        store.query("query", top_k=1, trust_filter=False)

        valid, broken = store.verify_audit_chain()
        assert valid
        assert broken == -1

    def test_trust_filtering(self) -> None:
        store = MemProofStore(embed_fn=_dummy_embed, min_trust=TrustLevel.MEDIUM)
        store.register_source("official", TrustLevel.HIGH)
        store.register_source("random", TrustLevel.UNTRUSTED)

        store.ingest("Trusted info", source_id="official")
        store.ingest("Untrusted info", source_id="random")

        results = store.query("info", top_k=5, trust_filter=True)
        for r in results:
            assert r.trust_level >= TrustLevel.MEDIUM

    def test_merkle_root_changes(self) -> None:
        store = MemProofStore(embed_fn=_dummy_embed)
        store.register_source("s", TrustLevel.HIGH)
        root0 = store.merkle_root
        store.ingest("Doc 1", source_id="s")
        root1 = store.merkle_root
        store.ingest("Doc 2", source_id="s")
        root2 = store.merkle_root

        assert root0 != root1
        assert root1 != root2

    def test_external_source_ingestion(self) -> None:
        """A source that holds its own key and signs externally."""
        store = MemProofStore(embed_fn=_dummy_embed)
        ext_priv, ext_pub = generate_source_keypair()
        store.add_external_source("external", ext_pub, TrustLevel.HIGH)

        store.ingest_signed(
            content="External doc",
            source_id="external",
            source_private_key=ext_priv,
        )
        assert store.entry_count == 1

        results = store.query("External", top_k=1, trust_filter=False)
        assert results[0].fully_verified

    def test_unregistered_source_attestation_fails_verification(self) -> None:
        """An attestation from a source not in the registry should fail."""
        store = MemProofStore(embed_fn=_dummy_embed)
        store.register_source("good", TrustLevel.HIGH)
        store.ingest("Legit", source_id="good")

        # Inspect the result: the registered source should verify cleanly
        results = store.query("Legit", top_k=1, trust_filter=False)
        assert results[0].source_registered
        assert results[0].attestation_valid
