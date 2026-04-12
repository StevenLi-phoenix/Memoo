"""Tests for embedding and semantic search."""

from core.embeddings import _local_embed, cosine_similarity, deserialize_embedding, serialize_embedding


class TestLocalEmbed:
    def test_produces_vector(self):
        vec = _local_embed("hello world")
        assert isinstance(vec, list)
        assert len(vec) == 256

    def test_unit_vector(self):
        import math

        vec = _local_embed("test input")
        norm = math.sqrt(sum(v * v for v in vec))
        assert abs(norm - 1.0) < 0.01

    def test_similar_texts(self):
        a = _local_embed("the cat sat on the mat")
        b = _local_embed("the cat sat on the rug")
        c = _local_embed("quantum physics equations")
        # a and b should be more similar than a and c
        assert cosine_similarity(a, b) > cosine_similarity(a, c)

    def test_identical_texts(self):
        a = _local_embed("exact same text")
        b = _local_embed("exact same text")
        assert abs(cosine_similarity(a, b) - 1.0) < 0.001


class TestSerialization:
    def test_roundtrip(self):
        vec = _local_embed("test")
        serialized = serialize_embedding(vec)
        restored = deserialize_embedding(serialized)
        assert len(restored) == len(vec)
        for a, b in zip(vec, restored):
            assert abs(a - b) < 1e-5
