"""Tests for embedding and retrieval."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.embedding.embedder import TextEmbedder
from src.embedding.index import FAISSIndex
from src.schema import Incident, IncidentSource


class TestTextEmbedder:
    """Tests for TextEmbedder."""

    @pytest.fixture
    def embedder(self):
        """Create embedder instance."""
        return TextEmbedder(model_name="all-MiniLM-L6-v2")

    def test_embed_single_text(self, embedder):
        """Test embedding a single text."""
        text = "High latency on API endpoint"
        embedding = embedder.embed(text)

        assert isinstance(embedding, np.ndarray)
        assert embedding.ndim == 1
        assert embedding.shape[0] == embedder.dimension

    def test_embed_normalized(self, embedder):
        """Test that embeddings are normalized."""
        text = "Test text for normalization"
        embedding = embedder.embed(text)

        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 0.01  # Should be unit norm

    def test_embed_batch(self, embedder):
        """Test batch embedding."""
        texts = [
            "First incident about latency",
            "Second incident about outage",
            "Third incident about deployment",
        ]

        embeddings = embedder.embed_batch(texts)

        assert embeddings.shape[0] == 3
        assert embeddings.shape[1] == embedder.dimension

    def test_similar_texts_have_similar_embeddings(self, embedder):
        """Test that similar texts produce similar embeddings."""
        text1 = "High latency on payment API"
        text2 = "Elevated response times on payment service"
        text3 = "Database migration failed"

        emb1 = embedder.embed(text1)
        emb2 = embedder.embed(text2)
        emb3 = embedder.embed(text3)

        # Cosine similarity (embeddings are normalized)
        sim_12 = np.dot(emb1, emb2)
        sim_13 = np.dot(emb1, emb3)

        # text1 and text2 should be more similar than text1 and text3
        assert sim_12 > sim_13


class TestFAISSIndex:
    """Tests for FAISSIndex."""

    @pytest.fixture
    def sample_incidents(self):
        """Create sample incidents."""
        return [
            Incident(
                id="inc-1",
                title="High latency on API",
                description="Response times increased to 5000ms",
                source=IncidentSource.SYNTHETIC,
            ),
            Incident(
                id="inc-2",
                title="Service outage",
                description="Complete outage of payment service",
                source=IncidentSource.SYNTHETIC,
            ),
            Incident(
                id="inc-3",
                title="Database connection errors",
                description="Unable to connect to primary database",
                source=IncidentSource.SYNTHETIC,
            ),
            Incident(
                id="inc-4",
                title="Elevated API response times",
                description="P99 latency increased significantly",
                source=IncidentSource.SYNTHETIC,
            ),
        ]

    def test_build_index(self, sample_incidents):
        """Test building index from incidents."""
        index = FAISSIndex()
        index.build(sample_incidents)

        assert index.size == 4
        assert len(index.id_map) == 4
        assert len(index.incident_map) == 4

    def test_search(self, sample_incidents):
        """Test searching the index."""
        index = FAISSIndex()
        index.build(sample_incidents)

        # Search for latency-related incidents
        results = index.search("high latency API response time", k=2)

        assert len(results) == 2
        # First result should be one of the latency incidents
        assert results[0][0] in ["inc-1", "inc-4"]
        # Score should be between 0 and 1
        assert 0 <= results[0][1] <= 1

    def test_search_with_threshold(self, sample_incidents):
        """Test search with similarity threshold."""
        index = FAISSIndex()
        index.build(sample_incidents)

        # High threshold should filter results
        results = index.search("completely unrelated query xyz", k=4, threshold=0.9)

        # Most results should be filtered out
        assert len(results) < 4

    def test_search_by_incident(self, sample_incidents):
        """Test searching by incident."""
        index = FAISSIndex()
        index.build(sample_incidents)

        # Search for similar to first incident
        query_incident = sample_incidents[0]
        results = index.search_by_incident(query_incident, k=2, exclude_self=True)

        # Should not include the query incident itself
        result_ids = [r[0] for r in results]
        assert query_incident.id not in result_ids

    def test_save_and_load(self, sample_incidents):
        """Test saving and loading index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir)

            # Build and save
            index1 = FAISSIndex()
            index1.build(sample_incidents)
            index1.save(save_path)

            # Load into new index
            index2 = FAISSIndex(index_path=save_path)

            assert index2.size == index1.size
            assert index2.id_map == index1.id_map

            # Search should work on loaded index
            results = index2.search("latency", k=2)
            assert len(results) == 2

    def test_empty_index_raises(self):
        """Test that empty index raises on search."""
        index = FAISSIndex()

        with pytest.raises(ValueError):
            index.search("test query")

    def test_build_empty_list_raises(self):
        """Test that building with empty list raises."""
        index = FAISSIndex()

        with pytest.raises(ValueError):
            index.build([])
