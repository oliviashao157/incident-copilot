"""Tests for incident classifier."""

import pytest

from src.classifier.categories import (
    CATEGORY_KEYWORDS,
    infer_category_from_labels,
    infer_category_from_text,
)
from src.classifier.model import IncidentClassifier
from src.schema import Category, Incident, IncidentSource


class TestCategoryKeywords:
    """Tests for category keyword mappings."""

    def test_all_categories_have_keywords(self):
        """Test that all categories (except unknown) have keywords."""
        for category in Category:
            if category != Category.UNKNOWN:
                assert category in CATEGORY_KEYWORDS
                assert len(CATEGORY_KEYWORDS[category]) > 0

    def test_keywords_are_lowercase(self):
        """Test that all keywords are lowercase for matching."""
        for category, keywords in CATEGORY_KEYWORDS.items():
            for keyword in keywords:
                assert keyword == keyword.lower(), f"Keyword '{keyword}' should be lowercase"


class TestInferCategoryFromLabels:
    """Tests for label-based category inference."""

    def test_exact_label_match(self):
        """Test exact label matching."""
        assert infer_category_from_labels(["performance"]) == Category.LATENCY
        assert infer_category_from_labels(["bug"]) == Category.OUTAGE
        assert infer_category_from_labels(["security"]) == Category.SECURITY

    def test_case_insensitive(self):
        """Test case-insensitive matching."""
        assert infer_category_from_labels(["PERFORMANCE"]) == Category.LATENCY
        assert infer_category_from_labels(["Security"]) == Category.SECURITY

    def test_multiple_labels(self):
        """Test with multiple labels (first match wins)."""
        result = infer_category_from_labels(["enhancement", "performance", "bug"])
        assert result == Category.LATENCY  # "performance" matches first

    def test_unknown_labels(self):
        """Test with unknown labels."""
        result = infer_category_from_labels(["unknown-label", "random"])
        assert result == Category.UNKNOWN

    def test_empty_labels(self):
        """Test with empty labels."""
        assert infer_category_from_labels([]) == Category.UNKNOWN


class TestInferCategoryFromText:
    """Tests for text-based category inference."""

    def test_latency_keywords(self):
        """Test latency category inference."""
        text = "High latency detected, p99 response times increased"
        category, confidence = infer_category_from_text(text)
        assert category == Category.LATENCY
        assert confidence > 0

    def test_outage_keywords(self):
        """Test outage category inference."""
        text = "Service is down, returning 503 errors, complete outage"
        category, confidence = infer_category_from_text(text)
        assert category == Category.OUTAGE

    def test_security_keywords(self):
        """Test security category inference."""
        text = "Security vulnerability CVE-2024-1234 found in authentication"
        category, confidence = infer_category_from_text(text)
        assert category == Category.SECURITY

    def test_no_match(self):
        """Test text with no matching keywords."""
        text = "Something happened somewhere"
        category, confidence = infer_category_from_text(text)
        assert category == Category.UNKNOWN
        assert confidence == 0.0

    def test_case_insensitive(self):
        """Test case-insensitive matching."""
        text = "HIGH LATENCY on PRODUCTION"
        category, _ = infer_category_from_text(text)
        assert category == Category.LATENCY


class TestIncidentClassifier:
    """Tests for ML classifier."""

    @pytest.fixture
    def sample_incidents(self):
        """Create sample incidents for testing."""
        incidents = []

        # Create incidents for each category
        category_samples = {
            Category.LATENCY: [
                ("High API latency", "Response times increased to 5000ms"),
                ("Slow database queries", "Query times degraded significantly"),
            ],
            Category.OUTAGE: [
                ("Service down", "Complete outage of payment service"),
                ("502 errors on gateway", "All requests returning 502"),
            ],
            Category.DEPLOYMENT: [
                ("Failed deployment", "Rollout stuck at 50%"),
                ("Canary errors", "New version showing high error rate"),
            ],
            Category.CONFIG: [
                ("Wrong config", "Misconfigured database connection"),
                ("Missing env var", "SERVICE_URL not set"),
            ],
            Category.SECURITY: [
                ("SSL expired", "Certificate expired on production"),
                ("Auth failures", "Spike in failed login attempts"),
            ],
        }

        for category, samples in category_samples.items():
            for i, (title, desc) in enumerate(samples):
                incidents.append(
                    Incident(
                        id=f"{category.value}-{i}",
                        title=title,
                        description=desc,
                        category=category,
                        source=IncidentSource.SYNTHETIC,
                    )
                )

        return incidents

    def test_classifier_init(self):
        """Test classifier initialization."""
        classifier = IncidentClassifier()
        assert classifier.pipeline is None

    def test_train_requires_minimum_data(self, sample_incidents):
        """Test that training requires minimum data."""
        classifier = IncidentClassifier()

        # Should fail with too few incidents
        with pytest.raises(ValueError):
            classifier.train(sample_incidents[:5])

    def test_predict_requires_trained_model(self):
        """Test that predict requires a trained model."""
        classifier = IncidentClassifier()
        incident = Incident(
            id="test",
            title="Test",
            description="Test",
            source=IncidentSource.MANUAL,
        )

        with pytest.raises(ValueError):
            classifier.predict(incident)

    def test_train_and_predict(self, sample_incidents):
        """Test full train and predict cycle."""
        # Duplicate samples to meet minimum threshold
        incidents = sample_incidents * 3  # 30 incidents

        classifier = IncidentClassifier()
        metrics = classifier.train(incidents, test_size=0.2)

        # Check metrics exist
        assert "accuracy" in metrics
        assert "macro_f1" in metrics
        assert metrics["accuracy"] >= 0
        assert metrics["accuracy"] <= 1

        # Test prediction
        test_incident = Incident(
            id="test",
            title="High latency on API",
            description="Response times are very slow",
            source=IncidentSource.MANUAL,
        )

        category, confidence = classifier.predict(test_incident)
        assert isinstance(category, Category)
        assert 0 <= confidence <= 1

    def test_predict_batch(self, sample_incidents):
        """Test batch prediction."""
        incidents = sample_incidents * 3
        classifier = IncidentClassifier()
        classifier.train(incidents, test_size=0.2)

        test_incidents = [
            Incident(
                id=f"test-{i}",
                title=f"Test incident {i}",
                description="Some description",
                source=IncidentSource.MANUAL,
            )
            for i in range(3)
        ]

        results = classifier.predict_batch(test_incidents)
        assert len(results) == 3
        for category, confidence in results:
            assert isinstance(category, Category)
            assert 0 <= confidence <= 1
