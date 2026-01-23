"""TF-IDF + Logistic Regression classifier for incidents."""

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.schema import Category, Incident


class IncidentClassifier:
    """TF-IDF + Logistic Regression classifier for incident categorization."""

    def __init__(self, model_path: Optional[Path] = None):
        """Initialize classifier, optionally loading from path."""
        self.pipeline: Optional[Pipeline] = None
        self.categories: list[str] = [c.value for c in Category if c != Category.UNKNOWN]

        if model_path and model_path.exists():
            self.load(model_path)

    def _prepare_text(self, incident: Incident) -> str:
        """Prepare incident text for classification."""
        return f"{incident.title} {incident.description}"

    def train(
        self,
        incidents: list[Incident],
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> dict:
        """Train the classifier on a list of incidents.

        Returns a dict with training metrics.
        """
        # Filter out unknown categories for training
        labeled_incidents = [i for i in incidents if i.category != Category.UNKNOWN]

        if len(labeled_incidents) < 10:
            raise ValueError(f"Need at least 10 labeled incidents, got {len(labeled_incidents)}")

        texts = [self._prepare_text(i) for i in labeled_incidents]
        labels = [i.category.value for i in labeled_incidents]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=random_state, stratify=labels
        )

        # Create pipeline
        self.pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=5000,
                        ngram_range=(1, 2),
                        stop_words="english",
                        min_df=2,
                        max_df=0.95,
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=random_state,
                        multi_class="multinomial",
                    ),
                ),
            ]
        )

        # Train
        self.pipeline.fit(X_train, y_train)

        # Evaluate
        y_pred = self.pipeline.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=self.categories)

        return {
            "accuracy": report["accuracy"],
            "macro_f1": report["macro avg"]["f1-score"],
            "weighted_f1": report["weighted avg"]["f1-score"],
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "categories": self.categories,
        }

    def predict(self, incident: Incident) -> tuple[Category, float]:
        """Predict category for an incident. Returns (category, confidence)."""
        if self.pipeline is None:
            raise ValueError("Model not trained or loaded")

        text = self._prepare_text(incident)
        proba = self.pipeline.predict_proba([text])[0]
        pred_idx = np.argmax(proba)
        confidence = proba[pred_idx]

        category_value = self.pipeline.classes_[pred_idx]
        category = Category(category_value)

        return category, float(confidence)

    def predict_batch(self, incidents: list[Incident]) -> list[tuple[Category, float]]:
        """Predict categories for multiple incidents."""
        return [self.predict(incident) for incident in incidents]

    def save(self, path: Path) -> None:
        """Save the trained model to disk."""
        if self.pipeline is None:
            raise ValueError("No model to save")

        path.parent.mkdir(parents=True, exist_ok=True)
        model_path = path / "model.joblib"
        meta_path = path / "metadata.json"

        joblib.dump(self.pipeline, model_path)

        metadata = {
            "categories": self.categories,
            "feature_names": self.pipeline.named_steps["tfidf"].get_feature_names_out().tolist()[
                :100
            ],  # First 100 features
        }
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def load(self, path: Path) -> None:
        """Load a trained model from disk."""
        model_path = path / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        self.pipeline = joblib.load(model_path)

        meta_path = path / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                metadata = json.load(f)
                self.categories = metadata.get("categories", self.categories)
