from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
from .preprocessing import Preprocessor


class FeatureExtractor:
    """TF-IDF feature extraction with optional preprocessing."""

    def __init__(self, max_features=5000, preprocessor: Preprocessor = None):
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.preprocessor = preprocessor or Preprocessor()

    def fit_transformation(self, texts: List[str]):
        if not isinstance(texts, list):
            raise TypeError("fit_transformation expects a list of strings.")
        if len(texts) < 5:
            warnings.warn("Small dataset: TF-IDF may perform poorly.")
        cleaned = [self.preprocessor.clean(t) for t in texts]
        return self.vectorizer.fit_transform(cleaned)

    def transform(self, texts: List[str]):
        if not hasattr(self.vectorizer, "vocabulary_"):
            raise RuntimeError("Vectorizer not fitted. Call fit_transformation first.")
        cleaned = [self.preprocessor.clean(t) for t in texts]
        return self.vectorizer.transform(cleaned)

    def __repr__(self):
        return f"FeatureExtractor(max_features={self.max_features})"
