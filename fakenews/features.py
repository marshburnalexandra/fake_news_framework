from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings

class FeatureExtractor:
    def __init__(self, max_features=5000):
        self.vectorizer = TfidfVectorizer(max_features=max_features)

    def fit_transformation(self, texts: List[str]):
        if not isinstance(texts, list):
            raise TypeError("fit_transform() expects a list of strings.")
        
        if len(texts) < 5:
            warnings.warn("Dataset is too small. TF-IDF may perform poorly.")

        return self.vectorizer.fit_transform(texts)
    
    def transform(self, texts: List[str]):
        if not hasattr(self.vectorizer, "vocabulary_"):
            raise RuntimeError(
                "trasnform() called before fit_transform(). Train the vectorizer first."
            )
        return self.vectorizer.transform(texts)