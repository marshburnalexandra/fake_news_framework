from typing import Iterable
from .preprocessing import Preprocessor
from .features import FeatureExtractor
from .models import LogisticNewsModel


class FakeNewsPipeline:
    """Simple pipeline using Preprocessor, TF-IDF, and a classifier."""

    def __init__(self, max_features: int = 5000):
        self.preprocessor = Preprocessor()
        self.extractor = FeatureExtractor(max_features=max_features, preprocessor=self.preprocessor)
        self.model = LogisticNewsModel()
        self.is_fitted = False

    def fit(self, texts: Iterable[str], labels: Iterable[str]):
        cleaned = [self.preprocessor.clean(t) for t in texts]
        X = self.extractor.fit_transformation(cleaned)
        self.model.train(X, labels)
        self.is_fitted = True
        return self

    def evaluate(self, texts: Iterable[str], labels: Iterable[str]):
        if not self.is_fitted:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")
        cleaned = [self.preprocessor.clean(t) for t in texts]
        X = self.extractor.transform(cleaned)
        return self.model.evaluate(X, labels)

    def predict(self, text: str):
        if not self.is_fitted:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")
        X = self.extractor.transform([self.preprocessor.clean(text)])
        return self.model.predict_single(X)

    def __repr__(self):
        return f"FakeNewsPipeline(model={self.model}, extractor={self.extractor})"
