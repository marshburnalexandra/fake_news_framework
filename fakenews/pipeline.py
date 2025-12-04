from .preprocessing import Preprocessor
from .features import FeatureExtractor
from .models import LogisticNewsModel

class FakeNewsPipeline:

    def __init__(self, model=None, feature_max_features=3000,
                 remove_stopwords=True, stem=True):

        self.preprocessor = Preprocessor(
            remove_stopwords=remove_stopwords,
            stem=stem
        )

        self.feature_extractor = FeatureExtractor(
            max_features=feature_max_features,
            preprocessor=self.preprocessor
        )

        self.model = model or LogisticNewsModel()

    def fit(self, texts, labels):
        cleaned = [self.preprocessor.clean(t) for t in texts]
        X = self.feature_extractor.fit_transform(cleaned)
        self.model.train(X, labels)

    def predict(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        cleaned = [self.preprocessor.clean(t) for t in texts]
        X = self.feature_extractor.transform(cleaned)
        return self.model.predict(X)

    def evaluate(self, texts, labels):
        preds = self.predict(texts)
        accuracy = sum(p == l for p, l in zip(preds, labels)) / len(labels)
        return {"accuracy": accuracy}

    def __repr__(self):
        return f"FakeNewsPipeline(model={self.model})"
