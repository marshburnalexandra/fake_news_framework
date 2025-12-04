from .preprocessing import Preprocessor
from .features import FeatureExtractor

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

        self.model = model

        self.is_fitted = False

    def fit(self, texts, labels):
        cleaned_texts = [self.preprocessor.clean(t) for t in texts]
        X = self.feature_extractor.fit_transform(cleaned_texts)
        self.model.train(X, labels)
        self.is_fitted = True

    def predict(self, texts):
        if not self.is_fitted:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")
        
        if isinstance(texts, str):
            texts = [texts]

        cleaned_texts = [self.preprocessor.clean(t) for t in texts]
        X = self.feature_extractor.transform(cleaned_texts)
        
        return [self.model.predict_single(X[i]) for i in range(X.shape[0])]

    def evaluate(self, texts, labels):
        preds = self.predict(texts)
        accuracy = sum(p == l for p, l in zip(preds, labels)) / len(labels)
        return {"accuracy": accuracy, "predictions": preds}

    def __repr__(self):
        return f"FakeNewsPipeline(model={self.model}, preprocessor={self.preprocessor})"
