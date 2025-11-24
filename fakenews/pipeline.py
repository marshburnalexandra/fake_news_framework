from typing import Iterable
from .preprocessing import Preprocessor
from .features import FeatureExtractor
from .models import LogisticNewsModel


class FakeNewsPipeline:
    def __init__(self, model=None, feature_max_features=3000, remove_stopwords=True, stem=True):
        self.preprocessor = Preprocessor(remove_stopwords=remove_stopwords, stem=stem)
        self.feature_extractor = FeatureExtractor(max_features=feature_max_features, preprocessor=self.preprocessor)
        self.model = model or LogisticNewsModel()  # default model

    def fit(self, texts, labels):
        # Preprocess texts
        cleaned = [self.preprocessor.clean(t) for t in texts]
        X = self.feature_extractor.fit_transform(cleaned)
        self.model.train(X, labels)

    def predict(self, text):
        x = self.feature_extractor.transform([self.preprocessor.clean(text)])
        return self.model.predict_single(x)
    
    def evaluate(self, texts, labels):
        cleaned = [self.preprocessor(t) for t in texts]
        X = self.feature_extractor.transform(cleaned)
        accuracy = self.model.evaluate(X, labels)["accuracy"]
        return {"accuracy": accuracy}

    def __repr__(self):
        return f"FakeNewsPipeline(model={repr(self.model)}, extractor={repr(self.feature_extractor)})"