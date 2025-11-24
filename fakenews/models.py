from abc import ABC, abstractmethod
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
import warnings


class BaseNewsModel(ABC):
    """Abstract base model."""

    @abstractmethod
    def train(self, X, y):
        pass

    @abstractmethod
    def predict_single(self, x_vector):
        pass

    @abstractmethod
    def evaluate(self, X, y):
        pass


class LogisticNewsModel(BaseNewsModel):
    def __init__(self, **kwargs):
        self.model = LogisticRegression(max_iter=2000, **kwargs)

    def train(self, X, y):
        if X.shape[0] != len(y):
            raise ValueError("X and y lengths mismatch")
        if len(set(y)) < 2:
            raise ValueError("Training labels must contain at least two classes")
        self.model.fit(X, y)
        if X.shape[0] < 20:
            warnings.warn("Small training set; results may be unreliable")
        return self

    def predict_single(self, x_vector):
        return self.model.predict(x_vector)[0]

    def evaluate(self, X, y):
        preds = self.model.predict(X)
        return {
            "accuracy": accuracy_score(y, preds),
            "report": classification_report(y, preds, zero_division=0)
        }

    def __repr__(self):
        return "LogisticNewsModel(LogisticRegression)"


class NaiveBayesNewsModel(BaseNewsModel):
    def __init__(self):
        self.model = MultinomialNB()

    def train(self, X, y):
        self.model.fit(X, y)
        return self

    def predict_single(self, x_vector):
        return self.model.predict(x_vector)[0]

    def evaluate(self, X, y):
        preds = self.model.predict(X)
        return {
            "accuracy": accuracy_score(y, preds),
            "report": classification_report(y, preds, zero_division=0)
        }

    def __repr__(self):
        return "NaiveBayesNewsModel(MultinomialNB)"


class SVMNewsModel(BaseNewsModel):
    def __init__(self):
        self.model = LinearSVC(max_iter=20000)

    def train(self, X, y):
        self.model.fit(X, y)
        return self

    def predict_single(self, x_vector):
        return self.model.predict(x_vector)[0]

    def evaluate(self, X, y):
        preds = self.model.predict(X)
        return {
            "accuracy": accuracy_score(y, preds),
            "report": classification_report(y, preds, zero_division=0)
        }

    def __repr__(self):
        return "SVMNewsModel(LinearSVC)"
