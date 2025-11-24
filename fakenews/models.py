import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

class FakeNewsModel:
    def __init__(self):
        self.model = LogisticRegression(max_iter=2000)

    def train(self, X, y):
        if X.shape[0] != len(y):
            raise ValueError("X and y lengths do not match.")
        
        if len(set(y)) < 2:
            raise ValueError("Training labels must contain at least 2 classes.")
        
        self.model.fit(X, y)

        if X.shape[0] < 20:
            warnings.warn("Training with small dataset. Accuracy may be misleading.")

        return self.model
    
    def evaluate(self, X, y):
        preds = self.model.predict(X)

        return {
            "accuracy": accuracy_score(y, preds),
            "report": classification_report(y, preds, zero_division=0)
        }
    
    def predict_single(self, text_vector):
        """Predict a single vectorized article.""" 
        return self.model.predict(text_vector)[0]