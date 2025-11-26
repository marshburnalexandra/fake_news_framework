from abc import ABC, abstractmethod
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
import warnings


class BaseNewsModel(ABC):
    """
    Abstract base class for all fake news classification models.

    This ensures that all model subclasses implement a consistent interface
    consisting of training, single-sample prediction, and evaluation methods.
    """

    @abstractmethod
    def train(self, X, y):
        """
        Train the model using the provided feature matrix and labels.

        Parameters
        ----------
        X : sparse matrix or ndarray
            Feature matrix used for training.
        y : list or array-like
            Corresponding labels for each sample.
        """
        pass

    @abstractmethod
    def predict_single(self, x_vector):
        """
        Predict the label for a single feature vector.

        Parameters
        ----------
        x_vector : sparse matrix or ndarray
            The vector representing a single sample.

        Returns
        -------
        label : str or int
            Predicted class label.
        """
        pass

    @abstractmethod
    def evaluate(self, X, y):
        """
        Evaluate the model’s performance on the given dataset.

        Parameters
        ----------
        X : sparse matrix or ndarray
            Feature matrix to evaluate on.
        y : list or array-like
            True labels.

        Returns
        -------
        dict
            A dictionary containing accuracy and a classification report.
        """
        pass


class LogisticNewsModel(BaseNewsModel):
    """
    Logistic Regression classifier for fake news detection.

    Uses scikit-learn's ``LogisticRegression`` with a high iteration limit to
    ensure convergence on high-dimensional TF-IDF features.
    """

    def __init__(self, **kwargs):
        """
        Initialize the Logistic Regression model.

        Parameters
        ----------
        **kwargs : dict
            Optional keyword arguments passed to ``LogisticRegression``.
        """
        self.model = LogisticRegression(max_iter=2000, **kwargs)

    def train(self, X, y):
        """
        Fit the logistic regression model on the training data.

        Parameters
        ----------
        X : sparse matrix or ndarray
            Training feature matrix.
        y : list or array-like
            Training labels.

        Returns
        -------
        self : LogisticNewsModel

        Raises
        ------
        ValueError
            If ``X`` and ``y`` have mismatched lengths or less than 2 classes.
        """
        if X.shape[0] != len(y):
            raise ValueError("X and y lengths mismatch")
        if len(set(y)) < 2:
            raise ValueError("Training labels must contain at least two classes")
        self.model.fit(X, y)
        if X.shape[0] < 20:
            warnings.warn("Small training set; results may be unreliable")
        return self

    def predict_single(self, x_vector):
        """
        Predict the label of a single TF-IDF feature vector.

        Parameters
        ----------
        x_vector : sparse matrix or ndarray
            Single sample feature vector.

        Returns
        -------
        label : int or str
        """
        return self.model.predict(x_vector)[0]

    def evaluate(self, X, y):
        """
        Evaluate the logistic regression model on the given dataset.

        Parameters
        ----------
        X : sparse matrix or ndarray
            Evaluation feature matrix.
        y : list or array-like
            True labels.

        Returns
        -------
        dict
            Contains:
                - ``accuracy`` : float  
                - ``report`` : str (classification report)
        """
        preds = self.model.predict(X)
        return {
            "accuracy": accuracy_score(y, preds),
            "report": classification_report(y, preds, zero_division=0)
        }

    def __repr__(self):
        """Return a readable representation of the model."""
        return "LogisticNewsModel(LogisticRegression)"


class NaiveBayesNewsModel(BaseNewsModel):
    """
    Multinomial Naive Bayes classifier for fake news detection.

    Well-suited for TF-IDF or word count features, providing fast and
    interpretable predictions for text classification tasks.
    """

    def __init__(self):
        """Initialize the MultinomialNB classifier."""
        self.model = MultinomialNB()

    def train(self, X, y):
        """
        Fit the Naive Bayes model on the training data.

        Parameters
        ----------
        X : sparse matrix or ndarray
            Training features.
        y : list or array-like
            Training labels.

        Returns
        -------
        self : NaiveBayesNewsModel
        """
        self.model.fit(X, y)
        return self

    def predict_single(self, x_vector):
        """
        Predict the label of a single feature vector.

        Parameters
        ----------
        x_vector : sparse matrix or ndarray

        Returns
        -------
        label : int or str
        """
        return self.model.predict(x_vector)[0]

    def evaluate(self, X, y):
        """
        Evaluate the Naive Bayes model.

        Parameters
        ----------
        X : sparse matrix or ndarray
        y : list or array-like

        Returns
        -------
        dict
            Accuracy and classification report.
        """
        preds = self.model.predict(X)
        return {
            "accuracy": accuracy_score(y, preds),
            "report": classification_report(y, preds, zero_division=0)
        }

    def __repr__(self):
        """Return a simple string representation."""
        return "NaiveBayesNewsModel(MultinomialNB)"


class SVMNewsModel(BaseNewsModel):
    """
    Support Vector Machine (Linear SVC) classifier for fake news detection.

    Uses ``LinearSVC`` for efficient training on high-dimensional sparse
    TF-IDF features, making it well suited for text classification problems.
    """

    def __init__(self):
        """Initialize the LinearSVC classifier."""
        self.model = LinearSVC(max_iter=20000)

    def train(self, X, y):
        """
        Train the SVM model on the dataset.

        Parameters
        ----------
        X : sparse matrix or ndarray
        y : list or array-like

        Returns
        -------
        self : SVMNewsModel
        """
        self.model.fit(X, y)
        return self

    def predict_single(self, x_vector):
        """
        Predict the label for a single feature vector.

        Parameters
        ----------
        x_vector : sparse matrix or ndarray

        Returns
        -------
        label : int or str
        """
        return self.model.predict(x_vector)[0]

    def evaluate(self, X, y):
        """
        Evaluate the SVM model on the dataset.

        Parameters
        ----------
        X : sparse matrix or ndarray
        y : list or array-like

        Returns
        -------
        dict
            Accuracy and classification report.
        """
        preds = self.model.predict(X)
        return {
            "accuracy": accuracy_score(y, preds),
            "report": classification_report(y, preds, zero_division=0)
        }

    def __repr__(self):
        """Return a model description string."""
        return "SVMNewsModel(LinearSVC)"
