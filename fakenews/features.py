from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
from .preprocessing import Preprocessor


class FeatureExtractor:
    """
    A TF-IDF–based feature extraction component for transforming cleaned text
    into numerical feature vectors.

    This class wraps scikit-learn's ``TfidfVectorizer`` and integrates seamlessly
    with the custom ``Preprocessor`` used in the fake news detection pipeline.
    It supports fitting a vocabulary from training text and transforming new text
    using the learned TF-IDF representation.

    Parameters
    ----------
    max_features : int, default=5000
        Maximum number of features (vocabulary size) to keep. Higher values
        may improve performance but increase memory cost.
    preprocessor : Preprocessor, optional
        An instance of the ``Preprocessor`` class used to clean text before
        applying TF-IDF. If not provided, a new default ``Preprocessor`` is created.

    Attributes
    ----------
    vectorizer : TfidfVectorizer
        The underlying scikit-learn vectorizer.
    preprocessor : Preprocessor
        The preprocessing object responsible for text cleaning.
    """

    def __init__(self, max_features=5000, preprocessor: Preprocessor = None):
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.preprocessor = preprocessor or Preprocessor()

    def fit_transform(self, texts: List[str]):
        """
        Fit the TF-IDF vectorizer to the provided text data and return the
        transformed TF-IDF feature matrix.

        This method both learns the vocabulary and computes the TF-IDF values.
        Texts are automatically cleaned using the assigned ``Preprocessor``.

        Parameters
        ----------
        texts : List[str]
            A list of raw input text strings.

        Returns
        -------
        sparse matrix (scipy.sparse.csr_matrix)
            The TF-IDF feature matrix representing the cleaned input text.

        Raises
        ------
        TypeError
            If ``texts`` is not a list of strings.
        Warning
            If the dataset contains fewer than 5 samples, which may lead to
            unstable TF-IDF behavior.
        """
        if not isinstance(texts, list):
            raise TypeError("fit_transform expects a list of strings.")
        if len(texts) < 5:
            warnings.warn("Small dataset: TF-IDF may perform poorly.")
        cleaned = [self.preprocessor.clean(t) for t in texts]
        return self.vectorizer.fit_transform(cleaned)

    def transform(self, texts: List[str]):
        """
        Transform new text data into TF-IDF feature vectors using the already
        fitted vocabulary.

        All input text is cleaned using the same preprocessing logic as during
        training.

        Parameters
        ----------
        texts : List[str]
            A list of raw text samples to transform.

        Returns
        -------
        sparse matrix (scipy.sparse.csr_matrix)
            TF-IDF feature representation of the cleaned text.

        Raises
        ------
        RuntimeError
            If the vectorizer has not been fitted with ``fit_transform``.
        """
        if not hasattr(self.vectorizer, "vocabulary_"):
            raise RuntimeError("Vectorizer not fitted. Call fit_transformation first.")
        cleaned = [self.preprocessor.clean(t) for t in texts]
        return self.vectorizer.transform(cleaned)

    def __repr__(self):
        return f"FeatureExtractor(max_features={self.max_features})"
