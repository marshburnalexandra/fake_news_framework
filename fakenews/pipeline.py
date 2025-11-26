from typing import Iterable
from .preprocessing import Preprocessor
from .features import FeatureExtractor
from .models import LogisticNewsModel


class FakeNewsPipeline:
    """
    End-to-end pipeline for fake news detection.

    This pipeline combines text preprocessing, TF-IDF feature extraction, and
    model inference into a single, user-friendly interface. It enables training,
    prediction, and evaluation using any model that follows the
    ``BaseNewsModel`` interface. By default, it uses a logistic regression
    classifier.

    Components
    ----------
    preprocessor : Preprocessor
        Cleans and normalizes raw text (stopword removal, stemming, etc.).
    feature_extractor : FeatureExtractor
        Converts cleaned text into TF-IDF feature vectors.
    model : BaseNewsModel
        A compatible classifier (default: ``LogisticNewsModel``).

    Parameters
    ----------
    model : BaseNewsModel, optional
        Custom model to use for classification. If None, defaults to
        ``LogisticNewsModel``.
    feature_max_features : int, default=3000
        Maximum vocabulary size for the TF-IDF vectorizer.
    remove_stopwords : bool, default=True
        Whether the preprocessor removes stopwords.
    stem : bool, default=True
        Whether the preprocessor applies stemming.
    """

    def __init__(self, model=None, feature_max_features=3000, remove_stopwords=True, stem=True):
        self.preprocessor = Preprocessor(remove_stopwords=remove_stopwords, stem=stem)
        self.feature_extractor = FeatureExtractor(max_features=feature_max_features, preprocessor=self.preprocessor)
        self.model = model or LogisticNewsModel()

    def fit(self, texts, labels):
        """
        Train the full pipeline on a dataset.

        This method preprocesses the raw text, extracts TF-IDF features, and then
        trains the model on those features.

        Parameters
        ----------
        texts : Iterable[str]
            A collection of raw text samples.
        labels : Iterable
            Corresponding labels for each text sample.

        Returns
        -------
        None
        """
        cleaned = [self.preprocessor.clean(t) for t in texts]
        X = self.feature_extractor.fit_transform(cleaned)
        self.model.train(X, labels)

    def predict(self, text):
        """
        Predict the label of a single text sample.

        The text is cleaned, transformed into TF-IDF features, and then passed to
        the trained model for prediction.

        Parameters
        ----------
        text : str
            Raw text input to classify.

        Returns
        -------
        label : int or str
            Predicted class label.
        """
        x = self.feature_extractor.transform([self.preprocessor.clean(text)])
        return self.model.predict_single(x)
    
    def evaluate(self, texts, labels):
        """
        Evaluate the pipeline on a labeled dataset.

        Texts are cleaned and transformed using the already-fitted feature
        extractor. The method returns the accuracy score from the underlying
        model.

        Parameters
        ----------
        texts : Iterable[str]
            Raw text samples to evaluate.
        labels : Iterable
            True labels corresponding to each text sample.

        Returns
        -------
        dict
            A dictionary containing:
                - ``accuracy`` : float
        """
        cleaned = [self.preprocessor(t) for t in texts]
        X = self.feature_extractor.transform(cleaned)
        accuracy = self.model.evaluate(X, labels)["accuracy"]
        return {"accuracy": accuracy}

    def __repr__(self):
        """
        Return a concise string representation of the pipeline, including the
        model and feature extractor used.

        Returns
        -------
        str
        """
        return f"FakeNewsPipeline(model={repr(self.model)}, extractor={repr(self.feature_extractor)})"
