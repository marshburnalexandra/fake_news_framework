"""
Fake News Detection Package
===========================

This package provides a complete toolkit for building and evaluating
fake news detection systems. It exposes components for preprocessing,
feature extraction, model training, model evaluation, and running a full
classification pipeline.

Exposed Components
------------------

Preprocessing:
    - ``Preprocessor``  
      Cleans and normalizes raw news text (lowercasing, stopword removal, etc.)
      to prepare inputs for feature extraction and machine learning models.

Feature Extraction:
    - ``FeatureExtractor``  
      Converts cleaned text into numerical TF-IDF feature vectors, with support
      for configurable vocabulary size and preprocessing.

Models:
    - ``LogisticNewsModel``  
      Logistic Regression classifier for fake news detection.

    - ``NaiveBayesNewsModel``  
      Multinomial Naive Bayes classifier optimized for text classification tasks.

    - ``SVMNewsModel``  
      Support Vector Machine (SVM) classifier using linear kernels for high-dimensional text data.

Pipeline:
    - ``FakeNewsPipeline``  
      End-to-end pipeline that integrates preprocessing, feature extraction,
      model training, and prediction into a single unified interface.

Utilities:
    - ``load_dataset``  
      Loads a CSV dataset into a pandas DataFrame.

    - ``preview``  
      Displays a formatted preview of the dataset for inspection.

Usage
-----

These components are typically used together in a workflow such as:

    1. Load dataset with ``load_dataset``
    2. Preprocess text using ``Preprocessor``
    3. Extract TF-IDF features with ``FeatureExtractor``
    4. Train models (Logistic, Naive Bayes, SVM)
    5. Or use ``FakeNewsPipeline`` for a full end-to-end solution.

All exported names are listed in ``__all__`` for clean imports when using:

    from fakenews import *

This design ensures that external users have access only to the primary
public-facing classes and functions.
"""
__version__ = "0.1.1"
print("fakenews __init__.py loaded")

from .preprocessing import Preprocessor
from .features import FeatureExtractor
from .models import LogisticNewsModel, NaiveBayesNewsModel, SVMNewsModel
from .pipeline import FakeNewsPipeline
from .utils import load_dataset, preview

__all__ = [
    "Preprocessor",
    "FeatureExtractor",
    "LogisticNewsModel",
    "NaiveBayesNewsModel",
    "SVMNewsModel",
    "FakeNewsPipeline",
    "load_dataset",
    "preview",
]
