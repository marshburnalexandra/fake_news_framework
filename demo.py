"""
Fake News Detection Script
--------------------------

This script demonstrates a full workflow for fake news classification using the
fakenews library. It loads a dataset, preprocesses text, extracts features,
trains multiple machine learning models, evaluates them, and tests a complete
pipeline.

Workflow:
    1. Load dataset from CSV
    2. Preview data
    3. Initialize and test the Preprocessor
    4. Extract TF-IDF features
    5. Train Logistic Regression, Naive Bayes, and SVM models
    6. Evaluate models
    7. Initialize and train the unified FakeNewsPipeline
    8. Make a sample prediction

Run:
    python script.py
"""
import os
from fakenews.utils import load_dataset, preview
from fakenews.preprocessing import Preprocessor
from fakenews.features import FeatureExtractor
from fakenews.models import LogisticNewsModel, NaiveBayesNewsModel, SVMNewsModel
from fakenews.pipeline import FakeNewsPipeline

def main():
    """
    Execute the fake news detection workflow.
    """
    # ------------------------
    # Load dataset
    # ------------------------
    csv_path = os.path.join("News", "news.csv")
    df = load_dataset(csv_path)

    print("Dataset preview:")
    print(preview(df, 5))

    # ------------------------
    # Initialize Preprocessor
    # ------------------------
    preprocessor = Preprocessor()
    example_text = "Breaking news: Scientists announce a new study showing X cures Y."
    print("\nOriginal:", example_text)
    print("Cleaned:", preprocessor(example_text))
    print("Preprocessor repr:", repr(preprocessor))

    # ------------------------
    # Feature extraction
    # ------------------------
    fe = FeatureExtractor(max_features=3000, preprocessor=preprocessor)
    cleaned_texts = df["text"].apply(preprocessor).tolist()
    X = fe.fit_transform(cleaned_texts)
    y = df["label"].tolist()

    print("\nFeature matrix shape:", X.shape)
    print("FeatureExtractor repr:", repr(fe))

    # ------------------------
    # Train models individually
    # ------------------------
    print("\n--- Model Training ---")
    models = [LogisticNewsModel(), NaiveBayesNewsModel(), SVMNewsModel()]

    for model in models:
        print(f"\nTraining {repr(model)}")
        model.train(X, y)  # note: use .train(), not .fit()
        results = model.evaluate(X, y)
        print("Accuracy:", results["accuracy"])

    # ------------------------
    # Test full pipeline
    # ------------------------
    pipeline = FakeNewsPipeline(feature_max_features=3000)  # remove `model=...`
    pipeline.fit(df["text"].tolist(), df["label"].tolist())
    print("\nPipeline repr:", repr(pipeline))

    # Single prediction
    prediction = pipeline.predict(example_text)
    print("\nPrediction for example article:", prediction)


if __name__ == "__main__":
    main()
