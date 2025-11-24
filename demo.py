import os
from fakenews.utils import load_dataset, preview
from fakenews.preprocessing import Preprocessor
from fakenews.features import FeatureExtractor
from fakenews.models import LogisticNewsModel, NaiveBayesNewsModel, SVMNewsModel
from fakenews.pipeline import FakeNewsPipeline

def main():
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
