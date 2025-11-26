from fakenews.pipeline import FakeNewsPipeline
import joblib
import os
import pandas as pd

MODEL_PATH = "saved_model.pkl"


def save_pipeline(pipeline):
    """
    Save a trained FakeNewsPipeline object to disk.

    Args:
        pipeline (FakeNewsPipeline): The trained pipeline to save.

    Saves:
        saved_model.pkl: Serialized model file created with joblib.
    """
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Pipeline saved to {MODEL_PATH}")


def load_pipeline():
    """
    Load a previously saved FakeNewsPipeline model from disk.

    Returns:
        FakeNewsPipeline or None:
            - Loaded model if the file exists.
            - None if no saved model is found.
    """
    if not os.path.exists(MODEL_PATH):
        print("No saved model found! Training a new one...")
        return None

    print(f"Loaded saved model from {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def train_and_save():
    """
    Train a new FakeNewsPipeline model on the dataset and save it.

    Returns:
        FakeNewsPipeline: The trained model.
    """
    print("Training model and saving...")

    df = pd.read_csv("News/news.csv").dropna(subset=["text", "label"])

    pipeline = FakeNewsPipeline(
        model=None,
        feature_max_features=3000,
        remove_stopwords=True,
        stem=True
    )

    pipeline.fit(df["text"].tolist(), df["label"].tolist())

    save_pipeline(pipeline)
    return pipeline


def main():
    """
    Load or train the model, then predict user-provided news text.
    """
    pipeline = load_pipeline()

    if pipeline is None:
        pipeline = train_and_save()

    print("\nType or paste any news article text:")
    news_text = input("> ").strip()

    if not news_text:
        print("Please enter some text.")
        return

    prediction = pipeline.predict(news_text)
    print(f"\nPrediction: {prediction.upper()}")


if __name__ == "__main__":
    main()
