from fakenews.pipeline import FakeNewsPipeline
import joblib
import os
import pandas as pd

MODEL_PATH = "saved_model.pkl"


def save_pipeline(pipeline):
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Pipeline saved to {MODEL_PATH}")


def load_pipeline():
    if not os.path.exists(MODEL_PATH):
        print("No saved model found! Training a new one...")
        return None

    print(f"Loaded saved model from {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def train_and_save():
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
