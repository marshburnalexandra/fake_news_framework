from fakenews import Preprocessor, FeatureExtractor, FakeNewsPipeline, load_dataset, preview

def main():
    """
    Run a demonstration of the FakeNews detection workflow.

    This includes:
    - Loading and previewing the dataset
    - Preprocessing a sample text
    - Extracting features from processed text
    - Training the FakeNewsPipeline on the dataset
    - Predicting the label of a sample news sentence
    - Displaying a preview of the dataset

    This function provides a simple example of how preprocessing,
    feature extraction, model training, and prediction work together
    within the FakeNews pipeline.
    """
    print("=== Loading dataset ===")
    df = load_dataset("News/news.csv")
    print(df.head())

    print("\n=== Preprocessing text ===")
    preprocessor = Preprocessor()
    sample_text = df['text'].iloc[0]
    cleaned_text = preprocessor(sample_text)
    print(f"Original text:\n{sample_text}\n")
    print(f"Cleaned text:\n{cleaned_text}\n")

    print("=== Feature extraction ===")
    extractor = FeatureExtractor(max_features=3000)
    features = extractor.fit_transform([cleaned_text])
    print(f"Feature vector shape: {features.shape}\n")

    print("=== Training pipeline ===")
    pipeline = FakeNewsPipeline(model=None, feature_max_features=3000)
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    pipeline.fit(texts, labels)
    print("Pipeline trained!")

    print("\n=== Making prediction on sample text ===")
    test_sample = "Breaking news: Scientists discovered a new planet in the solar system."
    prediction = pipeline.predict(test_sample)
    print(f"Sample text prediction: {prediction.upper()}\n")

    print("=== Preview dataset ===")
    preview(df)

if __name__ == "__main__":
    main()
