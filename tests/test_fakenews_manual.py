from fakenews import load_dataset, FakeNewsPipeline

def main():
    print("Loading dataset...")
    df = load_dataset("News/news.csv")
    print(f"Dataset loaded with {len(df)} entries.\n")

    print("Training FakeNewsPipeline on the dataset...")
    pipeline = FakeNewsPipeline(model=None, feature_max_features=3000)
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    pipeline.fit(texts, labels)
    print("Pipeline trained!\n")

    while True:
        news_text = input("Type or paste a news article (or 'exit' to quit):\n> ").strip()
        if news_text.lower() == "exit":
            print("Exiting.")
            break
        if not news_text:
            print("Please enter some text.")
            continue
        prediction = pipeline.predict(news_text)
        print(f"Prediction: {prediction.upper()}\n")

if __name__ == "__main__":
    main()
