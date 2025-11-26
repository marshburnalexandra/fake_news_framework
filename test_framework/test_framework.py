"""
Fake News Detection Script

This script loads a news dataset, cleans the text, extracts TF-IDF features,
trains multiple machine-learning models, evaluates them, and tests a complete
FakeNewsPipeline. It demonstrates preprocessing, feature extraction, model
training, and making a single prediction.

Steps:
1. Load and clean dataset
2. Test Preprocessor
3. Extract features with FeatureExtractor
4. Train Logistic Regression, Naive Bayes, and SVM models
5. Train and evaluate FakeNewsPipeline
6. Predict label for an example article
"""
import os
from fakenews.utils import load_dataset, preview
from fakenews.pipeline import FakeNewsPipeline
from fakenews.preprocessing import Preprocessor
from fakenews.features import FeatureExtractor
from fakenews.models import LogisticNewsModel, NaiveBayesNewsModel, SVMNewsModel

# -----------------------------
# 1. Load dataset
# -----------------------------
csv_path = os.path.join(os.path.dirname(__file__), "..", "News", "news.csv")
print("Loading CSV from:", csv_path)

df = load_dataset(csv_path)
df = df.dropna(subset=["text", "label"])
df = df[df["label"].astype(str).str.strip() != ""]
df = df.reset_index(drop=True)

print("\nDataset preview:")
print(preview(df, 5))

# -----------------------------
# 2. Demonstrate Preprocessor
# -----------------------------
print("\n--- Preprocessor Test ---")
preprocessor = Preprocessor(remove_stopwords=True, stem=True)
sample_text = "Breaking News: Scientists discover cure for COVID-19! Visit https://example.com"
cleaned_text = preprocessor(sample_text)  # Using __call__ dunder
print("Original:", sample_text)
print("Cleaned:", cleaned_text)
print("Preprocessor repr:", repr(preprocessor))

# -----------------------------
# 3. Feature extraction
# -----------------------------
print("\n--- FeatureExtractor Test ---")
feature_extractor = FeatureExtractor(max_features=3000, preprocessor=preprocessor)
cleaned_texts = [preprocessor(t) for t in df["text"].tolist()]
X = feature_extractor.fit_transform(cleaned_texts)
print("Feature matrix shape:", X.shape)
print("FeatureExtractor repr:", repr(feature_extractor))

# -----------------------------
# 4. Model tests (Inheritance + Polymorphism)
# -----------------------------
print("\n--- Model Tests ---")
y = df["label"].tolist()

# Using different models
models = [
    LogisticNewsModel(),
    NaiveBayesNewsModel(),
    SVMNewsModel()
]

for model in models:
    print(f"\nTraining {repr(model)}")
    model.train(X, y)
    results = model.evaluate(X, y)
    print("Accuracy:", results["accuracy"])
    # print(results["report"])  # Optional: full classification report

# -----------------------------
# 5. FakeNewsPipeline (Composition)
# -----------------------------
print("\n--- FakeNewsPipeline Test ---")
pipeline = FakeNewsPipeline(model=LogisticNewsModel(), feature_max_features=3000)

pipeline.fit(df["text"].tolist(), df["label"].tolist())
pipeline_results = pipeline.evaluate(df["text"].tolist(), df["label"].tolist())
print("Pipeline Accuracy:", pipeline_results["accuracy"])
print("Pipeline repr:", repr(pipeline))

# -----------------------------
# 6. Predict single article
# -----------------------------
print("\n--- Single Prediction ---")
example_article = "Breaking news: Scientists announce a new study showing X cures Y."
prediction = pipeline.predict(example_article)
print("Article:", example_article)
print("Predicted label:", prediction)
