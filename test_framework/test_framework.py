import os
from fakenews.utils import load_dataset, preview
from fakenews.preprocessing import preprocess_pipeline
from fakenews.features import FeatureExtractor
from fakenews.models import FakeNewsModel

csv_path = os.path.join(os.path.dirname(__file__), "..", "News", "news.csv")
print("Trying to load CSV from:", csv_path)

df = load_dataset(csv_path)
df = df.dropna(subset=["text", "label"])
df = df[df["label"].astype(str).str.strip() != ""]
df = df.reset_index(drop=True)
print("Dataset preview:")
print(preview(df))

df["cleaned"] = df["text"].apply(preprocess_pipeline)

print(type(df["cleaned"]))       
print(type(df["cleaned"].tolist())) 

fe = FeatureExtractor()
X = fe.fit_transformation(df["cleaned"].tolist()) 
y = df["label"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = FakeNewsModel()
model.train(X_train, y_train)

results = model.evaluate(X_test, y_test)
print("\nModel Accuracy:", results["accuracy"])
print(results["report"])

new_article = "Breaking news: Scientists discover cure for XYZ."
processed = preprocess_pipeline(new_article)
vector = fe.transform([processed])
prediction = model.predict_single(vector)
print("Prediction for new article:", prediction)