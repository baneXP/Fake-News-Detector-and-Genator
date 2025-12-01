import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

class FakeNewsDetector:
    def __init__(self, model_path=None):
        # initialize vectorizer and model, optionally load from file
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        self.model = LogisticRegression()
        if model_path:
            self.load(model_path)

    def train(self, dataset_path):
        df = pd.read_csv(dataset_path)
        X_train, X_test, y_train, y_test = train_test_split(
            df["text"], df["label"], test_size=0.2, random_state=42
        )

        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)

        self.model.fit(X_train_tfidf, y_train)
        preds = self.model.predict(X_test_tfidf)
        print(classification_report(y_test, preds))

        joblib.dump((self.vectorizer, self.model), "detector/fake_news_model.pkl")

    def load(self, model_path="detector/fake_news_model.pkl"):
        self.vectorizer, self.model = joblib.load(model_path)

    def predict(self, text):
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model not loaded. Call load() first or pass model_path to constructor.")
        features = self.vectorizer.transform([text])
        return self.model.predict(features)[0]
