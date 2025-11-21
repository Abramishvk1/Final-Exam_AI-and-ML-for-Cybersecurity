#!/usr/bin/env python3
import argparse, pickle, json
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path

def main(data_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_path)
    X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])
    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_df=0.9, min_df=3)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_tfidf, y_train)
    with open(out_dir/"vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open(out_dir/"model.pkl", "wb") as f:
        pickle.dump(clf, f)
    y_pred = clf.predict(X_test_tfidf)
    rep = classification_report(y_test, y_pred, output_dict=True)
    import json
    with open(out_dir/"classification_report.json", "w") as f:
        json.dump(rep, f, indent=2)
    print("Training complete. Artifacts saved to", out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out-dir", default=".")
    args = parser.parse_args()
    main(args.data, args.out_dir)
