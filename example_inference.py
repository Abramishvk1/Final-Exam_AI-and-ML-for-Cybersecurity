#!/usr/bin/env python3
import argparse, pickle
from pathlib import Path
def main(model_path, vectorizer_path, text):
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    X = vectorizer.transform([text])
    proba = model.predict_proba(X)[0][1]
    pred = model.predict(X)[0]
    print(f"Prediction: {pred} (1=spam,0=ham) | spam probability: {proba:.3f}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--vectorizer", required=True)
    p.add_argument("--text", required=True)
    args = p.parse_args()
    main(args.model, args.vectorizer, args.text)
