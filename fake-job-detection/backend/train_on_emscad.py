"""
train_on_emscad.py
==================
Retrain FraudShield on the real EMSCAD dataset (or any CSV with the same schema).

EMSCAD download:
  https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction
  File: fake_job_postings.csv

Usage:
  python train_on_emscad.py                             # uses fake_job_postings.csv in current dir
  python train_on_emscad.py --csv path/to/dataset.csv  # custom path
  python train_on_emscad.py --csv fake_job_postings.csv --rows 5000  # limit rows (faster)

Expected CSV columns (EMSCAD schema):
  title, location, department, salary_range, company_profile, description,
  requirements, benefits, telecommuting, has_company_logo, has_questions,
  employment_type, required_experience, required_education, industry, function,
  fraudulent   ← label column (0=real, 1=fake)
"""

import argparse
import pickle
import logging
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

from models.detector import FeatureExtractor, _to_struct_vec, MODEL_PATH

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_emscad(csv_path: str, max_rows: int = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path, nrows=max_rows)
    # Normalise column names to lowercase
    df.columns = [c.lower().strip() for c in df.columns]

    required = {"fraudulent"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV missing required column 'fraudulent'. Found: {list(df.columns)}")

    logger.info("Loaded %d rows — %d fake, %d real",
                len(df), df["fraudulent"].sum(), (df["fraudulent"] == 0).sum())
    return df


def row_to_job_dict(row: pd.Series) -> dict:
    """Map EMSCAD row → job_data dict understood by FeatureExtractor."""
    salary = str(row.get("salary_range", "")) if pd.notna(row.get("salary_range")) else ""
    return {
        "title":        str(row.get("title", "")) if pd.notna(row.get("title")) else "",
        "company":      str(row.get("company_profile", "")) if pd.notna(row.get("company_profile")) else "",
        "description":  str(row.get("description", "")) if pd.notna(row.get("description")) else "",
        "requirements": str(row.get("requirements", "")) if pd.notna(row.get("requirements")) else "",
        "salary":       salary,
        "location":     str(row.get("location", "")) if pd.notna(row.get("location")) else "",
    }


def build_feature_matrix(df: pd.DataFrame, extractor: FeatureExtractor, tfidf: TfidfVectorizer = None, fit_tfidf: bool = True):
    struct_rows, text_rows = [], []
    for _, row in df.iterrows():
        job_dict = row_to_job_dict(row)
        feats = extractor.extract(job_dict)
        struct_rows.append(_to_struct_vec(feats))
        text_rows.append(feats["full_text"])

    X_struct = np.array(struct_rows)

    if fit_tfidf:
        tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=1000, sublinear_tf=True)
        X_tfidf = tfidf.fit_transform(text_rows)
    else:
        X_tfidf = tfidf.transform(text_rows)

    X_combined = sp.hstack([sp.csr_matrix(X_struct), X_tfidf]).toarray()
    return X_combined, tfidf


def train(csv_path: str, max_rows: int = None):
    df = load_emscad(csv_path, max_rows)
    y = df["fraudulent"].values.astype(int)

    extractor = FeatureExtractor()
    logger.info("Extracting features from %d samples...", len(df))
    X, tfidf = build_feature_matrix(df, extractor, fit_tfidf=True)

    logger.info("Training ensemble (RF + LR)...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, n_jobs=-1,
                                 random_state=42, class_weight="balanced")
    lr = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced",
                             random_state=42, solver="lbfgs", n_jobs=-1)
    ensemble = VotingClassifier([("rf", rf), ("lr", lr)], voting="soft", weights=[0.6, 0.4])

    # Cross-validation
    logger.info("Running 5-fold cross-validation (this takes ~1-2 min on full dataset)...")
    cv_scores = cross_val_score(ensemble, X, y, cv=5, scoring="f1", n_jobs=-1)
    logger.info("CV F1 scores: %s  →  mean=%.3f  std=%.3f",
                cv_scores.round(3), cv_scores.mean(), cv_scores.std())

    # Final fit on all data
    ensemble.fit(X, y)

    y_pred = ensemble.predict(X)
    print("\n=== Training Set Classification Report ===")
    print(classification_report(y, y_pred, target_names=["REAL", "FAKE"]))

    bundle = {"ensemble": ensemble, "tfidf": tfidf}
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as fh:
        pickle.dump(bundle, fh)
    logger.info("✅ Model saved to %s", MODEL_PATH)
    logger.info("Restart your FastAPI server to pick up the new model.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FraudShield on EMSCAD dataset")
    parser.add_argument("--csv", default="fake_job_postings.csv", help="Path to EMSCAD CSV")
    parser.add_argument("--rows", type=int, default=None, help="Max rows to use (default: all)")
    args = parser.parse_args()

    if not Path(args.csv).exists():
        print(f"ERROR: CSV not found at '{args.csv}'")
        print("Download from: https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction")
        exit(1)

    train(args.csv, args.rows)
