"""Train a regression/ranking model to predict aesthetic score from CLIP embeddings."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("data/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def build_target(df: pd.DataFrame) -> pd.Series:
    """Log-normalised upvote score as aesthetic proxy."""
    return np.log1p(df["score"])


def train_and_evaluate(embeddings: np.ndarray, df: pd.DataFrame) -> dict:
    y = build_target(df).values
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, y, test_size=0.2, random_state=42
    )

    models = {
        "ridge": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=10.0))]),
        "rf": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "gbm": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42),
    }

    results = {}
    best_r2, best_name, best_model = -np.inf, None, None

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        results[name] = {"r2": round(r2, 4), "mae": round(mae, 4)}
        print(f"{name:8s}  R²={r2:.4f}  MAE={mae:.4f}")

        if r2 > best_r2:
            best_r2, best_name, best_model = r2, name, model

    print(f"\nBest model: {best_name}")
    joblib.dump(best_model, MODELS_DIR / "scorer.pkl")
    return results


def run():
    embeddings = np.load(PROCESSED_DIR / "embeddings.npy")
    df = pd.read_csv(PROCESSED_DIR / "clustered.csv")

    results = train_and_evaluate(embeddings, df)
    print(results)


if __name__ == "__main__":
    run()
