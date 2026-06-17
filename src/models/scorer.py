"""Train regression and binary classification models on CLIP embeddings."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("data/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def build_targets(df: pd.DataFrame):
    """Return continuous log-score and binary high/low quality label."""
    y_reg = np.log1p(df["score"]).values
    threshold = np.median(y_reg)
    y_clf = (y_reg > threshold).astype(int)
    return y_reg, y_clf, threshold


def train_regression(embeddings, y):
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, y, test_size=0.2, random_state=42
    )
    models = {
        "ridge": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=10.0))]),
        "rf":    RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "gbm":   GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42),
    }

    results = {}
    best_r2, best_name, best_model = -np.inf, None, None

    print("=== Regression ===")
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        r2  = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        results[name] = {"r2": round(r2, 4), "mae": round(mae, 4)}
        print(f"  {name:6s}  R²={r2:.4f}  MAE={mae:.4f}")
        if r2 > best_r2:
            best_r2, best_name, best_model = r2, name, model

    print(f"\n  Best: {best_name} (R²={best_r2:.4f})")
    joblib.dump(best_model, MODELS_DIR / "scorer.pkl")
    np.save(MODELS_DIR / "score_range.npy", np.array([y.min(), y.max()]))
    return results, X_test, y_test


def train_classifier(embeddings, y_clf):
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )
    models = {
        "logreg": Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000, C=0.1))]),
        "rf":     RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "gbm":    GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42),
    }

    results = {}
    best_auc, best_name, best_model = -np.inf, None, None

    print("\n=== Binary Classification (above/below median score) ===")
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds     = model.predict(X_test)
        proba     = model.predict_proba(X_test)[:, 1]
        auc       = roc_auc_score(y_test, proba)
        avg_prec  = average_precision_score(y_test, proba)
        cm        = confusion_matrix(y_test, preds)
        results[name] = {
            "auc_roc": round(auc, 4),
            "avg_precision": round(avg_prec, 4),
            "confusion_matrix": cm.tolist(),
        }
        print(f"  {name:8s}  AUC-ROC={auc:.4f}  Avg-Precision={avg_prec:.4f}")
        print(f"  Confusion matrix:\n{cm}")
        print(f"  {classification_report(y_test, preds, target_names=['low','high'])}")
        if auc > best_auc:
            best_auc, best_name, best_model = auc, name, model

    print(f"  Best classifier: {best_name} (AUC={best_auc:.4f})")
    joblib.dump(best_model, MODELS_DIR / "classifier.pkl")
    return results


def run():
    embeddings = np.load(PROCESSED_DIR / "embeddings.npy")
    df = pd.read_csv(PROCESSED_DIR / "clustered.csv")

    y_reg, y_clf, threshold = build_targets(df)
    print(f"Binary threshold (median log-score): {threshold:.3f}  →  score ≈ {int(np.expm1(threshold))}")
    print(f"High quality: {y_clf.sum()} photos  |  Low quality: {(1-y_clf).sum()} photos\n")

    reg_results, _, _ = train_regression(embeddings, y_reg)
    clf_results       = train_classifier(embeddings, y_clf)

    print("\n=== Summary ===")
    print("Regression:", reg_results)
    print("Classification:", {k: {m: v for m, v in r.items() if m != "confusion_matrix"} for k, r in clf_results.items()})


if __name__ == "__main__":
    run()
