from __future__ import annotations

import joblib
import pandas as pd

from credit_scorecard.config.config import logger

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

# optional dependency; if not installed, we fall back safely
try:
    from imblearn.over_sampling import RandomOverSampler
    _HAS_IMBLEARN = True
except Exception:
    RandomOverSampler = None
    _HAS_IMBLEARN = False


def train_model(data_path: str, model_path: str) -> float:
    """
    Train a PD model (logistic regression) on the fully preprocessed dataset and save it.

    Expects the CSV to already be model-ready:
      - contains target column 'bad_loan'
      - all remaining columns are numeric (dummy / binned)
    Returns ROC-AUC on the held-out test set.
    """
    df = pd.read_csv(data_path)
    logger.info(f"Loaded training data: {data_path} shape={df.shape}")

    if "bad_loan" not in df.columns:
        raise ValueError("Expected target column 'bad_loan' not found. Did you run preprocess_data()?")

    # Features/target
    X = df.drop(columns=["bad_loan"])
    y = df["bad_loan"].astype(int)

    # Safety: ensure all features are numeric
    non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if non_numeric:
        raise ValueError(
            f"Non-numeric feature columns found (should be all numeric after preprocessing): {non_numeric[:20]}"
        )

    # Train/test split (stratify keeps class proportions)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Oversample minority class on TRAIN ONLY (kernel behavior)
    if _HAS_IMBLEARN:
        ros = RandomOverSampler(random_state=42)
        X_train_bal, y_train_bal = ros.fit_resample(X_train, y_train)
        logger.info(
            "Applied RandomOverSampler. "
            f"Train class counts: {y_train.value_counts().to_dict()} -> {pd.Series(y_train_bal).value_counts().to_dict()}"
        )
    else:
        X_train_bal, y_train_bal = X_train, y_train
        logger.warning(
            "imblearn not installed; training without oversampling. "
            "Install with: pip install imbalanced-learn"
        )

    # Logistic regression PD model
    # Use a solver that handles lots of features well.
    model = LogisticRegression(
        solver="liblinear",
        max_iter=2000,
        n_jobs=None,   # liblinear ignores n_jobs
    )
    model.fit(X_train_bal, y_train_bal)

    # Evaluate on test
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)

    y_pred = (y_proba >= 0.5).astype(int)
    report = classification_report(y_test, y_pred, digits=3)

    logger.info(f"PD LogisticRegression ROC-AUC (test): {auc:.4f}")
    logger.info("Classification report (threshold=0.5):\n" + report)

    # Save everything needed for consistent inference later
    artifact = {
        "model": model,
        "feature_columns": list(X.columns),
        "metrics": {
            "roc_auc_test": float(auc),
        },
        "meta": {
            "target": "bad_loan",
            "positive_label": 1,  # IMPORTANT: your preprocessing defines 1 as "good"
        },
    }
    joblib.dump(artifact, model_path)
    logger.info(f"Saved model artifact to {model_path}")

    return float(auc)