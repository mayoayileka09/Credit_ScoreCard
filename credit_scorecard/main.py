import warnings
import typer
import mlflow
from pathlib import Path
import pandas as pd
import joblib

from sklearn.metrics import roc_auc_score, classification_report

from credit_scorecard.config.config import logger
from credit_scorecard import data, train, report  # plotter removed for now to avoid breaking

app = typer.Typer()
warnings.filterwarnings("ignore")

DATA_PATH = "data2/raw/loan_data_2007_2014.csv"


@app.command()
def run_pipeline(
    data_path: str = DATA_PATH,
    report_path: str = "report.zip",
    fig_path: str = "eval_plot.png",  # kept for CLI compatibility; may not be used yet
    test_run: bool = True,
):
    # --------------------
    # Step 1: Preprocess
    # --------------------
    clean_path = data.preprocess_data(data_path)
    logger.info(f"clean_path resolved to: {clean_path}")

    if not test_run:
        mlflow.log_artifact(clean_path)
        df_clean = pd.read_csv(clean_path)
        mlflow.log_param("rows_cleaned", int(df_clean.shape[0]))
        mlflow.log_param("cols_cleaned", int(df_clean.shape[1]))

    # --------------------
    # Step 2: Train
    # --------------------
    Path("models").mkdir(parents=True, exist_ok=True)
    model_path = "models/credit_risk_pd_model.joblib"

    score = train.train_model(clean_path, model_path)  # score is ROC-AUC (per updated train.py)
    logger.info(f"Training score (ROC-AUC test): {score:.4f}")

    if not test_run:
        mlflow.log_artifact(model_path)
        mlflow.log_metric("roc_auc_test", float(score))

    # --------------------
    # Step 3: Predict
    # --------------------
    artifact = joblib.load(model_path)

    # Support both formats: artifact dict OR raw sklearn model
    if isinstance(artifact, dict) and "model" in artifact:
        model = artifact["model"]
        feature_cols = artifact.get("feature_columns", None)
        target_col = artifact.get("meta", {}).get("target", "bad_loan")
        positive_label = artifact.get("meta", {}).get("positive_label", 1)
    else:
        model = artifact
        feature_cols = None
        target_col = "bad_loan"
        positive_label = 1

    df = pd.read_csv(clean_path)

    if target_col not in df.columns:
        raise ValueError(
            f"Expected target column '{target_col}' not found in cleaned data. "
            "Did preprocess_data() run correctly?"
        )

    if feature_cols is not None:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Cleaned data missing required model features: {missing[:20]}")
        X = df[feature_cols]
    else:
        X = df.drop(columns=[target_col], errors="ignore")

    if not hasattr(model, "predict_proba"):
        raise ValueError("Loaded model does not support predict_proba; expected LogisticRegression-like model.")

    # Your preprocessing defines 1 = GOOD, 0 = BAD (kernel behavior).
    proba_good = model.predict_proba(X)[:, 1]

    # PD = P(default/bad) = 1 - P(good)
    df["pd"] = 1.0 - proba_good

    pred_path = clean_path.replace(".csv", "_pred.csv")
    df.to_csv(pred_path, index=False)
    logger.info(f"Saved predictions to {pred_path}")

    if not test_run:
        mlflow.log_artifact(pred_path)
        mlflow.log_metric("mean_pd", float(df["pd"].mean()))

    # --------------------
    # Step 4: Evaluate
    # --------------------
    y_true = df[target_col].astype(int)

    # For ROC-AUC we want probability of the positive class.
    # If positive_label==1 means "good", then we should use proba_good;
    # but if you want AUC for "bad", you use pd (prob bad).
    # Here we log BOTH to avoid confusion.
    auc_good = roc_auc_score(y_true, proba_good)
    auc_bad = roc_auc_score(1 - y_true, df["pd"])  # probability of bad vs (1-y_true)

    logger.info(f"ROC-AUC (good as positive, label=1): {auc_good:.4f}")
    logger.info(f"ROC-AUC (bad as positive): {auc_bad:.4f}")

    # Optional threshold report for "bad" using pd >= 0.5
    y_pred_bad = (df["pd"] >= 0.5).astype(int)  # 1 means predicted bad
    report_txt = classification_report((1 - y_true), y_pred_bad, digits=3)
    logger.info("Classification report for BAD (threshold pd>=0.5):\n" + report_txt)

    if not test_run:
        mlflow.log_metric("roc_auc_good", float(auc_good))
        mlflow.log_metric("roc_auc_bad", float(auc_bad))

    # Note: plotter.plot_regression removed because this is classification now.
    # If you later add plotter.plot_roc(...) or plot_pr(...), you can log fig_path again.

    # --------------------
    # Step 5: Build report
    # --------------------
    # Assumes your report builder can accept a CSV path + optional image path.
    # If report.build_report expects y/pred columns, update it to use 'bad_loan' and 'pd'.
    zip_bytes = report.build_report(pred_path, fig_path)
    with open(report_path, "wb") as f:
        f.write(zip_bytes.read())
    logger.info(f"Report saved to {report_path}")


if __name__ == "__main__":
    app()