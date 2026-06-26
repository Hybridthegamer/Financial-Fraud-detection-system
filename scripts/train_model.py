#!/usr/bin/env python3
"""
AFDS-xAI Training Pipeline
===========================
Implements the CRISP-DM data science lifecycle described in Chapter 3.

Steps:
  1. Data Understanding  — load and inspect the Kaggle Credit Card Fraud dataset
  2. Data Preparation    — stratified train/test split, StandardScaler, SMOTE
  3. Modelling           — 5-fold CV + final XGBoost fit (hyperparams: Ch 3.7.1)
  4. Evaluation          — AUC-ROC, F1, precision, recall, confusion matrix, ROC curve
  5. Explainability      — TreeSHAP global feature importance (beeswarm plot)
  6. Deployment prep     — pickle artefacts for Streamlit app

Prerequisites:
  Download the Kaggle Credit Card Fraud Detection dataset and place it at:
      data/creditcard.csv
  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
"""

import json
import logging
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import roc_curve

# Make src importable when run from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    BACKGROUND_SAMPLE_SIZE,
    DATASET_PATH,
    DEMO_SAMPLES_PER_CLASS,
    EVAL_RESULTS_PATH,
    EXPLAINER_ARTIFACT,
    FEATURES,
    MODEL_ARTIFACT,
    MODELS_DIR,
    N_CV_FOLDS,
    PLOTS_DIR,
    PREPROCESSOR_ARTIFACT,
    RANDOM_STATE,
    SAMPLE_TRANSACTIONS_PATH,
    SCALE_FEATURES,
    TARGET,
    TEST_SHAP_SAMPLE_SIZE,
    TEST_SIZE,
)
from src.explainer import SHAPExplainer
from src.model import FraudDetectionModel
from src.preprocessing import TransactionProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    if not DATASET_PATH.exists():
        log.error("Dataset not found at %s", DATASET_PATH)
        log.error("Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        sys.exit(1)

    log.info("Loading dataset from %s", DATASET_PATH)
    df = pd.read_csv(DATASET_PATH)
    log.info("Shape: %s   Fraud rate: %.4f%%", df.shape, df[TARGET].mean() * 100)

    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        log.error("Missing columns: %s", missing)
        sys.exit(1)

    return df[FEATURES], df[TARGET]


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def plot_roc_curve(y_true, y_score, save_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = float(np.trapz(tpr, fpr))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="#c0392b", lw=2, label=f"XGBoost (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="ROC Curve — AFDS-xAI", xlim=[0, 1], ylim=[0, 1.02])
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("ROC curve → %s", save_path)


def plot_confusion_matrix(tn: int, fp: int, fn: int, tp: int, save_path: Path) -> None:
    cm = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    labels = ["Legitimate", "Fraud"]
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=labels, yticklabels=labels,
           xlabel="Predicted Label", ylabel="True Label",
           title="Confusion Matrix — AFDS-xAI")
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Confusion matrix → %s", save_path)


def plot_shap_summary(shap_values: np.ndarray, X_sample: pd.DataFrame, save_path: Path) -> None:
    fig = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, max_display=20, show=False)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("SHAP summary plot → %s", save_path)


def plot_shap_bar(shap_values: np.ndarray, X_sample: pd.DataFrame, save_path: Path) -> None:
    fig = plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", max_display=20, show=False)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("SHAP bar chart → %s", save_path)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load ──────────────────────────────────────────────────────────────
    X, y = load_dataset()

    # ── 2. Train / test split ────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    log.info(
        "Train: %d rows (%.4f%% fraud)   Test: %d rows (%.4f%% fraud)",
        len(X_train), y_train.mean() * 100,
        len(X_test),  y_test.mean()  * 100,
    )

    # ── 3. Preprocessing ─────────────────────────────────────────────────────
    log.info("Fitting TransactionProcessor (StandardScaler on Amount, Time)…")
    processor = TransactionProcessor()
    X_train_scaled = processor.fit_transform(X_train, scale_cols=SCALE_FEATURES)
    X_test_scaled  = processor.transform(X_test)

    # ── 4. SMOTE oversampling on training set only ───────────────────────────
    log.info("Applying SMOTE to training set…")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_res_arr, y_res_arr = smote.fit_resample(X_train_scaled, y_train)
    X_res = pd.DataFrame(X_res_arr, columns=FEATURES)
    y_res = pd.Series(y_res_arr, name=TARGET)
    log.info(
        "After SMOTE: %d rows   fraud rate: %.2f%%",
        len(X_res), y_res.mean() * 100,
    )

    # ── 5. Cross-validation ──────────────────────────────────────────────────
    log.info("Running %d-fold stratified cross-validation…", N_CV_FOLDS)
    cv_clf = FraudDetectionModel(scale_pos_weight=1.0)
    skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(cv_clf.model, X_res, y_res, cv=skf, scoring="roc_auc", n_jobs=-1)
    log.info("CV AUC-ROC: %.4f ± %.4f", cv_scores.mean(), cv_scores.std())

    # ── 6. Final model training ──────────────────────────────────────────────
    log.info("Training final XGBoost model…")
    model = FraudDetectionModel(scale_pos_weight=1.0)
    model.fit(X_res, y_res, X_val=X_test_scaled, y_val=y_test)
    log.info("Training complete.")

    # ── 7. Evaluation ────────────────────────────────────────────────────────
    log.info("Evaluating on held-out test set…")
    results = model.evaluate(X_test_scaled, y_test)
    results["cv_auc_mean"] = float(cv_scores.mean())
    results["cv_auc_std"]  = float(cv_scores.std())

    log.info(
        "AUC-ROC: %.4f  |  Avg-Precision: %.4f  |  F1: %.4f  |  "
        "Precision: %.4f  |  Recall: %.4f",
        results["auc_roc"], results["average_precision"],
        results["f1"], results["precision"], results["recall"],
    )
    log.info(
        "Confusion matrix  TN=%d  FP=%d  FN=%d  TP=%d",
        results["tn"], results["fp"], results["fn"], results["tp"],
    )

    proba = model.predict_proba(X_test_scaled)

    # ── 8. SHAP explainer ────────────────────────────────────────────────────
    log.info("Building TreeSHAP explainer (background n=%d)…", BACKGROUND_SAMPLE_SIZE)
    background = X_train_scaled.sample(
        n=min(BACKGROUND_SAMPLE_SIZE, len(X_train_scaled)), random_state=RANDOM_STATE
    )
    explainer = SHAPExplainer(model, background_data=background)

    log.info("Computing global SHAP values on test sample (n=%d)…", TEST_SHAP_SAMPLE_SIZE)
    test_sample = X_test_scaled.sample(
        n=min(TEST_SHAP_SAMPLE_SIZE, len(X_test_scaled)), random_state=RANDOM_STATE
    )
    shap_values = explainer.explain_batch(test_sample)

    # ── 9. Plots ─────────────────────────────────────────────────────────────
    plot_roc_curve(y_test, proba, PLOTS_DIR / "roc_curve.png")
    plot_confusion_matrix(
        results["tn"], results["fp"], results["fn"], results["tp"],
        PLOTS_DIR / "confusion_matrix.png",
    )
    plot_shap_summary(shap_values, test_sample, PLOTS_DIR / "shap_summary.png")
    plot_shap_bar(shap_values, test_sample, PLOTS_DIR / "shap_bar.png")

    # ── 10. Save sample transactions for demo UI ─────────────────────────────
    fraud_samples  = X_test[y_test == 1].head(DEMO_SAMPLES_PER_CLASS).reset_index(drop=True)
    legit_samples  = X_test[y_test == 0].head(DEMO_SAMPLES_PER_CLASS).reset_index(drop=True)
    with open(SAMPLE_TRANSACTIONS_PATH, "wb") as fh:
        pickle.dump({"fraud": fraud_samples, "legitimate": legit_samples}, fh)
    log.info("Sample transactions → %s", SAMPLE_TRANSACTIONS_PATH)

    # ── 11. Save artefacts ───────────────────────────────────────────────────
    log.info("Saving model artefacts…")
    model.save(MODEL_ARTIFACT)
    processor.save(PREPROCESSOR_ARTIFACT)
    explainer.save(EXPLAINER_ARTIFACT)

    with open(EVAL_RESULTS_PATH, "w") as fh:
        json.dump(results, fh, indent=2)

    log.info("=" * 60)
    log.info("All artefacts saved to  models/")
    log.info("  AUC-ROC : %.4f", results["auc_roc"])
    log.info("  F1-Score: %.4f", results["f1"])
    log.info("  Recall  : %.4f", results["recall"])
    log.info("Launch the app:  streamlit run app.py")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
