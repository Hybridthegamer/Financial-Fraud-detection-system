"""
FraudDetectionModel: XGBoost binary classifier for financial fraud detection.

Implements the algorithm described in Chapter 3, Section 3.7.1 with the
hyperparameters selected via 5-fold cross-validated grid search.
"""
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class FraudDetectionModel:
    """XGBoost-based fraud detection model with evaluation and persistence."""

    def __init__(self, scale_pos_weight: float = 1.0, **overrides: Any) -> None:
        params: Dict[str, Any] = {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "auc",
            "scale_pos_weight": scale_pos_weight,
            "random_state": 42,
            "n_jobs": -1,
            "tree_method": "hist",
        }
        params.update(overrides)
        self.model = xgb.XGBClassifier(**params)
        self.params = params
        self.threshold: float = 0.5
        self.feature_names: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "FraudDetectionModel":
        self.feature_names = list(X_train.columns)
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, X) -> np.ndarray:
        """Return fraud probability scores (class 1)."""
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X, threshold: Optional[float] = None) -> np.ndarray:
        """Return binary predictions using the configured decision threshold."""
        t = threshold if threshold is not None else self.threshold
        return (self.predict_proba(X) >= t).astype(int)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, X_test, y_test) -> Dict[str, Any]:
        """Return a comprehensive evaluation dictionary for the test set."""
        proba = self.predict_proba(X_test)
        preds = self.predict(X_test)
        cm = confusion_matrix(y_test, preds)
        tn, fp, fn, tp = cm.ravel()

        return {
            "auc_roc": float(roc_auc_score(y_test, proba)),
            "average_precision": float(average_precision_score(y_test, proba)),
            "f1": float(f1_score(y_test, preds, zero_division=0)),
            "precision": float(precision_score(y_test, preds, zero_division=0)),
            "recall": float(recall_score(y_test, preds, zero_division=0)),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "confusion_matrix": cm.tolist(),
            "classification_report": classification_report(
                y_test,
                preds,
                target_names=["Legitimate", "Fraud"],
                output_dict=True,
                zero_division=0,
            ),
        }

    def get_feature_importance(self) -> pd.DataFrame:
        importance = self.model.feature_importances_
        return pd.DataFrame(
            {"feature": self.feature_names, "importance": importance}
        ).sort_values("importance", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @classmethod
    def load(cls, path: Path) -> "FraudDetectionModel":
        with open(path, "rb") as fh:
            return pickle.load(fh)
