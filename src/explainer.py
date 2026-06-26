"""
SHAPExplainer: TreeSHAP-based explanation engine for the XGBoost model.

Implements the TreeSHAP algorithm described in Chapter 3, Section 3.7.2
(Lundberg et al., 2018). Produces both local (per-transaction) and global
(dataset-level) Shapley value explanations.
"""
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import shap


class SHAPExplainer:
    """
    Wraps shap.TreeExplainer to provide consistent local and global
    SHAP explanations for the FraudDetectionModel.
    """

    def __init__(
        self,
        model,
        background_data: Optional[pd.DataFrame] = None,
    ) -> None:
        self.feature_names: List[str] = model.feature_names or []
        self._explainer = shap.TreeExplainer(
            model.model,
            data=background_data,
            feature_perturbation=(
                "interventional" if background_data is not None else "tree_path_dependent"
            ),
        )
        self.expected_value: float = self._resolve_expected_value()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_expected_value(self) -> float:
        ev = self._explainer.expected_value
        if isinstance(ev, (list, np.ndarray)):
            return float(np.asarray(ev).ravel()[-1])
        return float(ev)

    def _raw_shap(self, X) -> np.ndarray:
        """Return SHAP values as a 2-D array (n_samples × n_features)."""
        raw = self._explainer.shap_values(X)
        if isinstance(raw, list):
            return np.array(raw[-1])
        return np.array(raw)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain_instance(self, X_single: pd.DataFrame) -> shap.Explanation:
        """Return a shap.Explanation object for a single transaction row."""
        values = self._raw_shap(X_single)[0]
        data_vals = X_single.values[0] if hasattr(X_single, "values") else X_single[0]
        return shap.Explanation(
            values=values,
            base_values=self.expected_value,
            data=data_vals,
            feature_names=self.feature_names,
        )

    def explain_batch(self, X: pd.DataFrame) -> np.ndarray:
        """Return SHAP values matrix for a batch (n_samples × n_features)."""
        return self._raw_shap(X)

    def get_waterfall_data(self, X_single: pd.DataFrame) -> pd.DataFrame:
        """
        Return a DataFrame with columns [feature, value, shap_value] sorted
        by absolute SHAP contribution (descending) for a single transaction.
        """
        values = self._raw_shap(X_single)[0]
        data_vals = X_single.values[0] if hasattr(X_single, "values") else X_single[0]
        df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "value": data_vals,
                "shap_value": values,
            }
        )
        df["_abs"] = df["shap_value"].abs()
        return df.sort_values("_abs", ascending=False).drop("_abs", axis=1).reset_index(drop=True)

    def get_global_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Return mean absolute SHAP values across a dataset as a sorted DataFrame
        with columns [feature, mean_abs_shap].
        """
        shap_vals = self._raw_shap(X)
        mean_abs = np.abs(shap_vals).mean(axis=0)
        return (
            pd.DataFrame({"feature": self.feature_names, "mean_abs_shap": mean_abs})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @classmethod
    def load(cls, path: Path) -> "SHAPExplainer":
        with open(path, "rb") as fh:
            return pickle.load(fh)
