"""
TransactionProcessor: feature scaling and validation for raw transaction inputs.

Applies StandardScaler to Amount and Time (V1-V28 are already PCA-normalised
in the Kaggle dataset). Serialisable via pickle for consistent inference.
"""
import pickle
from pathlib import Path
from typing import List

import pandas as pd
from sklearn.preprocessing import StandardScaler


class TransactionProcessor:
    """Fits and applies StandardScaler to Amount and Time features."""

    def __init__(self) -> None:
        self._scaler = StandardScaler()
        self._feature_names: List[str] = []
        self._scale_cols: List[str] = []
        self._fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, scale_cols: List[str] = None) -> "TransactionProcessor":
        self._feature_names = list(X.columns)
        self._scale_cols = [c for c in (scale_cols or ["Amount", "Time"]) if c in X.columns]
        if self._scale_cols:
            self._scaler.fit(X[self._scale_cols])
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("TransactionProcessor must be fitted before transform.")
        out = X[self._feature_names].copy()
        if self._scale_cols:
            out[self._scale_cols] = self._scaler.transform(out[self._scale_cols])
        return out

    def fit_transform(self, X: pd.DataFrame, scale_cols: List[str] = None) -> pd.DataFrame:
        self.fit(X, scale_cols)
        return self.transform(X)

    @property
    def feature_names(self) -> List[str]:
        return list(self._feature_names)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @classmethod
    def load(cls, path: Path) -> "TransactionProcessor":
        with open(path, "rb") as fh:
            return pickle.load(fh)
