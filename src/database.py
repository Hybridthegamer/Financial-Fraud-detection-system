"""
DatabaseManager: SQLite-backed audit logging and analyst account management.

Schema mirrors the ER diagram in Chapter 3, Section 3.6:
  - analyst           : user accounts
  - transaction_record: raw feature vectors submitted for analysis
  - prediction        : model outputs (fraud probability + classification)
  - shap_explanation  : per-feature SHAP values for each prediction

All timestamps stored as UTC ISO-8601 strings.
Primary keys for transactions and analysts use UUID4 to prevent enumeration.
"""
import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


class DatabaseManager:
    """Manages the SQLite audit database for AFDS-xAI."""

    def __init__(self, db_path: Path) -> None:
        self._path = str(db_path)
        self._init_schema()
        self._seed_demo_analyst()

    # ------------------------------------------------------------------
    # Connection helper
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    # Schema initialisation
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS analyst (
                    analyst_id    TEXT PRIMARY KEY,
                    name          TEXT NOT NULL,
                    email         TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role          TEXT NOT NULL DEFAULT 'analyst',
                    created_at    TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS transaction_record (
                    transaction_id TEXT PRIMARY KEY,
                    analyst_id     TEXT NOT NULL,
                    feature_vector TEXT NOT NULL,
                    ingested_at    TEXT NOT NULL,
                    FOREIGN KEY (analyst_id) REFERENCES analyst(analyst_id)
                );

                CREATE TABLE IF NOT EXISTS prediction (
                    prediction_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                    transaction_id    TEXT UNIQUE NOT NULL,
                    fraud_probability REAL NOT NULL,
                    is_fraud          INTEGER NOT NULL,
                    model_version     TEXT NOT NULL,
                    predicted_at      TEXT NOT NULL,
                    FOREIGN KEY (transaction_id)
                        REFERENCES transaction_record(transaction_id)
                );

                CREATE TABLE IF NOT EXISTS shap_explanation (
                    explanation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_id  INTEGER NOT NULL,
                    feature_name   TEXT NOT NULL,
                    feature_value  REAL,
                    shap_value     REAL NOT NULL,
                    rank           INTEGER NOT NULL,
                    FOREIGN KEY (prediction_id)
                        REFERENCES prediction(prediction_id)
                );
                """
            )

    def _seed_demo_analyst(self) -> None:
        """Create a demo analyst account on first run."""
        with self._connect() as conn:
            if conn.execute("SELECT COUNT(*) FROM analyst").fetchone()[0] == 0:
                self.create_analyst(
                    name="Demo Analyst",
                    email="demo@fraudsystem.com",
                    password="demo123",
                    role="admin",
                )

    # ------------------------------------------------------------------
    # Password hashing
    # ------------------------------------------------------------------

    @staticmethod
    def _hash(password: str) -> str:
        return hashlib.sha256(password.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Analyst management
    # ------------------------------------------------------------------

    def create_analyst(
        self, name: str, email: str, password: str, role: str = "analyst"
    ) -> str:
        analyst_id = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO analyst VALUES (?,?,?,?,?,?)",
                (
                    analyst_id,
                    name,
                    email,
                    self._hash(password),
                    role,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
        return analyst_id

    def authenticate(self, email: str, password: str) -> Optional[Dict[str, str]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT analyst_id, name, role FROM analyst "
                "WHERE email=? AND password_hash=?",
                (email, self._hash(password)),
            ).fetchone()
        return dict(row) if row else None

    def email_exists(self, email: str) -> bool:
        with self._connect() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM analyst WHERE email=?", (email,)
            ).fetchone()[0] > 0

    # ------------------------------------------------------------------
    # Audit logging
    # ------------------------------------------------------------------

    def log_analysis(
        self,
        analyst_id: str,
        feature_vector: Dict[str, float],
        fraud_probability: float,
        is_fraud: bool,
        shap_df: pd.DataFrame,
        model_version: str = "1.0.0",
    ) -> str:
        """
        Persist one complete fraud analysis event:
          transaction_record → prediction → shap_explanation (one row per feature)
        Returns the new transaction_id UUID.
        """
        transaction_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        with self._connect() as conn:
            conn.execute(
                "INSERT INTO transaction_record VALUES (?,?,?,?)",
                (transaction_id, analyst_id, json.dumps(feature_vector), now),
            )
            cursor = conn.execute(
                "INSERT INTO prediction "
                "(transaction_id, fraud_probability, is_fraud, model_version, predicted_at) "
                "VALUES (?,?,?,?,?)",
                (transaction_id, float(fraud_probability), int(is_fraud), model_version, now),
            )
            prediction_id = cursor.lastrowid

            ranked = (
                shap_df.assign(_abs=shap_df["shap_value"].abs())
                .sort_values("_abs", ascending=False)
                .drop("_abs", axis=1)
                .reset_index(drop=True)
            )
            for rank, row in ranked.iterrows():
                conn.execute(
                    "INSERT INTO shap_explanation "
                    "(prediction_id, feature_name, feature_value, shap_value, rank) "
                    "VALUES (?,?,?,?,?)",
                    (
                        prediction_id,
                        row["feature"],
                        float(row["value"]),
                        float(row["shap_value"]),
                        int(rank) + 1,
                    ),
                )
        return transaction_id

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_recent_predictions(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    p.transaction_id,
                    a.name          AS analyst_name,
                    p.fraud_probability,
                    p.is_fraud,
                    p.model_version,
                    p.predicted_at
                FROM prediction p
                JOIN transaction_record tr ON p.transaction_id = tr.transaction_id
                JOIN analyst a             ON tr.analyst_id     = a.analyst_id
                ORDER BY p.predicted_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_prediction_shap(self, transaction_id: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT se.feature_name, se.feature_value, se.shap_value, se.rank
                FROM shap_explanation se
                JOIN prediction p ON se.prediction_id = p.prediction_id
                WHERE p.transaction_id = ?
                ORDER BY se.rank
                """,
                (transaction_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_summary_stats(self) -> Dict[str, int]:
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM prediction").fetchone()[0]
            fraud = conn.execute(
                "SELECT COUNT(*) FROM prediction WHERE is_fraud=1"
            ).fetchone()[0]
        return {"total": total, "fraud": fraud, "legitimate": total - fraud}
