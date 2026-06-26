"""
Central configuration for AFDS-xAI.
All path constants, model hyperparameters, and thresholds live here.
"""
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

DB_PATH = BASE_DIR / "fraud_detection.db"
DATASET_PATH = DATA_DIR / "creditcard.csv"

MODEL_ARTIFACT = MODELS_DIR / "xgboost_model.pkl"
PREPROCESSOR_ARTIFACT = MODELS_DIR / "preprocessor.pkl"
EXPLAINER_ARTIFACT = MODELS_DIR / "shap_explainer.pkl"
EVAL_RESULTS_PATH = MODELS_DIR / "eval_results.json"
SAMPLE_TRANSACTIONS_PATH = MODELS_DIR / "sample_transactions.pkl"

PLOTS_DIR = MODELS_DIR

# Dataset feature specification (Kaggle Credit Card Fraud Detection dataset)
V_FEATURES = [f"V{i}" for i in range(1, 29)]
FEATURES = V_FEATURES + ["Amount", "Time"]
TARGET = "Class"
SCALE_FEATURES = ["Amount", "Time"]  # V1-V28 are already PCA-scaled

# XGBoost hyperparameters (from Chapter 3, Section 3.7.1)
XGBOOST_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "auc",
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",
}

# Risk thresholds (from Chapter 3, Section 3.5.2)
FRAUD_THRESHOLD_HIGH = 0.70    # Red — likely fraudulent
FRAUD_THRESHOLD_MEDIUM = 0.40  # Amber — requires review
# Below FRAUD_THRESHOLD_MEDIUM → Green — likely legitimate

MODEL_VERSION = "1.0.0"
TEST_SIZE = 0.20
RANDOM_STATE = 42
N_CV_FOLDS = 5
SMOTE_SAMPLES = None  # None = auto-balance to 50/50
BACKGROUND_SAMPLE_SIZE = 500   # samples for SHAP background
TEST_SHAP_SAMPLE_SIZE = 500    # samples for global SHAP plots
DEMO_SAMPLES_PER_CLASS = 5
