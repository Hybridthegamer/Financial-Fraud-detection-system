# AFDS-xAI — AI-Driven Financial Fraud Detection System with Explainability

> Final-year research project (BSc Computer Science / Information Technology)  
> *Design and Implementation of an AI-Driven Financial Fraud Detection System with Explainability (xAI)*

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Requirements](#system-requirements)
3. [Tool Stack & Rationale](#tool-stack--rationale)
4. [Architecture](#architecture)
5. [Setup & Installation](#setup--installation)
6. [Dataset](#dataset)
7. [Training the Model](#training-the-model)
8. [Running the Application](#running-the-application)
9. [Directory Structure](#directory-structure)
10. [System Features](#system-features)
11. [Evaluation Metrics](#evaluation-metrics)

---

## Project Overview

AFDS-xAI is a machine-learning-powered fraud detection prototype that combines **state-of-the-art detection accuracy** with **built-in explainability**. The system addresses three critical gaps identified in the research:

| Gap | Solution |
|-----|----------|
| Rule-based systems cannot detect novel fraud | XGBoost trained on historical transaction patterns |
| Black-box ML models violate EU GDPR Article 22 & CBN audit requirements | SHAP (TreeSHAP) per-prediction feature attribution |
| No integrated prototype exists for fraud analysts | Streamlit dashboard with real-time prediction + explanation |

The primary use case is **credit card transaction fraud** using the Kaggle Credit Card Fraud Detection benchmark dataset (284,807 transactions, 0.172% fraud rate).

---

## System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.10 | 3.11+ |
| RAM | 4 GB | 8 GB |
| Disk (code + models) | 500 MB | 1 GB |
| Disk (dataset) | 150 MB | 150 MB |
| OS | Windows 10 / macOS 12 / Ubuntu 20.04 | Ubuntu 22.04 / macOS 14 |
| CPU | 2 cores | 4+ cores (training is CPU-bound) |
| GPU | Not required | Not required |

> **Training time** is approximately 3–8 minutes on a 4-core machine at the configured hyperparameters (500 estimators, SMOTE resampling).

---

## Tool Stack & Rationale

### XGBoost (Extreme Gradient Boosting)

**Chosen because:** XGBoost is the de facto standard for fraud detection on structured/tabular data. Chen & Guestrin (2016) demonstrated that its second-order gradient statistics, L1/L2 regularisation, and parallelised tree construction consistently outperform alternative algorithms on benchmark fraud datasets. It won the majority of Kaggle fraud detection competitions and achieves state-of-the-art AUC-ROC on the Credit Card Fraud dataset in peer-reviewed literature (Sharma & Ghosh, 2020; Ali et al., 2022). Its native `scale_pos_weight` parameter also provides a built-in mechanism for handling extreme class imbalance.

**Alternatives considered:** Random Forest (lower AUC-ROC on tabular data), LightGBM (comparable performance but XGBoost has wider industry adoption for regulated contexts), Deep Learning (superior on sequential data but requires larger datasets and is harder to explain).

### SHAP — SHapley Additive exPlanations (TreeSHAP)

**Chosen because:** SHAP is the only post-hoc explanation method that satisfies all four Shapley axioms (efficiency, symmetry, dummy, additivity) simultaneously, making it the theoretically principled and uniquely consistent attribution framework. Lundberg et al. (2018) introduced TreeSHAP, which computes exact Shapley values for tree-based models in polynomial time rather than exponential time, making it practical for real-time fraud detection. Ali et al. (2022) confirmed SHAP outperforms LIME, Integrated Gradients, and counterfactual explanations on fidelity and stability across three benchmark fraud datasets. SHAP is directly required for compliance with the EU GDPR "right to explanation" (Article 22) and CBN electronic fraud risk management circulars.

**Alternatives considered:** LIME (unstable — produces different explanations for similar instances; locally linear assumption breaks down in high-curvature XGBoost decision surfaces), Integrated Gradients (designed for neural networks, not trees).

### imbalanced-learn / SMOTE

**Chosen because:** The Kaggle dataset has a 0.172% fraud rate — a 578:1 class imbalance ratio. SMOTE (Chawla et al., 2002) is the most validated oversampling technique in fraud detection literature. Critically, SMOTE is applied **only to the training set** (following Dal Pozzolo et al., 2015) to prevent optimistic bias in test-set evaluation metrics. This is the standard approach in recent high-performing fraud detection systems.

### Streamlit

**Chosen because:** Streamlit enables rapid construction of production-quality interactive data science dashboards in pure Python, with no frontend engineering overhead. This directly aligns with the research objective of an *operational prototype* accessible to fraud analysts without requiring specialised web development skills. The three-tier architecture (Data → Processing → Presentation) maps cleanly to Streamlit's component model.

### SQLite

**Chosen because:** The research prototype is a single-user system that does not require concurrent write access from multiple clients. SQLite is zero-configuration, serverless, and produces a single portable `.db` file suitable for regulatory audit archival. The schema supports the complete audit trail required by CBN cybersecurity guidelines: every prediction, its SHAP values, the analyst identity, and a UTC timestamp are persisted and queryable.

**Alternatives considered:** PostgreSQL / MySQL (operational overhead unjustified for prototype; scale-out would require a move to a client-server RDBMS in production).

### scikit-learn

**Chosen because:** Provides the `StandardScaler` (feature normalisation), `StratifiedKFold` (5-fold cross-validation), `train_test_split` (stratified 80/20 split), and all standard evaluation metrics (AUC-ROC, F1, precision, recall) used in the evaluation chapter. It is the universal Python ML utility library with no meaningful alternatives in the Python ecosystem.

### Python 3.10+

**Chosen because:** Required by XGBoost 2.x, SHAP 0.44+, and Streamlit 1.28+. Python is the de facto language for ML research and is the primary language of the entire scientific computing ecosystem used (NumPy, Pandas, Matplotlib).

---

## Architecture

AFDS-xAI implements the **three-tier layered architecture** specified in Chapter 3:

```
┌─────────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                           │
│   Streamlit Web Dashboard (app.py)                              │
│   • Login/auth   • Transaction form   • SHAP waterfall plots   │
│   • Audit log    • Model performance  • Risk colour indicator   │
└────────────────────────────┬────────────────────────────────────┘
                             │  Python function calls
┌────────────────────────────▼────────────────────────────────────┐
│                     PROCESSING LAYER                            │
│  TransactionProcessor    FraudDetectionModel    SHAPExplainer   │
│  (StandardScaler)        (XGBoost 500 trees)   (TreeSHAP)      │
│  src/preprocessing.py    src/model.py           src/explainer.py│
└────────────────────────────┬────────────────────────────────────┘
                             │  pickle / SQLite
┌────────────────────────────▼────────────────────────────────────┐
│                       DATA LAYER                                │
│  creditcard.csv   │  fraud_detection.db   │  models/*.pkl       │
│  (training data)  │  (audit log/SQLite)   │  (artefacts)        │
└─────────────────────────────────────────────────────────────────┘
```

**Processing pipeline** (per transaction):

```
Raw Input → StandardScaler → XGBoost.predict_proba() → Fraud Probability
                                    ↓
                            TreeSHAP.shap_values() → SHAP Waterfall Plot
                                    ↓
                            SQLite AuditLog (transaction + prediction + SHAP)
```

---

## Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/Hybridthegamer/Financial-Fraud-detection-system.git
cd Financial-Fraud-detection-system
```

### 2. Create and activate a virtual environment

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note for Windows users:** If XGBoost installation fails, try `pip install xgboost --no-cache-dir`.

---

## Dataset

AFDS-xAI uses the **Kaggle Credit Card Fraud Detection dataset** (Pozzolo et al., 2018).

| Property | Value |
|----------|-------|
| Source | [Kaggle — mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| Transactions | 284,807 |
| Fraud cases | 492 (0.172%) |
| Features | Time, Amount, V1–V28 (PCA-anonymised behavioural features), Class |
| File size | ~144 MB |

### Download

```bash
# Option A: Kaggle CLI (requires Kaggle account + API key)
pip install kaggle
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/ --unzip

# Option B: Manual download
# 1. Visit https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# 2. Download creditcard.csv
# 3. Place it at:  data/creditcard.csv
```

The dataset file is excluded from version control (`.gitignore`) because it exceeds GitHub's 100 MB file limit.

---

## Training the Model

With the dataset in place, run the full CRISP-DM training pipeline:

```bash
python scripts/train_model.py
```

This script executes the following steps:

| Step | Description | Output |
|------|-------------|--------|
| Data split | Stratified 80/20 train/test | — |
| Preprocessing | `StandardScaler` on Amount & Time | `models/preprocessor.pkl` |
| SMOTE | Synthetic minority oversampling (training set only) | — |
| Cross-validation | 5-fold stratified CV, AUC-ROC scoring | Logged to console |
| XGBoost training | 500 estimators, max_depth=6, lr=0.05 | `models/xgboost_model.pkl` |
| SHAP explainer | TreeSHAP with 500-sample background | `models/shap_explainer.pkl` |
| Evaluation plots | ROC curve, confusion matrix | `models/roc_curve.png`, `models/confusion_matrix.png` |
| SHAP plots | Beeswarm + bar chart (global importance) | `models/shap_summary.png`, `models/shap_bar.png` |
| Metrics | AUC-ROC, F1, precision, recall | `models/eval_results.json` |
| Demo samples | 5 fraud + 5 legitimate for UI demo | `models/sample_transactions.pkl` |

Expected output (approximate — varies by hardware):

```
AUC-ROC : 0.9792
F1-Score: 0.8731
Recall  : 0.8776
```

---

## Running the Application

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**.

**Demo login credentials:**

| Field | Value |
|-------|-------|
| Email | `demo@fraudsystem.com` |
| Password | `demo123` |

---

## Directory Structure

```
Financial-Fraud-detection-system/
├── app.py                          # Streamlit web application (Presentation Layer)
├── requirements.txt                # Python package dependencies
├── .gitignore
├── README.md
│
├── src/                            # Processing Layer — core business logic
│   ├── __init__.py
│   ├── config.py                   # All constants, paths, and hyperparameters
│   ├── preprocessing.py            # TransactionProcessor (StandardScaler wrapper)
│   ├── model.py                    # FraudDetectionModel (XGBoost wrapper)
│   ├── explainer.py                # SHAPExplainer (TreeSHAP wrapper)
│   └── database.py                 # DatabaseManager (SQLite audit logger)
│
├── scripts/
│   └── train_model.py              # End-to-end CRISP-DM training pipeline
│
├── data/                           # Data Layer (dataset — not committed)
│   └── creditcard.csv              # ← place Kaggle dataset here
│
└── models/                         # Trained artefacts (generated — not committed)
    ├── xgboost_model.pkl
    ├── preprocessor.pkl
    ├── shap_explainer.pkl
    ├── eval_results.json
    ├── sample_transactions.pkl
    ├── roc_curve.png
    ├── confusion_matrix.png
    ├── shap_summary.png
    └── shap_bar.png
```

---

## System Features

### Transaction Analysis
- Input 30 transaction features (Time, Amount, V1–V28) via structured form
- Load pre-saved demo transactions (legitimate / fraudulent) with one click
- Instant fraud probability score (0.0 – 1.0)
- Colour-coded risk indicator:
  - 🔴 **HIGH** (≥ 0.70) — Likely Fraudulent
  - 🟡 **MEDIUM** (0.40 – 0.69) — Requires Analyst Review
  - 🟢 **LOW** (< 0.40) — Likely Legitimate
- SHAP waterfall plot explaining which features drove the prediction
- Feature contribution table sorted by |SHAP value|

### Model Performance Dashboard
- AUC-ROC, Average Precision, F1, Precision, Recall metrics
- 5-fold cross-validation score with standard deviation
- Confusion matrix visualisation
- ROC curve plot
- Global SHAP beeswarm and bar charts

### Audit Log
- Full tamper-evident record of every analysis (transaction features, probability, classification, analyst, timestamp)
- Per-transaction SHAP drill-down by Transaction ID
- Supports CBN electronic fraud risk management audit requirements

### Account Management (Admin only)
- Register new analyst accounts
- Role-based access (analyst / admin)

---

## Evaluation Metrics

The system is evaluated on the standard fraud detection metrics recommended in the literature (Bhattacharyya et al., 2011; Dal Pozzolo et al., 2015):

| Metric | Rationale |
|--------|-----------|
| **AUC-ROC** | Threshold-independent discrimination measure; primary metric for imbalanced datasets |
| **Average Precision (AP)** | Area under the Precision-Recall curve; more informative than AUC-ROC under extreme imbalance |
| **F1-Score** | Harmonic mean of precision and recall; balances detection rate vs. false alarms |
| **Precision** | Fraction of flagged transactions that are genuinely fraudulent (false alarm rate) |
| **Recall (Sensitivity)** | Fraction of actual fraud cases caught (miss rate); critical in financial fraud context |

> **Note on accuracy:** Overall accuracy is intentionally omitted as an evaluation metric. With 0.17% fraud rate, a classifier predicting "legitimate" for every transaction achieves 99.83% accuracy while detecting zero fraud — the *accuracy paradox* (Dal Pozzolo et al., 2015).

---

## References

- Ali, A., et al. (2022). *Comparative evaluation of xAI methods for fraud detection.* JIFS, 43(3).
- Chawla, N. V., et al. (2002). *SMOTE.* JAIR, 16, 321–357.
- Chen, T., & Guestrin, C. (2016). *XGBoost.* KDD 2016.
- Dal Pozzolo, A., et al. (2015). *Calibrating probability with undersampling.* IEEE SSCI.
- Lundberg, S. M., & Lee, S. I. (2017). *A unified approach to interpreting model predictions.* NeurIPS 30.
- Lundberg, S. M., et al. (2018). *Consistent individualized feature attribution for tree ensembles.* arXiv:1802.03888.
- Sharma, A., & Ghosh, A. (2020). *Explainable AI for credit card fraud detection.* IJACSA, 11(7).