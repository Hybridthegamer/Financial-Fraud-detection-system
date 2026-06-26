"""
AFDS-xAI — AI-Driven Financial Fraud Detection System with Explainability
==========================================================================
Streamlit web application (Presentation Layer).

Pages:
  🏠 Home               — Dashboard with summary statistics
  🔍 Analyse Transaction — Input form → XGBoost prediction + SHAP waterfall
  📊 Model Performance   — AUC-ROC, confusion matrix, global SHAP plots
  📋 Audit Log           — Regulatorily-compliant prediction history
  👤 Account             — Register new analyst accounts (admin only)
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

# ── Page configuration (must be first Streamlit call) ────────────────────────
st.set_page_config(
    page_title="AFDS-xAI | Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Local imports ─────────────────────────────────────────────────────────────
from src.config import (
    DB_PATH,
    EVAL_RESULTS_PATH,
    EXPLAINER_ARTIFACT,
    FEATURES,
    FRAUD_THRESHOLD_HIGH,
    FRAUD_THRESHOLD_MEDIUM,
    MODEL_ARTIFACT,
    MODEL_VERSION,
    MODELS_DIR,
    PREPROCESSOR_ARTIFACT,
    SAMPLE_TRANSACTIONS_PATH,
)
from src.database import DatabaseManager
from src.explainer import SHAPExplainer
from src.model import FraudDetectionModel
from src.preprocessing import TransactionProcessor

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        .risk-high   { color: #c0392b; font-weight: bold; font-size: 1.3rem; }
        .risk-medium { color: #e67e22; font-weight: bold; font-size: 1.3rem; }
        .risk-low    { color: #27ae60; font-weight: bold; font-size: 1.3rem; }
        .metric-card { background: #f8f9fa; border-radius: 8px; padding: 1rem; }
        div[data-testid="stMetric"] { background: #f8f9fa; border-radius: 8px; padding: 0.6rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Artifact loading (cached across re-runs) ──────────────────────────────────

@st.cache_resource(show_spinner="Loading model artefacts…")
def load_ml_artifacts():
    model       = FraudDetectionModel.load(MODEL_ARTIFACT)
    preprocessor = TransactionProcessor.load(PREPROCESSOR_ARTIFACT)
    explainer   = SHAPExplainer.load(EXPLAINER_ARTIFACT)
    return model, preprocessor, explainer


@st.cache_data(show_spinner=False)
def load_eval_results():
    if EVAL_RESULTS_PATH.exists():
        return json.loads(EVAL_RESULTS_PATH.read_text())
    return None


@st.cache_data(show_spinner=False)
def load_sample_transactions():
    if SAMPLE_TRANSACTIONS_PATH.exists():
        with open(SAMPLE_TRANSACTIONS_PATH, "rb") as fh:
            return pickle.load(fh)
    return None


def artifacts_ready() -> bool:
    return all(p.exists() for p in [MODEL_ARTIFACT, PREPROCESSOR_ARTIFACT, EXPLAINER_ARTIFACT])


# ── Risk helpers ──────────────────────────────────────────────────────────────

def risk_css_class(prob: float) -> str:
    if prob >= FRAUD_THRESHOLD_HIGH:
        return "risk-high"
    if prob >= FRAUD_THRESHOLD_MEDIUM:
        return "risk-medium"
    return "risk-low"


def risk_label(prob: float) -> str:
    if prob >= FRAUD_THRESHOLD_HIGH:
        return "🔴 HIGH RISK — Likely Fraudulent"
    if prob >= FRAUD_THRESHOLD_MEDIUM:
        return "🟡 MEDIUM RISK — Requires Review"
    return "🟢 LOW RISK — Likely Legitimate"


# ── Page renderers ────────────────────────────────────────────────────────────

def page_login(db: DatabaseManager) -> None:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("## 🛡️ AFDS-xAI")
        st.markdown("**AI-Driven Financial Fraud Detection System with Explainability**")
        st.divider()
        with st.form("login_form"):
            email    = st.text_input("Email", placeholder="analyst@bank.com")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", use_container_width=True, type="primary")

        if submitted:
            analyst = db.authenticate(email, password)
            if analyst:
                st.session_state.authenticated = True
                st.session_state.analyst = analyst
                st.rerun()
            else:
                st.error("Invalid credentials.")

        st.caption("Demo account: **demo@fraudsystem.com** / **demo123**")


def page_home(db: DatabaseManager) -> None:
    st.header("🏠 Dashboard")
    st.write(f"Welcome back, **{st.session_state.analyst['name']}**")

    stats = db.get_summary_stats()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Analyses", f"{stats['total']:,}")
    col2.metric("Fraud Detected", f"{stats['fraud']:,}")
    col3.metric("Legitimate", f"{stats['legitimate']:,}")
    rate = stats["fraud"] / max(stats["total"], 1) * 100
    col4.metric("Detection Rate", f"{rate:.1f}%")

    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("System Status")
        if artifacts_ready():
            st.success("✅ Model artefacts loaded and ready")
        else:
            st.error("❌ Model not trained — run `python scripts/train_model.py`")

        results = load_eval_results()
        if results:
            st.info(
                f"**Model Performance (test set)**  \n"
                f"AUC-ROC: `{results['auc_roc']:.4f}` &nbsp;|&nbsp; "
                f"F1: `{results['f1']:.4f}` &nbsp;|&nbsp; "
                f"Recall: `{results['recall']:.4f}`"
            )

    with col_b:
        st.subheader("Quick Guide")
        st.markdown(
            """
            1. **Analyse Transaction** — submit transaction features to get an instant fraud
               probability score with a full SHAP explanation.
            2. **Model Performance** — inspect ROC curves, confusion matrices, and global
               feature importance charts.
            3. **Audit Log** — review all historical predictions with complete SHAP drill-down
               for regulatory compliance.
            """
        )

    st.divider()
    st.subheader("Recent Activity")
    recent = db.get_recent_predictions(limit=10)
    if recent:
        df = pd.DataFrame(recent)
        df["is_fraud"] = df["is_fraud"].map({1: "🚨 Fraud", 0: "✅ Legitimate"})
        df["fraud_probability"] = df["fraud_probability"].round(4)
        df = df.rename(columns={
            "transaction_id": "Transaction ID",
            "analyst_name": "Analyst",
            "fraud_probability": "Fraud Prob.",
            "is_fraud": "Classification",
            "predicted_at": "Timestamp",
        })
        st.dataframe(df[["Transaction ID", "Analyst", "Fraud Prob.", "Classification", "Timestamp"]],
                     use_container_width=True, hide_index=True)
    else:
        st.info("No analyses logged yet.")


def page_analyse(
    model: FraudDetectionModel,
    preprocessor: TransactionProcessor,
    explainer: SHAPExplainer,
    db: DatabaseManager,
) -> None:
    st.header("🔍 Analyse Transaction")

    # ── Sample loader ─────────────────────────────────────────────────────────
    samples = load_sample_transactions()
    sample_options = ["Manual Entry"]
    if samples:
        sample_options += [
            f"Demo — Legitimate transaction #{i+1}"
            for i in range(len(samples["legitimate"]))
        ]
        sample_options += [
            f"Demo — Fraudulent transaction #{i+1}"
            for i in range(len(samples["fraud"]))
        ]

    choice = st.selectbox("Load a demo transaction or enter values manually:", sample_options)

    # Pre-fill feature values from sample if chosen
    init_vals: dict[str, float] = {f: 0.0 for f in FEATURES}
    if choice != "Manual Entry" and samples:
        if "Legitimate" in choice:
            idx = int(choice.split("#")[1]) - 1
            row = samples["legitimate"].iloc[idx]
        else:
            idx = int(choice.split("#")[1]) - 1
            row = samples["fraud"].iloc[idx]
        init_vals = {f: float(row[f]) for f in FEATURES}

    # ── Input form ───────────────────────────────────────────────────────────
    with st.form("analysis_form"):
        st.subheader("Transaction Details")
        c1, c2 = st.columns(2)
        with c1:
            init_vals["Time"] = st.number_input(
                "Time (seconds elapsed from first transaction in dataset)",
                value=init_vals["Time"], format="%.2f", min_value=0.0,
            )
        with c2:
            init_vals["Amount"] = st.number_input(
                "Transaction Amount (€)",
                value=init_vals["Amount"], format="%.2f", min_value=0.0,
            )

        st.subheader("Anonymised PCA Behavioural Features (V1–V28)")
        st.caption(
            "These features are PCA-transformed by the issuing bank to protect "
            "cardholder privacy. Provide the values supplied by your data pipeline."
        )
        cols = st.columns(4)
        for i, feat in enumerate([f"V{j}" for j in range(1, 29)]):
            with cols[i % 4]:
                init_vals[feat] = st.number_input(
                    feat, value=init_vals[feat], format="%.6f", key=f"inp_{feat}"
                )

        submitted = st.form_submit_button(
            "🔍  Analyse Transaction", type="primary", use_container_width=True
        )

    if not submitted:
        return

    # ── Prediction ───────────────────────────────────────────────────────────
    input_df     = pd.DataFrame([init_vals])[FEATURES]
    input_scaled = preprocessor.transform(input_df)
    fraud_prob   = float(model.predict_proba(input_scaled)[0])
    is_fraud     = fraud_prob >= model.threshold

    st.divider()
    st.subheader("Prediction Result")

    m1, m2, m3 = st.columns(3)
    m1.metric("Fraud Probability", f"{fraud_prob:.4f}")
    m2.metric("Binary Classification", "FRAUD" if is_fraud else "LEGITIMATE")
    m3.metric(
        "Risk Level",
        "HIGH"   if fraud_prob >= FRAUD_THRESHOLD_HIGH
        else "MEDIUM" if fraud_prob >= FRAUD_THRESHOLD_MEDIUM
        else "LOW",
    )

    css = risk_css_class(fraud_prob)
    st.markdown(
        f"<p class='{css}'>{risk_label(fraud_prob)}</p>", unsafe_allow_html=True
    )
    st.progress(float(fraud_prob), text=f"Fraud probability: {fraud_prob:.2%}")

    # ── SHAP Explanation ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("🧠 SHAP Explanation")
    st.write(
        "Each bar shows a feature's contribution to the fraud probability relative "
        "to the model baseline. **Red → increases fraud probability**, "
        "**Blue → decreases it**."
    )

    shap_df = explainer.get_waterfall_data(input_scaled)

    # Waterfall plot
    try:
        explanation = explainer.explain_instance(input_scaled)
        fig = plt.figure(figsize=(10, 7))
        shap.plots.waterfall(explanation, max_display=15, show=False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    except Exception as exc:
        st.warning(f"Waterfall plot could not be rendered: {exc}")

    # Feature contribution table
    st.write("**Top Feature Contributions (sorted by |SHAP value|):**")
    display = shap_df.head(15).copy()
    display["shap_value"] = display["shap_value"].round(6)
    display["value"]      = display["value"].round(4)
    display["direction"]  = display["shap_value"].apply(
        lambda x: "↑ Increases fraud prob." if x > 0 else "↓ Decreases fraud prob."
    )
    st.dataframe(display, use_container_width=True, hide_index=True)

    # ── Audit log ────────────────────────────────────────────────────────────
    tx_id = db.log_analysis(
        analyst_id=st.session_state.analyst["analyst_id"],
        feature_vector=init_vals,
        fraud_probability=fraud_prob,
        is_fraud=is_fraud,
        shap_df=shap_df,
        model_version=MODEL_VERSION,
    )
    st.success(f"✅ Analysis logged to audit database.  Transaction ID: `{tx_id}`")


def page_performance() -> None:
    st.header("📊 Model Performance")
    results = load_eval_results()

    if not results:
        st.warning("No evaluation results found. Run `python scripts/train_model.py` first.")
        return

    # ── Key metrics ──────────────────────────────────────────────────────────
    st.subheader("Performance Metrics (Held-out Test Set)")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("AUC-ROC",        f"{results['auc_roc']:.4f}")
    m2.metric("Avg. Precision", f"{results['average_precision']:.4f}")
    m3.metric("F1-Score",       f"{results['f1']:.4f}")
    m4.metric("Precision",      f"{results['precision']:.4f}")
    m5.metric("Recall",         f"{results['recall']:.4f}")

    if "cv_auc_mean" in results:
        st.info(
            f"**{5}-Fold Cross-Validation AUC-ROC:** "
            f"`{results['cv_auc_mean']:.4f}` ± `{results['cv_auc_std']:.4f}`"
        )

    # ── Confusion matrix + ROC ───────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Confusion Matrix")
        cm_path = MODELS_DIR / "confusion_matrix.png"
        if cm_path.exists():
            st.image(str(cm_path), use_container_width=True)
        else:
            tn, fp, fn, tp = results["tn"], results["fp"], results["fn"], results["tp"]
            st.dataframe(
                pd.DataFrame(
                    [[tn, fp], [fn, tp]],
                    index=["Actual Legitimate", "Actual Fraud"],
                    columns=["Predicted Legitimate", "Predicted Fraud"],
                ),
                use_container_width=True,
            )

    with col2:
        st.subheader("ROC Curve")
        roc_path = MODELS_DIR / "roc_curve.png"
        if roc_path.exists():
            st.image(str(roc_path), use_container_width=True)

    # ── SHAP global importance ───────────────────────────────────────────────
    st.subheader("Global SHAP Feature Importance")
    st.write(
        "Beeswarm plot shows the distribution of SHAP values for each feature across "
        "the test sample. Features are ordered by mean absolute SHAP value (highest impact at top)."
    )
    for fname, label in [
        ("shap_summary.png", "Beeswarm Plot"),
        ("shap_bar.png",     "Mean |SHAP| Bar Chart"),
    ]:
        path = MODELS_DIR / fname
        if path.exists():
            st.caption(label)
            st.image(str(path), use_container_width=True)

    # ── Classification report ────────────────────────────────────────────────
    if "classification_report" in results:
        st.subheader("Full Classification Report")
        cr = results["classification_report"]
        cr_df = pd.DataFrame(cr).transpose().round(4)
        st.dataframe(cr_df, use_container_width=True)


def page_audit_log(db: DatabaseManager) -> None:
    st.header("📋 Audit Log")
    st.write(
        "Complete, tamper-evident record of all fraud analyses — satisfies "
        "CBN electronic fraud risk management audit requirements."
    )

    preds = db.get_recent_predictions(limit=100)
    if not preds:
        st.info("No analyses logged yet. Use **Analyse Transaction** to run your first prediction.")
        return

    df = pd.DataFrame(preds)
    df["is_fraud"] = df["is_fraud"].map({1: "🚨 Fraud", 0: "✅ Legitimate"})
    df["fraud_probability"] = df["fraud_probability"].round(4)
    df = df.rename(columns={
        "transaction_id":   "Transaction ID",
        "analyst_name":     "Analyst",
        "fraud_probability":"Fraud Prob.",
        "is_fraud":         "Classification",
        "model_version":    "Model",
        "predicted_at":     "Timestamp (UTC)",
    })
    st.dataframe(df, use_container_width=True, hide_index=True)

    # ── SHAP drill-down ──────────────────────────────────────────────────────
    st.divider()
    st.subheader("SHAP Explanation Drill-Down")
    st.write("Paste a Transaction ID from the table above to retrieve its stored SHAP explanation.")
    tx_id = st.text_input("Transaction ID", placeholder="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx")
    if tx_id.strip():
        shap_data = db.get_prediction_shap(tx_id.strip())
        if shap_data:
            shap_df = pd.DataFrame(shap_data)
            shap_df["shap_value"] = shap_df["shap_value"].round(6)
            shap_df["feature_value"] = shap_df["feature_value"].round(4)
            st.dataframe(shap_df, use_container_width=True, hide_index=True)
        else:
            st.warning("No SHAP explanation found for this Transaction ID.")


def page_account(db: DatabaseManager) -> None:
    st.header("👤 Account Management")
    analyst = st.session_state.analyst

    st.subheader("Your Profile")
    st.write(f"**Name:** {analyst['name']}")
    st.write(f"**Role:** {analyst['role'].title()}")

    if analyst["role"] != "admin":
        st.info("Only administrators can register new analyst accounts.")
        return

    st.divider()
    st.subheader("Register New Analyst")
    with st.form("register_form"):
        new_name  = st.text_input("Full Name")
        new_email = st.text_input("Email")
        new_pwd   = st.text_input("Password", type="password")
        new_role  = st.selectbox("Role", ["analyst", "admin"])
        if st.form_submit_button("Create Account", type="primary"):
            if not (new_name and new_email and new_pwd):
                st.error("All fields are required.")
            elif db.email_exists(new_email):
                st.error("An account with that email already exists.")
            else:
                db.create_analyst(new_name, new_email, new_pwd, new_role)
                st.success(f"Account created for **{new_name}** ({new_email}).")


# ── Application entry point ───────────────────────────────────────────────────

def main() -> None:
    db = DatabaseManager(DB_PATH)

    # Initialise session state
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.analyst = None

    # Unauthenticated → login wall
    if not st.session_state.authenticated:
        page_login(db)
        return

    # ── Sidebar navigation ────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🛡️ AFDS-xAI")
        st.caption("AI Fraud Detection System")
        st.divider()
        st.write(f"**{st.session_state.analyst['name']}**")
        st.caption(st.session_state.analyst["role"].title())
        st.divider()

        page = st.radio(
            "Navigation",
            ["🏠 Home", "🔍 Analyse Transaction", "📊 Model Performance",
             "📋 Audit Log", "👤 Account"],
            label_visibility="collapsed",
        )

        st.divider()
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.analyst = None
            st.rerun()

    # ── Load model (only when needed) ────────────────────────────────────────
    if page in ("🔍 Analyse Transaction",) and not artifacts_ready():
        st.error("Model artefacts not found. Run `python scripts/train_model.py` first.")
        return

    if page == "🏠 Home":
        page_home(db)

    elif page == "🔍 Analyse Transaction":
        model, preprocessor, explainer = load_ml_artifacts()
        page_analyse(model, preprocessor, explainer, db)

    elif page == "📊 Model Performance":
        page_performance()

    elif page == "📋 Audit Log":
        page_audit_log(db)

    elif page == "👤 Account":
        page_account(db)


if __name__ == "__main__":
    main()
