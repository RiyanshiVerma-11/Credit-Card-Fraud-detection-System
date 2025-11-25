# app.py - Credit Card Fraud Detection Dashboard (Cleaned Version - Only Logistic Regression & Random Forest)
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             confusion_matrix, roc_curve, classification_report,
                             precision_score, recall_score)
from sklearn.utils import resample

import warnings
warnings.filterwarnings("ignore")

# --- Page config ---
st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background: #f7f9fb; }
    .big-title { font-size:28px; color:#0b6e99; font-weight:600; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="big-title">Credit Card Fraud Detection Dashboard</div>', unsafe_allow_html=True)
st.write("Load `creditcard.csv` (Kaggle) into the app folder. The app auto-detects it and provides EDA, training, and prediction tools.")

# --- Sidebar controls ---
st.sidebar.header("Dataset & Model Controls")
data_path = "creditcard.csv"
data_exists = os.path.exists(data_path)

if data_exists:
    if st.sidebar.button("📥 Load dataset (auto)"):
        try:
            df = pd.read_csv(data_path)
            # normalize expected target name
            if "Class" in df.columns and "is_fraud" not in df.columns:
                df = df.rename(columns={"Class": "is_fraud"})
            st.session_state["df"] = df
            st.session_state["data_loaded"] = True
            st.sidebar.success("Dataset loaded successfully")
        except Exception as e:
            st.sidebar.error(f"Load failed: {e}")
else:
    st.sidebar.info("Place creditcard.csv in app folder to load automatically.")

st.sidebar.markdown("---")
test_size_pct = st.sidebar.slider("Test size (%)", 10, 40, 30, 5)
oversample = st.sidebar.checkbox("Enable oversampling (minority)", value=True)
random_state = int(st.sidebar.number_input("Random seed", value=42, min_value=0, step=1))

st.sidebar.markdown("---")
st.sidebar.subheader("Models to include")
use_logreg = st.sidebar.checkbox("Logistic Regression", value=True)
use_rf = st.sidebar.checkbox("Random Forest", value=True)

if st.sidebar.button("🔁 Reset session state"):
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.experimental_rerun()

# --- Top metrics ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    if "data_loaded" in st.session_state:
        st.success("Dataset Loaded")
    else:
        st.warning("Dataset not loaded")
with col2:
    st.metric("Transactions", f"{len(st.session_state['df']):,}" if "df" in st.session_state else "—")
with col3:
    st.metric("Fraud cases", f"{int(st.session_state['df']['is_fraud'].sum()):,}" if "df" in st.session_state else "—")
with col4:
    if "df" in st.session_state:
        rate = st.session_state['df']['is_fraud'].mean() * 100
        st.metric("Fraud rate", f"{rate:.4f}%")
    else:
        st.metric("Fraud rate", "—")

st.markdown("---")

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 EDA", "🤖 Train", "📈 Results", "🔍 Predict"])

# --- EDA Tab ---
with tab1:
    st.header("Exploratory Data Analysis")
    if "df" not in st.session_state:
        st.info("Load dataset first")
    else:
        df = st.session_state["df"].copy()

        st.subheader("Preview")
        st.dataframe(df.head(10))

        st.subheader("Summary")
        st.write(df.describe().T)

        st.subheader("Class distribution")
        if "is_fraud" in df.columns:
            dist = df["is_fraud"].value_counts().rename(index={0: "Normal", 1: "Fraud"})
            fig = px.pie(values=dist.values, names=dist.index, hole=0.45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No 'is_fraud' column found in dataset preview.")

        amt_col = "Amount"
        if amt_col in df.columns:
            fig2 = px.box(df, x="is_fraud", y=amt_col)
            st.subheader("Amount distribution")
            st.plotly_chart(fig2, use_container_width=True)

# --- Train Tab ---
with tab2:
    st.header("Model Training")
    if "df" not in st.session_state:
        st.info("Load dataset first")
    else:
        # Work with numeric columns only to avoid errors
        df = st.session_state["df"].select_dtypes(include=[np.number]).copy()

        if "is_fraud" not in df.columns:
            st.error("Target column 'is_fraud' missing from numeric dataset. Ensure your CSV has 'Class' or 'is_fraud'.")
        else:
            st.write("Preparing dataset for training. Shape:", df.shape)
            st.dataframe(df.head())

            # Balance handling
            if oversample:
                st.info("Applying oversampling to minority class (fraud).")
                df_major = df[df.is_fraud == 0]
                df_minor = df[df.is_fraud == 1]
                if len(df_minor) < 2:
                    st.error("Too few fraud samples to oversample. Disable oversample or add more data.")
                else:
                    target_n = min(len(df_major), max(len(df_minor) * 50, 3000))
                    df_minor_up = resample(df_minor, replace=True, n_samples=int(target_n), random_state=random_state)
                    df_bal = pd.concat([df_major, df_minor_up]).sample(frac=1, random_state=random_state).reset_index(drop=True)
            else:
                df_bal = df.copy()

            st.write("Training shape after balancing:", df_bal.shape)

            # Features and target
            X = df_bal.drop("is_fraud", axis=1)
            y = df_bal["is_fraud"]

            # train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size_pct / 100.0, stratify=y, random_state=random_state)
            st.write("Train:", X_train.shape, "Test:", X_test.shape)

            # Save feature names to session so prediction uses same columns & order
            st.session_state["feature_names"] = X_train.columns.tolist()

            # scaling
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            # build model dict from sidebar choices
            models = {}
            if use_logreg:
                models["Logistic Regression"] = LogisticRegression(max_iter=1000, random_state=random_state)
            if use_rf:
                models["Random Forest"] = RandomForestClassifier(
                    n_estimators=60, max_depth=10, n_jobs=-1, random_state=random_state)

            if len(models) == 0:
                st.warning("Enable at least one model")
            else:
                if st.button("🚀 Train Models"):
                    progress = st.progress(0)
                    results = {}
                    total = len(models)
                    for i, (name, model) in enumerate(models.items()):
                        model.fit(X_train_s, y_train)
                        y_pred = model.predict(X_test_s)

                        # Probability / decision scores for AUC
                        try:
                            y_proba = model.predict_proba(X_test_s)[:, 1]
                        except Exception:
                            try:
                                dec = model.decision_function(X_test_s)
                                # scale to [0,1]
                                y_proba = (dec - dec.min()) / (dec.max() - dec.min() + 1e-9)
                            except Exception:
                                y_proba = np.zeros_like(y_pred, dtype=float)

                        # Compute metrics and convert to percentages (two decimals)
                        acc = accuracy_score(y_test, y_pred) * 100
                        prec = precision_score(y_test, y_pred, zero_division=0) * 100
                        rec = recall_score(y_test, y_pred, zero_division=0) * 100
                        f1 = f1_score(y_test, y_pred, zero_division=0) * 100
                        roc_auc = roc_auc_score(y_test, y_proba) * 100 if len(np.unique(y_test)) > 1 else 0.0

                        results[name] = {
                            "model": model,
                            "y_pred": y_pred,
                            "y_proba": y_proba,
                            "accuracy (%)": round(acc, 2),
                            "precision (%)": round(prec, 2),
                            "recall (%)": round(rec, 2),
                            "f1-score (%)": round(f1, 2),
                            "roc_auc (%)": round(roc_auc, 2),
                            "confusion_matrix": confusion_matrix(y_test, y_pred)
                        }

                        progress.progress((i + 1) / total)

                    # persist to session
                    st.session_state["results"] = results
                    st.session_state["scaler"] = scaler
                    st.session_state["X_test"] = X_test_s
                    st.session_state["y_test"] = y_test
                    st.success("Training complete and results saved in session.")

# --- Results Tab ---
with tab3:
    st.header("Model results & diagnostics")
    if "results" not in st.session_state:
        st.info("Train models in the Train tab to view results.")
    else:
        results = st.session_state["results"]
        y_test = st.session_state["y_test"]

        # Extract names and percentage metrics
        names = list(results.keys())
        accuracies = [results[n]["accuracy (%)"] for n in names]
        precisions = [results[n]["precision (%)"] for n in names]
        recalls = [results[n]["recall (%)"] for n in names]
        f1s = [results[n]["f1-score (%)"] for n in names]
        rocs = [results[n]["roc_auc (%)"] for n in names]

        # Model comparison chart (0-100 scale)
        figp = go.Figure()
        figp.add_trace(go.Bar(x=names, y=accuracies, name="Accuracy (%)"))
        figp.add_trace(go.Bar(x=names, y=precisions, name="Precision (%)"))
        figp.add_trace(go.Bar(x=names, y=recalls, name="Recall (%)"))
        figp.add_trace(go.Bar(x=names, y=f1s, name="F1-Score (%)"))
        figp.add_trace(go.Bar(x=names, y=rocs, name="ROC-AUC (%)"))
        figp.update_layout(
            barmode="group",
            title="Model Performance Comparison",
            yaxis=dict(range=[0, 100], title="Score (%)")
        )
        st.plotly_chart(figp, use_container_width=True)

        # Summary table (sorted by chosen metric)
        summary_df = pd.DataFrame({
            "Model": names,
            "Accuracy (%)": accuracies,
            "Precision (%)": precisions,
            "Recall (%)": recalls,
            "F1-Score (%)": f1s,
            "ROC-AUC (%)": rocs
        }).set_index("Model")
        st.subheader("Performance summary (percentages)")
        st.dataframe(summary_df.style.format("{:.2f}%"))

        # ROC Curve
        st.subheader("ROC Curves")
        fig_roc = go.Figure()
        for n in names:
            fpr, tpr, _ = roc_curve(y_test, results[n]["y_proba"])
            auc_label = results[n]["roc_auc (%)"]
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                name=f"{n} (AUC={auc_label:.2f}%)"
            ))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash")))
        fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", title="ROC Curves")
        st.plotly_chart(fig_roc, use_container_width=True)

        # Confusion Matrices (one column per model)
        st.subheader("Confusion Matrices")
        cols = st.columns(len(names))
        for i, n in enumerate(names):
            cm = results[n]["confusion_matrix"]
            with cols[i]:
                st.write(f"**{n}**")
                fig_cm = px.imshow(
                    cm, text_auto=True,
                    labels=dict(x="Predicted", y="Actual"),
                    x=["Normal", "Fraud"], y=["Normal", "Fraud"],
                    color_continuous_scale="Blues"
                )
                fig_cm.update_layout(title="")
                st.plotly_chart(fig_cm, use_container_width=True)

        # Feature Importance (if RF available)
        if "Random Forest" in results:
            rf = results["Random Forest"]["model"]
            if hasattr(rf, "feature_importances_"):
                st.subheader("Feature Importance (Random Forest)")
                feature_cols = st.session_state["df"].select_dtypes(include=[np.number]).drop("is_fraud", axis=1).columns.tolist()
                importances = rf.feature_importances_
                idxs = np.argsort(importances)[::-1][:15]
                fig_imp = go.Figure(go.Bar(x=importances[idxs], y=[feature_cols[i] for i in idxs], orientation="h"))
                fig_imp.update_layout(title="Top Feature Contributions (importance)")
                st.plotly_chart(fig_imp, use_container_width=True)

        # Show classification reports as percent table when requested
        if st.checkbox("Show classification reports (as percentages)"):
            for n in names:
                st.write("###", n)
                report = classification_report(y_test, results[n]["y_pred"], output_dict=True, zero_division=0)
                for label, stats in report.items():
                    if label != "accuracy" and isinstance(stats, dict):
                        stats["precision"] = round(stats.get("precision", 0) * 100, 2)
                        stats["recall"] = round(stats.get("recall", 0) * 100, 2)
                        stats["f1-score"] = round(stats.get("f1-score", 0) * 100, 2)
                if "accuracy" in report:
                    report["accuracy"] = round(report["accuracy"] * 100, 2)
                report_df = pd.DataFrame(report).T
                if "support" in report_df.columns:
                    report_df["support"] = report_df["support"].astype(int)
                st.dataframe(report_df)

# --- Predict Tab ---
with tab4:
    st.header("Predict Single Transaction")
    if "results" not in st.session_state:
        st.info("Train models first.")
    else:
        df = st.session_state["df"]
        # get the feature names that were used during training
        feature_names = st.session_state.get("feature_names", None)
        if feature_names is None:
            st.error("Feature names missing. Train a model first to establish feature set.")
        else:
            st.write("Enter values for the features below. Defaults are column means from the original dataset.")
            inputs = {}
            # show first 12 features compactly and allow user to expand to all
            short = feature_names[:12]
            with st.form("predict_form"):
                for c in short:
                    meanv = float(df[c].mean()) if c in df.columns else 0.0
                    inputs[c] = st.number_input(c, value=float(round(meanv, 6)), format="%f")
                if st.checkbox("Show all feature inputs"):
                    for c in feature_names[12:]:
                        meanv = float(df[c].mean()) if c in df.columns else 0.0
                        inputs[c] = st.number_input(c, value=float(round(meanv, 6)), format="%f")
                submit = st.form_submit_button("Predict")

            if submit:
                # build a full-row dict with exact feature names & order used during training
                tx_full = {}
                for col in feature_names:
                    if col in inputs:
                        tx_full[col] = inputs[col]
                    else:
                        # fallback to dataset column mean if user didn't supply the field
                        tx_full[col] = float(df[col].mean()) if col in df.columns else 0.0

                # Construct DataFrame with identical column order
                tx = pd.DataFrame([tx_full], columns=feature_names)

                scaler = st.session_state["scaler"]
                try:
                    tx_s = scaler.transform(tx)
                except Exception as e:
                    st.error(f"Scaler.transform failed: {e}")
                    st.stop()

                # Run predictions with each trained model
                for name, r in st.session_state["results"].items():
                    model = r["model"]
                    try:
                        pred = model.predict(tx_s)[0]
                    except Exception as e:
                        st.error(f"{name}: predict() failed: {e}")
                        continue

                    # Try to get a probability / confidence
                    proba = None
                    try:
                        proba = model.predict_proba(tx_s)[0][1]  # probability of class 1 (fraud)
                    except Exception:
                        try:
                            dec = model.decision_function(tx_s)
                            # decision_function may return scalar or array; convert to 0-1
                            dec = np.array(dec).ravel()
                            # map to 0-1
                            proba = float((dec - dec.min()) / (dec.max() - dec.min() + 1e-9)) if dec.size > 0 else 0.0
                        except Exception:
                            proba = 0.0

                    if pred == 1:
                        st.error(f"{name}: FRAUD (confidence {proba*100:.2f}%)")
                    else:
                        st.success(f"{name}: NORMAL (confidence {(1-proba)*100:.2f}%)")

st.markdown("---")
st.write("Tip: For production, persist trained models, serve with proper thresholds, and log alerts.")
