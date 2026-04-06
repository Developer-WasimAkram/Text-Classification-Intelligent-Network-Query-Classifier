#!/usr/bin/env python3
"""Streamlit GUI for HCO-AI text classification."""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import torch
import joblib
import streamlit as st
from transformers import BertTokenizer, BertModel
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import config


# ── Cached model loading ────────────────────────────────────────────────────


@st.cache_resource
def load_bert():
    """Load BERT tokenizer and model (cached across reruns)."""
    tokenizer = BertTokenizer.from_pretrained(config.BERT_MODEL_NAME)
    model = BertModel.from_pretrained(config.BERT_MODEL_NAME)
    model.to(config.DEVICE)
    model.eval()
    return tokenizer, model


@st.cache_resource
def load_classifier(name):
    """Load a saved classifier from disk."""
    path = (
        config.XGBOOST_MODEL_PATH if name == "xgboost" else config.ADABOOST_MODEL_PATH
    )
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def embed_texts(texts, tokenizer, model):
    """Get BERT [CLS] embeddings for a list of texts."""
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=config.MAX_SEQ_LENGTH,
        return_tensors="pt",
    )
    encoded = {k: v.to(config.DEVICE) for k, v in encoded.items()}
    with torch.no_grad():
        outputs = model(**encoded)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy().astype(np.float32)


# ── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(page_title="HCO-AI Text Classifier", page_icon="🔍", layout="wide")

st.title("HCO-AI Text Classifier")
st.markdown("BERT + XGBoost / AdaBoost multi-class text classification")

# ── Sidebar ─────────────────────────────────────────────────────────────────

st.sidebar.header("Settings")

model_choice = st.sidebar.selectbox(
    "Classifier",
    options=["Both", "XGBoost", "AdaBoost"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.subheader("Class Labels")
for idx, name in config.CLASS_NAMES.items():
    st.sidebar.markdown(f"**{idx}** — {name}")

# ── Check model availability ────────────────────────────────────────────────

xgb_available = os.path.exists(config.XGBOOST_MODEL_PATH)
ada_available = os.path.exists(config.ADABOOST_MODEL_PATH)

if not xgb_available and not ada_available:
    st.error(
        "No trained models found. Run `python main.py` first to train models."
    )
    st.stop()

# ── Tabs ────────────────────────────────────────────────────────────────────

tab_predict, tab_batch, tab_eval = st.tabs(
    ["Single Prediction", "Batch Prediction", "Model Evaluation"]
)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: Single Prediction
# ─────────────────────────────────────────────────────────────────────────────

with tab_predict:
    st.subheader("Classify a single text query")

    user_text = st.text_area(
        "Enter text to classify",
        height=120,
        placeholder="e.g. How many interfaces are up on router-01?",
    )

    if st.button("Classify", key="single_btn", type="primary"):
        if not user_text.strip():
            st.warning("Please enter some text.")
        else:
            with st.spinner("Loading models & computing embedding…"):
                tokenizer, bert_model = load_bert()
                embedding = embed_texts([user_text.strip()], tokenizer, bert_model)

            classifiers = {}
            if model_choice in ("XGBoost", "Both") and xgb_available:
                classifiers["XGBoost"] = load_classifier("xgboost")
            if model_choice in ("AdaBoost", "Both") and ada_available:
                classifiers["AdaBoost"] = load_classifier("adaboost")

            if not classifiers:
                st.error("Selected model is not available.")
            else:
                cols = st.columns(len(classifiers))
                for col, (name, clf) in zip(cols, classifiers.items()):
                    pred = int(clf.predict(embedding)[0])
                    label = config.CLASS_NAMES[pred]
                    col.metric(label=name, value=label, delta=f"Class {pred}")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: Batch Prediction
# ─────────────────────────────────────────────────────────────────────────────

with tab_batch:
    st.subheader("Classify multiple texts at once")

    texts = []

    batch_input = st.text_area(
        "Enter texts (one per line)",
        height=200,
        placeholder="How many BGP neighbors are active?\nIs interface GigabitEthernet0/0 up?",
    )
    if batch_input.strip():
        texts = [t.strip() for t in batch_input.strip().splitlines() if t.strip()]

    if texts:
        st.write(f"**{len(texts)}** texts ready for classification.")

    if st.button("Classify Batch", key="batch_btn", type="primary") and texts:
        with st.spinner("Computing embeddings…"):
            tokenizer, bert_model = load_bert()
            embeddings = embed_texts(texts, tokenizer, bert_model)

        classifiers = {}
        if model_choice in ("XGBoost", "Both") and xgb_available:
            classifiers["XGBoost"] = load_classifier("xgboost")
        if model_choice in ("AdaBoost", "Both") and ada_available:
            classifiers["AdaBoost"] = load_classifier("adaboost")

        if not classifiers:
            st.error("Selected model is not available.")
        else:
            rows = []
            for i, text in enumerate(texts):
                row = {"Text": text}
                for name, clf in classifiers.items():
                    pred = int(clf.predict(embeddings[i : i + 1])[0])
                    row[f"{name} Class"] = pred
                    row[f"{name} Label"] = config.CLASS_NAMES[pred]
                rows.append(row)

            df_results = pd.DataFrame(rows)
            st.dataframe(df_results, use_container_width=True)

            csv_data = df_results.to_csv(index=False)
            st.download_button(
                "Download results as CSV",
                data=csv_data,
                file_name="predictions.csv",
                mime="text/csv",
            )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: Model Evaluation
# ─────────────────────────────────────────────────────────────────────────────

with tab_eval:
    st.subheader("Evaluation on test set")

    embeddings_exist = os.path.exists(config.TEST_EMBEDDINGS_PATH) and os.path.exists(
        config.TEST_LABELS_PATH
    )

    if not embeddings_exist:
        st.warning(
            "Test embeddings not found. Run the training pipeline first "
            "(`python main.py`) to generate them."
        )
    else:
        if st.button("Run Evaluation", key="eval_btn", type="primary"):
            X_test = np.load(config.TEST_EMBEDDINGS_PATH).astype(np.float32)
            y_test = np.load(config.TEST_LABELS_PATH)
            label_names = [config.CLASS_NAMES[i] for i in sorted(config.CLASS_NAMES)]

            eval_models = {}
            if xgb_available:
                eval_models["XGBoost"] = load_classifier("xgboost")
            if ada_available:
                eval_models["AdaBoost"] = load_classifier("adaboost")

            for name, clf in eval_models.items():
                st.markdown(f"### {name}")
                y_pred = clf.predict(X_test)

                # Metrics table
                metrics = {
                    "Accuracy": accuracy_score(y_test, y_pred),
                    "Precision (macro)": precision_score(
                        y_test, y_pred, average="macro", zero_division=0
                    ),
                    "Precision (weighted)": precision_score(
                        y_test, y_pred, average="weighted", zero_division=0
                    ),
                    "Recall (macro)": recall_score(
                        y_test, y_pred, average="macro", zero_division=0
                    ),
                    "Recall (weighted)": recall_score(
                        y_test, y_pred, average="weighted", zero_division=0
                    ),
                    "F1 (macro)": f1_score(
                        y_test, y_pred, average="macro", zero_division=0
                    ),
                    "F1 (weighted)": f1_score(
                        y_test, y_pred, average="weighted", zero_division=0
                    ),
                }

                col_m, col_cm = st.columns([1, 1])

                with col_m:
                    st.markdown("**Metrics**")
                    df_metrics = pd.DataFrame(
                        {"Metric": metrics.keys(), "Score": metrics.values()}
                    )
                    df_metrics["Score"] = df_metrics["Score"].map("{:.4f}".format)
                    st.dataframe(df_metrics, use_container_width=True, hide_index=True)

                with col_cm:
                    st.markdown("**Confusion Matrix**")
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots(figsize=(6, 4.5))
                    sns.heatmap(
                        cm,
                        annot=True,
                        fmt="d",
                        cmap="Blues",
                        xticklabels=label_names,
                        yticklabels=label_names,
                        ax=ax,
                    )
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("Actual")
                    ax.set_title(f"{name} — Confusion Matrix")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                # Classification report
                with st.expander(f"{name} — Full Classification Report"):
                    report = classification_report(
                        y_test, y_pred, target_names=label_names, zero_division=0
                    )
                    st.code(report)

                st.markdown("---")
