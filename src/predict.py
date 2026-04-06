#!/usr/bin/env python3
"""Interactive inference: classify user-provided text using saved models."""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import numpy as np
import torch
import joblib
from transformers import BertTokenizer, BertModel

import config


def load_saved_model(model_name):
    """Load a saved classifier from disk.

    Parameters
    ----------
    model_name : str
        ``'xgboost'`` or ``'adaboost'``.

    Returns
    -------
    model : fitted classifier
    """
    path = (config.XGBOOST_MODEL_PATH if model_name == "xgboost"
            else config.ADABOOST_MODEL_PATH)
    if not os.path.exists(path):
        print(f"[predict] Model file not found: {path}")
        print("[predict] Run  python main.py --sample 1000  first to train models.")
        sys.exit(1)
    model = joblib.load(path)
    print(f"[predict] Loaded {model_name} model from {path}")
    return model


def embed_texts(texts, tokenizer, model):
    """Get BERT [CLS] embeddings for a list of texts.

    Returns
    -------
    np.ndarray, shape (N, 768), dtype float32
    """
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
    cls = outputs.last_hidden_state[:, 0, :].cpu().numpy().astype(np.float32)
    return cls


def predict(text, classifier, tokenizer, bert_model):
    """Classify a single text and return the predicted label + probabilities.

    Returns
    -------
    pred_label : int  (0-3)
    pred_name : str
    """
    embedding = embed_texts([text], tokenizer, bert_model)
    pred = classifier.predict(embedding)[0]
    pred = int(pred)
    return pred, config.CLASS_NAMES[pred]


def interactive_loop(classifier_name="both"):
    """Run an interactive loop that classifies user input."""

    # ── Load BERT ────────────────────────────────────────────────────────
    print("[predict] Loading BERT tokenizer and model …")
    tokenizer = BertTokenizer.from_pretrained(config.BERT_MODEL_NAME)
    bert_model = BertModel.from_pretrained(config.BERT_MODEL_NAME)
    bert_model.to(config.DEVICE)
    bert_model.eval()
    print("[predict] BERT ready.\n")

    # ── Load classifier(s) ──────────────────────────────────────────────
    classifiers = {}
    if classifier_name in ("xgboost", "both"):
        classifiers["XGBoost"] = load_saved_model("xgboost")
    if classifier_name in ("adaboost", "both"):
        classifiers["AdaBoost"] = load_saved_model("adaboost")

    # ── Interactive loop ────────────────────────────────────────────────
    print("=" * 60)
    print("  Interactive Text Classifier")
    print("  Type a query and press Enter to classify.")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 60)

    while True:
        try:
            text = input("\n>>> Enter text: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not text or text.lower() in ("quit", "exit", "q"):
            print("Exiting.")
            break

        embedding = embed_texts([text], tokenizer, bert_model)

        for name, clf in classifiers.items():
            pred = int(clf.predict(embedding)[0])
            label = config.CLASS_NAMES[pred]
            print(f"  [{name}]  →  Class {pred} : {label}")


def batch_predict(texts, classifier_name="both"):
    """Classify a list of texts (non-interactive).

    Parameters
    ----------
    texts : list[str]
    classifier_name : str
        ``'xgboost'``, ``'adaboost'``, or ``'both'``.
    """
    tokenizer = BertTokenizer.from_pretrained(config.BERT_MODEL_NAME)
    bert_model = BertModel.from_pretrained(config.BERT_MODEL_NAME)
    bert_model.to(config.DEVICE)
    bert_model.eval()

    classifiers = {}
    if classifier_name in ("xgboost", "both"):
        classifiers["XGBoost"] = load_saved_model("xgboost")
    if classifier_name in ("adaboost", "both"):
        classifiers["AdaBoost"] = load_saved_model("adaboost")

    embeddings = embed_texts(texts, tokenizer, bert_model)

    print(f"\n{'Text':<60} ", end="")
    for name in classifiers:
        print(f"| {name:<20}", end="")
    print()
    print("-" * (62 + 22 * len(classifiers)))

    for i, text in enumerate(texts):
        display = text[:57] + "…" if len(text) > 57 else text
        print(f"{display:<60} ", end="")
        for name, clf in classifiers.items():
            pred = int(clf.predict(embeddings[i : i + 1])[0])
            label = config.CLASS_NAMES[pred]
            print(f"| {label:<20}", end="")
        print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict text class using saved models")
    parser.add_argument(
        "--model",
        choices=["xgboost", "adaboost", "both"],
        default="both",
        help="Which model(s) to use (default: both)",
    )
    parser.add_argument(
        "--text",
        type=str,
        nargs="+",
        default=None,
        help="Text(s) to classify (non-interactive). Omit for interactive mode.",
    )
    args = parser.parse_args()

    if args.text:
        batch_predict(args.text, args.model)
    else:
        interactive_loop(args.model)
