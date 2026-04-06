import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

import config


def evaluate_model(model, X_test, y_test, class_names, model_name):
    """Evaluate a trained classifier and produce metrics + confusion matrix.

    Parameters
    ----------
    model : fitted sklearn-compatible classifier
    X_test : np.ndarray, shape (N, 768)
    y_test : np.ndarray, shape (N,)
    class_names : dict[int, str]
    model_name : str  – e.g. ``"XGBoost"`` or ``"AdaBoost"``

    Returns
    -------
    results : dict  – all computed metrics
    """
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    prec_weighted = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    rec_weighted = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    label_names = [class_names[i] for i in sorted(class_names.keys())]

    # ── Print classification report ──────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  {model_name} — Evaluation Results")
    print(f"{'=' * 60}")
    print(classification_report(y_test, y_pred, target_names=label_names, zero_division=0))
    print(f"  Accuracy          : {acc:.4f}")
    print(f"  Precision (macro) : {prec_macro:.4f}")
    print(f"  Precision (wt.)   : {prec_weighted:.4f}")
    print(f"  Recall    (macro) : {rec_macro:.4f}")
    print(f"  Recall    (wt.)   : {rec_weighted:.4f}")
    print(f"  F1-Score  (macro) : {f1_macro:.4f}")
    print(f"  F1-Score  (wt.)   : {f1_weighted:.4f}")
    print(f"{'=' * 60}\n")

    # ── Confusion matrix heatmap ─────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{model_name} — Confusion Matrix")
    plt.tight_layout()

    os.makedirs(config.DATA_DIR, exist_ok=True)
    cm_path = os.path.join(config.DATA_DIR, f"{model_name.lower()}_confusion_matrix.png")
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"[evaluate] Confusion matrix saved → {cm_path}")

    results = {
        "model_name": model_name,
        "accuracy": acc,
        "precision_macro": prec_macro,
        "precision_weighted": prec_weighted,
        "recall_macro": rec_macro,
        "recall_weighted": rec_weighted,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }
    return results


def compare_models(results_list):
    """Print a side-by-side comparison table for all evaluated models.

    Parameters
    ----------
    results_list : list[dict]
        Each dict is the return value of :func:`evaluate_model`.
    """
    metrics = [
        ("Accuracy", "accuracy"),
        ("Precision (macro)", "precision_macro"),
        ("Precision (weighted)", "precision_weighted"),
        ("Recall (macro)", "recall_macro"),
        ("Recall (weighted)", "recall_weighted"),
        ("F1-Score (macro)", "f1_macro"),
        ("F1-Score (weighted)", "f1_weighted"),
    ]

    names = [r["model_name"] for r in results_list]
    col_w = max(len(n) for n in names) + 2
    label_w = 22

    header = f"{'Metric':<{label_w}}" + "".join(f"{n:>{col_w}}" for n in names)
    sep = "-" * len(header)

    print(f"\n{sep}")
    print("  Model Comparison")
    print(sep)
    print(header)
    print(sep)
    for label, key in metrics:
        row = f"{label:<{label_w}}"
        for r in results_list:
            row += f"{r[key]:>{col_w}.4f}"
        print(row)
    print(sep)
    print()
