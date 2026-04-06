import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

import config


def load_data(csv_path=None, test_size=None, sample_n=None):
    """Load the CSV dataset, split into train/test, and compute sample weights.

    Parameters
    ----------
    csv_path : str, optional
        Path to the CSV file. Defaults to ``config.CSV_PATH``.
    test_size : float, optional
        Fraction of data reserved for testing. Defaults to ``config.TEST_SIZE``.
    sample_n : int or None
        If provided, randomly sample *n* rows before splitting (useful for
        quick pipeline validation).

    Returns
    -------
    X_train_texts : list[str]
    y_train : np.ndarray
    X_test_texts : list[str]
    y_test : np.ndarray
    class_names : dict[int, str]
    sample_weights : np.ndarray
    """
    csv_path = csv_path or config.CSV_PATH
    test_size = test_size or config.TEST_SIZE

    # ── Load ─────────────────────────────────────────────────────────────
    print(f"[data_loader] Loading dataset from {csv_path} …")
    df = pd.read_csv(csv_path, header=None, names=["text", "label"])
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)
    print(f"[data_loader] Total rows: {len(df):,}")

    # ── Subsample (optional) ────────────────────────────────────────────
    if sample_n is not None and sample_n < len(df):
        df = df.sample(n=sample_n, random_state=config.RANDOM_SEED).reset_index(drop=True)
        print(f"[data_loader] Subsampled to {len(df):,} rows")

    # ── Shift labels 1-4 → 0-3 (required by XGBoost/sklearn) ─────────
    df["label"] = df["label"] - 1

    # ── Class distribution ──────────────────────────────────────────────
    print("[data_loader] Class distribution:")
    for label_id, count in df["label"].value_counts().sort_index().items():
        name = config.CLASS_NAMES.get(label_id, f"class_{label_id}")
        pct = count / len(df) * 100
        print(f"  {label_id} ({name}): {count:,} ({pct:.1f}%)")

    # ── Stratified train / test split ───────────────────────────────────
    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        df["text"].tolist(),
        df["label"].values,
        test_size=test_size,
        random_state=config.RANDOM_SEED,
        stratify=df["label"].values,
    )
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    print(f"[data_loader] Train size: {len(X_train_texts):,}  |  Test size: {len(X_test_texts):,}")

    # ── Sample weights for class-imbalance handling ─────────────────────
    sample_weights = compute_sample_weight("balanced", y_train)

    return X_train_texts, y_train, X_test_texts, y_test, config.CLASS_NAMES, sample_weights
