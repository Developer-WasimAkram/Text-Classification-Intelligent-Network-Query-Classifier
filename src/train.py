import time
import joblib
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier

import config


def train_xgboost(X_train, y_train, sample_weight=None):
    """Train an XGBoost multi-class classifier on BERT embeddings.

    Parameters
    ----------
    X_train : np.ndarray, shape (N, 768)
    y_train : np.ndarray, shape (N,)
    sample_weight : np.ndarray or None

    Returns
    -------
    model : XGBClassifier (fitted)
    elapsed : float  – training time in seconds
    """
    print("[train] Training XGBoost …")
    model = XGBClassifier(
        objective="multi:softmax",
        num_class=len(config.CLASS_NAMES),
        n_estimators=config.XGB_N_ESTIMATORS,
        max_depth=config.XGB_MAX_DEPTH,
        learning_rate=config.XGB_LEARNING_RATE,
        eval_metric="mlogloss",
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
    )

    start = time.time()
    model.fit(X_train, y_train, sample_weight=sample_weight)
    elapsed = time.time() - start

    joblib.dump(model, config.XGBOOST_MODEL_PATH)
    print(f"[train] XGBoost trained in {elapsed:.1f}s  →  saved to {config.XGBOOST_MODEL_PATH}")
    return model, elapsed


def train_adaboost(X_train, y_train, sample_weight=None):
    """Train an AdaBoost multi-class classifier on BERT embeddings.

    Parameters
    ----------
    X_train : np.ndarray, shape (N, 768)
    y_train : np.ndarray, shape (N,)
    sample_weight : np.ndarray or None

    Returns
    -------
    model : AdaBoostClassifier (fitted)
    elapsed : float  – training time in seconds
    """
    print("[train] Training AdaBoost …")
    model = AdaBoostClassifier(
        n_estimators=config.ADA_N_ESTIMATORS,
        learning_rate=config.ADA_LEARNING_RATE,
        algorithm="SAMME",
        random_state=config.RANDOM_SEED,
    )

    start = time.time()
    model.fit(X_train, y_train, sample_weight=sample_weight)
    elapsed = time.time() - start

    joblib.dump(model, config.ADABOOST_MODEL_PATH)
    print(f"[train] AdaBoost trained in {elapsed:.1f}s  →  saved to {config.ADABOOST_MODEL_PATH}")
    return model, elapsed


def load_model(path):
    """Load a previously saved model from disk."""
    return joblib.load(path)
