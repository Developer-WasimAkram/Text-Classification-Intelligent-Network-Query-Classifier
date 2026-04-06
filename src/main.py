#!/usr/bin/env python3
"""End-to-end pipeline: BERT embeddings → XGBoost / AdaBoost classification."""

import os
# Fix macOS OpenMP conflict between PyTorch and XGBoost
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import gc
import random
import sys

import numpy as np
import torch

import config
import data_loader
import feature_extractor
import train as trainer
import evaluate


def set_seeds(seed=None):
    """Set random seeds for reproducibility across all libraries."""
    seed = seed or config.RANDOM_SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(
        description="BERT + XGBoost/AdaBoost multi-class text classification"
    )
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Use cached embeddings from data/ (skip BERT inference)",
    )
    parser.add_argument(
        "--model",
        choices=["xgboost", "adaboost", "both"],
        default="both",
        help="Which classifier(s) to train (default: both)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Use only N rows from dataset for quick testing",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seeds()

    os.makedirs(config.DATA_DIR, exist_ok=True)

    # ── Step 1: Load & split data ───────────────────────────────────────
    if args.skip_extraction:
        print("[main] --skip-extraction: loading cached embeddings …")
        X_train, y_train = feature_extractor.load_embeddings("train")
        X_test, y_test = feature_extractor.load_embeddings("test")
        class_names = config.CLASS_NAMES

        # Recompute sample weights from cached labels
        from sklearn.utils.class_weight import compute_sample_weight
        sample_weights = compute_sample_weight("balanced", y_train)
    else:
        X_train_texts, y_train, X_test_texts, y_test, class_names, _ = (
            data_loader.load_data(sample_n=args.sample)
        )

        # Save labels now (needed later); free what we can
        np.save(config.TRAIN_LABELS_PATH, y_train)
        np.save(config.TEST_LABELS_PATH, y_test)

        # ── Step 2a: Extract TRAIN embeddings ───────────────────────────
        tokenizer, model = feature_extractor.load_bert()

        print("\n[main] Extracting training embeddings …")
        X_train = feature_extractor.extract_embeddings(
            X_train_texts, tokenizer, model
        )
        feature_extractor.save_embeddings(X_train, y_train, "train")
        del X_train, X_train_texts
        gc.collect()

        # ── Step 2b: Extract TEST embeddings ────────────────────────────
        print("\n[main] Extracting test embeddings …")
        X_test = feature_extractor.extract_embeddings(
            X_test_texts, tokenizer, model
        )
        feature_extractor.save_embeddings(X_test, y_test, "test")
        del X_test, X_test_texts
        gc.collect()

        # Free BERT from memory before classifiers
        del tokenizer, model
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Reload embeddings from disk (memory-mapped, low RAM)
        print("\n[main] Reloading embeddings from disk …")
        X_train, y_train = feature_extractor.load_embeddings("train")
        X_test, y_test = feature_extractor.load_embeddings("test")

    # Cast to float32 for classifiers (XGBoost doesn't accept float16)
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # ── Step 3: Compute balanced class weights ──────────────────────────
    from sklearn.utils.class_weight import compute_sample_weight
    sample_weights = compute_sample_weight("balanced", y_train)
    print(f"[main] Class weights applied (balanced) for {len(np.unique(y_train))} classes")

    # ── Step 4: Train classifiers ───────────────────────────────────────
    results = []

    if args.model in ("xgboost", "both"):
        xgb_model, xgb_time = trainer.train_xgboost(X_train, y_train, sample_weights)
        res = evaluate.evaluate_model(xgb_model, X_test, y_test, class_names, "XGBoost")
        res["train_time"] = xgb_time
        results.append(res)

    if args.model in ("adaboost", "both"):
        ada_model, ada_time = trainer.train_adaboost(X_train, y_train, sample_weights)
        res = evaluate.evaluate_model(ada_model, X_test, y_test, class_names, "AdaBoost")
        res["train_time"] = ada_time
        results.append(res)

    # ── Step 5: Compare models ──────────────────────────────────────────
    if len(results) > 1:
        evaluate.compare_models(results)

    print("[main] Done.")


if __name__ == "__main__":
    main()
