# Implementation Guide: BERT + XGBoost/AdaBoost Multi-Class Text Classification

## Overview

This project implements a **multi-class text classification system** that categorizes device/network queries into **4 distinct classes** using BERT embeddings with XGBoost and AdaBoost classifiers.

### Classes

| Label | Description |
|-------|-------------|
| 1 | How many ports does a device have? |
| 2 | Is optics available in a device? |
| 3 | Provide details for optics |
| 4 | What are the different types of ports? |

---

## Project Structure

```
HCO-AI/
├── train_data.csv            # Raw dataset (~728k rows, no header)
├── data/                     # Cached embeddings, trained models, plots
│   ├── train_embeddings.npy  # (generated) BERT [CLS] embeddings for train set
│   ├── test_embeddings.npy   # (generated) BERT [CLS] embeddings for test set
│   ├── train_labels.npy      # (generated) Train labels
│   ├── test_labels.npy       # (generated) Test labels
│   ├── xgboost_model.joblib  # (generated) Trained XGBoost model
│   ├── adaboost_model.joblib # (generated) Trained AdaBoost model
│   ├── xgboost_confusion_matrix.png  # (generated)
│   └── adaboost_confusion_matrix.png # (generated)
├── requirements.txt          # Python dependencies
├── config.py                 # Hyperparameters, seeds, paths
├── data_loader.py            # CSV loading, train/test split
├── feature_extractor.py      # BERT tokenization + embedding extraction
├── train.py                  # XGBoost & AdaBoost training
├── evaluate.py               # Metrics computation & visualization
├── main.py                   # CLI orchestration script
└── implementation.md         # This file
```

---

## Technical Architecture

```
train_data.csv
      │
      ▼
┌──────────────┐
│  data_loader  │  Load CSV → stratified 80/20 split → labels 1-4 shifted to 0-3
└──────┬───────┘
       │ (texts[], labels[])
       ▼
┌───────────────────┐
│ feature_extractor  │  BERT (bert-base-uncased) → tokenize → [CLS] embedding (768-dim)
│   (PyTorch)        │  Batched inference with torch.no_grad()
└──────┬────────────┘
       │ (np.ndarray, shape: N×768)
       ▼
┌──────────────┐
│    train      │  XGBoost + AdaBoost trained on embeddings
└──────┬───────┘
       │ (fitted models)
       ▼
┌──────────────┐
│   evaluate    │  Accuracy, Precision, Recall, F1, Confusion Matrix
└──────────────┘
```

---

## Module Specifications

### 1. `config.py`

Central configuration module. All magic numbers and paths live here.

```python
# Key configurations:
RANDOM_SEED = 42
BERT_MODEL_NAME = "bert-base-uncased"
MAX_SEQ_LENGTH = 64          # Queries are short
BATCH_SIZE = 64
DEVICE = "cpu"
CSV_PATH = "train_data.csv"
TEST_SIZE = 0.2              # 80/20 split
EMBEDDING_DIM = 768          # BERT base hidden size

CLASS_NAMES = {
    0: "Count",
    1: "Boolean",
    2: "Details",
    3: "Finer  details"
}

# Output paths (all under data/)
TRAIN_EMBEDDINGS_PATH = "data/train_embeddings.npy"
TEST_EMBEDDINGS_PATH = "data/test_embeddings.npy"
TRAIN_LABELS_PATH = "data/train_labels.npy"
TEST_LABELS_PATH = "data/test_labels.npy"
XGBOOST_MODEL_PATH = "data/xgboost_model.joblib"
ADABOOST_MODEL_PATH = "data/adaboost_model.joblib"
```

### 2. `data_loader.py`

**Function**: `load_data(csv_path, test_size, sample_n=None)`

- Reads `train_data.csv` using `pandas.read_csv(path, header=None, names=["text", "label"])`
- Strips whitespace from text column
- Shifts labels from 1–4 → 0–3 (required by XGBoost/sklearn)
- Optional `sample_n` parameter to subsample for quick testing
- Performs stratified train/test split via `sklearn.model_selection.train_test_split(stratify=labels)`
- Computes `sample_weight` using `sklearn.utils.class_weight.compute_sample_weight('balanced', y_train)` to handle potential class imbalance
- **Returns**: `(X_train_texts, y_train, X_test_texts, y_test, class_names, sample_weights)`

### 3. `feature_extractor.py`

**Function**: `extract_embeddings(texts, tokenizer, model, batch_size, max_len, device)`

- Loads `BertTokenizer` and `BertModel` from `transformers` library (PyTorch backend)
- Sets model to `eval()` mode
- Processes texts in batches of `BATCH_SIZE`:
  1. Tokenizes batch with `tokenizer(batch, padding=True, truncation=True, max_length=MAX_SEQ_LENGTH, return_tensors='pt')`
  2. Forward pass through BERT under `torch.no_grad()` context
  3. Extracts `[CLS]` token embedding: `outputs.last_hidden_state[:, 0, :]` (768-dim vector)
  4. Converts to NumPy, appends to list
- Concatenates all batches → single `np.ndarray` of shape `(N, 768)` in **float16** to save memory
- Prints progress every 100 batches

**Function**: `save_embeddings(embeddings, labels, prefix)` / `load_embeddings(prefix)`
- Saves/loads `.npy` files to `data/` directory for caching
- Avoids re-running BERT inference on subsequent runs

**Memory Considerations**:
- Full dataset: 728k × 768 × 2 bytes (float16) ≈ **1.05 GB** per embedding array
- Use `--sample N` flag to test with smaller subsets first

### 4. `train.py`

**Function**: `train_xgboost(X_train, y_train, sample_weight)`

```python
XGBClassifier(
    objective='multi:softmax',
    num_class=4,
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    eval_metric='mlogloss',
    use_label_encoder=False,
    random_state=42,
    n_jobs=-1                # Use all CPU cores
)
```
- Fits with `sample_weight` parameter
- Saves model via `joblib.dump()`
- **Returns**: fitted model

**Function**: `train_adaboost(X_train, y_train, sample_weight)`

```python
AdaBoostClassifier(
    n_estimators=200,
    learning_rate=0.5,
    algorithm='SAMME',
    random_state=42
)
```
- Fits with `sample_weight` parameter
- Saves model via `joblib.dump()`
- **Returns**: fitted model

### 5. `evaluate.py`

**Function**: `evaluate_model(model, X_test, y_test, class_names, model_name)`

Computes and prints:
- **Accuracy**: `accuracy_score(y_test, y_pred)`
- **Classification Report**: `classification_report(y_test, y_pred)` — includes per-class Precision, Recall, F1
- **F1-Score (macro)**: `f1_score(y_test, y_pred, average='macro')`
- **F1-Score (weighted)**: `f1_score(y_test, y_pred, average='weighted')`
- **Confusion Matrix**: `confusion_matrix(y_test, y_pred)` visualized as heatmap via `seaborn.heatmap()` and saved as PNG

**Returns**: dict with all metrics for comparison

**Function**: `compare_models(results_dict)`

Prints a formatted side-by-side comparison table:

```
┌──────────────────┬───────────┬───────────┐
│ Metric           │  XGBoost  │  AdaBoost │
├──────────────────┼───────────┼───────────┤
│ Accuracy         │   0.XXXX  │   0.XXXX  │
│ F1 (macro)       │   0.XXXX  │   0.XXXX  │
│ F1 (weighted)    │   0.XXXX  │   0.XXXX  │
│ Precision (macro)│   0.XXXX  │   0.XXXX  │
│ Recall (macro)   │   0.XXXX  │   0.XXXX  │
└──────────────────┴───────────┴───────────┘
```

### 6. `main.py`

End-to-end orchestration with CLI arguments:

```
python main.py [OPTIONS]

Options:
  --skip-extraction    Use cached embeddings from data/ (skip BERT inference)
  --model {xgboost,adaboost,both}  Which classifier(s) to train (default: both)
  --sample N           Use only N rows from dataset for quick testing
```

**Execution flow**:
1. Set random seeds (Python `random`, `numpy`, `torch.manual_seed`)
2. Load and split data via `data_loader.load_data()`
3. Extract or load cached BERT embeddings via `feature_extractor`
4. Train selected model(s) via `train`
5. Evaluate and compare via `evaluate`

---

## Setup & Execution

### Prerequisites

```bash
pip install -r requirements.txt
```

### Quick Test (recommended first run)

```bash
python main.py --sample 1000
```

This uses only 1000 rows to validate the entire pipeline works correctly. Should complete in a few minutes.

### Full Run

```bash
python main.py
```

**Warning**: BERT feature extraction for ~728k rows on CPU will take **several hours**. Embeddings are cached after the first run.

### Subsequent Runs (skip BERT extraction)

```bash
python main.py --skip-extraction
python main.py --skip-extraction --model xgboost
python main.py --skip-extraction --model adaboost
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | PyTorch — BERT model inference |
| `transformers` | HuggingFace — BERT tokenizer & model |
| `xgboost` | XGBoost classifier |
| `scikit-learn` | AdaBoost, train/test split, metrics |
| `numpy` | Array operations |
| `pandas` | CSV loading |
| `matplotlib` | Plotting |
| `seaborn` | Confusion matrix heatmaps |

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Embedding strategy | `[CLS]` from last hidden layer | More reliable than pooler output; pooler has an extra linear layer that may not be well-trained for this domain |
| Max sequence length | 64 tokens | Device/optics queries are short; 64 covers virtually all inputs without truncation |
| Label encoding | Shift 1–4 → 0–3 | XGBoost and sklearn require 0-indexed labels |
| Embedding dtype | float16 | Halves memory from ~2.1GB to ~1.05GB per array |
| Caching | `.npy` files in `data/` | BERT inference is the bottleneck; caching avoids hours of re-computation |
| Class imbalance | `compute_sample_weight('balanced')` | Defensive handling even if dataset is balanced |
| BERT variant | `bert-base-uncased` | Good balance of embedding quality and CPU inference speed |

---

## Reproducibility

All random seeds are set to **42** at the entry point:
- `random.seed(42)`
- `numpy.random.seed(42)`
- `torch.manual_seed(42)`
- XGBoost: `random_state=42`
- AdaBoost: `random_state=42`
- `train_test_split`: `random_state=42`

---

## Expected Output

After a successful run, the following files will be generated in `data/`:

- `train_embeddings.npy` — BERT embeddings for training set
- `test_embeddings.npy` — BERT embeddings for test set
- `train_labels.npy` — Training labels (0–3)
- `test_labels.npy` — Test labels (0–3)
- `xgboost_model.joblib` — Trained XGBoost model
- `adaboost_model.joblib` — Trained AdaBoost model
- `xgboost_confusion_matrix.png` — Confusion matrix plot
- `adaboost_confusion_matrix.png` — Confusion matrix plot

Console output will include:
- Class distribution summary
- BERT extraction progress
- Per-model classification reports
- Side-by-side model comparison table
