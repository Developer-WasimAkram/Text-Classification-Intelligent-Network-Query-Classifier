import os

# ── Paths ────────────────────────────────────────────────────────────────────
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── BERT Configuration ───────────────────────────────────────────────────────
BERT_MODEL_NAME = "bert-base-uncased"
MAX_SEQ_LENGTH = 64
BATCH_SIZE = 64
DEVICE = "cpu"
EMBEDDING_DIM = 768

# ── Dataset ──────────────────────────────────────────────────────────────────
CSV_PATH = os.path.join(_SRC_DIR, "train_data_balanced.csv")
TEST_SIZE = 0.2

CLASS_NAMES = {
    0: "COUNT",
    1: "BOOLEAN",
    2: "Details",
    3: "Fine-details",
}

# ── Output Paths ─────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(_PROJECT_ROOT, "data")

TRAIN_EMBEDDINGS_PATH = os.path.join(DATA_DIR, "train_embeddings.npy")
TEST_EMBEDDINGS_PATH = os.path.join(DATA_DIR, "test_embeddings.npy")
TRAIN_LABELS_PATH = os.path.join(DATA_DIR, "train_labels.npy")
TEST_LABELS_PATH = os.path.join(DATA_DIR, "test_labels.npy")
XGBOOST_MODEL_PATH = os.path.join(DATA_DIR, "xgboost_model.joblib")
ADABOOST_MODEL_PATH = os.path.join(DATA_DIR, "adaboost_model.joblib")

# ── XGBoost Hyperparameters ──────────────────────────────────────────────────
XGB_N_ESTIMATORS = 300
XGB_MAX_DEPTH = 6
XGB_LEARNING_RATE = 0.1

# ── AdaBoost Hyperparameters ─────────────────────────────────────────────────
ADA_N_ESTIMATORS = 200
ADA_LEARNING_RATE = 0.5
