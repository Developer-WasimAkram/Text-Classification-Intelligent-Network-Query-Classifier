import os
import numpy as np
import torch
from transformers import BertTokenizer, BertModel

import config


def load_bert(model_name=None, device=None):
    """Load pretrained BERT tokenizer and model.

    Returns
    -------
    tokenizer : BertTokenizer
    model : BertModel  (already on *device*, eval mode)
    """
    model_name = model_name or config.BERT_MODEL_NAME
    device = device or config.DEVICE

    print(f"[feature_extractor] Loading BERT model: {model_name} …")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model


def extract_embeddings(texts, tokenizer, model, batch_size=None, max_len=None, device=None):
    """Extract [CLS] token embeddings from BERT for a list of texts.

    Uses a memory-mapped file to avoid holding all embeddings in RAM.

    Parameters
    ----------
    texts : list[str]
        Raw input texts.
    tokenizer : BertTokenizer
    model : BertModel
    batch_size : int, optional
    max_len : int, optional
    device : str, optional

    Returns
    -------
    embeddings : np.ndarray, shape (N, 768), dtype float16
    """
    batch_size = batch_size or config.BATCH_SIZE
    max_len = max_len or config.MAX_SEQ_LENGTH
    device = device or config.DEVICE

    total = len(texts)
    n_batches = (total + batch_size - 1) // batch_size

    # Pre-allocate a memory-mapped temp file to avoid RAM pressure
    os.makedirs(config.DATA_DIR, exist_ok=True)
    tmp_path = os.path.join(config.DATA_DIR, "_tmp_embeddings.npy")
    mmap = np.lib.format.open_memmap(
        tmp_path, mode="w+", dtype=np.float16,
        shape=(total, config.EMBEDDING_DIM),
    )

    print(f"[feature_extractor] Extracting embeddings for {total:,} texts "
          f"({n_batches} batches, batch_size={batch_size}) …")

    with torch.no_grad():
        for i in range(0, total, batch_size):
            batch_texts = texts[i : i + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}

            outputs = model(**encoded)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            batch_np = cls_embeddings.cpu().numpy().astype(np.float16)

            end = min(i + batch_size, total)
            mmap[i:end] = batch_np

            batch_idx = i // batch_size + 1
            if batch_idx % 100 == 0 or batch_idx == n_batches:
                print(f"  Batch {batch_idx}/{n_batches} done")

    # Flush to disk and re-open as read-only
    del mmap
    embeddings = np.load(tmp_path, mmap_mode="r")
    print(f"[feature_extractor] Embeddings shape: {embeddings.shape}")
    return embeddings


def save_embeddings(embeddings, labels, split):
    """Save embeddings and labels as .npy files under data/.

    Parameters
    ----------
    embeddings : np.ndarray
    labels : np.ndarray
    split : str
        Either ``'train'`` or ``'test'``.
    """
    os.makedirs(config.DATA_DIR, exist_ok=True)

    if split == "train":
        emb_path = config.TRAIN_EMBEDDINGS_PATH
        lbl_path = config.TRAIN_LABELS_PATH
    else:
        emb_path = config.TEST_EMBEDDINGS_PATH
        lbl_path = config.TEST_LABELS_PATH

    np.save(emb_path, embeddings)
    np.save(lbl_path, labels)
    print(f"[feature_extractor] Saved {split} embeddings → {emb_path}")
    print(f"[feature_extractor] Saved {split} labels    → {lbl_path}")


def load_embeddings(split):
    """Load cached embeddings and labels from .npy files.

    Parameters
    ----------
    split : str
        Either ``'train'`` or ``'test'``.

    Returns
    -------
    embeddings : np.ndarray
    labels : np.ndarray
    """
    if split == "train":
        emb_path = config.TRAIN_EMBEDDINGS_PATH
        lbl_path = config.TRAIN_LABELS_PATH
    else:
        emb_path = config.TEST_EMBEDDINGS_PATH
        lbl_path = config.TEST_LABELS_PATH

    embeddings = np.load(emb_path)
    labels = np.load(lbl_path)
    print(f"[feature_extractor] Loaded {split} embeddings from {emb_path}  shape={embeddings.shape}")
    return embeddings, labels
