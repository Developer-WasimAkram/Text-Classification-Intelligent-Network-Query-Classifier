# HCO-AI: Intelligent Network Query Classifier
## Project Implementation Plan

**Version:** 1.0  
**Date:** 7 April 2026  
**Status:** Model Trained & Validated

---

## 1. What This Project Does

This system **automatically understands and classifies questions** that engineers or automated tools ask about network devices. Instead of a human reading each question and deciding what type of answer is needed, the AI does it instantly.

**Example:** When someone asks *"how many ports does device ffmdls1 have?"*, the system instantly recognises this is a **counting question** and routes it to the right handler that can look up and return a number.

---

## 2. The 4 Types of Questions (Classes)

| Class | Name | What It Means | Example Question |
|-------|------|---------------|------------------|
| **0** | **COUNT** | Questions asking for a number/quantity | *"How many ports does device ffmdls1 have?"* |
| **1** | **BOOLEAN** | Questions asking yes/no or availability | *"Is optics0/4/0/3 available in ffmdls1?"* |
| **2** | **Details** | Questions asking for information about a component | *"Please provide details for optics0/4/0/3 for ffmdls1"* |
| **3** | **Fine-details** | Questions asking for specific attributes or status | *"What is the operational state for port optics0/4/0/3 in device ffmdls1?"* |

---

## 3. How It Works (Non-Technical Summary)

The system works in two stages, similar to how a human would process a question:

```
  User Question
       |
       v
  ┌─────────────────────┐
  │  Stage 1: READING    │   The AI "reads" the question and converts
  │  (BERT Language AI)  │   it into a numerical representation that
  │                      │   captures the meaning of the sentence.
  └──────────┬──────────┘
             |
             v
  ┌─────────────────────┐
  │  Stage 2: DECIDING   │   A fast decision-making algorithm looks at
  │  (XGBoost/AdaBoost) │   the meaning and classifies it into one of
  │                      │   the 4 categories above.
  └──────────┬──────────┘
             |
             v
     Classification Result
     (e.g. "This is a COUNT question")
```

**Stage 1 — BERT (Language Understanding):** A pre-trained AI model from Google that has been trained on billions of words of text. It reads the question and produces a 768-number fingerprint that captures what the question *means*, not just what words it contains. This is why it can correctly classify questions about devices it has never seen before.

**Stage 2 — XGBoost / AdaBoost (Classification):** Two industry-standard machine learning algorithms that have been trained on our labelled data to map the meaning-fingerprint to one of the 4 classes. We train both and compare to pick the best performer.

---

## 4. Training Data

### 4.1 Original Dataset
- **Source file:** `train_data.csv`
- **Total rows:** 1,199,369
- **Format:** Each row has a question and a label (1–4)

### 4.2 Data Quality Issue Identified & Resolved
The original dataset contained massive redundancy — the same question template repeated thousands of times with only the device name or port number changed:

| | Before | After Balancing |
|---|---|---|
| Total rows | 1,199,369 | **208,000** |
| Class 0 (COUNT) | 52,515 (4.4%) | **52,000 (25%)** |
| Class 1 (BOOLEAN) | 468,352 (39.1%) | **52,000 (25%)** |
| Class 2 (Details) | 364,562 (30.4%) | **52,000 (25%)** |
| Class 3 (Fine-details) | 313,940 (26.2%) | **52,000 (25%)** |

**What we did:**
- Created a **balanced dataset** (`train_data_balanced.csv`) with exactly **52,000 rows per class**
- Larger classes were randomly down-sampled; the smallest class (COUNT) was kept as-is
- This ensures the AI pays equal attention to all 4 question types and doesn't develop a bias toward the more common ones

### 4.3 Train/Test Split
- **80% Training** (166,400 rows) — used to teach the models
- **20% Testing** (41,600 rows) — held back to evaluate how well the models perform on unseen data
- Split is **stratified**: equal class proportions maintained in both sets

---

## 5. Model Performance Results

### 5.1 Overall Accuracy

| Model | Accuracy | F1-Score (Macro) | F1-Score (Weighted) |
|-------|----------|------------------|---------------------|
| **XGBoost** | **99.99%** | **99.99%** | **99.99%** |
| AdaBoost | 99.85% | 99.85% | 99.85% |

Both models achieve near-perfect accuracy. **XGBoost is the recommended model** with only **4 misclassifications out of 41,600 test questions**.

### 5.2 Per-Class Breakdown (XGBoost)

| Class | Precision | Recall | F1-Score | Test Samples |
|-------|-----------|--------|----------|--------------|
| COUNT | 100% | 100% | 100% | 10,400 |
| BOOLEAN | 100% | 100% | 100% | 10,400 |
| Details | 100% | 100% | 100% | 10,400 |
| Fine-details | 100% | 100% | 100% | 10,400 |

**What these metrics mean:**
- **Precision** = When the model says "this is a COUNT question," how often is it correct? → **100%**
- **Recall** = Of all actual COUNT questions, how many did the model find? → **100%**
- **F1-Score** = The combined measure of precision and recall → **100%**

### 5.3 Error Analysis (XGBoost — 4 total errors out of 41,600)

| Actual Class | Misclassified As | Count |
|---|---|---|
| COUNT | Fine-details | 1 |
| Details | Fine-details | 1 |
| Fine-details | Details | 2 |

The only minor confusion is between **Details** and **Fine-details** — which are the two most similar categories. This is expected and represents a negligible 0.01% error rate.

### 5.4 AdaBoost Error Analysis (61 total errors out of 41,600)

| Actual Class | Misclassified As | Count |
|---|---|---|
| COUNT | Fine-details | 10 |
| BOOLEAN | Details | 2 |
| BOOLEAN | Fine-details | 3 |
| Details | Fine-details | 7 |
| Fine-details | Details | 39 |

AdaBoost shows a slightly higher confusion between Details and Fine-details, but the overall error rate is still only 0.15%.

---

## 6. System Components

| Component | Purpose | Status |
|-----------|---------|--------|
| `config.py` | Central settings (model parameters, file paths) | Done |
| `data_loader.py` | Loads CSV data, splits into train/test sets | Done |
| `feature_extractor.py` | Converts text questions into AI-readable format using BERT | Done |
| `train.py` | Trains both XGBoost and AdaBoost classifiers | Done |
| `evaluate.py` | Measures model accuracy and generates reports | Done |
| `main.py` | Runs the complete pipeline end-to-end | Done |
| `predict.py` | Interactive mode — type a question, get instant classification | Done |
| `Dockerfile` | Containerised deployment packaging | Done |
| `deduplicate.py` | Data cleaning utility — removes same-meaning duplicates | Done |
| `balance_data.py` | Data balancing utility — equalises class sizes | Done |

---

## 7. How to Use the System

### 7.1 Training (Already Completed)
The models have been trained and saved. Re-training is only needed if new question types are added or the dataset changes.

```
python main.py
```

### 7.2 Interactive Prediction
Type any network query and get an instant classification:

```
python predict.py
>>> Enter text: how many ports does device xyz123 have?
  [XGBoost]  →  Class 0 : COUNT
  [AdaBoost] →  Class 0 : COUNT
```

### 7.3 Batch Prediction
The system can also classify questions in bulk via the `batch_predict` function in `predict.py`.

---

## 8. Key Decisions & Rationale

| Decision | Rationale |
|----------|-----------|
| **BERT for text understanding** | Industry-leading language model that understands sentence meaning, not just keywords. Handles unseen device names and ports gracefully. |
| **XGBoost + AdaBoost (two models)** | Training two different algorithms lets us compare and pick the best. XGBoost was the winner. |
| **Balanced dataset (52K per class)** | Prevents the model from being biased toward more common question types. Ensures equal accuracy across all 4 categories. |
| **80/20 train/test split** | Industry standard. Provides enough test data (41,600 questions) for reliable accuracy measurement. |
| **CPU-based inference** | BERT inference on CPU is sufficient for this use case (short queries, moderate throughput). No GPU required for deployment. |

---

## 9. Deployment Readiness

| Item | Status |
|------|--------|
| Models trained and validated | Yes |
| Confusion matrices generated | Yes |
| Docker configuration available | Yes |
| Interactive prediction working | Yes |
| Handles unseen device names | Yes (BERT generalises to new entities) |
| All dependencies documented | Yes (`requirements.txt`) |

---

## 10. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| New question types added in future | Medium | Model won't recognise them | Retrain with new labelled examples; modular pipeline makes this straightforward |
| Questions in different languages | Low | Model may misclassify | Current BERT model is English-only; multilingual BERT can be swapped in if needed |
| Very ambiguous questions | Low | Misclassification between Details/Fine-details | These two classes have the most overlap; further labelling guidelines could reduce ambiguity |
| Model drift over time | Low | Gradual accuracy decline | Periodic evaluation against new production data recommended |

---

## 11. Summary

- The system classifies network device queries into 4 categories with **99.99% accuracy**
- **XGBoost** is the recommended production model (only 4 errors in 41,600 test questions)
- Training data was cleaned and balanced to **208,000 rows** (52K per class) ensuring fair treatment of all query types
- The system is ready for deployment and can handle questions about devices and ports it has never seen before
- The complete pipeline (data loading → BERT embedding → classification → evaluation) runs end-to-end with a single command
