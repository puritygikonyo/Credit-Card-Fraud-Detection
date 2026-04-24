# CREDIT CARD FRAUD DETECTION
### Exploratory Data Analysis
Dataset: 284,807 transactions × 31 features. All features are numeric (float64/int64). No missing values across any column. Target variable (Class) is binary: 0 = legitimate, 1 = fraud. Data is complete, well-structured, and ready for modelling with no cleaning or gap-filling required.


**Key Findings:**

1. **Severe class imbalance** — only 492 fraud cases (0.17%) vs. 284,315 legitimate transactions (a 579:1 ratio). Standard accuracy metrics will be misleading; requires SMOTE, undersampling, or class weighting before modelling. Evaluate using F1-score and AUC-ROC.
2. **V17, V14, and V12 are the strongest fraud predictors** — correlation coefficients of −0.33, −0.30, and −0.26 with Class respectively. Fraudulent transactions cluster at strongly negative values of these features, providing clear statistical separation.
3. **V1–V28 are mutually independent by PCA design** — near-zero inter-feature correlation throughout. No redundant features to drop; all 28 carry distinct, non-overlapping signal.
4. **Amount is a weak standalone predictor** (r = +0.06) — transaction size alone does not reliably flag fraud and should not be used as a primary signal.
5. **Time has negligible predictive value** (r = −0.01 with Class) — despite showing a bimodal daily distribution, it contributes little to fraud detection.

**Notable observations affecting modelling:**

- 'Amount' is heavily right-skewed — apply log transform before scaling.
  
- Class imbalance is extreme — a naïve model predicting "legitimate" for every transaction scores 99.83% accuracy while catching zero fraud cases. Oversampling or class weighting is essential before training.


### Feature Engineering
There are 12 new engineered features to the original dataframe (31 --> 43 total columns). Zero null introduced.

#### Category A — Domain-Specific (5 features)

| Feature | Logic |
|---|---|
| `hour_of_day` | Time in seconds converted to 0–23 hour |
| `is_night_transaction` | 1 if hour < 5 (midnight–5am = higher fraud risk) |
| `log_amount` | `log1p(Amount)` — fixes right skew for scaling |
| `is_round_amount` | 1 if Amount is a whole number (card-testing pattern) |
| `is_small_amount` | 1 if Amount < $1.00 (card verification tactic) |

#### Category B — Statistical (3 features)

| Feature | Logic |
|---|---|
| `top3_fraud_signal` | V17 + V14 + V12 — composite fraud pressure score |
| `pca_vector_magnitude` | L2 norm of V1–V28 — distance from normal cluster |
| `pca_signal_std` | Std dev of V1–V28 per row — anomalous dispersion signal |

#### Category C — Interaction (4 features)

| Feature | Logic |
|---|---|
| `v14_v17_interaction` | V14 × V17 — joint extreme negative = compounded fraud signal |
| `v12_v14_interaction` | V12 × V14 — same logic for 2nd and 3rd predictors |
| `amount_v17_ratio` | Amount / (abs(V17) + ε) — disproportionate amount vs anomaly |
| `log_amount_x_fraud_signal` | `log_amount` × `top3_fraud_signal` — large transaction + fraud pressure |

#### Feature selection
Not all engineered features help, some hurt

### Credit Card Fraud Detection — Feature Pipeline

## What This Pipeline Does

This pipeline prepares raw transaction data for a fraud detection
machine learning model. Think of it as a preparation stage — the
raw data comes in, gets enriched with new signals, trimmed of
redundant information, and saved in a clean format ready for the
model to learn from.

The pipeline runs automatically with one command:

```bash
python features/run_features.py
```

---

## The Data

The pipeline works on credit card transaction records. Each row
is one transaction made by a cardholder. The data contains:

- **283,726 transactions** — nearly 284,000 real credit card purchases
- **31 original columns** — including the transaction amount, a
  timestamp, 28 anonymised security signals (V1–V28), and a label
  telling us whether each transaction was fraudulent or not

The anonymised signals (V1–V28) were scrambled by the data provider
to protect cardholder privacy, but they still carry patterns that
a fraud detection model can learn from.

---

## Stage 1 — Loading the Data
Source : data/cleaned.csv
Rows   : 283,726 transactions
Columns: 31
Time   : 7.63 seconds

The pipeline first reads the cleaned dataset from disk. The file
loads in under 8 seconds. Every transaction and every column is
present — the data is complete and ready to work with.

---

## Stage 2 — Feature Engineering
Columns before : 31
Columns after  : 43
New signals    : 12
Missing values : 0 across all new columns
Time taken     : 2.47 seconds



### What is feature engineering?

Feature engineering means creating new signals from the existing
data that make it easier for the model to spot fraud. The raw data
tells you the transaction amount — but it does not tell you whether
that amount was unusually small (a sign of card testing), or
whether the transaction happened at 3am (a higher-risk window).
Feature engineering adds those extra clues.

Think of it like briefing a detective. The raw data is the crime
scene. Feature engineering translates the evidence into clear,
labelled observations — "this happened at night", "this amount
was suspiciously small", "two of our strongest fraud signals were
both extreme at the same time."

### The 12 new signals created

**Category A — Behaviour-based signals (5 features)**
These translate business knowledge about fraud into measurable flags:

| New Signal | What it means |
|---|---|
| `hour_of_day` | What hour of the day did this transaction happen? (0–23) |
| `is_night_transaction` | Did it happen between midnight and 5am? (yes=1, no=0) |
| `log_amount` | The transaction amount on a compressed scale — makes large and small amounts comparable |
| `is_round_amount` | Was the amount a perfectly round number like $50.00 or $100.00? Fraudsters often test cards with round amounts |
| `is_small_amount` | Was the amount under $1.00? A known card-verification tactic used by fraudsters |

**Category B — Statistical signals (3 features)**
These measure how unusual a transaction looks compared to normal ones:

| New Signal | What it means |
|---|---|
| `top3_fraud_signal` | A combined score from the three strongest fraud indicators (V12 + V14 + V17). The more negative this score, the more suspicious the transaction |
| `pca_vector_magnitude` | How far this transaction sits from the "normal" cluster of transactions. Most transactions cluster tightly together — fraudulent ones often sit far outside |
| `pca_signal_std` | How spread out the 28 security signals are for this transaction. Unusual spread can indicate an anomalous transaction |

**Category C — Combination signals (4 features)**
These capture situations where two signals together are more revealing than either alone:

| New Signal | What it means |
|---|---|
| `v14_v17_interaction` | What happens when both V14 and V17 are extreme at the same time — a compounded fraud warning |
| `v12_v14_interaction` | Same idea for V12 and V14 together |
| `amount_v17_ratio` | Is the transaction amount disproportionate relative to how suspicious the V17 signal looks? |
| `log_amount_x_fraud_signal` | Are large transactions accompanied by a strong fraud pressure score? This is the highest-risk combination |

All 12 signals were created successfully with zero missing values
across all 283,726 transactions.

---
### Stage 3 — Feature Selection (Removing Redundancy)

| Detail | Value |
|---|---|
| Signals entering selection | 42 |
| Signals dropped | 1 |
| Signals kept | 41 |
| Time taken | 9.80 seconds |

> **What is feature selection?** Not every signal is equally useful. Some signals carry almost identical information — keeping both is like hiring two employees to do the exact same job. Feature selection finds and removes the redundant ones so the model only receives signals that each contribute something unique.

One signal was correctly removed:

| Signal Dropped | Reason | Signal Kept Instead |
|---|---|---|
| pca_signal_std | 99.89% identical to pca_vector_magnitude — both were measuring the same thing | pca_vector_magnitude |

> **Important — What Changed vs. the Previous Run:** In the previous version, the near-constant filter was incorrectly discarding 12 legitimate signals at this stage. That bug has been fixed. In this updated run, zero signals were dropped by the near-constant filter, meaning all 12 engineered fraud signals successfully pass through to the model.

---

### Stage 4 — Saving the Output

| Detail | Value |
|---|---|
| Output file | data/features.csv |
| Rows saved | 283,726 |
| Columns saved | 42 (41 signals + Class fraud label) |
| Time taken | 87.76 seconds |

The final, prepared dataset is saved to disk. This file is the direct input to model training. Every one of the 283,726 transactions is saved with all 41 input signals and the fraud label. The model training step will read this file and use it to learn the difference between legitimate and fraudulent transactions.

---

## Full Pipeline Summary

| Stage | What Happened | Outcome | Time |
|---|---|---|---|
| 1 — Load Data | 283,726 transaction records read from disk | All rows and columns present | 14.84s |
| 2 — Feature Engineering | 12 new fraud signals created from existing data | Columns: 31 → 43 | 5.44s |
| 3 — Feature Selection | 1 duplicate signal removed; 0 incorrectly dropped (bug fixed) | Columns: 43 → 42 | 9.80s |
| 4 — Save Output | Final dataset written to data/features.csv | 283,726 rows, 42 columns saved | 87.76s |
| **Total** | **Complete pipeline run** | **data/features.csv ready** | **118.76s** |

---

## Full List of Signals Passed to the Model (41 Total)

| # | Signal | Type | What It Represents |
|---|---|---|---|
| 1 | Time | Original | When the transaction occurred |
| 2 | Amount | Original | The transaction amount in dollars |
| 3–30 | V1 to V28 | Original | 28 anonymised security indicators carrying fraud patterns |
| 31 | hour_of_day | Engineered | Hour of the day (0–23) |
| 32 | is_night_transaction | Engineered | 1 if between midnight and 5am, else 0 |
| 33 | log_amount | Engineered | Transaction amount on a compressed scale |
| 34 | is_round_amount | Engineered | 1 if a perfectly round amount, else 0 |
| 35 | is_small_amount | Engineered | 1 if under $1.00, else 0 |
| 36 | top3_fraud_signal | Engineered | Combined score from V12 + V14 + V17 |
| 37 | pca_vector_magnitude | Engineered | How far the transaction sits from the normal cluster |
| 38 | v14_v17_interaction | Engineered | Combined extreme signal from V14 and V17 |
| 39 | v12_v14_interaction | Engineered | Combined extreme signal from V12 and V14 |
| 40 | amount_v17_ratio | Engineered | Amount disproportionate to V17 suspicion level |
| 41 | log_amount_x_fraud_signal | Engineered | Large amount combined with high fraud pressure score |

---

## What Comes Next — Model Training

The `data/features.csv` file is the starting point for the next phase: training the fraud-detection model.

A machine learning algorithm will read the 41 signals for every transaction and look for patterns that separate fraudulent transactions from legitimate ones. Once trained, the model will be able to assess a brand-new, unseen transaction and decide — based on those same 41 signals — whether it is likely to be fraud.

**Handoff summary:**
- Input file: `data/features.csv`
- Transactions: 283,726
- Signals available to the model: 41
- Fraud labels included: Yes (Class column)

---

## Key Numbers at a Glance

| Metric | Value |
|---|---|
| Transactions processed | 283,726 |
| Original signals in raw data | 31 |
| New signals created by engineering | 12 |
| Signals passed to model (final) | 41 |
| Signals incorrectly dropped in previous version | 12 (bug now fixed) |
| Missing values introduced | 0 |
| Total pipeline processing time | 118.76 seconds |
| Output file location | data/features.csv |

---

## Glossary

| Term | Plain-English Meaning |
|---|---|
| Feature / Signal | A single piece of information about a transaction (e.g. the amount, the time, whether it was a round number) |
| Feature Engineering | Creating new, more informative signals from existing raw data to help the model spot patterns |
| Feature Selection | Removing signals that are redundant or unhelpful so the model is not confused by duplicate information |
| Model / Machine Learning Model | A computer program that learns patterns from historical data and uses them to make predictions on new data |
| PCA Signals (V1–V28) | 28 anonymised security indicators scrambled to protect privacy but still carrying fraud-related patterns |
| Class Label | The column that tells us the answer — 0 = legitimate transaction, 1 = confirmed fraud |
| Near-Constant Signal | A signal with almost the same value for every transaction — useless to the model because it provides no variation to learn from |
| Correlation | A measure of how similar two signals are. If two signals are 99% correlated, they are nearly identical and one can be safely removed |
| Pipeline | A sequence of automated steps that transform raw data into a clean, prepared output file — each step feeds into the next |

---
