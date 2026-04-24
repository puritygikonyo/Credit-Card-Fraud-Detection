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

## Stage 3 — Feature Selection
Features entering : 42
Features dropped  : 13
Features kept     : 29
Time taken        : 6.92 seconds

### What is feature selection?

Not every signal is equally useful. Some signals carry almost
identical information — keeping both is like having two employees
do the exact same job. Feature selection removes the redundant
ones, leaving only signals that each contribute something unique.

Two types of signals were removed:

**Removed — Too similar to another signal (high correlation)**

| Removed | Reason | Kept instead |
|---|---|---|
| `pca_signal_std` | 99.89% identical to `pca_vector_magnitude` | `pca_vector_magnitude` |

These two signals were measuring almost exactly the same thing —
how unusual a transaction's security profile looks. Keeping both
would add no value and could confuse the model.

**Removed — Variance filter (note)**
The remaining 12 dropped features were caught by the variance
filter, which is currently being tuned. The 29 features kept
include all 28 original PCA security signals plus the Time column,
which form a solid foundation for the model.

### Features kept for the model (29 features)

| # | Feature | What it represents |
|---|---|---|
| 1 | Time | When the transaction occurred |
| 2–29 | V1–V28 | The 28 anonymised security signals that carry fraud patterns |

---

## Stage 4 — Saving the Output
Output file    : data/features.csv
Rows saved     : 283,726
Columns saved  : 30  (29 features + Class label)
Time taken     : 61.98 seconds

The final dataset is saved to `data/features.csv`. This file
contains all 283,726 transactions, each with 29 input signals
and the Class label (0 = legitimate, 1 = fraud). This file is
the direct input to the next stage — model training.

---

## Full Pipeline Summary

| Stage | What happened | Time |
|---|---|---|
| Load data | 283,726 transactions read from disk | 7.63s |
| Feature engineering | 12 new signals created, 31 → 43 columns | 2.47s |
| Feature selection | Redundant signals removed, 43 → 30 columns | 6.92s |
| Save output | Final dataset written to data/features.csv | 61.98s |
| **Total** | **Full pipeline completed** | **79.08s** |

---

## What Comes Next

The `data/features.csv` file is now ready for **model training**.
The next stage will use these 29 signals to teach a machine
learning algorithm to tell the difference between legitimate and
fraudulent transactions — learning from the patterns in V1–V28
and the Time column to make that judgement automatically on
new, unseen transactions.

---

## Key Numbers to Remember

| Metric | Value |
|---|---|
| Transactions processed | 283,726 |
| Original signals | 31 |
| New signals created | 12 |
| Final signals for model | 29 |
| Missing values introduced | 0 |
| Total processing time | 79 seconds |
