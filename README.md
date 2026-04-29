# 🛡️ Credit Card Fraud Detection
### A Full-Stack Machine Learning Project — Built in 7 Days, Deployed to the World

**[👉 Open the Live Dashboard](https://credit-card-fraud-detection-portfolio-project.streamlit.app/)**

---

> *"283,726 transactions. 473 hidden fraud cases. One model to find them —
> catching 80 out of every 100 before a single euro leaves a customer's account."*

---

## 📌 Table of Contents

1. [What Is This Project?](#what-is-this-project)
2. [Live Dashboard](#live-dashboard)
3. [Why Fraud Detection Is Hard](#why-fraud-detection-is-hard)
4. [The Data](#the-data)
5. [What the Dashboard Shows](#what-the-dashboard-shows)
6. [The Pipeline — How Raw Data Becomes a Prediction](#the-pipeline)
7. [Feature Engineering — Teaching the Model What to Look For](#feature-engineering)
8. [Model Training — Finding the Best Algorithm](#model-training)
9. [Final Results](#final-results)
10. [The Business Decision](#the-business-decision)
11. [Automated Tests](#automated-tests)
12. [How to Run This Project](#how-to-run-this-project)
13. [Project Structure](#project-structure)
14. [Tools Used](#tools-used)
15. [What Comes Next](#what-comes-next)
16. [Glossary](#glossary)

---

## What Is This Project?

Imagine you work at a bank. Every day, hundreds of thousands of people tap
their cards to buy groceries, pay bills, and book flights. Most of those
transactions are completely normal.

But hidden inside that sea of normal activity are fraudsters — people using
stolen card details, testing cards with tiny amounts before making large
withdrawals, or making purchases at 3am in unusual locations.

Your job is to find them. Without stopping real customers in the process.

This project builds a machine learning system that reads the patterns in
transaction data and decides, for every single transaction:

> *Is this fraud — or is this a genuine customer?*

It then translates every result into plain business language — euros protected,
customers inconvenienced, analyst time required — so that anyone, technical
or not, can understand and act on the findings.

**This is not a notebook. This is a complete, deployed system:**

- A trained fraud detection model with measurable, honest results
- An interactive public dashboard accessible to anyone with a browser
- A fully automated data pipeline from raw data to prediction-ready features
- A Docker container that runs identically on any machine in the world
- 41 automated tests that verify every component works correctly

---

## Live Dashboard

👉 **[credit-card-fraud-detection-portfolio-project.streamlit.app](https://credit-card-fraud-detection-portfolio-project.streamlit.app)**

*No login required. Works on your phone. Opens in your browser.*

The dashboard has five pages:

| Page | What It Shows |
|---|---|
| 📊 Executive Summary | Euro value of fraud stopped, false alarms, net value delivered |
| 🔭 Project Overview | What the project does and the tech stack behind it |
| 🔬 Explore the Data | Interactive charts — when fraud happens, what it costs |
| 🏆 Model Results | How four models compare and why the winner was chosen |
| 🛠️ How I Built This | Architecture diagram, 7-day timeline, honest lessons learned |

---

## Why Fraud Detection Is Hard

Most people assume fraud detection works like a simple filter. It does not.

Here is the core problem:

> **Out of every 600 transactions, only 1 is fraud.**

This is called **class imbalance** — the thing you are trying to find is
extremely rare compared to everything else.

Because of this, if a model simply approved every single transaction —
doing absolutely nothing — it would still score **99.83% accuracy.**

That number sounds impressive. It means every fraud case gets through.
Every stolen card. Every fraudulent withdrawal. All of it approved.

**Accuracy is the wrong measure for this problem.**

The right measures are:

| Metric | The Question It Answers | Why It Matters |
|---|---|---|
| **Recall** | What percentage of actual fraud did we catch? | Missing fraud costs real money |
| **Precision** | Of everything we flagged, what percentage was actually fraud? | False alarms block real customers and damage trust |
| **F1 Score** | Are we balancing catching fraud and protecting customers? | A single number combining both concerns |
| **AUC-ROC** | How well can the model separate fraud from legitimate overall? | The fairest way to compare models regardless of threshold |

These are the metrics this project optimises for and reports honestly.

---

## The Data

**Source:** [Kaggle — Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

> The raw dataset is not included in this repository due to file size limits.
> Download it from Kaggle and place it in `data/` before running the pipeline.

| Detail | Value |
|---|---|
| Total transactions | 284,807 |
| Legitimate transactions | 284,315 (99.83%) |
| Confirmed fraud cases | 492 (0.17%) |
| Features in raw data | 31 |
| Missing values | 0 — data is complete |
| Feature types | All numeric |

**What the columns mean:**

- **Time** — seconds elapsed since the first transaction in the dataset
- **Amount** — the transaction value in dollars
- **V1–V28** — 28 anonymised security signals. These were mathematically
  scrambled by the data provider using a technique called PCA to protect
  cardholder privacy. The patterns are real. The identities are protected.
- **Class** — the answer: 0 = legitimate, 1 = fraud

**Key findings from exploring the data:**

- V17, V14, and V12 are the strongest fraud predictors — fraudulent
  transactions cluster at strongly negative values of these signals,
  providing clear separation a model can learn from
- Amount is a weak standalone predictor — transaction size alone does
  not reliably flag fraud, but becomes powerful when combined with other signals
- Time has almost no predictive value on its own — but the hour of day
  extracted from it does (fraud peaks between midnight and 5am)
- The 28 V-features are mathematically independent of each other by design —
  all 28 carry distinct, non-overlapping information. None can be safely dropped

---

## What the Dashboard Shows

### Page 1 — Executive Summary

This page answers the question a CFO or Head of Fraud would ask in a meeting:
**what did this model actually do for the business?**

Every metric is translated from a percentage into a euro value.

| What You See | What It Means |
|---|---|
| Fraud Value Stopped | Total euros of fraud the model caught before money left accounts |
| Fraud Still at Risk | Euros of fraud the model missed — reported honestly |
| False Alarms | Genuine customers incorrectly flagged and blocked |
| Net Value Delivered | Fraud stopped minus the cost of analysts reviewing alerts |
| Gain vs Old System | Extra fraud cases stopped compared to the previous approach |

The model catches **77.9% of all fraud** while incorrectly blocking fewer
than **1 in 1,000 genuine customers.** The industry benchmark for false
alarms is 3–5 per 1,000. This model is 3–5× better.

---

### Page 3 — Explore the Data

Interactive charts that update in real time as you filter by time of day,
transaction amount, or transaction type. Key patterns visible in the data:

🌙 **Fraud peaks at night** — fraud rates are 3–4× higher between midnight
and 5am, when fraud operations teams are smallest.

💳 **Fraudsters test before striking** — small amounts (under $1) and
perfectly round amounts ($50.00, $100.00) are associated with card-testing
behaviour before larger fraudulent withdrawals.

📐 **V14 is the dominant signal** — its distribution for fraud cases sits
4.8 standard deviations away from legitimate transactions. No other single
feature comes close to this separation.

---

### Page 4 — Model Results and The Threshold Decision

Four models are compared side by side, with every metric translated into
banking language. At the bottom of this page is the most important
interactive feature: **the Threshold Slider.**

Moving it left → the model flags more transactions → catches more fraud
→ but also blocks more genuine customers.

Moving it right → only high-confidence fraud is flagged → almost no genuine
customers are blocked → but some fraud slips through.

Every position shows the exact euro impact in real time. This is the
decision a Head of Fraud Operations makes every quarter — and this tool
makes it transparent and measurable.

---

## The Pipeline

The full system runs in four automated stages:

```
Raw Data (CSV from Kaggle)
      ↓
Stage 1 — Load & Validate       "Are the ingredients safe to use?"
      ↓
Stage 2 — Feature Engineering   "Translate raw data into fraud signals"
      ↓
Stage 3 — Feature Selection     "Remove redundant signals"
      ↓
Stage 4 — Save Output           "Hand clean data to the model"
      ↓
Model Training
      ↓
Predictions + Dashboard
```

Run the full pipeline with one command:

```bash
python features/run_features.py
```

**Pipeline performance:**

| Stage | What Happened | Outcome | Time |
|---|---|---|---|
| 1 — Load Data | 283,726 transactions read from disk | All rows and columns present | 14.84s |
| 2 — Feature Engineering | 12 new fraud signals created | Columns: 31 → 43 | 5.44s |
| 3 — Feature Selection | 1 duplicate signal removed | Columns: 43 → 42 | 9.80s |
| 4 — Save Output | Final dataset written to disk | 283,726 rows, 42 columns | 87.76s |
| **Total** | **Complete pipeline** | **data/features.csv ready** | **118.76s** |

---

## Feature Engineering

Feature engineering means creating new, more informative signals from the
existing raw data — signals that make it easier for the model to spot fraud.

The raw data tells you the transaction amount. It does not tell you whether
that amount was suspiciously small (a card-testing sign), or whether the
transaction happened at 3am (a higher-risk window), or whether two of the
strongest fraud signals fired simultaneously.

Feature engineering adds those extra clues.

Think of it like briefing a detective. The raw data is the crime scene.
Feature engineering translates the evidence into clear, labelled observations
the model can learn from.

**12 new signals were created across three categories. Zero missing values
were introduced across all 283,726 transactions.**

---

### Category A — Behaviour-Based Signals (5 features)

These translate business knowledge about fraud into measurable flags:

| New Signal | What It Detects | Why It Matters |
|---|---|---|
| `hour_of_day` | What hour did this transaction happen? (0–23) | Fraud peaks between midnight and 5am |
| `is_night_transaction` | Did it happen between midnight and 5am? (1=yes, 0=no) | Direct flag for the highest-risk time window |
| `log_amount` | Transaction amount on a compressed scale | Fixes right-skew in raw amounts so large and small values are comparable |
| `is_round_amount` | Was it a perfectly round amount like $50.00 or $100.00? | Fraudsters often test stolen cards with round numbers |
| `is_small_amount` | Was it under $1.00? | A known card-verification tactic used before large fraudulent withdrawals |

---

### Category B — Statistical Signals (3 features)

These measure how unusual a transaction looks compared to normal ones:

| New Signal | What It Detects | Why It Matters |
|---|---|---|
| `top3_fraud_signal` | Combined score from V12 + V14 + V17 | The three strongest fraud indicators combined into one pressure score |
| `pca_vector_magnitude` | How far this transaction sits from the normal cluster | Most transactions cluster tightly — fraudulent ones often sit far outside |
| `pca_signal_std` | How spread out the 28 security signals are for this transaction | Unusual dispersion across signals can indicate an anomalous transaction |

---

### Category C — Combination Signals (4 features)

These capture situations where two signals together reveal more than either alone:

| New Signal | What It Detects | Why It Matters |
|---|---|---|
| `v14_v17_interaction` | Both V14 and V17 extreme at the same time | A compounded fraud warning — two major signals firing simultaneously |
| `v12_v14_interaction` | Both V12 and V14 extreme at the same time | Same logic for the 2nd and 3rd strongest predictors |
| `amount_v17_ratio` | Transaction amount disproportionate to V17 suspicion level | Is a large amount paired with a suspicious security signal? |
| `log_amount_x_fraud_signal` | Large transaction paired with a high fraud pressure score | The highest-risk combination — high value and high suspicion together |

---

### Feature Selection — Removing Redundancy

Not all engineered features help. Some carry almost identical information —
keeping both is like hiring two employees to do the exact same job.

One signal was removed:

| Signal Dropped | Reason | Signal Kept Instead |
|---|---|---|
| `pca_signal_std` | 99.89% identical to `pca_vector_magnitude` — both measuring the same thing | `pca_vector_magnitude` |

**Final count: 41 signals passed to the model.**

---

### Full Signal List Passed to the Model (41 Total)

| # | Signal | Type | What It Represents |
|---|---|---|---|
| 1 | Time | Original | When the transaction occurred |
| 2 | Amount | Original | The transaction amount in dollars |
| 3–30 | V1 to V28 | Original | 28 anonymised security indicators |
| 31 | hour_of_day | Engineered | Hour of day (0–23) |
| 32 | is_night_transaction | Engineered | 1 if midnight–5am, else 0 |
| 33 | log_amount | Engineered | Amount on a compressed scale |
| 34 | is_round_amount | Engineered | 1 if perfectly round amount, else 0 |
| 35 | is_small_amount | Engineered | 1 if under $1.00, else 0 |
| 36 | top3_fraud_signal | Engineered | Combined V12 + V14 + V17 score |
| 37 | pca_vector_magnitude | Engineered | Distance from normal transaction cluster |
| 38 | v14_v17_interaction | Engineered | Combined extreme signal from V14 × V17 |
| 39 | v12_v14_interaction | Engineered | Combined extreme signal from V12 × V14 |
| 40 | amount_v17_ratio | Engineered | Amount relative to V17 suspicion level |
| 41 | log_amount_x_fraud_signal | Engineered | Large amount × high fraud pressure score |

---

## Model Training

### Step 1 — The Baseline

Before training any complex model, a simple one is trained first.
This gives a performance benchmark to beat and confirms the pipeline works.

**Model used:** Logistic Regression — the simplest possible binary classifier.

**Why start simple?**

- A clear benchmark to beat. If XGBoost scores the same as Logistic Regression,
  it adds no value despite being far more complex and harder to explain
- Simple models expose data problems. Suspiciously good or bad results signal
  something is wrong with the data before hours are spent on complex training
- Fast results in seconds, confirming the pipeline works before investing hours
- Sets stakeholder expectations early with a working, explainable result

**Baseline results:**

| Metric | Score | What It Means |
|---|---|---|
| Accuracy | 0.9993 | ⚠️ Misleading — do not use as the primary metric |
| Precision | 0.8732 | 87.3% of flagged transactions are genuinely fraud |
| **Recall** | **0.6526** | ⚠️ Key weakness — misses 34.7% of actual fraud |
| F1 Score | 0.7470 | Moderate balance between catching fraud and precision |
| AUC-ROC | 0.9279 | Strong overall ability to separate fraud from legitimate |

> **Why accuracy is misleading:** 99.83% of transactions are legitimate. A model
> that labels everything as legitimate scores 99.83% accuracy while catching zero
> fraud. Recall and AUC-ROC are the honest metrics for this problem.

**The key weakness:** The baseline misses 35 out of every 100 fraud cases.
That is the number to improve.

---

### Step 2 — The Train/Test Split

The data is divided 80% for training, 20% for testing — with **stratification.**

**Why stratification matters here:**

Without it, a random split could accidentally place most of the 473 fraud cases
into one subset. With only 473 fraud cases total, losing even 50 from the
training set would meaningfully hurt the model's ability to learn fraud patterns.

Stratification guarantees both sets maintain exactly the same fraud rate (0.17%)
as the full dataset — making the test results genuinely representative:

```
Training set : 226,980 transactions (80%) — proportional fraud rate maintained
Test set     :  56,746 transactions  (20%) — proportional fraud rate maintained
```

A fixed random seed (random_state=42) ensures anyone who runs this project
gets the exact same split every time — making results reproducible and comparable.

---

### Step 3 — Model Comparison

Three models were trained against the baseline. Each handles the class imbalance
problem differently.

**The imbalance problem:** There are 599 legitimate transactions for every fraud
case. Without compensation, the model learns to ignore fraud entirely because
approving everything is already 99.83% accurate.

| Model | How It Handles Imbalance | Plain English |
|---|---|---|
| XGBoost | `scale_pos_weight = 598.8` | Missing one fraud case is treated as 599× more costly than missing a legitimate transaction |
| Random Forest | `class_weight = 'balanced'` | Fraud cases automatically given 599× more weight during training |
| LightGBM | `class_weight = 'balanced'` | Same weighting approach, different underlying algorithm |

**Why these three models:**

**XGBoost** builds decision trees one after another, each learning from the
mistakes of the previous one — like a team of investigators where each new
member is briefed specifically on the cases the previous one got wrong. It has
a built-in imbalance setting designed for exactly this type of problem.

**Random Forest** builds many independent decision trees and combines their
votes — like asking 100 analysts to review a transaction and going with the
majority verdict. Robust against overfitting. Chosen to test whether simple
ensemble averaging can compete with gradient boosting.

**LightGBM** is Microsoft's faster gradient boosting implementation. Its
leaf-wise tree growth strategy finds deeper, more specific fraud patterns
and trains significantly faster on large datasets. Included as a genuine
alternative to XGBoost rather than a guaranteed loser.

---

### Step 4 — Results

**Cross-validation scores** (how consistently each model performs across
5 different training rounds — the most reliable indicator of real-world stability):

| Model | CV Mean AUC | CV Std | What This Means |
|---|---|---|---|
| Logistic Regression | — | — | Baseline only — CV not performed |
| **XGBoost** | **0.9787** | **0.0036** | Highest and most consistent — barely moved between rounds |
| Random Forest | 0.9471 | 0.0154 | Noticeably lower — performance changes significantly by round |
| LightGBM | 0.9615 | 0.0156 | Good average but high variance — similar instability to Random Forest |

**Test set scores** (performance on the 20% of data no model ever saw during training):

| Model | Test AUC | Recall | Precision | F1 | Train Time |
|---|---|---|---|---|---|
| Logistic Regression (baseline) | 0.9279 | 65.3% | 87.3% | 0.7470 | 136s |
| **XGBoost** | **0.9725** | **77.9%** | **96.1%** | **0.8605** | **106s** |
| Random Forest | 0.9300 | 69.5% | 95.7% | 0.8049 | ~50 min |
| LightGBM | 0.9754 | 80.0% | 88.4% | 0.8398 | 191s |

**Reading every number:**

**Recall** — LightGBM catches the most fraud: 80 out of every 100 cases.
XGBoost catches 78 out of 100. Both are substantial improvements over the
baseline's 65. Random Forest barely improved over the baseline despite
50 minutes of training — eliminated immediately.

**Precision** — XGBoost raises the fewest false alarms: 96.1% of its flags
are genuine fraud. LightGBM flags more genuine customers (88.4% precision) —
meaning for every 100 alerts raised, 12 are real customers incorrectly blocked
versus only 4 with XGBoost.

**F1** — XGBoost wins the balance score (0.8605) — the best overall
combination of catching fraud and protecting customers.

**CV Std** — XGBoost is 4× more consistent than LightGBM and Random Forest.
Its score barely moved between training rounds. This is the decisive factor
for production deployment.

**Train Time** — Random Forest takes 50 minutes versus 2–3 minutes for the others,
for almost no performance benefit. Fraud patterns evolve weekly. A 50-minute
retraining cycle makes nightly updates impractical.

---

### Step 5 — Model Selection

**Winner: XGBoost.**

LightGBM caught slightly more fraud (80.0% vs 77.9%) but showed 4× more
variance in cross-validation — meaning its performance is less predictable
across different data samples. In a production fraud system, a model that
consistently catches 77.9% every day is safer than one that sometimes hits
80% and sometimes drops to 65%.

XGBoost also raises significantly fewer false alarms (96.1% precision vs
88.4%) — meaning fewer genuine customers are incorrectly blocked per fraud
case caught.

| Factor | XGBoost | LightGBM | Winner |
|---|---|---|---|
| Consistency (CV Std) | 0.0036 | 0.0156 | ✅ XGBoost — 4× more stable |
| Fraud caught (Recall) | 77.9% | 80.0% | LightGBM — +2.1pp |
| False alarms (Precision) | 96.1% | 88.4% | ✅ XGBoost — far fewer |
| Overall balance (F1) | 0.8605 | 0.8398 | ✅ XGBoost |
| Training speed | 106s | 191s | ✅ XGBoost — faster retraining |
| Unseen data (Test AUC) | 0.9725 | 0.9754 | Near tie |

XGBoost wins 4 of 6 categories. Selected for deployment.

---

## Final Results

The final XGBoost model, evaluated on the held-out test set:

| Metric | Score | What It Means in Practice |
|---|---|---|
| **Recall** | **77.9%** | 78 out of every 100 fraud cases caught before money leaves accounts |
| **Precision** | **96.1%** | 96% of flagged transactions are genuinely fraud — very few false alarms |
| **F1 Score** | **0.8605** | Strong balance between catching fraud and protecting customers |
| **AUC-ROC** | **0.9725** | Near-perfect ability to separate fraud from legitimate transactions |
| **False Alarm Rate** | **~0.09%** | Fewer than 1 in 1,000 genuine customers incorrectly blocked |
| **Improvement vs Baseline** | **+12.6pp recall** | 13 additional fraud cases caught per 100 vs the starting model |
| **Training Time** | **106 seconds** | Fast enough to retrain nightly as fraud patterns evolve |
| **CV Consistency** | **4× more stable** | Lower variance than any other model tested |

---

## The Business Decision

Every fraud system faces one fundamental trade-off:

> **Do we catch more fraud and inconvenience more customers,
> or protect more customers and let some fraud through?**

There is no objectively correct answer. It depends on the business.

The dashboard's **Threshold Slider** makes this decision interactive and
transparent. The model produces a fraud probability score (0–100%) for every
transaction. The threshold is the line above which a transaction gets flagged.

| Threshold Setting | Fraud Caught | False Alarm Rate | Right Context |
|---|---|---|---|
| 30% — Aggressive | ~85%+ | Higher | High-risk portfolios, post-breach response |
| 50% — Default | 77.9% | ~0.09% | Retail banking — balanced for high volume |
| 70% — Conservative | ~65% | Near-zero | Premium banking where a wrongly declined card causes immediate reputational damage |

This decision requires no retraining. The business can adjust the threshold
in minutes based on current fraud patterns, operational capacity, or regulatory
requirements. The dashboard shows the exact euro impact of every setting.

**The key business insight:** Moving from the default 50% threshold to 30%
would push recall to approximately 85% with zero model retraining required.
That is a 2-hour configuration change, not a 2-week modelling project.

---

## Automated Tests

Before deployment, 41 automated tests were written across three files.
Think of them as a health inspection before opening a restaurant.

```bash
$ pytest tests/ -v

tests/test_data_quality.py::TestQualityGatePassesOnCleanData::test_gate_passes              PASSED
tests/test_data_quality.py::TestQualityGateCatchesBrokenData::test_gate_fails_on_broken     PASSED
tests/test_data_quality.py::TestQualityGateCatchesBrokenData::test_catches_missing_cols     PASSED
tests/test_features.py::TestFeatureColumnCount::test_expected_number_of_columns             PASSED
tests/test_features.py::TestFeatureNoNaNs::test_no_nans_in_any_engineered_column            PASSED
tests/test_features.py::TestFeatureValueRanges::test_is_night_is_binary                     PASSED
tests/test_model.py::TestModelLoads::test_model_has_predict_proba_method                    PASSED
tests/test_model.py::TestPredictionRanges::test_fraud_probability_between_0_and_1           PASSED
tests/test_model.py::TestPredictionRanges::test_suspicious_transaction_scores_higher        PASSED
...

41 passed in 4.23s ✅
```

**What each file checks:**

`test_data_quality.py` — does the quality gate catch broken data before it
reaches the model? Tests that clean data passes and broken data (negative
amounts, missing columns, NaN values) is caught and rejected with a clear
explanation rather than silently corrupting results.

`test_features.py` — is the feature engineering working correctly? Checks
the exact column count, zero NaN values, and mathematical bounds on every
engineered signal (e.g. `Is_night` must only ever be 0 or 1 — never 0.7).

`test_model.py` — does the model actually work? Most importantly: does a
suspicious transaction (large amount, 3am, extreme V14 score) score higher
than a normal daytime purchase? If not, the model has learned nothing useful
regardless of what its other metrics say.

When all 41 tests pass, the project can be deployed with **evidence** that it
works — not just hope.

---

## How to Run This Project

**Option 1 — Just open the dashboard (no installation)**

```
👉 https://credit-card-fraud-detection-portfolio-project.streamlit.app/
```

**Option 2 — Run with Docker (one command, works on any machine)**

```bash
docker build -t my-ml-project .
docker run -p 8501:8501 my-ml-project
```

Then open `http://localhost:8501` in your browser.

**Option 3 — Run locally with Python**

```bash
# 1. Clone the repository
git clone https://github.com/your-username/my-ml-project
cd my-ml-project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the dataset from Kaggle and place in data/raw/

# 4. Run the feature pipeline
python features/run_features.py

# 5. Train the baseline model
python models/baseline.py

# 6. Run model comparison
python models/compare_models.py

# 7. Launch the dashboard
streamlit run app/streamlit_app.py
```

**Run the automated tests**

```bash
pytest tests/ -v
```

---

## Project Structure

```
my-ml-project/
│
├── 📊 app/                          The Streamlit dashboard
│   ├── streamlit_app.py             Five-page interactive application
│   └── data/
│       ├── predictions.csv          Model predictions on the test set
│       └── model_results.json       Metrics, confusion matrix, feature importance
│
├── ⚙️ features/                     The data preparation pipeline
│   └── run_features.py              Runs all 4 pipeline stages in sequence
│
├── 🤖 models/                       Model training scripts and saved artifacts
│   ├── baseline.py                  Logistic Regression baseline
│   ├── compare_models.py            XGBoost, Random Forest, LightGBM comparison
│   └── xgboost_model.pkl            Trained and saved final model
│
├── 🧪 tests/                        Automated test suite — 41 tests
│   ├── test_data_quality.py         Catches bad data before it reaches the model
│   ├── test_features.py             Verifies feature engineering is correct
│   └── test_model.py                Confirms model loads and predicts correctly
│
├── 🐳 Dockerfile                    Blueprint for the Docker container
├── 🐳 docker-compose.yml            Container manager — volumes, ports, restart
├── 📋 requirements.txt              Dashboard dependencies (9 packages)
├── 📋 requirements-train.txt        Training pipeline dependencies
├── 📋 requirements-dev.txt          Full development environment
├── 🚫 .dockerignore                 Keeps build context at 1.3KB not 697MB
├── 🚫 .gitignore                    Keeps sensitive data off GitHub
└── 📖 README.md                     This file
```

---

## Tools Used

| Category | Tools | Why |
|---|---|---|
| Data handling | Python, Pandas, NumPy | Industry standard for tabular data |
| Machine learning | Scikit-learn, XGBoost, LightGBM | Best-in-class for structured fraud detection |
| Imbalance handling | Class weighting, scale_pos_weight | Compensates for the 599:1 legitimate-to-fraud ratio |
| Dashboard | Streamlit, Plotly | Interactive, deployable, accessible without front-end expertise |
| Containerisation | Docker | Runs identically on any machine — no setup conflicts |
| Testing | Pytest | 41 automated tests across data, features, and model |
| Experiment tracking | MLflow | Logs every training run for reproducibility |

---

## What Comes Next

These are honest gaps between the current project and a fully production-ready
system at a real bank. They are listed on the dashboard itself because showing
awareness of limitations is more valuable than pretending they do not exist.

| Priority | What | Plain English | Estimated Effort |
|---|---|---|---|
| 🔴 High | Lower detection threshold | Moving from 0.50 → 0.30 adds ~8pp recall with zero retraining | 2 hours |
| 🔴 High | SHAP explainability | EU regulations require explaining why a transaction was blocked. SHAP produces this automatically for every prediction | 1–2 days |
| 🔴 High | FastAPI prediction endpoint | Allows external systems — mobile apps, payment terminals — to query the model in real time | 1–2 days |
| 🟡 Medium | Model monitoring | Detects when fraud patterns shift and triggers automatic retraining | 3–5 days |
| 🟡 Medium | Real transaction loss data | Links predictions to actual dollar values for exact ROI rather than averages | 1 day |
| 🟢 Low | A/B shadow deployment | Runs new model in parallel before switching production traffic | 3–5 days |

---

## About the Dataset

The data comes from real European credit card transactions — 284,807
transactions recorded over two days, with 492 confirmed fraud cases.

To protect cardholder privacy, all transaction details have been
mathematically transformed using PCA. The patterns are real.
The identities are protected.

The dataset is publicly available on Kaggle and is one of the most
widely studied benchmarks in fraud detection research worldwide.

---

## Glossary

| Term | Plain-English Meaning |
|---|---|
| **Feature / Signal** | A single piece of information about a transaction (e.g. the amount, the hour, whether it was a round number) |
| **Feature Engineering** | Creating new, more informative signals from existing raw data to help the model spot patterns |
| **Feature Selection** | Removing signals that are redundant or unhelpful so the model is not confused by duplicate information |
| **Model / Machine Learning Model** | A program that learns patterns from historical data and uses them to make predictions on new data |
| **PCA Signals (V1–V28)** | 28 anonymised security indicators scrambled to protect privacy but still carrying fraud-related patterns |
| **Class Label** | The column that tells us the answer — 0 = legitimate transaction, 1 = confirmed fraud |
| **Class Imbalance** | When one outcome is far rarer than the other — here, 1 fraud for every 599 legitimate transactions |
| **Recall** | The percentage of actual fraud cases the model caught. Missing fraud = money lost for real customers |
| **Precision** | Of everything the model flagged, what percentage was actually fraud. Low precision = too many false alarms |
| **F1 Score** | A single number balancing recall and precision. 1.0 is perfect, 0.0 is completely useless |
| **AUC-ROC** | How well the model separates fraud from legitimate across all possible thresholds. Higher is better |
| **Threshold** | The probability score above which a transaction gets flagged. Moving it changes the recall/precision trade-off |
| **Cross-Validation** | Testing the model 5 times on different data slices to confirm it performs consistently, not just on one lucky sample |
| **Class Weighting** | A technique that compensates for imbalance by making fraud cases count more heavily during training |
| **Pipeline** | A sequence of automated steps that transform raw data into a clean, prediction-ready output |
| **Docker** | A tool that packages the entire application so it runs identically on any machine — no setup required |
| **Stratification** | Splitting data so both training and test sets maintain the same fraud rate as the full dataset |
| **Baseline** | The simplest possible model trained first, used as a performance benchmark for all subsequent models to beat |

---

*Built in 7 days · April 2026 · Credit Card Fraud Detection · Purity Gikonyo ML Portfolio Project*

*Questions or feedback → [open a GitHub issue](https://github.com/puritygikonyo/Credit-Card-Fraud-Detection/issues)*