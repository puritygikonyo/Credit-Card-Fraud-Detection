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
