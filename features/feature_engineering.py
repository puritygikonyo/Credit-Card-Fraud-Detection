"""
feature_engineering.py
=======================
Feature engineering pipeline for credit card fraud detection.

Dataset context:
  - 284,807 transactions, 31 columns (Time, V1–V28, Amount, Class)
  - V1–V28 are PCA-transformed anonymised signals (original features hidden for privacy)
  - Class: 0 = legitimate, 1 = fraud  (0.17% fraud rate — severely imbalanced)
  - Top fraud predictors from EDA: V17 (r=−0.33), V14 (r=−0.30), V12 (r=−0.26)

Feature categories engineered:
  A. Domain-Specific   (5 features) — business logic around time, amount, and known fraud signals
  B. Statistical       (3 features) — deviation, magnitude, and spread of PCA signals
  C. Interaction       (4 features) — products/ratios of features that matter more together

Usage:
  python feature_engineering.py
"""

import pandas as pd
import numpy as np


# =============================================================================
# MAIN FEATURE ENGINEERING FUNCTION
# =============================================================================

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers 12 new features for credit card fraud detection across three
    categories: domain-specific, statistical, and interaction features.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe containing original dataset columns:
        Time, V1–V28, Amount, Class (Class is optional at inference time)

    Returns
    -------
    pd.DataFrame
        Original dataframe with 12 additional engineered feature columns appended.
        Original columns are preserved unchanged.
    """

    df = df.copy()  # Never mutate the original dataframe

    # =========================================================================
    # CATEGORY A — DOMAIN-SPECIFIC FEATURES (5 features)
    # Business logic grounded in how fraud actually behaves in the real world
    # =========================================================================

    # A1: Hour of day
    # Fraud rates are not uniform across the day. Transactions made in the
    # early hours (midnight–5am) have historically higher fraud risk because
    # cardholders are asleep and less likely to notice or dispute immediately.
    # Time is in seconds elapsed since first transaction — convert to hour (0–23).
    df["hour_of_day"] = (df["Time"] // 3600) % 24

    # A2: Is nighttime transaction flag
    # Binary flag for transactions occurring between midnight and 5am.
    # A direct operationalisation of the night-hour fraud risk above.
    # Useful for tree-based models that benefit from explicit binary splits.
    df["is_night_transaction"] = df["hour_of_day"].apply(
        lambda h: 1 if h < 5 else 0
    )

    # A3: Log-transformed transaction amount
    # Raw Amount is heavily right-skewed (most transactions are small, a long
    # tail of large ones). Log transform compresses the scale, reduces the
    # influence of extreme values, and makes the feature behave more normally —
    # which improves performance of linear models and distance-based algorithms.
    # Adding 1 avoids log(0) errors on zero-value transactions.
    df["log_amount"] = np.log1p(df["Amount"])

    # A4: Amount rounded flag
    # Fraudsters often test stolen cards with suspiciously round amounts
    # (e.g. exactly $100.00, $200.00) to verify the card works before making
    # larger purchases. A round amount is a subtle but real behavioural signal.
    df["is_round_amount"] = (df["Amount"] % 1 == 0).astype(int)

    # A5: Small amount flag
    # A common fraud pattern is "card testing" — making very small transactions
    # (under $1) to verify a stolen card is active without triggering alerts.
    # This is a known fraud typology used by fraud analysts in practice.
    df["is_small_amount"] = (df["Amount"] < 1.0).astype(int)

    # =========================================================================
    # CATEGORY B — STATISTICAL FEATURES (3 features)
    # Derived from the mathematical properties of the PCA signals themselves,
    # capturing overall signal strength and anomaly magnitude
    # =========================================================================

    # B1: Top-3 predictor composite score
    # V17, V14, and V12 are the three features most correlated with fraud
    # (r = −0.33, −0.30, −0.26 respectively from EDA). Fraudulent transactions
    # cluster at strongly negative values for all three simultaneously.
    # Summing them creates a single composite "fraud pressure" score — the more
    # negative this sum, the higher the collective fraud signal.
    df["top3_fraud_signal"] = df["V17"] + df["V14"] + df["V12"]

    # B2: L2 norm (magnitude) of all PCA signals
    # Computes the geometric distance of the transaction's PCA vector from the
    # origin. Fraudulent transactions often sit far from the dense cluster of
    # normal transactions in PCA space — a high magnitude can indicate an outlier.
    # Uses V1–V28 only (excludes Time and Amount which are on different scales).
    v_cols = [f"V{i}" for i in range(1, 29)]
    df["pca_vector_magnitude"] = np.sqrt((df[v_cols] ** 2).sum(axis=1))

    # B3: Standard deviation of PCA signals per transaction
    # Measures how "spread out" a single transaction's 28 PCA values are.
    # Normal transactions cluster tightly in PCA space; anomalous (potentially
    # fraudulent) ones may show unusual dispersion across components.
    df["pca_signal_std"] = df[v_cols].std(axis=1)

    # =========================================================================
    # CATEGORY C — INTERACTION FEATURES (4 features)
    # Products and ratios of features whose joint behaviour predicts fraud
    # better than either feature independently
    # =========================================================================

    # C1: V14 × V17 interaction (product)
    # The two single strongest fraud predictors. When BOTH are simultaneously
    # at extreme negative values, fraud probability is amplified beyond what
    # each signals individually. Their product captures this joint extreme —
    # a large positive product means both are strongly negative (fraud-like),
    # a near-zero product means at least one is unremarkable.
    df["v14_v17_interaction"] = df["V14"] * df["V17"]

    # C2: V12 × V14 interaction (product)
    # Second and third strongest predictors interacted. Captures transactions
    # where both V12 and V14 are simultaneously anomalous — a compounding
    # signal that is stronger than either alone.
    df["v12_v14_interaction"] = df["V12"] * df["V14"]

    # C3: Amount × V17 ratio
    # Examines whether the transaction amount is disproportionate relative to
    # the strongest fraud signal. Fraud at unusual amounts combined with an
    # anomalous V17 reading is a more specific pattern than either alone.
    # A small epsilon is added to avoid division by zero if V17 == 0.
    df["amount_v17_ratio"] = (df["Amount"] / (df["V17"].abs() + 1e-6)).clip(upper=1e4)
    # C4: Log amount × top-3 fraud signal interaction
    # Combines the size of the transaction (log-scaled for linearity) with the
    # composite fraud pressure score. Large transactions accompanied by strong
    # negative PCA signals are a particularly high-risk combination that neither
    # feature captures on its own.
    df["log_amount_x_fraud_signal"] = df["log_amount"] * df["top3_fraud_signal"]

    return df


# =============================================================================
# FEATURE SUMMARY UTILITY
# =============================================================================

def summarise_features(df_original: pd.DataFrame, df_engineered: pd.DataFrame) -> None:
    """Prints a concise summary of what was added by feature engineering."""

    new_cols = [c for c in df_engineered.columns if c not in df_original.columns]

    print("=" * 60)
    print("FEATURE ENGINEERING SUMMARY")
    print("=" * 60)
    print(f"Original features : {len(df_original.columns)}")
    print(f"Engineered features added : {len(new_cols)}")
    print(f"Total features : {len(df_engineered.columns)}")
    print()
    print("New features created:")
    print("-" * 60)

    categories = {
        "A — Domain-Specific": [
            "hour_of_day",
            "is_night_transaction",
            "log_amount",
            "is_round_amount",
            "is_small_amount",
        ],
        "B — Statistical": [
            "top3_fraud_signal",
            "pca_vector_magnitude",
            "pca_signal_std",
        ],
        "C — Interaction": [
            "v14_v17_interaction",
            "v12_v14_interaction",
            "amount_v17_ratio",
            "log_amount_x_fraud_signal",
        ],
    }

    for category, features in categories.items():
        print(f"\n  {category}")
        for f in features:
            dtype = df_engineered[f].dtype
            missing = df_engineered[f].isna().sum()
            print(f"    ✓ {f:<35} dtype={dtype}   nulls={missing}")

    print()
    print("Sample statistics on new features:")
    print("-" * 60)
    print(df_engineered[new_cols].describe().round(4).to_string())
    print("=" * 60)


# =============================================================================
# MAIN BLOCK
# =============================================================================

if __name__ == "__main__":

    import os

    # ------------------------------------------------------------------
    # 1. Load cleaned data
    #    Update DATA_PATH to point to your creditcard.csv location.
    # ------------------------------------------------------------------
    DATA_PATH = os.getenv("DATA_PATH", "data/creditcard.csv")

    print(f"Loading data from: {DATA_PATH}")
    df_raw = pd.read_csv(DATA_PATH)
    print(f"Raw data shape    : {df_raw.shape}")

    # ------------------------------------------------------------------
    # 2. Run feature engineering
    # ------------------------------------------------------------------
    df_features = create_features(df_raw)
    print(f"Engineered shape  : {df_features.shape}")

    # ------------------------------------------------------------------
    # 3. Print summary report
    # ------------------------------------------------------------------
    summarise_features(df_raw, df_features)

    # ------------------------------------------------------------------
    # 4. Save to file for use in modelling pipeline
    # ------------------------------------------------------------------
    OUTPUT_PATH = "data/creditcard_engineered.csv"
    df_features.to_csv(OUTPUT_PATH, index=False)
    print(f"\nEngineered dataset saved to: {OUTPUT_PATH}")




# =============================================================================
# FEATURE SELECTION FUNCTION
# =============================================================================

def select_features(
    df: pd.DataFrame,
    target_col: str = "Class",
    corr_threshold: float = 0.95,
    variance_threshold_pct: float = 0.01,
) -> (list, pd.DataFrame):
    """
    Selects features by removing only highly correlated features.
    Variance filter is intentionally scoped to near-constant features only
    (variance < 0.001) — a hard floor, not a relative threshold — because:
      - V1-V28 have variance ~1 by PCA design (not a flaw)
      - amount_v17_ratio has variance in the millions (ratio scale)
      - A relative threshold based on mean variance breaks when one feature
        lives on a completely different scale from all others

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with original + engineered features.
    target_col : str
        Target column — never dropped, excluded from all filtering.
    corr_threshold : float
        Drop features correlated above this value. Default 0.95.
    variance_threshold_pct : float
        Kept as parameter for API compatibility but not used for
        relative threshold. A hard floor of 0.001 is used instead.

    Returns
    -------
    list : Selected feature names (excludes target_col).
    pd.DataFrame : Reduced dataframe with selected features + target_col.
    """

    # ------------------------------------------------------------------
    # Build feature list — exclude target from ALL filtering
    # ------------------------------------------------------------------
    feature_cols = [c for c in df.columns if c != target_col]
    print(f"\n  Total features entering selection : {len(feature_cols)}")

    # ------------------------------------------------------------------
    # STEP 1 — Correlation filter
    # Drop one feature from any pair where correlation > corr_threshold.
    # Keep the first encountered, drop the second.
    # ------------------------------------------------------------------
    corr_matrix  = df[feature_cols].corr().abs()
    to_drop_corr = set()

    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if corr_matrix.iloc[i, j] > corr_threshold:
                col_to_drop  = corr_matrix.columns[i]
                kept_instead = corr_matrix.columns[j]
                if col_to_drop not in to_drop_corr:
                    to_drop_corr.add(col_to_drop)
                    print(f"  [CORR DROP]  '{col_to_drop}'"
                          f"  →  corr={corr_matrix.iloc[i,j]:.4f}"
                          f"  with '{kept_instead}'")

    print(f"\n  Dropped (high correlation) : {len(to_drop_corr)}")
    print(f"  Features                   : "
          f"{to_drop_corr if to_drop_corr else 'none'}")

    remaining_cols = [c for c in feature_cols if c not in to_drop_corr]

    # ------------------------------------------------------------------
    # STEP 2 — Hard floor variance filter
    # Only drop features that are genuinely near-constant across all rows.
    # A hard floor of 0.001 catches truly useless columns (e.g. a column
    # that is the same value for 99.99% of rows) without being misled
    # by scale differences between features.
    # ------------------------------------------------------------------
    HARD_FLOOR = 0.001  # any feature with variance below this is near-constant

    to_drop_var = []
    for col in remaining_cols:
        v = df[col].var()
        if v < HARD_FLOOR:
            to_drop_var.append(col)
            print(f"  [VAR  DROP]  '{col}'"
                  f"  →  variance={v:.8f} < hard floor {HARD_FLOOR}")

    print(f"\n  Dropped (near-constant)    : {len(to_drop_var)}")
    print(f"  Features                   : "
          f"{to_drop_var if to_drop_var else 'none'}")

    # ------------------------------------------------------------------
    # STEP 3 — Build final output
    # ------------------------------------------------------------------
    all_dropped = to_drop_corr.union(set(to_drop_var))
    all_dropped.discard(target_col)

    df_reduced        = df.drop(columns=list(all_dropped))
    selected_features = [c for c in df_reduced.columns if c != target_col]

    return selected_features, df_reduced