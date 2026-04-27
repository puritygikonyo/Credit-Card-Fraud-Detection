"""
Tests for feature engineering pipeline.
Verifies column count, NaN absence, and value ranges.
"""

import pytest
import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def raw_dataframe():
    """
    Minimal raw input — what your pipeline receives before feature engineering.
    Contains only the original columns, no engineered features yet.
    """
    np.random.seed(42)
    n = 300

    return pd.DataFrame({
        "Time":   np.arange(n, dtype=float),
        "Amount": np.abs(np.random.normal(88, 110, n)).clip(0.01, 5000),
        "Class":  np.where(np.random.rand(n) < 0.02, 1, 0),
        "Hour":   np.random.randint(0, 24, n),
        **{f"V{i}": np.random.normal(0, 1, n) for i in range(1, 29)},
    })


@pytest.fixture
def edge_case_dataframe():
    """
    Edge cases — zero amounts, midnight transactions, single row.
    Feature engineering must handle these without crashing.
    """
    return pd.DataFrame({
        "Time":   [0.0, 86399.0, 43200.0],
        "Amount": [0.01, 0.0, 9999.99],     # near-zero and large amount
        "Class":  [1, 0, 0],
        "Hour":   [0, 23, 12],              # midnight, last hour, noon
        **{f"V{i}": [0.0, 0.0, 0.0] for i in range(1, 29)},
    })


# ─────────────────────────────────────────────────────────────────────────────
#  FEATURE ENGINEERING FUNCTION
#  In production this lives in src/features.py — defined here for portability.
# ─────────────────────────────────────────────────────────────────────────────

# Expected columns after engineering — update this list if you add features
ORIGINAL_COLS    = ["Time", "Amount", "Class", "Hour"] + [f"V{i}" for i in range(1, 29)]
ENGINEERED_COLS  = ["Amount_log", "Amount_zscore", "Hour_sin", "Hour_cos", "Is_night"]
EXPECTED_COLS    = ORIGINAL_COLS + ENGINEERED_COLS
EXPECTED_N_COLS  = len(EXPECTED_COLS)   # 37 total


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies all feature transformations.
    Returns a new DataFrame — never modifies the input in place.
    """
    df = df.copy()

    # ── Numerical transforms on Amount ────────────────────────────────────────
    df["Amount_log"]    = np.log1p(df["Amount"])
    mean_amt            = df["Amount"].mean()
    std_amt             = df["Amount"].std()
    df["Amount_zscore"] = (df["Amount"] - mean_amt) / std_amt.clip(lower=1e-8)

    # ── Cyclical encoding for Hour ────────────────────────────────────────────
    # sin/cos encoding preserves the circular nature of time
    # (hour 23 and hour 0 are adjacent, not far apart)
    df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)

    # ── Binary flag for night-time transactions ───────────────────────────────
    df["Is_night"] = ((df["Hour"] >= 22) | (df["Hour"] <= 5)).astype(int)

    return df


# ─────────────────────────────────────────────────────────────────────────────
#  TESTS — COLUMN COUNT
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureColumnCount:
    """Feature engineering must produce exactly the right number of columns."""

    def test_expected_number_of_columns(self, raw_dataframe):
        """Output must have exactly EXPECTED_N_COLS columns."""
        result = engineer_features(raw_dataframe)
        assert result.shape[1] == EXPECTED_N_COLS, \
            f"Expected {EXPECTED_N_COLS} columns, got {result.shape[1]}"

    def test_all_engineered_columns_present(self, raw_dataframe):
        """Every engineered column must exist in the output."""
        result = engineer_features(raw_dataframe)
        for col in ENGINEERED_COLS:
            assert col in result.columns, \
                f"Engineered column '{col}' missing from output"

    def test_original_columns_preserved(self, raw_dataframe):
        """Feature engineering must not drop any original columns."""
        result = engineer_features(raw_dataframe)
        for col in ORIGINAL_COLS:
            assert col in result.columns, \
                f"Original column '{col}' was dropped during feature engineering"

    def test_row_count_unchanged(self, raw_dataframe):
        """Feature engineering must not add or remove rows."""
        result = engineer_features(raw_dataframe)
        assert len(result) == len(raw_dataframe), \
            f"Row count changed: {len(raw_dataframe)} → {len(result)}"

    def test_input_not_modified(self, raw_dataframe):
        """Engineer features must not mutate the original DataFrame."""
        original_cols = set(raw_dataframe.columns)
        engineer_features(raw_dataframe)
        assert set(raw_dataframe.columns) == original_cols, \
            "engineer_features modified the input DataFrame in place"


# ─────────────────────────────────────────────────────────────────────────────
#  TESTS — NO NaN VALUES
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureNoNaNs:
    """No NaN values should appear in engineered features."""

    def test_no_nans_in_amount_log(self, raw_dataframe):
        result = engineer_features(raw_dataframe)
        assert result["Amount_log"].isnull().sum() == 0, \
            "Amount_log contains NaN values"

    def test_no_nans_in_amount_zscore(self, raw_dataframe):
        result = engineer_features(raw_dataframe)
        assert result["Amount_zscore"].isnull().sum() == 0, \
            "Amount_zscore contains NaN values"

    def test_no_nans_in_hour_features(self, raw_dataframe):
        result = engineer_features(raw_dataframe)
        for col in ["Hour_sin", "Hour_cos"]:
            assert result[col].isnull().sum() == 0, \
                f"{col} contains NaN values"

    def test_no_nans_in_any_engineered_column(self, raw_dataframe):
        """Single sweep — no engineered column should have any NaN."""
        result = engineer_features(raw_dataframe)
        for col in ENGINEERED_COLS:
            nan_count = result[col].isnull().sum()
            assert nan_count == 0, \
                f"Column '{col}' has {nan_count} NaN values"

    def test_no_nans_on_edge_cases(self, edge_case_dataframe):
        """NaN check must hold even for zero amounts and boundary hours."""
        result = engineer_features(edge_case_dataframe)
        for col in ENGINEERED_COLS:
            assert result[col].isnull().sum() == 0, \
                f"Column '{col}' has NaN on edge case input"


# ─────────────────────────────────────────────────────────────────────────────
#  TESTS — VALUE RANGES
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureValueRanges:
    """Engineered features must stay within mathematically correct bounds."""

    def test_amount_log_non_negative(self, raw_dataframe):
        """log1p(x) >= 0 for all x >= 0."""
        result = engineer_features(raw_dataframe)
        assert (result["Amount_log"] >= 0).all(), \
            "Amount_log contains negative values (implies negative Amount input)"

    def test_hour_sin_within_bounds(self, raw_dataframe):
        """sin values must be in [-1, 1]."""
        result = engineer_features(raw_dataframe)
        assert result["Hour_sin"].between(-1, 1).all(), \
            "Hour_sin has values outside [-1, 1]"

    def test_hour_cos_within_bounds(self, raw_dataframe):
        """cos values must be in [-1, 1]."""
        result = engineer_features(raw_dataframe)
        assert result["Hour_cos"].between(-1, 1).all(), \
            "Hour_cos has values outside [-1, 1]"

    def test_is_night_is_binary(self, raw_dataframe):
        """Is_night must contain only 0 and 1."""
        result = engineer_features(raw_dataframe)
        assert result["Is_night"].isin([0, 1]).all(), \
            "Is_night contains values other than 0 and 1"

    def test_night_hours_flagged_correctly(self, raw_dataframe):
        """Hours 22–23 and 0–5 must all have Is_night == 1."""
        result = engineer_features(raw_dataframe)
        night_hours = result[result["Hour"].isin(list(range(0, 6)) + [22, 23])]
        assert (night_hours["Is_night"] == 1).all(), \
            "Night hours (22-23, 0-5) not correctly flagged as Is_night=1"

    def test_day_hours_not_flagged_as_night(self, raw_dataframe):
        """Hours 6–21 must have Is_night == 0."""
        result = engineer_features(raw_dataframe)
        day_hours = result[result["Hour"].between(6, 21)]
        assert (day_hours["Is_night"] == 0).all(), \
            "Day hours (6-21) incorrectly flagged as Is_night=1"

    def test_amount_zscore_has_zero_mean(self, raw_dataframe):
        """Z-score normalisation must produce a near-zero mean."""
        result = engineer_features(raw_dataframe)
        assert abs(result["Amount_zscore"].mean()) < 0.01, \
            f"Amount_zscore mean is {result['Amount_zscore'].mean():.4f}, expected ~0"

    def test_amount_log_monotonic_with_amount(self, raw_dataframe):
        """Larger amounts must produce larger log values."""
        result  = engineer_features(raw_dataframe)
        corr    = result[["Amount", "Amount_log"]].corr().iloc[0, 1]
        assert corr > 0.99, \
            f"Amount_log not monotonically increasing with Amount (corr={corr:.4f})"