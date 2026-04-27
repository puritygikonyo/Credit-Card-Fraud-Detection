"""
Tests for data quality gates.
Verifies that the quality checker passes clean data and catches broken data.
"""

import pytest
import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  FIXTURES
#  pytest fixtures are reusable setup functions. Any test that lists a fixture
#  name as a parameter automatically receives the return value — no manual
#  setup/teardown needed.
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def clean_dataframe():
    """
    A realistic, well-formed transaction dataset.
    Mirrors the schema your pipeline produces — same columns, valid ranges.
    Used by tests that expect the quality gate to PASS.
    """
    np.random.seed(42)
    n = 200

    return pd.DataFrame({
        "Time":   np.arange(n, dtype=float),
        "Amount": np.abs(np.random.normal(88, 110, n)).clip(0.01, 5000),
        "Class":  np.where(np.random.rand(n) < 0.02, 1, 0),  # ~2% fraud
        **{f"V{i}": np.random.normal(0, 1, n) for i in range(1, 29)},
    })


@pytest.fixture
def broken_dataframe():
    """
    A deliberately broken dataset — missing critical columns, NaNs everywhere.
    Used by tests that expect the quality gate to FAIL/raise.
    """
    return pd.DataFrame({
        "Time":   [1, 2, None, None],        # NaNs in time
        "Amount": [-99, None, 0, -1],         # negatives and nulls
        # Class column missing entirely
        # All V-features missing
    })


@pytest.fixture
def empty_dataframe():
    """Zero rows — edge case the quality gate must handle."""
    return pd.DataFrame(columns=["Time", "Amount", "Class"] +
                        [f"V{i}" for i in range(1, 29)])


# ─────────────────────────────────────────────────────────────────────────────
#  QUALITY GATE FUNCTION
#  In a real project this would live in src/data_quality.py and be imported.
#  Defined here so the tests are self-contained and always runnable.
# ─────────────────────────────────────────────────────────────────────────────

def run_quality_gate(df: pd.DataFrame) -> dict:
    """
    Runs a suite of data quality checks and returns a results dict.

    Returns:
        {
            "passed":  bool,
            "errors":  list of str,   # blocking issues
            "warnings": list of str,  # non-blocking notices
        }
    """
    errors   = []
    warnings = []

    required_cols = ["Time", "Amount", "Class"] + [f"V{i}" for i in range(1, 29)]

    # ── Check 1: required columns present ────────────────────────────────────
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")

    # ── Check 2: no empty dataset ─────────────────────────────────────────────
    if len(df) == 0:
        errors.append("Dataset is empty — 0 rows")

    # ── Check 3: no NaN values in critical columns ────────────────────────────
    critical = [c for c in ["Time", "Amount", "Class"] if c in df.columns]
    for col in critical:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            errors.append(f"Column '{col}' has {null_count} NaN values")

    # ── Check 4: Amount must be non-negative ──────────────────────────────────
    if "Amount" in df.columns:
        neg_count = (df["Amount"] < 0).sum()
        if neg_count > 0:
            errors.append(f"Amount has {neg_count} negative values")

    # ── Check 5: Class must be binary (0 or 1 only) ───────────────────────────
    if "Class" in df.columns and len(df) > 0:
        invalid_classes = ~df["Class"].isin([0, 1])
        if invalid_classes.any():
            errors.append("Class column contains values other than 0 and 1")

    # ── Check 6: fraud rate sanity check (warning, not error) ────────────────
    if "Class" in df.columns and len(df) > 0:
        fraud_rate = df["Class"].mean()
        if fraud_rate > 0.20:
            warnings.append(f"Unusually high fraud rate: {fraud_rate:.1%} (expected < 20%)")

    return {
        "passed":   len(errors) == 0,
        "errors":   errors,
        "warnings": warnings,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  TESTS — CLEAN DATA (all should PASS the quality gate)
# ─────────────────────────────────────────────────────────────────────────────

class TestQualityGatePassesOnCleanData:
    """Quality gate must give the green light on well-formed data."""

    def test_gate_passes(self, clean_dataframe):
        """The overall result must be passed=True on clean data."""
        result = run_quality_gate(clean_dataframe)
        assert result["passed"] is True, \
            f"Quality gate failed on clean data. Errors: {result['errors']}"

    def test_no_errors_on_clean_data(self, clean_dataframe):
        """Error list must be empty — no blocking issues."""
        result = run_quality_gate(clean_dataframe)
        assert result["errors"] == [], \
            f"Unexpected errors on clean data: {result['errors']}"

    def test_all_required_columns_present(self, clean_dataframe):
        """Clean fixture must contain every required column."""
        result = run_quality_gate(clean_dataframe)
        column_errors = [e for e in result["errors"] if "Missing" in e]
        assert column_errors == [], \
            f"Column errors on clean data: {column_errors}"

    def test_amount_non_negative(self, clean_dataframe):
        """All Amount values in clean data must be >= 0."""
        assert (clean_dataframe["Amount"] >= 0).all(), \
            "Clean fixture contains negative Amount values"

    def test_class_is_binary(self, clean_dataframe):
        """Class column must contain only 0 and 1."""
        assert clean_dataframe["Class"].isin([0, 1]).all(), \
            "Clean fixture contains non-binary Class values"

    def test_no_nulls_in_critical_columns(self, clean_dataframe):
        """Time, Amount, Class must have zero nulls in clean data."""
        for col in ["Time", "Amount", "Class"]:
            assert clean_dataframe[col].isnull().sum() == 0, \
                f"Clean fixture has NaNs in '{col}'"


# ─────────────────────────────────────────────────────────────────────────────
#  TESTS — BROKEN DATA (all should FAIL the quality gate)
# ─────────────────────────────────────────────────────────────────────────────

class TestQualityGateCatchesBrokenData:
    """Quality gate must catch every class of data problem."""

    def test_gate_fails_on_broken_data(self, broken_dataframe):
        """Overall result must be passed=False on broken data."""
        result = run_quality_gate(broken_dataframe)
        assert result["passed"] is False, \
            "Quality gate incorrectly passed broken data"

    def test_catches_missing_columns(self, broken_dataframe):
        """Must detect that V1–V28 and Class columns are absent."""
        result = run_quality_gate(broken_dataframe)
        column_errors = [e for e in result["errors"] if "Missing" in e]
        assert len(column_errors) > 0, \
            "Quality gate did not catch missing columns"

    def test_catches_negative_amounts(self, broken_dataframe):
        """Must flag negative Amount values as an error."""
        result = run_quality_gate(broken_dataframe)
        amount_errors = [e for e in result["errors"] if "negative" in e.lower()]
        assert len(amount_errors) > 0, \
            "Quality gate did not catch negative Amount values"

    def test_catches_nan_in_amount(self, broken_dataframe):
        """Must detect NaN values in the Amount column."""
        result = run_quality_gate(broken_dataframe)
        nan_errors = [e for e in result["errors"] if "NaN" in e and "Amount" in e]
        assert len(nan_errors) > 0, \
            "Quality gate did not catch NaN values in Amount"

    def test_errors_list_is_not_empty(self, broken_dataframe):
        """Errors list must contain at least one entry for broken data."""
        result = run_quality_gate(broken_dataframe)
        assert len(result["errors"]) > 0, \
            "Errors list was empty for broken data"

    def test_gate_fails_on_empty_dataframe(self, empty_dataframe):
        """An empty DataFrame must fail the quality gate."""
        result = run_quality_gate(empty_dataframe)
        assert result["passed"] is False, \
            "Quality gate incorrectly passed an empty DataFrame"

    def test_catches_empty_dataset_error(self, empty_dataframe):
        """Must produce a specific error message for 0-row datasets."""
        result = run_quality_gate(empty_dataframe)
        empty_errors = [e for e in result["errors"] if "empty" in e.lower()]
        assert len(empty_errors) > 0, \
            "Quality gate did not report an error for empty dataset"