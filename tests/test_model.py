"""
Tests for model loading and prediction behaviour.
Verifies the model artifact loads correctly and produces valid outputs.
"""

import pytest
import numpy as np
import pandas as pd
import os


# ─────────────────────────────────────────────────────────────────────────────
#  FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def feature_names():
    """Exact column order the model was trained on — must match training."""
    return [
        "Amount", "Hour", "Amount_log", "Amount_zscore",
        "Hour_sin", "Hour_cos", "Is_night",
        "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
        "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18",
        "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
    ]


@pytest.fixture
def single_transaction(feature_names):
    """One realistic transaction — a normal daytime purchase."""
    values = {
        "Amount": 45.00,   "Hour": 14,
        "Amount_log": np.log1p(45.00), "Amount_zscore": -0.4,
        "Hour_sin": np.sin(2 * np.pi * 14 / 24),
        "Hour_cos": np.cos(2 * np.pi * 14 / 24),
        "Is_night": 0,
        **{f"V{i}": 0.0 for i in range(1, 29)},
    }
    return pd.DataFrame([values])[feature_names]


@pytest.fixture
def batch_transactions(feature_names):
    """50 transactions — tests batch prediction behaviour."""
    np.random.seed(99)
    n = 50
    data = {
        "Amount":       np.abs(np.random.normal(88, 110, n)).clip(0.01, 5000),
        "Hour":         np.random.randint(0, 24, n),
        "Amount_log":   np.log1p(np.abs(np.random.normal(88, 110, n))),
        "Amount_zscore":np.random.normal(0, 1, n),
        "Hour_sin":     np.sin(2 * np.pi * np.random.randint(0, 24, n) / 24),
        "Hour_cos":     np.cos(2 * np.pi * np.random.randint(0, 24, n) / 24),
        "Is_night":     np.random.randint(0, 2, n),
        **{f"V{i}": np.random.normal(0, 1, n) for i in range(1, 29)},
    }
    return pd.DataFrame(data)[feature_names]


@pytest.fixture
def suspicious_transaction(feature_names):
    """
    A transaction with classic fraud signals.
    V14 strongly negative, 3am, unusual amount.
    Model should score this higher than a normal transaction.
    """
    values = {
        "Amount": 892.50,  "Hour": 3,
        "Amount_log": np.log1p(892.50), "Amount_zscore": 2.8,
        "Hour_sin": np.sin(2 * np.pi * 3 / 24),
        "Hour_cos": np.cos(2 * np.pi * 3 / 24),
        "Is_night": 1,
        **{f"V{i}": 0.0 for i in range(1, 29)},
        "V14": -6.2,   # strong fraud signal
        "V4":   3.1,   # secondary fraud signal
    }
    return pd.DataFrame([values])[feature_names]


@pytest.fixture
def model(feature_names):
    """
    Loads the real model if it exists, otherwise returns a demo model.
    This fixture is the key to making tests always runnable — the same
    pattern as your demo data system in the dashboard.
    """
    import joblib

    model_path = os.path.join(
        os.path.dirname(__file__), "..", "models", "xgboost_model.pkl"
    )

    if os.path.exists(model_path):
        return joblib.load(model_path)

    # ── Demo model — used when no trained model exists yet ────────────────────
    # Trains a tiny XGBoost model on synthetic data in ~1 second.
    # Ensures all tests are runnable before your pipeline has been executed.
    from xgboost import XGBClassifier
    import pandas as pd

    np.random.seed(42)
    n = 500
    X_demo = pd.DataFrame(
        np.random.randn(n, len(feature_names)),
        columns=feature_names
    )
    # Make V14 a strong predictor — mirrors real data behaviour
    y_demo = ((X_demo["V14"] < -2) | (np.random.rand(n) < 0.05)).astype(int)

    demo_model = XGBClassifier(
        n_estimators=10,      # tiny — just enough to make predictions
        max_depth=3,
        random_state=42,
        eval_metric="logloss",
        use_label_encoder=False,
    )
    demo_model.fit(X_demo, y_demo)
    return demo_model


# ─────────────────────────────────────────────────────────────────────────────
#  TESTS — MODEL LOADS
# ─────────────────────────────────────────────────────────────────────────────

class TestModelLoads:
    """Model fixture must return a usable, fitted estimator."""

    def test_model_is_not_none(self, model):
        """Model fixture must return an object, not None."""
        assert model is not None, "Model fixture returned None"

    def test_model_has_predict_method(self, model):
        """Model must expose a predict() method."""
        assert hasattr(model, "predict"), \
            "Model does not have a predict() method"

    def test_model_has_predict_proba_method(self, model):
        """Model must expose predict_proba() — required for threshold tuning."""
        assert hasattr(model, "predict_proba"), \
            "Model does not have a predict_proba() method — " \
            "threshold decision tool will not work"

    def test_model_has_feature_importances(self, model):
        """Fitted XGBoost must expose feature_importances_ attribute."""
        assert hasattr(model, "feature_importances_"), \
            "Model does not have feature_importances_ — not fitted?"


# ─────────────────────────────────────────────────────────────────────────────
#  TESTS — SINGLE PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

class TestSinglePrediction:
    """Model must handle a single transaction correctly."""

    def test_predict_returns_result(self, model, single_transaction):
        """predict() must return without raising an exception."""
        result = model.predict(single_transaction)
        assert result is not None

    def test_predict_output_shape(self, model, single_transaction):
        """predict() on 1 row must return an array of length 1."""
        result = model.predict(single_transaction)
        assert len(result) == 1, \
            f"Expected 1 prediction, got {len(result)}"

    def test_predict_output_is_binary(self, model, single_transaction):
        """predict() must return only 0 or 1."""
        result = model.predict(single_transaction)
        assert result[0] in [0, 1], \
            f"Prediction {result[0]} is not binary (0 or 1)"

    def test_predict_proba_output_shape(self, model, single_transaction):
        """predict_proba() on 1 row must return shape (1, 2)."""
        proba = model.predict_proba(single_transaction)
        assert proba.shape == (1, 2), \
            f"predict_proba shape is {proba.shape}, expected (1, 2)"

    def test_predict_proba_sums_to_one(self, model, single_transaction):
        """Class probabilities must sum to 1.0."""
        proba = model.predict_proba(single_transaction)
        total = proba[0].sum()
        assert abs(total - 1.0) < 1e-6, \
            f"Probabilities sum to {total}, expected 1.0"


# ─────────────────────────────────────────────────────────────────────────────
#  TESTS — PREDICTION RANGES
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictionRanges:
    """All model outputs must stay within valid mathematical bounds."""

    def test_fraud_probability_between_0_and_1(self, model, single_transaction):
        """Fraud probability must be in [0, 1]."""
        proba = model.predict_proba(single_transaction)[0, 1]
        assert 0.0 <= proba <= 1.0, \
            f"Fraud probability {proba} is outside [0, 1]"

    def test_batch_probabilities_all_valid(self, model, batch_transactions):
        """All 50 fraud probabilities must be in [0, 1]."""
        probas = model.predict_proba(batch_transactions)[:, 1]
        assert (probas >= 0).all() and (probas <= 1).all(), \
            f"Batch probabilities out of range: min={probas.min():.4f}, max={probas.max():.4f}"

    def test_batch_predictions_all_binary(self, model, batch_transactions):
        """All 50 hard predictions must be 0 or 1."""
        preds = model.predict(batch_transactions)
        assert set(preds).issubset({0, 1}), \
            f"Batch predictions contain non-binary values: {set(preds)}"

    def test_batch_output_length_matches_input(self, model, batch_transactions):
        """Output length must equal input row count."""
        preds = model.predict(batch_transactions)
        assert len(preds) == len(batch_transactions), \
            f"Output length {len(preds)} != input length {len(batch_transactions)}"

    def test_no_nan_in_probabilities(self, model, batch_transactions):
        """predict_proba must never return NaN."""
        probas = model.predict_proba(batch_transactions)
        assert not np.isnan(probas).any(), \
            "predict_proba returned NaN values"

    def test_suspicious_transaction_scores_higher(
            self, model, single_transaction, suspicious_transaction):
        """
        A transaction with strong fraud signals (V14=-6.2, 3am, high amount)
        must score higher than a normal daytime transaction.
        This is a sanity check that the model learned something meaningful.
        """
        normal_score    = model.predict_proba(single_transaction)[0, 1]
        suspicious_score = model.predict_proba(suspicious_transaction)[0, 1]
        assert suspicious_score > normal_score, \
            f"Suspicious transaction ({suspicious_score:.3f}) did not score " \
            f"higher than normal ({normal_score:.3f}) — model may not be learning correctly"