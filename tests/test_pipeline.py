import pytest
import pandas as pd
import joblib
from sklearn.utils.validation import check_is_fitted
import numpy as np
from src.config import NUMERICAL_COLS, CATEGORICAL_COLS, TARGET_COL
from src.pipeline import build_pipeline
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture
def sample_data():
    """Sample dataframe matching the expected input schema"""
    return pd.DataFrame(
        {
            "Amount": [100, 200, 300, 400, 500],
            "Value": [1, 2, 3, 4, 5],
            "CustomerId": ["C1", "C1", "C2", "C2", "C3"],
            "ProductCategory": ["A", "B", "A", "C", "B"],
            "CountryCode": [256, 256, 100, 100, 840],
            "CurrencyCode": ["UGX", "UGX", "USD", "USD", "EUR"],
            "TransactionStartTime": [
                "2023-01-01 10:00",
                "2023-01-02 11:00",
                "2023-01-03 12:00",
                "2023-01-04 13:00",
                "2023-01-05 14:00",
            ],
            "FraudResult": [0, 0, 1, 0, 1],
        }
    )


@pytest.fixture
def trained_pipeline(sample_data):
    """Pipeline fitted on sample data"""
    pipeline = build_pipeline()
    X = sample_data.drop(columns=[TARGET_COL])
    pipeline.fit(X)
    return pipeline


def test_pipeline_structure(trained_pipeline):
    """Verify pipeline contains expected steps"""
    steps = list(trained_pipeline.named_steps.keys())
    assert steps == ["feature_engineering", "preprocessor"]
    assert hasattr(trained_pipeline, "transform")


def test_feature_engineering_output(trained_pipeline, sample_data):
    """Test intermediate feature engineering outputs"""
    X = sample_data.drop(columns=[TARGET_COL])
    transformed = trained_pipeline.named_steps["feature_engineering"].transform(X)

    # Verify aggregate features exist
    assert "CustomerId_TransactionCount" in transformed.columns
    assert "ProductCategory_AvgAmount" in transformed.columns

    # Verify datetime features
    assert "TransactionHour" in transformed.columns
    assert transformed["TransactionHour"].between(0, 23).all()


def test_preprocessor_fitted(trained_pipeline):
    """Verify preprocessing components are properly fitted"""
    preprocessor = trained_pipeline.named_steps["preprocessor"]
    check_is_fitted(preprocessor)

    # Verify one-hot encoder categories
    ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    assert len(ohe.categories_) == len(CATEGORICAL_COLS)


def test_pipeline_output_shape(trained_pipeline, sample_data):
    """Test final output shape and consistency"""
    X = sample_data.drop(columns=[TARGET_COL])
    transformed = trained_pipeline.transform(X)

    # Verify no data loss
    assert transformed.shape[0] == sample_data.shape[0]

    # Verify numeric features are scaled
    assert transformed[:, : len(NUMERICAL_COLS)].std(axis=0).round(2) == pytest.approx(
        1.0
    )


def test_handles_missing_values(trained_pipeline):
    """Test robustness to missing data"""
    import numpy as np

    X_missing = pd.DataFrame(
        {
            "Amount": [100, np.nan, 300],
            "Value": [np.nan, 2, 3],
            "CustomerId": ["C1", "C1", np.nan],
            "ProductCategory": ["A", np.nan, "B"],
            "CountryCode": [256, 100, np.nan],
            "CurrencyCode": ["UGX", np.nan, "USD"],
            "TransactionStartTime": ["2023-01-01 10:00", np.nan, "2023-01-03 12:00"],
        }
    )

    # Should not raise exceptions
    transformed = trained_pipeline.transform(X_missing)
    assert not np.isnan(transformed).any()


def test_model_saving_loading(trained_pipeline, tmp_path):
    """Test serialization round-trip"""
    save_path = tmp_path / "pipeline.pkl"
    joblib.dump(trained_pipeline, save_path)
    loaded = joblib.load(save_path)

    # Verify identical output
    X_sample = pd.DataFrame(
        {
            "Amount": [150],
            "Value": [2],
            "CustomerId": ["C1"],
            "ProductCategory": ["A"],
            "CountryCode": [256],
            "CurrencyCode": ["UGX"],
            "TransactionStartTime": ["2023-01-01 10:00"],
        }
    )

    original_output = trained_pipeline.transform(X_sample)
    loaded_output = loaded.transform(X_sample)
    assert np.allclose(original_output, loaded_output)


def test_handles_new_categories(trained_pipeline):
    """Test behavior with unseen categories"""
    X_new = pd.DataFrame(
        {
            "Amount": [100],
            "Value": [1],
            "CustomerId": ["NEW_CUSTOMER"],
            "ProductCategory": ["NEW_CATEGORY"],
            "CountryCode": [999],
            "CurrencyCode": ["NEW_CURRENCY"],
            "TransactionStartTime": ["2023-01-01 10:00"],
        }
    )

    # Should not raise errors (due to handle_unknown='ignore')
    transformed = trained_pipeline.transform(X_new)
    assert transformed.shape[0] == 1
