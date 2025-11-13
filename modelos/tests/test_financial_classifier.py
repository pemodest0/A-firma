import numpy as np
import pandas as pd

from src.financial_classifier import (
    DatasetSlices,
    evaluate_direction_classifier,
    prepare_features,
    split_train_test,
    train_direction_classifier,
)


def _build_dummy_dataframe(rows: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    dates = pd.date_range("2020-01-01", periods=rows, freq="B")
    base = pd.DataFrame(
        {
            "symbol": np.where(np.arange(rows) % 2 == 0, "SPY", "IBOV"),
            "date": dates,
            "price_real": rng.normal(100, 5, size=rows),
            "price_pred": rng.normal(100, 5, size=rows),
            "phase": rng.choice(["difusiva", "transicao", "coerente"], size=rows),
            "expected_return": rng.normal(0, 0.01, size=rows),
            "actual_return": rng.normal(0, 0.01, size=rows),
            "alpha": rng.normal(-0.03, 0.02, size=rows),
            "entropy": rng.normal(3.5, 0.1, size=rows),
            "vol_realized_short": rng.uniform(0.05, 0.2, size=rows),
            "vol_realized_long": rng.uniform(0.05, 0.2, size=rows),
            "vol_ratio": rng.uniform(0.4, 1.6, size=rows),
            "vol_ewm_30": rng.uniform(0.05, 0.2, size=rows),
            "abs_return_short": rng.uniform(0.001, 0.02, size=rows),
            "drawdown_long": rng.uniform(-0.2, 0.0, size=rows),
            "vol_of_vol": rng.uniform(0.01, 0.05, size=rows),
            "noise_used": rng.uniform(0.01, 0.08, size=rows),
            "direction_match": rng.integers(0, 2, size=rows),
        }
    )
    return base


def test_prepare_features_generates_expected_shapes():
    df = _build_dummy_dataframe()
    features = [
        "expected_return",
        "alpha",
        "entropy",
        "vol_realized_short",
        "vol_realized_long",
        "vol_ratio",
        "vol_ewm_30",
        "abs_return_short",
        "drawdown_long",
        "vol_of_vol",
        "noise_used",
    ]
    dataset: DatasetSlices = prepare_features(df, features)
    assert dataset.features.shape == (df.shape[0], len(features))
    assert dataset.target.shape[0] == df.shape[0]
    assert not dataset.features.isnull().any().any()


def test_training_pipeline_runs():
    df = _build_dummy_dataframe()
    features = [
        "expected_return",
        "alpha",
        "entropy",
        "vol_realized_short",
        "vol_realized_long",
        "vol_ratio",
        "vol_ewm_30",
        "abs_return_short",
        "drawdown_long",
        "vol_of_vol",
        "noise_used",
    ]
    dataset = prepare_features(df, features)
    X_train, X_test, y_train, y_test, train_idx, test_idx = split_train_test(
        dataset, df["date"], test_start=pd.Timestamp("2020-04-30")
    )
    assert X_train.shape[0] == train_idx.shape[0]
    assert X_test.shape[0] == test_idx.shape[0]
    clf, scaler = train_direction_classifier(
        X_train, y_train, model_type="rf", n_estimators=50, max_depth=5
    )
    metrics = evaluate_direction_classifier(clf, scaler, X_test, y_test, features)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert "confusion_matrix" in metrics
    assert len(metrics["feature_importances"]) == len(features)
