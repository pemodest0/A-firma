import numpy as np
import pandas as pd

from hybrid_forecast import (
    DEFAULT_FEATURES,
    apply_residual_model,
    evaluate_adjusted_predictions,
    prepare_residual_dataset,
    time_series_grid_search,
    train_residual_model,
)


def _dummy_frame(rows: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    dates = pd.date_range("2023-01-01", periods=rows, freq="B")
    frame = pd.DataFrame(
        {
            "date": dates,
            "mode": "classical",
            "price_real": 1000 + rng.normal(0, 20, size=rows).cumsum(),
            "price_pred": 1000 + rng.normal(0, 20, size=rows).cumsum(),
            "expected_return": rng.normal(0, 0.01, size=rows),
            "actual_return": rng.normal(0, 0.01, size=rows),
            "alpha": rng.normal(-0.03, 0.02, size=rows),
            "entropy": rng.normal(3.3, 0.1, size=rows),
            "vol_realized_short": rng.uniform(0.05, 0.2, size=rows),
            "vol_realized_long": rng.uniform(0.05, 0.2, size=rows),
            "vol_ratio": rng.uniform(0.5, 1.5, size=rows),
            "vol_ewm_30": rng.uniform(0.05, 0.2, size=rows),
            "abs_return_short": rng.uniform(0.001, 0.02, size=rows),
            "drawdown_long": rng.uniform(-0.2, 0.0, size=rows),
            "vol_of_vol": rng.uniform(0.01, 0.05, size=rows),
            "noise_used": rng.uniform(0.01, 0.07, size=rows),
        }
    )
    frame["expected_minus_actual"] = frame["expected_return"] - frame["actual_return"]
    frame["error_abs"] = (frame["price_pred"] - frame["price_real"]).abs()
    frame["error_pct_abs"] = frame["error_abs"] / frame["price_real"].clip(lower=1e-6)

    # simple lags
    for col in ["expected_return", "vol_ratio", "noise_used"]:
        frame[f"{col}_lag1"] = frame[col].shift(1).bfill()
        frame[f"{col}_lag2"] = frame[col].shift(2).bfill()

    frame["rolling_alpha_mean_5"] = frame["alpha"].rolling(5, min_periods=1).mean()
    frame["rolling_entropy_mean_5"] = frame["entropy"].rolling(5, min_periods=1).mean()
    frame["rolling_vol_ratio_mean_5"] = frame["vol_ratio"].rolling(5, min_periods=1).mean()
    return frame


def test_residual_training_and_application():
    frame = _dummy_frame()
    dataset = prepare_residual_dataset(frame, DEFAULT_FEATURES)
    # train on first 80 rows, test on rest
    mask = dataset.frame.index < 80
    reg, scaler, metrics = train_residual_model(dataset, train_mask=mask, model="gbr")
    assert metrics["train_mae"] >= 0.0
    adjusted = apply_residual_model(dataset, reg, scaler)
    eval_stats = evaluate_adjusted_predictions(adjusted)
    assert "mae_base" in eval_stats
    assert adjusted.shape[0] == frame.shape[0]


def test_time_series_grid_search_returns_params():
    frame = _dummy_frame()
    dataset = prepare_residual_dataset(frame, DEFAULT_FEATURES)
    grid = {
        "learning_rate": [0.02, 0.05],
        "n_estimators": [100, 150],
    }
    params, score = time_series_grid_search(dataset, grid, model_type="gbr", n_splits=3)
    assert isinstance(params, dict)
    assert "learning_rate" in params
    assert score >= 0.0
