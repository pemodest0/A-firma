from __future__ import annotations

import itertools

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import StandardScaler

from explainability import ExplainabilityHelper


DEFAULT_MODEL_PARAMS = {
    "gbr": {
        "learning_rate": 0.03,
        "n_estimators": 500,
        "max_depth": 3,
        "subsample": 0.7,
        "validation_fraction": 0.1,
        "n_iter_no_change": 20,
        "tol": 1e-4,
    },
    "rf": {
        "n_estimators": 300,
        "max_depth": None,
        "n_jobs": -1,
    },
}

def _make_regressor(model_type: str, params: Optional[Dict[str, Any]], random_state: int) -> object:
    params = params or {}
    defaults = DEFAULT_MODEL_PARAMS.get(model_type, {}).copy()
    defaults.update(params)
    if model_type == "gbr":
        return GradientBoostingRegressor(random_state=random_state, **defaults)
    if model_type == "rf":
        return RandomForestRegressor(random_state=random_state, **defaults)
    raise ValueError("Unsupported model type. Use 'gbr' or 'rf'.")

def time_series_grid_search(
    dataset: ResidualDataset,
    param_grid: Dict[str, Sequence[object]],
    model_type: str = "gbr",
    n_splits: int = 3,
) -> Tuple[Dict[str, Any], float]:
    X = dataset.features.values
    y = dataset.target.values
    if X.shape[0] < n_splits + 2:
        raise ValueError("Not enough samples for TimeSeriesSplit.")
    keys = list(param_grid.keys())
    candidates = [dict(zip(keys, values)) for values in itertools.product(*param_grid.values())]
    best_params: Optional[Dict[str, Any]] = None
    best_score = float("inf")
    splitter = TimeSeriesSplit(n_splits=n_splits)
    for params in candidates:
        scores: List[float] = []
        for train_idx, val_idx in splitter.split(X):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_val = scaler.transform(X[val_idx])
            model = _make_regressor(model_type, params, random_state=42)
            model.fit(X_train, y[train_idx])
            pred = model.predict(X_val)
            score = mean_absolute_error(y[val_idx], pred)
            scores.append(score)
        mean_score = float(np.mean(scores)) if scores else float("inf")
        if mean_score < best_score:
            best_score = mean_score
            best_params = params
    if best_params is None:
        raise RuntimeError("Grid search did not evaluate any parameter set.")
    return best_params, best_score

DEFAULT_FEATURES = [
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
    "macd",
    "macd_signal",
    "macd_hist",
    "ppo",
    "cci",
    "trix",
    "kama",
    "williams_r",
    "tsi",
    "rsi_sma_diff",
    "price_sma_ratio",
    "volat_ratio_change",
    "sma_10",
    "sma_20",
    "ema_10",
    "rsi_14",
    "bollinger_bandwidth",
    "momentum_10",
    "rolling_max_20",
    "rolling_min_20",
    "price_zscore_20",
    "rolling_alpha_mean_5",
    "rolling_entropy_mean_5",
    "rolling_vol_ratio_mean_5",
    "expected_return_lag1",
    "expected_return_lag2",
    "vol_ratio_lag1",
    "vol_ratio_lag2",
    "noise_used_lag1",
    "quantum_grover_expected_return",
    "quantum_grover_price_pred",
    "quantum_grover_delta_price",
    "quantum_hadamard_expected_return",
    "quantum_hadamard_price_pred",
    "quantum_hadamard_delta_price",
]


@dataclass
class ResidualDataset:
    frame: pd.DataFrame
    features: pd.DataFrame
    target: pd.Series
    target_scaling: str = "points"


def load_metrics(path: Path, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df[df["mode"] == "classical"].copy()
    if df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_convert("UTC").dt.tz_localize(None)
    if start is not None:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["date"] <= pd.Timestamp(end)]
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def prepare_residual_dataset(
    df: pd.DataFrame,
    feature_cols: Sequence[str] = DEFAULT_FEATURES,
    residual_column: str = "price_residual",
    target_scaling: str = "points",
) -> ResidualDataset:
    frame = df.copy()
    if "price_residual" not in frame.columns:
        frame["price_residual"] = frame["price_real"] - frame["price_pred"]
    for col in feature_cols:
        if col not in frame.columns:
            frame[col] = np.nan
    features = frame[list(feature_cols)].copy()
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.ffill().bfill().fillna(0.0)
    if target_scaling == "pct":
        denom = frame["price_real"].abs().clip(lower=1e-6)
        frame["price_residual_pct"] = frame[residual_column] / denom
        target = frame["price_residual_pct"].astype(float)
    elif target_scaling == "points":
        target = frame[residual_column].astype(float)
    else:
        raise ValueError(f"target_scaling inválido: {target_scaling}")
    return ResidualDataset(frame=frame, features=features, target=target, target_scaling=target_scaling)


def train_residual_model(
    dataset: ResidualDataset,
    train_mask: Optional[pd.Series] = None,
    model: str = "gbr",
    test_size: float = 0.2,
    random_state: int = 42,
    model_params: Optional[Dict[str, Any]] = None,
) -> Tuple[object, StandardScaler, Dict[str, float]]:
    X = dataset.features.values
    y = dataset.target.values

    if train_mask is not None:
        mask_array = np.asarray(train_mask)
        X_train, X_test = X[mask_array], X[~mask_array]
        y_train, y_test = y[mask_array], y[~mask_array]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=False
        )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    reg = _make_regressor(model, model_params, random_state=random_state)
    reg.fit(X_train_scaled, y_train)
    pred_train = reg.predict(X_train_scaled)
    pred_test = reg.predict(X_test_scaled)

    metrics = {
        "train_mae": float(mean_absolute_error(y_train, pred_train)),
        "test_mae": float(mean_absolute_error(y_test, pred_test)),
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, pred_train))),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred_test))),
    }
    return reg, scaler, metrics


def apply_residual_model(
    dataset: ResidualDataset,
    reg: object,
    scaler: StandardScaler,
    base_column: str = "price_pred",
    residual_column: str = "price_residual",
    adjusted_column: str = "price_pred_adjusted",
    explainability_output: Optional[Path] = None,
    calibration_mask: Optional[Sequence[bool]] = None,
    calibrate: bool = True,
) -> pd.DataFrame:
    X_scaled = scaler.transform(dataset.features.values)
    residual_pred = reg.predict(X_scaled)
    residual_pred_original = residual_pred.copy()
    frame = dataset.frame.copy()

    calibration_info: Optional[Dict[str, float]] = None
    if calibrate and calibration_mask is not None:
        mask = np.asarray(calibration_mask, dtype=bool)
        if mask.shape[0] != residual_pred.shape[0]:
            raise ValueError("calibration_mask size mismatch with dataset.")
        finite_mask = np.isfinite(residual_pred)
        mask = mask & finite_mask
        if mask.any():
            residual_slice = residual_pred_original[mask].astype(float)
            spread = np.nanstd(residual_slice)
            if spread > 1e-12:
                base_prices = frame.loc[mask, base_column].astype(float).values
                actual_prices = frame.loc[mask, "price_real"].astype(float).values

                mae_base = float(np.mean(np.abs(base_prices - actual_prices)))
                if dataset.target_scaling == "pct":
                    price_scale = frame.loc[mask, "price_real"].abs().clip(lower=1e-6).values
                    residual_slice_points = residual_slice * price_scale
                else:
                    residual_slice_points = residual_slice

                mae_full = float(np.mean(np.abs((base_prices + residual_slice_points) - actual_prices)))

                def _evaluate_gamma(gamma_values: np.ndarray) -> Tuple[float, float]:
                    best_gamma_local = 1.0
                    best_mae_local = mae_full
                    for gamma in gamma_values:
                        preds = base_prices + gamma * residual_slice_points
                        mae = float(np.mean(np.abs(preds - actual_prices)))
                        if (mae + 1e-9) < best_mae_local or (
                            abs(mae - best_mae_local) <= 1e-9 and gamma < best_gamma_local
                        ):
                            best_mae_local = mae
                            best_gamma_local = float(gamma)
                    return best_gamma_local, best_mae_local

                coarse_grid = np.linspace(0.0, 1.0, 21)
                best_gamma, best_mae = _evaluate_gamma(coarse_grid)

                fine_start = max(0.0, best_gamma - 0.1)
                fine_end = min(1.0, best_gamma + 0.1)
                fine_grid = np.linspace(fine_start, fine_end, 41)
                best_gamma, best_mae = _evaluate_gamma(fine_grid)

                if np.isfinite(best_mae) and best_mae <= mae_full:
                    residual_pred = best_gamma * residual_pred_original
                    calibration_info = {
                        "gamma": float(best_gamma),
                        "mae_base": mae_base,
                        "mae_residual": mae_full,
                        "mae_calibrated": best_mae,
                        "samples": int(mask.sum()),
                    }
                else:
                    residual_pred = residual_pred_original.copy()

    if dataset.target_scaling == "pct":
        price_scale_full = frame["price_real"].abs().clip(lower=1e-6).values
        residual_pred_points = residual_pred * price_scale_full
    else:
        residual_pred_points = residual_pred

    frame[adjusted_column] = frame[base_column] + residual_pred_points
    frame[f"{residual_column}_pred"] = residual_pred_points
    if dataset.target_scaling == "pct":
        frame[f"{residual_column}_pct_pred"] = residual_pred
    if calibration_info is not None:
        frame.attrs["residual_calibration"] = calibration_info
    elif calibrate:
        frame.attrs["residual_calibration"] = None

    if explainability_output is not None:
        helper = ExplainabilityHelper(dataset.features.columns)
        helper.export_feature_importance(reg, explainability_output / "feature_importance.csv")

    return frame


def evaluate_adjusted_predictions(
    frame: pd.DataFrame,
    date_col: str = "date",
    base_column: str = "price_pred",
    adjusted_column: str = "price_pred_adjusted",
    actual_column: str = "price_real",
) -> Dict[str, float]:
    base_error = (frame[base_column] - frame[actual_column]).abs()
    adj_error = (frame[adjusted_column] - frame[actual_column]).abs()
    delta = frame[adjusted_column] - frame[actual_column]
    return {
        "mae_base": float(base_error.mean()),
        "mae_adjusted": float(adj_error.mean()),
        "median_base": float(base_error.median()),
        "median_adjusted": float(adj_error.median()),
        "max_over_adj": float(delta.max()),
        "max_under_adj": float(delta.min()),
    }
