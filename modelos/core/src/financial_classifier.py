from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


RANDOM_SEED = 42


@dataclass
class DatasetSlices:
    features: pd.DataFrame
    target: pd.Series
    meta: pd.DataFrame


def load_financial_metrics(
    paths: Iterable[Path],
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    modes: Sequence[str] = ("classical",),
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Metrics file not found: {path}")
        df = pd.read_csv(path, parse_dates=["date"])
        df = df[df["mode"].isin(modes)].copy()
        if df.empty:
            continue
        tz = df["date"].dt.tz
        if tz is not None:
            df["date"] = df["date"].dt.tz_convert("UTC").dt.tz_localize(None)
        if start is not None:
            df = df[df["date"] >= start]
        if end is not None:
            df = df[df["date"] <= end]
        frames.append(df)
    if not frames:
        raise ValueError("No data available after filtering.")
    return pd.concat(frames, ignore_index=True)


def prepare_features(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str = "direction_match",
) -> DatasetSlices:
    missing = [col for col in feature_columns if col not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {', '.join(missing)}")
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found.")

    features = df[list(feature_columns)].copy()
    target = df[target_column].astype(int)
    meta_cols = ["symbol", "date", "price_real", "price_pred", "phase", "expected_return", "actual_return"]
    meta = df[[col for col in meta_cols if col in df.columns]].copy()
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.ffill().bfill().fillna(0.0)
    return DatasetSlices(features=features, target=target, meta=meta)


def augment_with_derived_features(
    df: pd.DataFrame,
    group_col: str = "symbol",
    date_col: str = "date",
    base_columns: Optional[Sequence[str]] = None,
    lags: Sequence[int] = (1, 2, 5),
) -> pd.DataFrame:
    if base_columns is None:
        base_columns = [
            "expected_return",
            "actual_return",
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

    df = df.sort_values([group_col, date_col]).copy()
    grouped = df.groupby(group_col, group_keys=False)

    for col in base_columns:
        if col not in df.columns:
            continue
        for lag in lags:
            df[f"{col}_lag{lag}"] = grouped[col].shift(lag)

    df["expected_minus_actual"] = df["expected_return"] - df.get("actual_return", 0.0)
    df["error_abs"] = (df["price_pred"] - df["price_real"]).abs()
    if "error_pct" in df.columns:
        df["error_pct_abs"] = df["error_pct"].abs()

    df["rolling_alpha_mean_5"] = grouped["alpha"].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df["rolling_entropy_mean_5"] = grouped["entropy"].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df["rolling_vol_ratio_mean_5"] = grouped["vol_ratio"].transform(lambda x: x.rolling(5, min_periods=1).mean())

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.ffill().bfill()
    return df


def split_train_test(
    dataset: DatasetSlices,
    date_series: Optional[pd.Series],
    test_start: Optional[pd.Timestamp] = None,
    test_size: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = dataset.features.values
    y = dataset.target.values
    if test_start is not None and date_series is not None:
        mask = date_series >= test_start
        X_train, X_test = X[~mask], X[mask]
        y_train, y_test = y[~mask], y[mask]
        train_idx = np.where(~mask)[0]
        test_idx = np.where(mask)[0]
    else:
        X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
            X,
            y,
            np.arange(len(y)),
            test_size=test_size,
            random_state=RANDOM_SEED,
            stratify=y,
        )
    return X_train, X_test, y_train, y_test, train_idx, test_idx


def train_direction_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    model_type: str = "rf",
    n_estimators: int = 300,
    max_depth: Optional[int] = None,
    sample_weight: Optional[np.ndarray] = None,
) -> Tuple[object, StandardScaler]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    if model_type == "rf":
        clf: object = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
    elif model_type == "gb":
        clf = GradientBoostingClassifier(random_state=RANDOM_SEED)
    elif model_type == "logistic":
        clf = LogisticRegression(max_iter=200, class_weight="balanced", random_state=RANDOM_SEED)
    else:
        raise ValueError("model_type must be 'rf', 'gb', or 'logistic'.")

    clf.fit(X_scaled, y_train, sample_weight=sample_weight)
    return clf, scaler


def evaluate_direction_classifier(
    clf: object,
    scaler: StandardScaler,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: Sequence[str],
    *,
    calibrate: bool = False,
    X_train: Optional[np.ndarray] = None,
    y_train: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    X_scaled = scaler.transform(X_test)
    y_pred = clf.predict(X_scaled)
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X_scaled)[:, 1]
    else:
        decision = clf.decision_function(X_scaled)
        proba = 1.0 / (1.0 + np.exp(-decision))

    calibration = None
    if calibrate:
        if X_train is None or y_train is None:
            raise ValueError("X_train e y_train são necessários para calibrar o classificador.")
        X_train_scaled = scaler.transform(X_train)
        if hasattr(clf, "predict_proba"):
            proba_train = clf.predict_proba(X_train_scaled)[:, 1]
        else:
            decision_train = clf.decision_function(X_train_scaled)
            proba_train = 1.0 / (1.0 + np.exp(-decision_train))
        calibrator = LogisticRegression(max_iter=200, class_weight="balanced", random_state=RANDOM_SEED)
        calibrator.fit(proba_train.reshape(-1, 1), y_train)
        proba = calibrator.predict_proba(proba.reshape(-1, 1))[:, 1]
        calibration = {
            "calibrator": "logistic",
            "coeff": calibrator.coef_.ravel().tolist(),
            "intercept": calibrator.intercept_.tolist(),
        }

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")
    roc_auc = roc_auc_score(y_test, proba)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf = confusion_matrix(y_test, y_pred)

    if hasattr(clf, "feature_importances_"):
        importance_values = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importance_values = np.abs(clf.coef_[0])
    else:
        importance_values = np.zeros(len(feature_names))
    feature_importances = dict(zip(feature_names, importance_values))

    coverage = {}
    for threshold in (0.55, 0.6, 0.7, 0.8):
        mask = proba >= threshold
        if mask.sum() == 0:
            coverage[f"coverage_{threshold}"] = 0.0
            coverage[f"accuracy_{threshold}"] = float("nan")
        else:
            coverage[f"coverage_{threshold}"] = float(mask.mean())
            coverage[f"accuracy_{threshold}"] = accuracy_score(y_test[mask], y_pred[mask])

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "classification_report": report,
        "confusion_matrix": conf.tolist(),
        "feature_importances": feature_importances,
        "coverage": coverage,
        "probabilities": proba.tolist(),
        "predictions": y_pred.tolist(),
        "calibration": calibration,
    }


def save_metrics(metrics: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "classifier_metrics.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
