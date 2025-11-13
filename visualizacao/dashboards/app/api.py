from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except Exception:  # pragma: no cover - fastapi optional
    FastAPI = None  # type: ignore
    BaseModel = object  # type: ignore
    FASTAPI_AVAILABLE = False

import pandas as pd


class ForecastResponse(BaseModel):  # type: ignore
    asset: str
    date: datetime
    price_pred: float
    price_real: Optional[float]
    direction_match: Optional[bool]
    residual_adjusted: Optional[float]


def load_latest_forecast(path: Path) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["date"])
    if df.empty:
        raise ValueError("Dataset vazio")
    df.sort_values("date", inplace=True)
    return df.iloc[-1]


def create_app(metrics_dir: str = "results"):
    if not FASTAPI_AVAILABLE:
        raise RuntimeError("fastapi não instalado neste ambiente")

    app = FastAPI(title="Forecast API", version="1.0")

    @app.get("/health")
    def health_check() -> dict:
        return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

    @app.get("/forecast/{asset}", response_model=ForecastResponse)
    def get_forecast(asset: str, residual: bool = True):
        base_path = Path(metrics_dir)
        metrics_path = base_path / asset.lower() / "daily_forecast_metrics.csv"
        if not metrics_path.exists():
            raise HTTPException(404, detail=f"Metrics não encontrado para {asset}")
        latest = load_latest_forecast(metrics_path)
        response = ForecastResponse(
            asset=asset.upper(),
            date=latest["date"],
            price_pred=float(latest.get("price_pred", float("nan"))),
            price_real=float(latest.get("price_real", float("nan"))) if "price_real" in latest else None,
            direction_match=bool(latest.get("direction_match")) if "direction_match" in latest else None,
            residual_adjusted=None,
        )
        if residual:
            residual_path = base_path / f"hybrid_residual_{asset.lower()}" / "residual_adjusted_predictions.csv"
            if residual_path.exists():
                latest_residual = load_latest_forecast(residual_path)
                response.residual_adjusted = float(latest_residual.get("price_pred_adjusted"))
        return response

    return app


app = create_app() if FASTAPI_AVAILABLE else None
