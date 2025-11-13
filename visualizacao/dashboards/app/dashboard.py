"""
Dashboard Streamlit para visualizar previsões diárias.

Uso local:
    streamlit run app/dashboard.py

O app tenta consumir a API FastAPI (se variável API_BASE estiver configurada)
e, como fallback, lê diretamente os CSVs gerados pelo pipeline.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import streamlit as st

try:
    import requests

    REQUESTS_AVAILABLE = True
except Exception:  # pragma: no cover
    requests = None  # type: ignore
    REQUESTS_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_ASSETS = {
    "IBOV": DEFAULT_RESULTS_DIR / "ibov_forecast" / "daily_forecast_metrics.csv",
    "SPY": DEFAULT_RESULTS_DIR / "spy_forecast" / "daily_forecast_metrics.csv",
}


def list_available_assets(results_dir: Path = DEFAULT_RESULTS_DIR) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for path in results_dir.glob("**/daily_forecast_metrics.csv"):
        label = path.parent.name.replace("_forecast", "").upper()
        mapping[label] = path
    return mapping or DEFAULT_ASSETS


def load_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    if df.empty:
        raise ValueError(f"Dataset vazio: {path}")
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _asset_label_from_path(path: Path) -> str:
    return path.parent.name.replace("_forecast", "")


def load_residual(path: Path) -> Optional[pd.DataFrame]:
    asset_label = _asset_label_from_path(path)
    residual_path = path.parent.parent / f"hybrid_residual_{asset_label}" / "residual_adjusted_predictions.csv"
    if residual_path.exists():
        df = pd.read_csv(residual_path, parse_dates=["date"])
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
    return None


def load_feature_importance(path: Path) -> Optional[pd.DataFrame]:
    asset_label = _asset_label_from_path(path)
    explain_path = (
        path.parent.parent / f"hybrid_residual_{asset_label}" / "explainability" / "feature_importance.csv"
    )
    if explain_path.exists():
        df = pd.read_csv(explain_path)
        df.sort_values("importance", ascending=False, inplace=True)
        return df
    return None


def fetch_from_api(asset: str, api_base: str, residual: bool = True) -> Optional[dict]:
    if not REQUESTS_AVAILABLE:
        return None
    try:
        resp = requests.get(f"{api_base}/forecast/{asset}", params={"residual": residual}, timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        return None
    return None


def main() -> None:
    st.set_page_config(page_title="Forecast Dashboard", layout="wide")
    st.title("Previsões Diárias")

    api_base = os.getenv("FORECAST_API_BASE")
    available_assets = list_available_assets()
    asset = st.sidebar.selectbox("Ativo", sorted(available_assets.keys()))
    residual_toggle = st.sidebar.checkbox("Mostrar ajuste residual", value=True)

    latest_data = None
    if api_base:
        latest_data = fetch_from_api(asset, api_base, residual=residual_toggle)

    metrics_path = available_assets[asset]
    df = load_metrics(metrics_path)
    st.subheader(f"{asset} – série diária ({metrics_path})")

    col1, col2 = st.columns(2)
    latest_row = df.iloc[-1]
    col1.metric("Preço previsto (walk)", f"{latest_row['price_pred']:.2f}")
    if "price_real" in latest_row:
        col2.metric("Preço real", f"{latest_row['price_real']:.2f}")

    # Residual info
    residual_df = None
    if residual_toggle:
        residual_df = load_residual(metrics_path)
        if residual_df is not None:
            latest_residual = residual_df.iloc[-1]
            st.metric("Preço ajustado (residual)", f"{latest_residual['price_pred_adjusted']:.2f}")

    # API snapshot
    if latest_data:
        st.info(
            f"API {api_base}: previsões atuais → "
            f"walk={latest_data.get('price_pred')}, residual={latest_data.get('residual_adjusted')}"
        )

    # Charts
    st.subheader("Preço real vs previsto")
    chart_df = df.set_index("date")[["price_real", "price_pred"]].tail(180)
    st.line_chart(chart_df)

    st.subheader("Erro em pontos")
    df["error_points"] = df["price_pred"] - df["price_real"]
    st.line_chart(df.set_index("date")[["error_points"]].tail(180))

    if residual_df is not None:
        residual_df = residual_df.set_index("date")
        combined = pd.concat(
            [
                df.set_index("date")[["price_real", "price_pred"]],
                residual_df[["price_pred_adjusted"]],
            ],
            axis=1,
        ).tail(180)
        st.subheader("Walk vs Residual")
        st.line_chart(combined)

    # Feature importance table
    st.subheader("Importância de features (Residual)")
    importance_df = load_feature_importance(metrics_path)
    if importance_df is not None:
        st.dataframe(importance_df.head(20))
    else:
        st.write("Ainda não há importâncias calculadas (execute train_residual_model com --tune).")

    # Data quality summary
    dq_path = Path("results/data_quality") / f"{asset.lower()}.json"
    if dq_path.exists():
        st.sidebar.markdown("### Qualidade de dados")
        dq = pd.read_json(dq_path, typ="series")
        st.sidebar.json(dq.to_dict())


if __name__ == "__main__":
    main()
