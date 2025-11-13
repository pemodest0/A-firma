#!/usr/bin/env python3
"""Streamlit interativo para visualizar previsões de ativos (clássico/quântico/blend)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

RESULTS_DIR = Path("results")
FORECAST_SUFFIX = "_forecast"
BLEND_DIR = RESULTS_DIR / "crypto_blend"


def list_assets() -> Dict[str, Path]:
    assets: Dict[str, Path] = {}
    for forecast_dir in RESULTS_DIR.glob(f"*{FORECAST_SUFFIX}"):
        for sub in forecast_dir.glob("*"):
            metrics_path = sub / "daily_forecast_metrics.csv"
            if metrics_path.exists():
                assets[sub.stem] = metrics_path
    return dict(sorted(assets.items()))


def prepare_classical(df: pd.DataFrame) -> pd.DataFrame:
    classical = df[df["mode"] == "classical"].copy()
    classical.sort_values("date", inplace=True)
    classical["price_pred"] = classical["price_pred"].astype(float)
    classical["price_real"] = classical["price_real"].astype(float)
    classical.dropna(subset=["price_pred", "price_real"], inplace=True)
    classical["classical_err"] = classical["price_real"] - classical["price_pred"]
    for mode in ("quantum_hadamard", "quantum_grover"):
        col = f"{mode}_price_pred"
        if col in classical.columns:
            classical[f"{mode}_err"] = classical["price_real"] - classical[col]
    if "coin_risk_flag" in classical.columns:
        classical["coin_risk_flag"] = classical["coin_risk_flag"].fillna(False)
    return classical


def load_blend_predictions(asset: str) -> Dict[str, pd.DataFrame]:
    outputs: Dict[str, pd.DataFrame] = {}
    for mode in ("quantum_hadamard", "quantum_grover"):
        path = BLEND_DIR / f"{asset}_{mode}_dynamic_blend_predictions.csv"
        if path.exists():
            df = pd.read_csv(path, parse_dates=["date"])
            df.sort_values("date", inplace=True)
            outputs[mode] = df
    return outputs


def compute_metrics(series_true: np.ndarray, series_pred: np.ndarray) -> Dict[str, float]:
    err = series_true - series_pred
    mae = float(np.mean(np.abs(err)))
    mpe = float(np.mean(err / np.maximum(np.abs(series_true), 1e-9)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    return {"MAE": mae, "RMSE": rmse, "MPE": mpe}


def build_animation(dates, actual, predictions: Dict[str, np.ndarray]) -> go.Figure:
    traces = []
    frames = []
    names_colors = {"real": ("Preço real", "#111111")}
    for idx, (label, values) in enumerate(predictions.items(), start=1):
        names_colors[label] = (label, f"C{idx}")
    initial_data = []
    initial_data.append(go.Scatter(x=dates[:1], y=actual[:1], mode="lines", name="Preço real", line=dict(color="#111111", width=2)))
    for label, values in predictions.items():
        initial_data.append(go.Scatter(x=dates[:1], y=values[:1], mode="lines", name=label))
    for step in range(len(dates)):
        data = [go.Scatter(x=dates[: step + 1], y=actual[: step + 1], mode="lines", name="Preço real", line=dict(color="#111111", width=2))]
        for label, values in predictions.items():
            data.append(go.Scatter(x=dates[: step + 1], y=values[: step + 1], mode="lines", name=label))
        frames.append(go.Frame(data=data, name=str(step)))
    fig = go.Figure(data=initial_data, frames=frames)
    fig.update_layout(
        height=500,
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="top", y=1.02, xanchor="right", x=1),
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "buttons": [
                    {
                        "label": "▶ Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 200, "redraw": True}, "fromcurrent": True}],
                    },
                    {
                        "label": "⏸ Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    },
                ],
            }
        ],
        sliders=[
            {
                "pad": {"b": 10, "t": 40},
                "currentvalue": {"prefix": "Passo: ", "font": {"size": 16}},
                "steps": [
                    {
                        "label": str(step),
                        "method": "animate",
                        "args": [[str(step)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}],
                    }
                    for step in range(len(dates))
                ],
            }
        ],
    )
    fig.update_xaxes(title="Data")
    fig.update_yaxes(title="Preço")
    return fig


def main() -> None:
    st.set_page_config(page_title="Dashboard de Previsões", layout="wide")
    st.title("📈 Modelo de previsão – clássico vs quântico")

    assets = list_assets()
    if not assets:
        st.error("Não foram encontrados dados em results/*_forecast.")
        return

    asset = st.selectbox("Ativo/cripto", list(assets.keys()))
    metrics_path = assets[asset]
    df = pd.read_csv(metrics_path, parse_dates=["date"])
    classical = prepare_classical(df)

    min_date = classical["date"].min().date()
    max_date = classical["date"].max().date()
    start_date, end_date = st.slider("Janela", min_value=min_date, max_value=max_date, value=(min_date, max_date))
    mask = (classical["date"].dt.date >= start_date) & (classical["date"].dt.date <= end_date)
    classical = classical[mask].copy()

    st.subheader("Distribuição do erro (últimos dados selecionados)")
    error_rows = []
    error_cols = {
        "Clássico": classical["classical_err"],
    }
    if "quantum_hadamard_err" in classical.columns:
        error_cols["Hadamard"] = classical["quantum_hadamard_err"]
    if "quantum_grover_err" in classical.columns:
        error_cols["Grover"] = classical["quantum_grover_err"]
    for label, series in error_cols.items():
        if series.notna().any():
            error_rows.append(
                {
                    "Modelo": label,
                    "Erro mínimo": float(series.min()),
                    "Erro máximo": float(series.max()),
                }
            )
    if error_rows:
        st.table(pd.DataFrame(error_rows))
    else:
        st.write("Erros não disponíveis para este ativo.")

    predictions = {
        "Clássico": classical["price_pred"].to_numpy(),
    }
    if "quantum_hadamard_price_pred" in classical.columns:
        predictions["Hadamard"] = classical["quantum_hadamard_price_pred"].to_numpy()
    if "quantum_grover_price_pred" in classical.columns:
        predictions["Grover"] = classical["quantum_grover_price_pred"].to_numpy()

    blend_preds = load_blend_predictions(asset)
    if blend_preds:
        st.subheader("Blend dinâmico – previsões adicionais")
        for mode, blend_df in blend_preds.items():
            merged = pd.merge(classical, blend_df[["date", "blend_price_pred"]], on="date", how="left", suffixes=("", "_blend"))
            classical[f"blend_{mode}"] = merged["blend_price_pred"]
            predictions[f"Blend {mode.split('_')[-1].capitalize()}"] = merged["blend_price_pred"].to_numpy()

    st.subheader("Indicadores de erro")
    rows = []
    true_values = classical["price_real"].to_numpy()
    for label, pred in predictions.items():
        stats = compute_metrics(true_values, pred)
        stats["Modelo"] = label
        rows.append(stats)
    st.table(pd.DataFrame(rows)[["Modelo", "MAE", "RMSE", "MPE"]])

    st.subheader("Curva real vs. prevista")
    fig_static = go.Figure()
    fig_static.add_trace(go.Scatter(x=classical["date"], y=classical["price_real"], mode="lines", name="Preço real", line=dict(color="#111111", width=2)))
    for label, values in predictions.items():
        fig_static.add_trace(go.Scatter(x=classical["date"], y=values, mode="lines", name=label))
    fig_static.update_layout(height=480, legend=dict(orientation="h", yanchor="top", y=1.02, xanchor="right", x=1))
    fig_static.update_xaxes(title="Data")
    fig_static.update_yaxes(title="Preço")
    st.plotly_chart(fig_static, use_container_width=True)

    st.subheader("Sinalizadores")
    if "coin_risk_flag" in classical.columns:
        risk_rate = classical["coin_risk_flag"].mean() * 100
        st.write(f"coin_risk_flag ativo em {risk_rate:.2f}% das datas selecionadas.")
    else:
        st.write("coin_risk_flag não disponível para este ativo.")


if __name__ == "__main__":
    main()
