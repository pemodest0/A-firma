import subprocess
import threading
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def run_forecast(asset_cfg: Dict[str, str], window: int, bins: int) -> subprocess.CompletedProcess:
    cmd = [
        "bash",
        "-lc",
        (
            "PYTHONPATH=modelos/core:.:modelos python scripts/run_daily_forecast.py"
            f" --csv {asset_cfg['data']}"
            f" --window {window} --bins {bins}"
            f" --output {asset_cfg['forecast_output']}"
            + (f" --start {asset_cfg.get('start')}" if asset_cfg.get('start') else '')
            + (f" --end {asset_cfg.get('end')}" if asset_cfg.get('end') else '')
        ),
    ]
    return subprocess.run(cmd, cwd=str(PROJECT_ROOT))


def load_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"metrics não encontrado: {path}")
    df = pd.read_csv(path, parse_dates=['date'])
    df.sort_values('date', inplace=True)
    return df


def main() -> None:
    st.set_page_config(page_title="Interactive Forecast", layout="wide")
    st.title("Painel Interativo de Parâmetros")

    # Configuracoes simples para exemplo
    assets = {
        "IBOV": {
            "data": "dados/brutos/ibov_investing.csv",
            "forecast_output": "results/ibov_interactive",
            "start": "2016-01-01",
            "end": "2025-12-31",
        },
        "SPY": {
            "data": "dados/brutos/spy_stooq.csv",
            "forecast_output": "results/spy_interactive",
            "start": "2016-01-01",
            "end": "2025-12-31",
        },
    }

    asset = st.sidebar.selectbox("Ativo", list(assets.keys()))
    window = st.sidebar.slider("Janela (dias)", min_value=10, max_value=90, step=5, value=30)
    bins = st.sidebar.slider("Bins", min_value=5, max_value=30, step=1, value=15)

    if st.sidebar.button("Executar previsão"):
        with st.spinner("Executando ..."):
            result = run_forecast(assets[asset], window, bins)
        if result.returncode != 0:
            st.error("Erro ao executar previsão. Verifique logs.")
        else:
            st.success("Pipeline concluído. Veja resultados abaixo.")

    metrics_path = Path(assets[asset]['forecast_output']) / 'daily_forecast_metrics.csv'
    if metrics_path.exists():
        df = load_metrics(metrics_path)
        st.subheader(f"{asset} – Janela {window}, Bins {bins}")
        st.line_chart(df.set_index('date')[['price_real', 'price_pred']].tail(120))

        df['error_pts'] = df['price_pred'] - df['price_real']
        st.line_chart(df.set_index('date')[['error_pts']].tail(120))

        mae_pts = float((df['error_pts'].abs()).mean())
        st.metric("MAE (pts)", f"{mae_pts:.2f}")
    else:
        st.info("Execute o pipeline para ver resultados.")


if __name__ == "__main__":
    main()
