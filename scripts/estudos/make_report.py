from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analysis.metrics_comparison as mc


@dataclass
class DomainConfig:
    label: str
    domain: str
    path: Path
    dm_key: Optional[str] = None


DOMAIN_MAP: Dict[str, DomainConfig] = {
    "Simulado": DomainConfig("Simulado", "Sintetico", Path("results_simulated/simulated_market")),
    "SPY": DomainConfig("SPY", "Financeiro", Path("results_real/SPY"), dm_key="SPY"),
    "BVSP": DomainConfig("BVSP", "Financeiro", Path("results_real/_BVSP"), dm_key="_BVSP"),
    "AAPL": DomainConfig("AAPL", "Financeiro", Path("results_real/AAPL"), dm_key="AAPL"),
    "BTC-USD": DomainConfig("BTC-USD", "Cripto", Path("results_real/BTC-USD"), dm_key="BTC-USD"),
    "Temperatura": DomainConfig("Temperatura", "Fisico", Path("results_physical_real/temperature_city"), dm_key="temperature_city"),
    "Carga eletrica": DomainConfig("Carga eletrica", "Fisico", Path("results_physical_real/power_load"), dm_key="power_load"),
    "Duffing": DomainConfig("Duffing", "Sintetico", Path("results_synthetic/duffing"), dm_key="duffing"),
}

DAILY_PATHS: Dict[str, Path] = {
    "SPY": Path("results_daily_real/SPY"),
    "BVSP": Path("results_daily_real/_BVSP"),
    "AAPL": Path("results_daily_extra/AAPL"),
    "BTC-USD": Path("results_daily_extra/BTC-USD"),
    "Temperatura": Path("results_daily_physical/temperature_city"),
    "Carga eletrica": Path("results_daily_physical/power_load"),
    "Duffing": Path("results_daily_synthetic/duffing"),
}

DM_FILES = {
    "Financeiro": Path("results_dm_real/dm_summary.csv"),
    "Financeiro-extra": Path("results_dm_extra/dm_summary.csv"),
    "Cripto": Path("results_dm_extra/dm_summary.csv"),
    "Fisico": Path("results_dm_physical/dm_summary.csv"),
    "Sintetico": Path("results_dm_synthetic/dm_summary.csv"),
}

MODES = {
    "Classico": "Classico",
    "Hadamard": "Quantico (Hadamard)",
}


def _read_dm() -> Dict[str, float]:
    dm_map: Dict[str, float] = {}
    for domain, path in DM_FILES.items():
        if not path.exists():
            continue
        df = pd.read_csv(path)
        mask = (df["model_a"] == "mode:quantum_hadamard") & (df["model_b"] == "mode:classical")
        filtered = df.loc[mask, ["dataset", "p_value"]]
        for _, row in filtered.iterrows():
            dm_map[row["dataset"]] = float(row["p_value"])
    return dm_map


def _load_historical_metrics(config: DomainConfig) -> Optional[pd.DataFrame]:
    path = config.path / "historical_forecast_metrics.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    df["error_abs"] = df["error_pct"].abs()
    df["direction_hit"] = (df["direction_pred"] == df["direction_real"]).astype(float)
    grouped = (
        df.groupby("mode_label")
        .agg(
            error=("error_abs", "mean"),
            direction=("direction_hit", "mean"),
            alpha_delta=("alpha_delta", "mean"),
            entropy_delta=("entropy_delta", "mean"),
        )
        .reset_index()
    )
    return grouped


def _load_daily_summary(label: str) -> Optional[pd.DataFrame]:
    path = DAILY_PATHS.get(label)
    if path is None:
        return None
    summary_path = path / "daily_forecast_summary.csv"
    if not summary_path.exists():
        return None
    df = pd.read_csv(summary_path)
    if df.empty:
        return None
    return df


def generate_summary_table(dm_map: Dict[str, float]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for label, cfg in DOMAIN_MAP.items():
        hist = _load_historical_metrics(cfg)
        if hist is None:
            continue
        row: Dict[str, object] = {"Dataset": label, "Dominio": cfg.domain}
        for mode_alias, mode_name in MODES.items():
            mode_df = hist[hist["mode_label"] == mode_name]
            if mode_df.empty:
                continue
            row[f"Erro_{mode_alias}"] = float(mode_df["error"].iloc[0])
            row[f"Direcao_{mode_alias}"] = float(mode_df["direction"].iloc[0])
            row[f"DeltaAlpha_{mode_alias}"] = float(mode_df["alpha_delta"].iloc[0])
            row[f"DeltaEntropy_{mode_alias}"] = float(mode_df["entropy_delta"].iloc[0])
        if cfg.dm_key and cfg.dm_key in dm_map:
            row["DM_p"] = dm_map[cfg.dm_key]
        else:
            row["DM_p"] = np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    df.sort_values(["Dominio", "Dataset"], inplace=True)
    return df


def plot_entropy_alpha() -> Path:
    records: List[Dict[str, object]] = []
    for label, cfg in DOMAIN_MAP.items():
        daily = _load_daily_summary(label)
        if daily is None:
            continue
        for _, row in daily.iterrows():
            mode = row["mode_label"]
            if mode not in ("Classico", "Quantico (Hadamard)"):
                continue
            records.append(
                {
                    "Dataset": label,
                    "Dominio": cfg.domain,
                    "mode": mode,
                    "alpha_mean": row["alpha_mean"],
                    "entropy_mean": row["entropy_mean"],
                }
            )
    if not records:
        print("Nenhum resumo diário encontrado para plotar entropia/alpha.")
        return Path("reports/domain_entropy_alpha.png")
    df = pd.DataFrame(records)
    agg = df.groupby(["Dominio", "mode"]).agg(alpha=("alpha_mean", "mean"), entropy=("entropy_mean", "mean")).reset_index()

    domains = sorted(agg["Dominio"].unique())
    modes = ["Classico", "Quantico (Hadamard)"]
    x = np.arange(len(domains))
    width = 0.35

    plt.style.use("seaborn-v0_8-darkgrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

    for idx, mode in enumerate(modes):
        offsets = x + (idx - 0.5) * width
        alpha_vals = [agg[(agg["Dominio"] == dom) & (agg["mode"] == mode)]["alpha"].mean() for dom in domains]
        entropy_vals = [agg[(agg["Dominio"] == dom) & (agg["mode"] == mode)]["entropy"].mean() for dom in domains]
        axes[0].bar(offsets, alpha_vals, width=width, label=mode if idx == 0 else mode)
        axes[1].bar(offsets, entropy_vals, width=width, label=mode if idx == 0 else mode)

    ticks = x
    axes[0].set_xticks(ticks)
    axes[0].set_xticklabels(domains, rotation=15)
    axes[1].set_xticks(ticks)
    axes[1].set_xticklabels(domains, rotation=15)

    axes[0].set_title("Alpha médio por domínio")
    axes[0].set_ylabel("alpha médio")
    axes[1].set_title("Entropia média por domínio")
    axes[1].set_ylabel("Entropia média (bits)")
    axes[0].legend()

    fig.tight_layout()
    output_path = Path("reports/domain_entropy_alpha.png")
    output_path.parent.mkdir(exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def format_summary_table(df: pd.DataFrame) -> str:
    formatted = df.copy()
    column_order = ["Dataset", "Dominio"] + [col for col in formatted.columns if col not in ("Dataset", "Dominio")]
    table_rows: List[List[str]] = []
    header = column_order
    table_rows.append(header)
    table_rows.append(["---"] * len(header))
    for _, row in formatted.iterrows():
        row_values: List[str] = []
        for col in column_order:
            value = row.get(col, "")
            if isinstance(value, (float, np.floating)) and np.isfinite(value):
                row_values.append(f"{value:.3f}")
            elif value == "" or value is None or (isinstance(value, float) and np.isnan(value)):
                row_values.append("")
            else:
                row_values.append(str(value))
        table_rows.append(row_values)
    lines = ["| " + " | ".join(r) + " |" for r in table_rows]
    return "\n".join(lines)


def main() -> None:
    # Painel principal já existente
    report = mc.build_report()
    mc.plot_report(report, Path("reports/metrics_comparison.png"))

    # Complemento: entropia e alpha por domínio
    entropy_plot = plot_entropy_alpha()
    print(f"[RESULT] Grafico entropia/alpha salvo em {entropy_plot}")

    # Tabela resumo
    dm_map = _read_dm()
    summary_df = generate_summary_table(dm_map)
    markdown_table = format_summary_table(summary_df)
    table_path = Path("reports/summary_table.md")
    table_path.write_text(markdown_table, encoding="utf-8")
    print(f"[RESULT] Tabela resumo salva em {table_path}")
    print("\nResumo (Markdown):\n")
    print(markdown_table)


if __name__ == "__main__":
    main()
