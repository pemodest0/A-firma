from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class DatasetConfig:
    path: Path
    label: str
    category: str


DATASETS_HISTORICAL: List[DatasetConfig] = [
    DatasetConfig(Path("results_simulated/simulated_market"), "Simulado", "historical"),
    DatasetConfig(Path("results_real/SPY"), "SPY", "historical"),
    DatasetConfig(Path("results_real/_BVSP"), "BVSP", "historical"),
    DatasetConfig(Path("results_physical_real/temperature_city"), "Temperatura", "historical"),
    DatasetConfig(Path("results_synthetic/duffing"), "Duffing", "historical"),
    DatasetConfig(Path("results_physical_real/power_load"), "Carga eletrica", "historical"),
]

DATASETS_DAILY: List[DatasetConfig] = [
    DatasetConfig(Path("results_daily_real/SPY"), "SPY", "daily"),
    DatasetConfig(Path("results_daily_real/_BVSP"), "BVSP", "daily"),
    DatasetConfig(Path("results_daily_physical/temperature_city"), "Temperatura", "daily"),
    DatasetConfig(Path("results_daily_physical/power_load"), "Carga eletrica", "daily"),
    DatasetConfig(Path("results_daily_synthetic/duffing"), "Duffing", "daily"),
]

DM_RESULTS: List[Path] = [
    Path("results_dm_real/dm_summary.csv"),
    Path("results_dm_physical/dm_summary.csv"),
    Path("results_dm_synthetic/dm_summary.csv"),
]


def _load_historical_metrics(configs: List[DatasetConfig]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for cfg in configs:
        metrics_path = cfg.path / "historical_forecast_metrics.csv"
        if not metrics_path.exists():
            continue
        df = pd.read_csv(metrics_path)
        if df.empty:
            continue
        df["abs_error"] = df["error_pct"].abs()
        df["direction_hit"] = (df["direction_pred"] == df["direction_real"]).astype(float)
        grouped = (
            df.groupby("mode_label")
            .agg(
                mean_abs_error=("abs_error", "mean"),
                mean_rmse=("rmse_pct", "mean"),
                direction_accuracy=("direction_hit", "mean"),
            )
            .reset_index()
        )
        grouped["dataset"] = cfg.label
        rows.extend(grouped.to_dict("records"))
    return pd.DataFrame(rows)


def _load_daily_metrics(configs: List[DatasetConfig]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for cfg in configs:
        summary_path = cfg.path / "daily_forecast_summary.csv"
        if not summary_path.exists():
            continue
        df = pd.read_csv(summary_path)
        if df.empty:
            continue
        df["dataset"] = cfg.label
        rows.extend(
            df[
                [
                    "dataset",
                    "mode_label",
                    "mae_pct",
                    "direction_accuracy",
                    "alpha_mean",
                    "entropy_mean",
                ]
            ].to_dict("records")
        )
    return pd.DataFrame(rows)


def _load_dm_results(paths: List[Path]) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for path in paths:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["source"] = path.parent.name
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    combined = pd.concat(rows, ignore_index=True)
    mask = (combined["model_a"] == "mode:quantum_hadamard") & (combined["model_b"] == "mode:classical")
    selected = combined.loc[mask, ["dataset", "dm_stat", "p_value", "source"]].copy()
    def _label(row: pd.Series) -> str:
        source = row["source"]
        dataset = row["dataset"]
        if source == "results_dm_real":
            return dataset
        if source == "results_dm_physical":
            return f"{dataset} (daily)"
        return f"{dataset} ({source})"
    selected["Dataset"] = selected.apply(_label, axis=1)
    return selected.rename(columns={"dm_stat": "DM_stat", "p_value": "p_value"}).drop(columns=["dataset", "source"]).set_index("Dataset")


def _format_pct(values: pd.Series) -> pd.Series:
    return (values * 100.0).round(1)


def build_report() -> Dict[str, pd.DataFrame]:
    historical = _load_historical_metrics(DATASETS_HISTORICAL)
    daily = _load_daily_metrics(DATASETS_DAILY)
    dm = _load_dm_results(DM_RESULTS)

    report: Dict[str, pd.DataFrame] = {}
    if not historical.empty:
        pivot = historical.pivot_table(
            index="dataset", columns="mode_label", values="mean_abs_error", aggfunc="mean"
        )
        report["historical_error"] = pivot
    if not daily.empty:
        daily_pivot = daily.pivot_table(index="dataset", columns="mode_label", values="mae_pct", aggfunc="mean")
        report["daily_error"] = daily_pivot
    if not historical.empty:
        classical = historical[historical["mode_label"] == "Classico"][["dataset", "mean_abs_error"]]
        hadamard = historical[historical["mode_label"] == "Quantico (Hadamard)"][["dataset", "mean_abs_error"]]
        merged = pd.merge(classical, hadamard, on="dataset", how="inner", suffixes=("_classico", "_hadamard"))
        merged["delta_hadamard_vs_classico"] = merged["mean_abs_error_hadamard"] - merged["mean_abs_error_classico"]
        report["historical_diff"] = merged.set_index("dataset")[
            ["mean_abs_error_classico", "mean_abs_error_hadamard", "delta_hadamard_vs_classico"]
        ]
    if not dm.empty:
        dm = dm.copy()
        dm["p_value"] = dm["p_value"].map(float)
        dm["log10_p"] = dm["p_value"].apply(lambda x: -np.log10(x) if x > 0 else np.nan)
        report["dm"] = dm
    return report


def plot_report(report: Dict[str, pd.DataFrame], output_path: Path) -> None:
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax_hist, ax_hist_dir, ax_daily, ax_dm = axes.flatten()

    if "historical_error" in report:
        hist_df = report["historical_error"]
        for mode in hist_df.columns:
            ax_hist.plot(hist_df.index, hist_df[mode], marker="o", label=mode)
        ax_hist.set_title("Erro médio histórico (|erro|)")
        ax_hist.set_ylabel("Erro")
        ax_hist.legend()
        ax_hist.tick_params(axis="x", rotation=20)
    else:
        ax_hist.set_visible(False)

    if "historical_diff" in report:
        diff_df = report["historical_diff"]
        ax_hist_dir.bar(diff_df.index, diff_df["delta_hadamard_vs_classico"], color="#6A0DAD")
        ax_hist_dir.axhline(0.0, color="black", linewidth=1.0)
        ax_hist_dir.set_title("Delta Hadamard vs Clássico (erro histórico)")
        ax_hist_dir.set_ylabel("Δ erro (Hadamard - Clássico)")
        ax_hist_dir.tick_params(axis="x", rotation=20)
    else:
        ax_hist_dir.set_visible(False)

    if "daily_error" in report:
        daily_df = report["daily_error"]
        width = 0.25
        x = np.arange(len(daily_df.index))
        for idx, mode in enumerate(daily_df.columns):
            offsets = x + (idx - (len(daily_df.columns) - 1) / 2) * width
            ax_daily.bar(offsets, daily_df[mode].values, width=width, label=mode)
        ax_daily.set_xticks(x)
        ax_daily.set_xticklabels(daily_df.index, rotation=20)
        ax_daily.set_ylabel("MAE diário (|erro_pct|)")
        ax_daily.set_title("Erro médio diário por modo")
        ax_daily.legend()
    else:
        ax_daily.set_visible(False)

    if "dm" in report:
        dm_df = report["dm"]
        ax_dm.bar(dm_df.index, dm_df["log10_p"], color="#1f77b4")
        ax_dm.axhline(-np.log10(0.05), color="red", linestyle="--", label="p = 0.05")
        ax_dm.set_ylabel("-log10(p)")
        ax_dm.set_title("Diebold–Mariano Hadamard vs Clássico")
        ax_dm.legend()
        ax_dm.tick_params(axis="x", rotation=20)
    else:
        ax_dm.set_visible(False)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    report = build_report()
    if not report:
        print("Nenhum dado encontrado para gerar relatório.")
        return
    output = Path("reports/metrics_comparison.png")
    plot_report(report, output)
    print(f"[RESULT] Grafico consolidado salvo em {output}")
    for name, df in report.items():
        print(f"\n[name={name}]")
        print(df.round(3).to_string())


if __name__ == "__main__":
    main()
