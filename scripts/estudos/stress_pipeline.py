from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dados.brutos.financial_loader import download_price_series, load_value_csv  # type: ignore


@dataclass
class DomainSpec:
    name: str
    domain: str
    kind: str  # "symbol" or "csv"
    symbol: Optional[str] = None
    csv_path: Optional[Path] = None
    date_col: str = "date"
    value_col: str = "price"
    month_zscore: bool = False
    return_method: str = "log"
    start: Optional[str] = None
    end: Optional[str] = None
    extra_args: Dict[str, str] = field(default_factory=dict)

    def build_robustness_args(self, outdir: Path) -> List[str]:
        args = [
            "--forecast-days",
            "5",
            "--bins-list",
            "6,8,10,12,15,20",
            "--window-list",
            "20,30,45,60,90",
            "--noise-list",
            "0.0,0.01,0.03,0.05,0.1",
            "--outdir",
            str(outdir),
        ]
        if self.kind == "symbol":
            args.extend(["--symbol", self.symbol or ""])
            if self.start:
                args.extend(["--start", self.start])
            if self.end:
                args.extend(["--end", self.end])
            args.extend(["--return-method", self.return_method])
        else:
            args.extend(
                [
                    "--csv",
                    str(self.csv_path),
                    "--date-col",
                    self.date_col,
                    "--value-col",
                    self.value_col,
                    "--return-method",
                    self.return_method,
                ]
            )
            if self.month_zscore:
                args.append("--month-zscore")
        if "walk_steps" in self.extra_args:
            args.extend(["--walk-steps", self.extra_args["walk_steps"]])
        return args

    def build_financial_args(
        self,
        output: Path,
        cutoffs: Iterable[pd.Timestamp],
        noise_csv: Optional[Path] = None,
    ) -> List[str]:
        args: List[str] = []
        if noise_csv is not None:
            args.extend(
                [
                    "--csv",
                    str(noise_csv),
                    "--date-col",
                    self.date_col,
                    "--value-col",
                    self.value_col,
                ]
            )
        elif self.kind == "symbol":
            args.extend(["--symbol", self.symbol or ""])
            if self.start:
                args.extend(["--start", self.start])
            if self.end:
                args.extend(["--end", self.end])
        else:
            args.extend(
                [
                    "--csv",
                    str(self.csv_path),
                    "--date-col",
                    self.date_col,
                    "--value-col",
                    self.value_col,
                ]
            )
        args.extend(
            [
                "--return-method",
                self.return_method,
                "--bins",
                "41",
                "--window",
                "60",
                "--step",
                "10",
                "--walk-steps",
                "35",
                "--forecast-steps",
                "15",
                "--output",
                str(output),
            ]
        )
        if self.month_zscore and noise_csv is None:
            args.append("--month-zscore")
        for cutoff in cutoffs:
            args.extend(["--cutoff", cutoff.strftime("%Y-%m-%d")])
        args.append("--compare-real")
        return args


DOMAINS: List[DomainSpec] = [
    DomainSpec("SPY", "Financeiro", "symbol", symbol="SPY", start="2010-01-01", end="2024-12-31"),
    DomainSpec("BVSP", "Financeiro", "symbol", symbol="^BVSP", start="2010-01-01", end="2024-12-31"),
    DomainSpec("AAPL", "Financeiro", "symbol", symbol="AAPL", start="2010-01-01", end="2024-12-31"),
    DomainSpec("BTC-USD", "Cripto", "symbol", symbol="BTC-USD", start="2016-01-01", end="2024-12-31"),
    DomainSpec(
        "Temperatura",
        "Fisico",
        "csv",
        csv_path=ROOT / "dados/brutos/temperature_city.csv",
        value_col="value",
        month_zscore=True,
        return_method="diff",
    ),
    DomainSpec(
        "Carga eletrica",
        "Fisico",
        "csv",
        csv_path=ROOT / "dados/brutos/power_load.csv",
        value_col="value",
        month_zscore=True,
        return_method="diff",
    ),
    DomainSpec(
        "Duffing",
        "Sintetico",
        "csv",
        csv_path=ROOT / "dados/brutos/duffing.csv",
    ),
    DomainSpec(
        "Synthetic2",
        "Sintetico",
        "csv",
        csv_path=ROOT / "dados/brutos/synthetic2.csv",
    ),
]


def run_command(args: List[str]) -> None:
    import subprocess

    cmd = [sys.executable] + args
    print(f"[CMD] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def ensure_dataframe(spec: DomainSpec) -> pd.DataFrame:
    if spec.kind == "symbol":
        df = download_price_series(
            spec.symbol or "",
            start=spec.start,
            end=spec.end,
            interval="1d",
            price_column="Adj Close",
        )
        df.rename(columns={"price": "value"}, inplace=True)
        return df
    return load_value_csv(spec.csv_path, date_column=spec.date_col, value_column=spec.value_col)


def quarter_cutoffs(df: pd.DataFrame) -> List[pd.Timestamp]:
    dates = pd.to_datetime(df["date"]).sort_values()
    start = dates.min()
    end = dates.max()
    quarters = pd.date_range(start=start, end=end, freq="QS")
    # pick only those with enough history (avoid earliest quarter)
    cutoffs = []
    for q in quarters[2:]:
        if q < end - pd.Timedelta(days=30):
            cutoffs.append(q)
    return cutoffs or [end - pd.Timedelta(days=90)]


def compute_delta(series: pd.Series) -> Tuple[float, float]:
    arr = series.to_numpy()
    sign_changes = 0
    prev = None
    for val in arr:
        if abs(val) < 1e-8:
            continue
        s = math.copysign(1, val)
        if prev is not None and s != prev:
            sign_changes += 1
        prev = s
    rate = sign_changes / max(len(arr) - 1, 1)
    return float(series.mean()), rate


def process_historical(path: Path) -> Dict[str, np.ndarray]:
    df = pd.read_csv(path)
    df["direction_hit"] = (df["direction_pred"] == df["direction_real"]).astype(float)
    pivot_mae = df.pivot_table(index="cutoff_date", columns="mode_label", values="error_pct", aggfunc="first")
    pivot_alpha = df.pivot_table(index="cutoff_date", columns="mode_label", values="alpha_pred", aggfunc="first")
    pivot_entropy = df.pivot_table(index="cutoff_date", columns="mode_label", values="entropy_pred", aggfunc="first")
    pivot_dir = df.pivot_table(index="cutoff_date", columns="mode_label", values="direction_hit", aggfunc="mean")

    mae_cls = pivot_mae["Classico"].astype(float).abs()
    mae_had = pivot_mae["Quantico (Hadamard)"].astype(float).abs()
    delta_mae = (mae_cls - mae_had).to_numpy()

    dir_cls = pivot_dir["Classico"].astype(float)
    dir_had = pivot_dir["Quantico (Hadamard)"].astype(float)
    delta_dir = (dir_had - dir_cls).to_numpy()

    alpha_cls = pivot_alpha["Classico"].astype(float)
    alpha_had = pivot_alpha["Quantico (Hadamard)"].astype(float)
    delta_alpha = (alpha_cls - alpha_had).to_numpy()

    if "Classico" in pivot_entropy.columns and "Quantico (Hadamard)" in pivot_entropy.columns:
        ent_cls = pivot_entropy["Classico"].astype(float)
        ent_had = pivot_entropy["Quantico (Hadamard)"].astype(float)
        delta_entropy = (ent_cls - ent_had).to_numpy()
    else:
        delta_entropy = np.full_like(delta_alpha, np.nan, dtype=float)
    return {
        "delta_mae": delta_mae,
        "delta_dir": delta_dir,
        "delta_alpha": delta_alpha,
        "delta_entropy": delta_entropy,
    }


def evaluate_robustness(df: pd.DataFrame) -> pd.DataFrame:
    combos = []
    grouped = df.groupby(["combination"])
    for combo, group in grouped:
        try:
            cls = group[group["mode_label"] == "Classico"].iloc[0]
            had = group[group["mode_label"] == "Quantico (Hadamard)"].iloc[0]
        except IndexError:
            continue
        combos.append(
            {
                "combination": combo,
                "bins": cls["bins"],
                "window": cls["window"],
                "noise": cls["noise"],
                "delta_mae": cls["mae_pct"] - had["mae_pct"],
                "delta_dir": had["direction_accuracy"] - cls["direction_accuracy"],
                "delta_alpha": cls.get("alpha_mean", np.nan) - had.get("alpha_mean", np.nan),
                "delta_entropy": cls.get("entropy_mean", np.nan) - had.get("entropy_mean", np.nan),
            }
        )
    return pd.DataFrame(combos)


def run_noise_experiments(spec: DomainSpec, noise_levels: List[float], base_df: pd.DataFrame, cutoffs: List[pd.Timestamp], outdir: Path) -> pd.DataFrame:
    results = []
    rng = np.random.default_rng(12345)
    base_prices = base_df["value"].astype(float).to_numpy()
    for noise in noise_levels:
        noisy_prices = base_prices * (1.0 + noise * rng.standard_normal(base_prices.shape))
        noisy_df = base_df.copy()
        noisy_df["value"] = noisy_prices
        tmp_path = ROOT / "tmp" / f"{spec.name}_noise_{noise:.3f}.csv"
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        noisy_df.to_csv(tmp_path, index=False)
        out = outdir / f"noise_{noise:.3f}".replace(".", "p")
        run_command(
            [
                "run_financial_analysis.py",
                *spec.build_financial_args(out, cutoffs, noise_csv=tmp_path),
            ]
        )
        hist_path = out / spec.name / "historical_forecast_metrics.csv"
        if hist_path.exists():
            hist = process_historical(hist_path)
            results.append(
                {
                    "noise": noise,
                    "delta_mae": float(np.nanmean(hist["delta_mae"])),
                    "delta_alpha": float(np.nanmean(hist["delta_alpha"])),
                }
            )
    return pd.DataFrame(results)


def run_shuffle(spec: DomainSpec, base_df: pd.DataFrame, cutoffs: List[pd.Timestamp], outdir: Path) -> Dict[str, float]:
    shuffled = base_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    tmp_path = ROOT / "tmp" / f"{spec.name}_shuffle.csv"
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    shuffled.to_csv(tmp_path, index=False)
    out = outdir / "shuffle"
    run_command(
        [
            "run_financial_analysis.py",
            *spec.build_financial_args(out, cutoffs, noise_csv=tmp_path),
        ]
    )
    hist_path = out / spec.name / "historical_forecast_metrics.csv"
    if hist_path.exists():
        hist = process_historical(hist_path)
        return {
            "delta_mae": float(np.nanmean(hist["delta_mae"])),
            "delta_dir": float(np.nanmean(hist["delta_dir"])),
        }
    return {"delta_mae": float("nan"), "delta_dir": float("nan")}


def compute_failure_flags(
    robustness_df: pd.DataFrame,
    inversion_rate: float,
    dm_pvalue: float,
) -> Dict[str, bool]:
    small_gain = (robustness_df["delta_mae"] < 0.5).mean() > 0.6
    alpha_failure = inversion_rate > 0.4
    dm_failure = dm_pvalue >= 0.05
    return {"small_gain": small_gain, "alpha_failure": alpha_failure, "dm_failure": dm_failure}


def main() -> None:
    parser = argparse.ArgumentParser(description="Stress test pipeline.")
    parser.add_argument("--domains", type=str, default="all", help="Comma separated domain names or 'all'.")
    parser.add_argument("--noise-levels", type=str, default="0.0,0.01,0.02,0.05")
    args = parser.parse_args()

    selected_names = [name.strip() for name in args.domains.split(",")] if args.domains != "all" else [spec.name for spec in DOMAINS]
    noise_levels = [float(x.strip()) for x in args.noise_levels.split(",") if x.strip()]

    dm_map_file = ROOT / "results_dm_map.json"
    dm_map: Dict[str, float]
    if dm_map_file.exists():
        dm_map = json.loads(dm_map_file.read_text(encoding="utf-8"))
    else:
        from analysis.make_report import _read_dm  # type: ignore

        dm_map = _read_dm()
        dm_map_file.write_text(json.dumps(dm_map, indent=2), encoding="utf-8")

    stress_rows: List[Dict[str, object]] = []
    noise_records: Dict[str, pd.DataFrame] = {}
    summary_lines: List[str] = []

    for spec in DOMAINS:
        if spec.name not in selected_names:
            continue
        print(f"\n=== Domain: {spec.name} ({spec.domain}) ===")
        base_df = ensure_dataframe(spec)
        cutoffs = quarter_cutoffs(base_df)

        domain_out = ROOT / "results_stress" / spec.name
        domain_out.mkdir(parents=True, exist_ok=True)

        # Robustness grid
        robustness_dir = domain_out / "robustness"
        run_command(["run_robustness_grid.py", *spec.build_robustness_args(robustness_dir)])
        robustness_path = robustness_dir / "robustness_grid.csv"
        robustness_df = pd.read_csv(robustness_path)
        combos_df = evaluate_robustness(robustness_df)
        win_mae_ratio = (combos_df["delta_mae"] > 0).mean()
        win_dir_ratio = (combos_df["delta_dir"] > 0).mean()

        # Consistency temporal (base data)
        financial_out = domain_out / "financial"
        run_command(["run_financial_analysis.py", *spec.build_financial_args(financial_out, cutoffs)])
        hist_path = financial_out / spec.name / "historical_forecast_metrics.csv"
        hist_metrics = process_historical(hist_path)
        _, inversion_rate = compute_delta(pd.Series(hist_metrics["delta_alpha"]))
        delta_mae_mean = float(np.nanmean(hist_metrics["delta_mae"]))
        delta_dir_mean = float(np.nanmean(hist_metrics["delta_dir"]))
        delta_alpha_mean = float(np.nanmean(hist_metrics["delta_alpha"]))

        # Noise sensitivity
        noise_df = run_noise_experiments(spec, noise_levels, base_df, cutoffs, domain_out / "noise")
        noise_records[spec.name] = noise_df

        # Shuffle test
        shuffle_stats = run_shuffle(spec, base_df, cutoffs, domain_out)

        dm_value = dm_map.get(spec.name, float("nan"))
        failures = compute_failure_flags(combos_df, inversion_rate, dm_value if not math.isnan(dm_value) else 0.0)

        for _, row in combos_df.iterrows():
            stress_rows.append(
                {
                    "domain": spec.name,
                    "bins": row["bins"],
                    "window": row["window"],
                    "noise": row["noise"],
                    "delta_mae": row["delta_mae"],
                    "delta_dir": row["delta_dir"],
                    "delta_alpha": row["delta_alpha"],
                    "delta_entropy": row["delta_entropy"],
                    "p_value_DM": dm_value,
                    "inversion_rate": inversion_rate,
                }
            )

        summary_lines.append(
            f"- {spec.name}: ΔMAE médio={delta_mae_mean:.2f}, Δα médio={delta_alpha_mean:.3f}, "
            f"robustez (MAE)={win_mae_ratio:.1%}, robustez (dir)={win_dir_ratio:.1%}, "
            f"shuffle ΔMAE={shuffle_stats['delta_mae']:.2f}, DM p={dm_value:.3g}, "
            f"inversão α={inversion_rate:.1%}, falhas={failures}"
        )

    if stress_rows:
        stress_df = pd.DataFrame(stress_rows)
        stress_df.to_csv(ROOT / "results_stress" / "stress_summary.csv", index=False)

    # plot noise sensitivity
    if noise_records:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        for name, df in noise_records.items():
            if df.empty:
                continue
            axes[0].plot(df["noise"], df["delta_mae"], marker="o", label=name)
            axes[1].plot(df["noise"], df["delta_alpha"], marker="o", label=name)
        axes[0].set_ylabel("ΔMAE (Clássico - Hadamard)")
        axes[1].set_ylabel("Δα (Clássico - Hadamard)")
        axes[1].set_xlabel("Ruído adicional (σ)")
        axes[0].axhline(0.5, color="red", linestyle="--", linewidth=1, alpha=0.5)
        axes[0].axhline(0.0, color="black", linestyle=":", linewidth=1, alpha=0.5)
        axes[0].legend()
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(ROOT / "results_stress" / "stress_performance.png", dpi=300)
        plt.close(fig)

    # report markdown
    report_path = ROOT / "results_stress" / "stress_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print("\nResumo:\n" + "\n".join(summary_lines))
    print(f"\n[RESULT] stress_summary.csv, stress_performance.png e stress_report.md salvos em {report_path.parent}")


if __name__ == "__main__":
    main()
