#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

try:
    from engine.structural.csd import rolling_ac1 as structural_rolling_ac1
    from engine.structural.forman_ricci import forman_edge_curvature, forman_summary
    from engine.structural.graph import corr_to_graph
    from engine.structural.score import fit_normalizer as structural_fit_normalizer
    from engine.structural.score import structural_score as structural_score_fusion
    from engine.structural.score import transform as structural_transform

    STRUCTURAL_OK = True
except Exception:
    STRUCTURAL_OK = False


ROOT = REPO_ROOT
DEFAULT_OUT_BASE = ROOT / "results" / "lab_corr_macro"
DEFAULT_FINANCE_BASE = ROOT / "results" / "finance_download"
DEFAULT_BASELINE_DIR = DEFAULT_OUT_BASE / "_official_baseline"
ERA_ORDER = ["2018_2019", "2020", "2021_2022", "2023_2026"]


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _find_latest_finance_run() -> Path:
    if not DEFAULT_FINANCE_BASE.exists():
        raise FileNotFoundError(f"Missing finance base dir: {DEFAULT_FINANCE_BASE}")
    runs = sorted([p for p in DEFAULT_FINANCE_BASE.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True)
    for d in runs:
        if (d / "panel_long_sector.csv").exists() and (d / "universe_fixed.csv").exists():
            return d
    raise FileNotFoundError("No finance run with panel_long_sector.csv + universe_fixed.csv found.")


def _ensure_cols(df: pd.DataFrame, cols: list[str], ctx: str) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"{ctx} missing columns: {miss}")


def _safe_sign(a: np.ndarray) -> np.ndarray:
    s = np.sign(np.asarray(a, dtype=float))
    s[np.abs(np.asarray(a, dtype=float)) < 1e-12] = 0.0
    s[~np.isfinite(np.asarray(a, dtype=float))] = 0.0
    return s


def _parse_windows(text: str) -> list[int]:
    vals: list[int] = []
    for token in str(text).split(","):
        t = token.strip()
        if not t:
            continue
        vals.append(int(t))
    out = sorted(set(int(max(20, x)) for x in vals))
    if not out:
        return [60, 120, 252]
    return out


def _cluster_metrics(corr: np.ndarray) -> tuple[np.ndarray, int, float, float]:
    c = np.clip(np.asarray(corr, dtype=float), -1.0, 1.0)
    dist = np.sqrt(np.clip(2.0 * (1.0 - c), 0.0, None))
    np.fill_diagonal(dist, 0.0)
    z = linkage(squareform(dist, checks=False), method="average")
    cid = fcluster(z, t=1.0, criterion="distance")
    _, counts = np.unique(cid, return_counts=True)
    p = counts.astype(float) / counts.sum()
    entropy = float(-np.sum(p * np.log(p + 1e-12)))
    return cid.astype(int), int(counts.size), float(np.max(p)), entropy


def _turnover_pair_frac(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    m = labels_a.size
    if m < 2:
        return float("nan")
    eq_a = labels_a[:, None] == labels_a[None, :]
    eq_b = labels_b[:, None] == labels_b[None, :]
    upper = np.triu(np.ones((m, m), dtype=bool), k=1)
    changed = np.logical_xor(eq_a, eq_b)[upper]
    return float(np.sum(changed) / max(int(upper.sum()), 1))


def _spectral_metrics(corr: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    eig = np.real(np.sort(np.linalg.eigvalsh(corr))[::-1])
    s = float(np.sum(eig))
    if s <= 0:
        return eig, float("nan"), float("nan"), float("nan")
    p = eig / s
    p1 = float(p[0])
    deff = float(1.0 / np.sum(np.square(p)))
    top5 = float(np.sum(p[: min(5, p.size)]))
    return eig, p1, deff, top5


def _spectral_metrics_with_v1(corr: np.ndarray) -> tuple[np.ndarray, float, float, float, np.ndarray]:
    w, v = np.linalg.eigh(np.asarray(corr, dtype=float))
    order = np.argsort(w)[::-1]
    eig = np.real(w[order]).astype(float)
    v1 = np.real(v[:, order[0]]).astype(float)
    nrm = float(np.linalg.norm(v1))
    if nrm > 1e-12:
        v1 = v1 / nrm
    s = float(np.sum(eig))
    if s <= 0:
        return eig, float("nan"), float("nan"), float("nan"), v1
    p = eig / s
    p1 = float(p[0])
    deff = float(1.0 / np.sum(np.square(p)))
    top5 = float(np.sum(p[: min(5, p.size)]))
    return eig, p1, deff, top5, v1


def _zscore_series(x: pd.Series) -> pd.Series:
    s = pd.to_numeric(x, errors="coerce")
    mu = float(s.mean(skipna=True))
    sd = float(s.std(ddof=0, skipna=True))
    if (not np.isfinite(sd)) or sd <= 1e-12:
        return pd.Series(np.zeros(len(s), dtype=float), index=s.index)
    z = (s - mu) / sd
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _zscore_expanding(x: pd.Series, min_history: int) -> pd.Series:
    s = pd.to_numeric(x, errors="coerce")
    k = int(max(20, min_history))
    mu = s.expanding(min_periods=k).mean().shift(1)
    sd = s.expanding(min_periods=k).std(ddof=0).shift(1)
    z = (s - mu) / sd
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _expanding_quantile(x: pd.Series, q: float, min_history: int) -> pd.Series:
    s = pd.to_numeric(x, errors="coerce")
    k = int(max(20, min_history))
    return s.expanding(min_periods=k).quantile(float(q)).shift(1)


def _block_bootstrap_col(x: np.ndarray, block_size: int, rng: np.random.Generator) -> np.ndarray:
    n = int(x.shape[0])
    if n <= 1:
        return x.copy()
    b = int(max(2, min(block_size, n)))
    out = np.empty(n, dtype=float)
    pos = 0
    while pos < n:
        start = int(rng.integers(0, n - b + 1))
        seg = x[start : start + b]
        take = min(seg.shape[0], n - pos)
        out[pos : pos + take] = seg[:take]
        pos += take
    return out


def _block_bootstrap_matrix(x: np.ndarray, block_size: int, rng: np.random.Generator) -> np.ndarray:
    y = np.empty_like(x)
    for j in range(x.shape[1]):
        y[:, j] = _block_bootstrap_col(x[:, j], block_size=block_size, rng=rng)
    return y


def _process_window(
    returns_wide: pd.DataFrame,
    sector_by_ticker: dict[str, str],
    window: int,
    cov_window: float,
    min_assets: int,
    noise_step: int,
    bootstrap_block: int,
    overlap_step: int,
    seed: int,
    compute_forman: bool = False,
    forman_topk: int = 10,
    capture_v1: bool = False,
    universe_name: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = returns_wide.index
    rows: list[dict[str, Any]] = []
    snaps: list[dict[str, Any]] = []
    sector_rows: list[dict[str, Any]] = []
    v1_rows: list[dict[str, Any]] = []
    cluster_assign: dict[pd.Timestamp, dict[str, int]] = {}
    min_obs = max(30, int(np.ceil(window * cov_window)))
    prev_v1: np.ndarray | None = None
    prev_cols: list[str] | None = None
    if (len(dates) - window + 1) <= 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    for i in range(window - 1, len(dates)):
        date = pd.Timestamp(dates[i])
        block = returns_wide.reindex(dates[i - window + 1 : i + 1])
        cov = block.notna().mean(axis=0)
        tickers_t = cov[cov >= cov_window].index.to_list()
        n_used_init = int(len(tickers_t))
        q_init = float(window) / max(float(n_used_init), 1.0)
        rec: dict[str, Any] = {
            "date": date.date().isoformat(),
            "T_window": int(window),
            "N_used": n_used_init,
            "Q": q_init,
            "p1": np.nan,
            "deff": np.nan,
            "top5": np.nan,
            "cluster_count": np.nan,
            "largest_share": np.nan,
            "entropy": np.nan,
            "turnover_pair_frac": np.nan,
            "p1_shuffle": np.nan,
            "deff_shuffle": np.nan,
            "structure_score": np.nan,
            "p1_bootstrap": np.nan,
            "deff_bootstrap": np.nan,
            "structure_score_bootstrap": np.nan,
            "eigvec_overlap_1d": np.nan,
            "eigvec_instability_1d": np.nan,
            "forman_mean": np.nan,
            "forman_share_negative": np.nan,
            "insufficient_universe": True,
        }
        for k in range(1, 11):
            rec[f"lambda{k}"] = np.nan

        if rec["N_used"] < min_assets:
            rows.append(rec)
            continue

        aligned = block[tickers_t].dropna(how="any")
        if aligned.shape[0] < min_obs or aligned.shape[1] < min_assets:
            rows.append(rec)
            continue
        x = aligned.to_numpy(dtype=float)
        std = np.nanstd(x, axis=0)
        keep = std > 1e-12
        if int(np.sum(keep)) < min_assets:
            rec["N_used"] = int(np.sum(keep))
            rec["Q"] = float(window) / max(float(rec["N_used"]), 1.0)
            rows.append(rec)
            continue
        if not np.all(keep):
            aligned = aligned.loc[:, keep]
            x = aligned.to_numpy(dtype=float)
            rec["N_used"] = int(aligned.shape[1])
            rec["Q"] = float(window) / max(float(rec["N_used"]), 1.0)

        corr = np.corrcoef(x, rowvar=False)
        if not np.all(np.isfinite(corr)):
            rows.append(rec)
            continue
        eig, p1, deff, top5 = _spectral_metrics(corr)
        if not (np.isfinite(p1) and np.isfinite(deff) and np.isfinite(top5)):
            rows.append(rec)
            continue
        rec["p1"] = p1
        rec["deff"] = deff
        rec["top5"] = top5
        rec["insufficient_universe"] = False
        for k in range(1, 11):
            rec[f"lambda{k}"] = float(eig[k - 1]) if eig.size >= k else np.nan
        calc_overlap = (int(max(1, overlap_step)) == 1) or (i % int(max(1, overlap_step)) == 0)
        calc_v1 = bool(capture_v1) or bool(calc_overlap)
        if calc_v1:
            try:
                _, _, _, _, v1 = _spectral_metrics_with_v1(corr)
                if bool(capture_v1):
                    for t, wgt in zip(aligned.columns.to_list(), v1.tolist()):
                        v1_rows.append(
                            {
                                "date": date.date().isoformat(),
                                "universe_name": str(universe_name),
                                "asset_id": str(t),
                                "weight": float(wgt),
                            }
                        )
                if bool(calc_overlap):
                    if (prev_v1 is not None) and (prev_cols is not None):
                        common = sorted(set(prev_cols).intersection(aligned.columns.to_list()))
                        if len(common) >= 3:
                            prev_idx = [prev_cols.index(c) for c in common]
                            cur_idx = [aligned.columns.get_loc(c) for c in common]
                            a = prev_v1[prev_idx]
                            b = v1[cur_idx]
                            na = float(np.linalg.norm(a))
                            nb = float(np.linalg.norm(b))
                            if (na > 1e-12) and (nb > 1e-12):
                                ov = float(abs(np.dot(a / na, b / nb)))
                                rec["eigvec_overlap_1d"] = ov
                                rec["eigvec_instability_1d"] = float(1.0 - ov)
                    prev_v1 = v1
                    prev_cols = aligned.columns.to_list()
            except Exception:
                pass

        if bool(compute_forman) and STRUCTURAL_OK:
            try:
                edges = corr_to_graph(corr, method="topk", k=int(max(1, forman_topk)), abs_weights=True)
                curv = forman_edge_curvature(edges, n_nodes=int(corr.shape[0]))
                fsum = forman_summary(curv)
                rec["forman_mean"] = float(fsum.get("mean", np.nan))
                rec["forman_share_negative"] = float(fsum.get("share_negative", np.nan))
            except Exception:
                rec["forman_mean"] = np.nan
                rec["forman_share_negative"] = np.nan

        if SCIPY_OK:
            cid, ccount, largest_share, entropy = _cluster_metrics(corr)
            rec["cluster_count"] = ccount
            rec["largest_share"] = largest_share
            rec["entropy"] = entropy
            amap = {t: int(c) for t, c in zip(aligned.columns.to_list(), cid.tolist())}
            cluster_assign[date] = amap
            is_eom = (i == len(dates) - 1) or (pd.Timestamp(dates[i + 1]).month != date.month)
            if is_eom:
                for t, c in amap.items():
                    snaps.append({"date": date.date().isoformat(), "ticker": t, "cluster_id": c})
            vc = pd.Series([sector_by_ticker.get(t, "unknown") for t in amap.keys()]).value_counts()
            for sec, cnt in vc.items():
                sector_rows.append({"date": date.date().isoformat(), "sector": str(sec), "count": int(cnt)})

        prev_i = i - 5
        if prev_i >= (window - 1):
            prev_date = pd.Timestamp(dates[prev_i])
            cm = cluster_assign.get(date)
            pm = cluster_assign.get(prev_date)
            if cm is not None and pm is not None:
                common = sorted(set(cm.keys()).intersection(pm.keys()))
                if len(common) >= 2:
                    rec["turnover_pair_frac"] = _turnover_pair_frac(
                        np.asarray([cm[t] for t in common], dtype=int),
                        np.asarray([pm[t] for t in common], dtype=int),
                    )

        if (i % max(noise_step, 1) == 0) and (aligned.shape[1] >= min_assets):
            rng = np.random.default_rng(seed + window * 10_000 + i)
            x_shuffle = aligned.to_numpy(dtype=float).copy()
            for j in range(x_shuffle.shape[1]):
                rng.shuffle(x_shuffle[:, j])
            keep_sh = np.nanstd(x_shuffle, axis=0) > 1e-12
            if int(np.sum(keep_sh)) >= 2:
                corr_shuffle = np.corrcoef(x_shuffle[:, keep_sh], rowvar=False)
                if np.all(np.isfinite(corr_shuffle)):
                    _, p1_sh, deff_sh, _ = _spectral_metrics(corr_shuffle)
                    if np.isfinite(p1_sh) and np.isfinite(deff_sh):
                        rec["p1_shuffle"] = p1_sh
                        rec["deff_shuffle"] = deff_sh
                        rec["structure_score"] = float((rec["p1"] - p1_sh) + (deff_sh - rec["deff"]))

            x_boot = _block_bootstrap_matrix(
                aligned.to_numpy(dtype=float),
                block_size=int(max(2, bootstrap_block)),
                rng=np.random.default_rng(seed + window * 20_000 + i),
            )
            keep_bs = np.nanstd(x_boot, axis=0) > 1e-12
            if int(np.sum(keep_bs)) >= 2:
                corr_boot = np.corrcoef(x_boot[:, keep_bs], rowvar=False)
                if np.all(np.isfinite(corr_boot)):
                    _, p1_bs, deff_bs, _ = _spectral_metrics(corr_boot)
                    if np.isfinite(p1_bs) and np.isfinite(deff_bs):
                        rec["p1_bootstrap"] = p1_bs
                        rec["deff_bootstrap"] = deff_bs
                        rec["structure_score_bootstrap"] = float((rec["p1"] - p1_bs) + (deff_bs - rec["deff"]))

        rows.append(rec)
    return pd.DataFrame(rows), pd.DataFrame(snaps), pd.DataFrame(sector_rows), pd.DataFrame(v1_rows)


def _summary_block(ts: pd.DataFrame, sector_daily: pd.DataFrame, window: int, outdir: Path) -> str:
    if ts.empty:
        return f"[T={window}] sem dados."
    suff = ts[~ts["insufficient_universe"]].copy()
    tail = ts.tail(60)
    tails = tail[~tail["insufficient_universe"]]
    if not sector_daily.empty:
        sec = (
            sector_daily.groupby("sector", as_index=False)
            .agg(min_count=("count", "min"), mean_count=("count", "mean"), max_count=("count", "max"))
            .sort_values("sector")
        )
        sec.to_csv(outdir / f"sector_presence_T{window}.csv", index=False)
        z = sec[sec["min_count"] <= 0]["sector"].tolist()
        sanity = f"setor_sanity={'OK' if len(z)==0 else 'WARN'}; sectors_with_zero={','.join(z) if z else 'none'}"
    else:
        sanity = "setor_sanity=unavailable"

    stress = "sem dias suficientes para stress check"
    if not suff.empty:
        p1q, dq = float(suff["p1"].quantile(0.95)), float(suff["deff"].quantile(0.05))
        s = suff[(suff["p1"] >= p1q) & (suff["deff"] <= dq)]
        stress = (
            f"stress_count={int(s.shape[0])}; p1_q95={p1q:.6f}; deff_q05={dq:.6f}; "
            f"last_dates={','.join(s['date'].tail(10).to_list()) if not s.empty else 'none'}"
        )

    noise_line = "noise_baseline: unavailable"
    if not suff.empty:
        b = suff.dropna(subset=["p1_shuffle", "deff_shuffle"])
        b2 = suff.dropna(subset=["p1_bootstrap", "deff_bootstrap"])
        noise_line = (
            "noise_baseline: "
            f"shuffle_p1_gap={float((b['p1'] - b['p1_shuffle']).mean()) if not b.empty else np.nan:.4f}, "
            f"shuffle_deff_gap={float((b['deff_shuffle'] - b['deff']).mean()) if not b.empty else np.nan:.4f}, "
            f"bootstrap_p1_gap={float((b2['p1'] - b2['p1_bootstrap']).mean()) if not b2.empty else np.nan:.4f}, "
            f"bootstrap_deff_gap={float((b2['deff_bootstrap'] - b2['deff']).mean()) if not b2.empty else np.nan:.4f}"
        )

    ov_mean = float(tails["eigvec_overlap_1d"].mean()) if ("eigvec_overlap_1d" in tails.columns and (not tails.empty)) else np.nan
    return "\n".join(
        [
            f"[T={window}]",
            f"N_used stats: min={float(ts['N_used'].min()):.0f}, mean={float(ts['N_used'].mean()):.2f}, max={float(ts['N_used'].max()):.0f}, Q_min={float((float(window) / ts['N_used'].astype(float).clip(lower=1.0)).min()):.4f}, Q_med={float((float(window) / ts['N_used'].astype(float).clip(lower=1.0)).median()):.4f}, insufficient_days={int(ts['insufficient_universe'].sum())}",
            f"ultimos_60_dias: p1_mean={float(tails['p1'].mean()) if not tails.empty else np.nan:.6f}, deff_mean={float(tails['deff'].mean()) if not tails.empty else np.nan:.6f}, turnover_mean={float(tails['turnover_pair_frac'].mean()) if not tails.empty else np.nan:.6f}, eigvec_overlap_mean={ov_mean:.6f}",
            f"stress: {stress}",
            noise_line,
            f"sanity: {sanity}",
        ]
    )


def _majority_same_direction(signs: np.ndarray) -> int:
    nz = signs[signs != 0]
    if nz.size < 2:
        return 0
    return int(abs(int(np.sum(nz))) >= 2)


def _build_robustness(ts_map: dict[int, pd.DataFrame], outdir: Path) -> tuple[pd.DataFrame, dict[str, float], str]:
    windows = sorted(ts_map.keys())
    windows_present: list[int] = []
    parts = []
    for w in windows:
        d = ts_map[w].copy()
        if d.empty or ("date" not in d.columns):
            continue
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        d = d.dropna(subset=["date"])
        if "insufficient_universe" not in d.columns:
            d["insufficient_universe"] = False
        d = d[~d["insufficient_universe"]].set_index("date")
        if d.empty:
            continue
        parts.append(d[["p1", "deff"]].rename(columns={"p1": f"p1_T{w}", "deff": f"deff_T{w}"}))
        windows_present.append(w)
    if not parts:
        return pd.DataFrame(), {}, "temporal_robustness=sem_dados"
    m = parts[0]
    for p in parts[1:]:
        m = m.join(p, how="inner")
    if m.empty:
        return pd.DataFrame(), {}, "temporal_robustness=sem_intersecao_datas"
    if not windows_present:
        return pd.DataFrame(), {}, "temporal_robustness=sem_janelas_validas"

    for w in windows_present:
        m[f"dp1_5_T{w}"] = m[f"p1_T{w}"].diff(5)
        m[f"ddeff_5_T{w}"] = m[f"deff_T{w}"].diff(5)

    dp1_sign = _safe_sign(m[[f"dp1_5_T{w}" for w in windows_present]].to_numpy())
    ddeff_sign = _safe_sign(m[[f"ddeff_5_T{w}" for w in windows_present]].to_numpy())
    m["p1_dir_consistent_5"] = ((np.all(dp1_sign != 0, axis=1)) & (np.abs(np.sum(dp1_sign, axis=1)) == len(windows_present))).astype(int)
    m["deff_dir_consistent_5"] = ((np.all(ddeff_sign != 0, axis=1)) & (np.abs(np.sum(ddeff_sign, axis=1)) == len(windows_present))).astype(int)
    m["joint_dir_consistent_5"] = ((m["p1_dir_consistent_5"] == 1) & (m["deff_dir_consistent_5"] == 1)).astype(int)

    p1_majority = np.array([_majority_same_direction(dp1_sign[i, :]) for i in range(dp1_sign.shape[0])], dtype=int)
    deff_majority = np.array([_majority_same_direction(ddeff_sign[i, :]) for i in range(ddeff_sign.shape[0])], dtype=int)
    m["p1_dir_majority_5"] = p1_majority
    m["deff_dir_majority_5"] = deff_majority
    m["joint_majority_5"] = ((p1_majority == 1) & (deff_majority == 1)).astype(int)

    out = m.reset_index().rename(columns={"index": "date"})
    out["date"] = out["date"].dt.date.astype(str)
    out.to_csv(outdir / "robustness_temporal.csv", index=False)

    tail = out.tail(60)
    latest = out.iloc[-1]
    signs = []
    for w in windows_present:
        sp = int(_safe_sign(np.asarray([latest[f"dp1_5_T{w}"]]))[0])
        sd = int(_safe_sign(np.asarray([latest[f"ddeff_5_T{w}"]]))[0])
        signs.append(f"T{w}(dp1={sp:+d},ddeff={sd:+d})")

    metrics = {
        "p1_consistency_60d": float(tail["p1_dir_consistent_5"].mean()) if not tail.empty else float("nan"),
        "deff_consistency_60d": float(tail["deff_dir_consistent_5"].mean()) if not tail.empty else float("nan"),
        "joint_consistency_60d": float(tail["joint_dir_consistent_5"].mean()) if not tail.empty else float("nan"),
        "p1_majority_60d": float(tail["p1_dir_majority_5"].mean()) if not tail.empty else float("nan"),
        "deff_majority_60d": float(tail["deff_dir_majority_5"].mean()) if not tail.empty else float("nan"),
        "joint_majority_60d": float(tail["joint_majority_5"].mean()) if not tail.empty else float("nan"),
        "latest_joint_majority_5": float(latest["joint_majority_5"]),
        "latest_joint_consistent_5": float(latest["joint_dir_consistent_5"]),
    }
    txt = (
        "temporal_robustness_60d: "
        f"joint_consistency={metrics['joint_consistency_60d']:.3f}, "
        f"joint_majority={metrics['joint_majority_60d']:.3f}, "
        f"latest_joint_majority={int(metrics['latest_joint_majority_5'])}; "
        f"latest_signs={'; '.join(signs)}"
    )
    return out, metrics, txt


def _apply_hysteresis(labels: list[str], min_persist: int) -> list[str]:
    if not labels:
        return []
    k = int(max(1, min_persist))
    current = labels[0]
    pending = ""
    cnt = 0
    out = [current]
    for raw in labels[1:]:
        if raw == current:
            pending = ""
            cnt = 0
            out.append(current)
            continue
        if raw == pending:
            cnt += 1
        else:
            pending = raw
            cnt = 1
        if cnt >= k:
            current = pending
            pending = ""
            cnt = 0
        out.append(current)
    return out


def _classify_regime(
    ts: pd.DataFrame,
    hysteresis_days: int,
    exp_stress: float,
    exp_transition: float,
    exp_stable: float,
    exp_dispersion: float,
    w_dp1: float,
    w_ddeff: float,
    w_overlap: float,
    threshold_mode: str = "full_sample",
    walkforward_min_history: int = 252,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    d = ts.copy()
    if d.empty or ("date" not in d.columns):
        return pd.DataFrame(), {}
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date"])
    d = d[~d["insufficient_universe"]].sort_values("date")
    if d.empty:
        return pd.DataFrame(), {}

    d["dp1_5"] = d["p1"].diff(5)
    d["ddeff_5"] = d["deff"].diff(5)
    d["overlap_instability"] = 1.0 - pd.to_numeric(d.get("eigvec_overlap_1d"), errors="coerce")
    mode = str(threshold_mode or "full_sample").strip().lower()
    use_walkforward = mode in {"walk_forward", "walkforward", "expanding", "causal"}
    min_hist = int(max(20, walkforward_min_history))

    if use_walkforward:
        z_dp1 = _zscore_expanding(d["dp1_5"].abs(), min_history=min_hist)
        z_ddeff = _zscore_expanding(d["ddeff_5"].abs(), min_history=min_hist)
        z_ov = _zscore_expanding(d["overlap_instability"], min_history=min_hist)
    else:
        z_dp1 = _zscore_series(d["dp1_5"].abs())
        z_ddeff = _zscore_series(d["ddeff_5"].abs())
        z_ov = _zscore_series(d["overlap_instability"])
    d["transition_score"] = float(w_dp1) * z_dp1 + float(w_ddeff) * z_ddeff + float(w_overlap) * z_ov

    if use_walkforward:
        p1_lo_s = _expanding_quantile(d["p1"], q=0.20, min_history=min_hist)
        p1_hi_s = _expanding_quantile(d["p1"], q=0.80, min_history=min_hist)
        deff_lo_s = _expanding_quantile(d["deff"], q=0.20, min_history=min_hist)
        deff_hi_s = _expanding_quantile(d["deff"], q=0.80, min_history=min_hist)
        dp1_thr_s = _expanding_quantile(d["dp1_5"].abs(), q=0.80, min_history=min_hist)
        ddeff_thr_s = _expanding_quantile(d["ddeff_5"].abs(), q=0.80, min_history=min_hist)
        tr_thr_s = _expanding_quantile(pd.to_numeric(d["transition_score"], errors="coerce"), q=0.80, min_history=min_hist)
    else:
        p1_lo = float(d["p1"].quantile(0.20))
        p1_hi = float(d["p1"].quantile(0.80))
        deff_lo = float(d["deff"].quantile(0.20))
        deff_hi = float(d["deff"].quantile(0.80))
        dp1_thr = float(d["dp1_5"].abs().quantile(0.80))
        ddeff_thr = float(d["ddeff_5"].abs().quantile(0.80))
        tr_thr = float(pd.to_numeric(d["transition_score"], errors="coerce").quantile(0.80))
        p1_lo_s = pd.Series(np.full(len(d), p1_lo), index=d.index, dtype=float)
        p1_hi_s = pd.Series(np.full(len(d), p1_hi), index=d.index, dtype=float)
        deff_lo_s = pd.Series(np.full(len(d), deff_lo), index=d.index, dtype=float)
        deff_hi_s = pd.Series(np.full(len(d), deff_hi), index=d.index, dtype=float)
        dp1_thr_s = pd.Series(np.full(len(d), dp1_thr), index=d.index, dtype=float)
        ddeff_thr_s = pd.Series(np.full(len(d), ddeff_thr), index=d.index, dtype=float)
        tr_thr_s = pd.Series(np.full(len(d), tr_thr), index=d.index, dtype=float)

    d["thr_p1_q20"] = p1_lo_s
    d["thr_p1_q80"] = p1_hi_s
    d["thr_deff_q20"] = deff_lo_s
    d["thr_deff_q80"] = deff_hi_s
    d["thr_abs_dp1_5_q80"] = dp1_thr_s
    d["thr_abs_ddeff_5_q80"] = ddeff_thr_s
    d["thr_transition_score_q80"] = tr_thr_s

    raw_reg = []
    for _, r in d.iterrows():
        p1_lo_t = float(r.get("thr_p1_q20", np.nan))
        p1_hi_t = float(r.get("thr_p1_q80", np.nan))
        deff_lo_t = float(r.get("thr_deff_q20", np.nan))
        deff_hi_t = float(r.get("thr_deff_q80", np.nan))
        dp1_thr_t = float(r.get("thr_abs_dp1_5_q80", np.nan))
        ddeff_thr_t = float(r.get("thr_abs_ddeff_5_q80", np.nan))
        tr_thr_t = float(r.get("thr_transition_score_q80", np.nan))

        if not (
            np.isfinite(p1_lo_t)
            and np.isfinite(p1_hi_t)
            and np.isfinite(deff_lo_t)
            and np.isfinite(deff_hi_t)
            and np.isfinite(dp1_thr_t)
            and np.isfinite(ddeff_thr_t)
            and np.isfinite(tr_thr_t)
        ):
            raw_reg.append("stable")
            continue

        if (float(r["p1"]) >= p1_hi_t) and (float(r["deff"]) <= deff_lo_t):
            raw_reg.append("stress")
        elif (float(r["p1"]) <= p1_lo_t) and (float(r["deff"]) >= deff_hi_t):
            raw_reg.append("dispersion")
        elif (
            (abs(float(r["dp1_5"]) if pd.notna(r["dp1_5"]) else 0.0) >= dp1_thr_t)
            or (abs(float(r["ddeff_5"]) if pd.notna(r["ddeff_5"]) else 0.0) >= ddeff_thr_t)
            or (
                float(r.get("transition_score", np.nan)) >= tr_thr_t
                if pd.notna(r.get("transition_score", np.nan))
                else False
            )
        ):
            raw_reg.append("transition")
        else:
            raw_reg.append("stable")

    reg_hys = _apply_hysteresis(raw_reg, min_persist=int(max(1, hysteresis_days)))
    d["regime_raw"] = raw_reg
    d["regime"] = reg_hys
    exp_map = {
        "stress": float(exp_stress),
        "transition": float(exp_transition),
        "stable": float(exp_stable),
        "dispersion": float(exp_dispersion),
    }
    d["exposure"] = d["regime"].map(exp_map).fillna(float(exp_stable)).astype(float).clip(0.0, 1.0)
    d["date"] = d["date"].dt.date.astype(str)
    p1_lo = float(pd.to_numeric(d["thr_p1_q20"], errors="coerce").dropna().iloc[-1]) if d["thr_p1_q20"].notna().any() else float("nan")
    p1_hi = float(pd.to_numeric(d["thr_p1_q80"], errors="coerce").dropna().iloc[-1]) if d["thr_p1_q80"].notna().any() else float("nan")
    deff_lo = float(pd.to_numeric(d["thr_deff_q20"], errors="coerce").dropna().iloc[-1]) if d["thr_deff_q20"].notna().any() else float("nan")
    deff_hi = float(pd.to_numeric(d["thr_deff_q80"], errors="coerce").dropna().iloc[-1]) if d["thr_deff_q80"].notna().any() else float("nan")
    dp1_thr = float(pd.to_numeric(d["thr_abs_dp1_5_q80"], errors="coerce").dropna().iloc[-1]) if d["thr_abs_dp1_5_q80"].notna().any() else float("nan")
    ddeff_thr = float(pd.to_numeric(d["thr_abs_ddeff_5_q80"], errors="coerce").dropna().iloc[-1]) if d["thr_abs_ddeff_5_q80"].notna().any() else float("nan")
    tr_thr = float(pd.to_numeric(d["thr_transition_score_q80"], errors="coerce").dropna().iloc[-1]) if d["thr_transition_score_q80"].notna().any() else float("nan")
    warmup_days = int(d["thr_transition_score_q80"].isna().sum())
    meta = {
        "threshold_mode": "walk_forward" if use_walkforward else "full_sample",
        "walkforward_min_history": float(min_hist),
        "warmup_days_without_threshold": float(warmup_days),
        "p1_q20": p1_lo,
        "p1_q80": p1_hi,
        "deff_q20": deff_lo,
        "deff_q80": deff_hi,
        "abs_dp1_5_q80": dp1_thr,
        "abs_ddeff_5_q80": ddeff_thr,
        "transition_score_q80": tr_thr,
        "hysteresis_days": float(max(1, hysteresis_days)),
        "w_dp1": float(w_dp1),
        "w_ddeff": float(w_ddeff),
        "w_overlap": float(w_overlap),
        "exp_stress": float(exp_stress),
        "exp_transition": float(exp_transition),
        "exp_stable": float(exp_stable),
        "exp_dispersion": float(exp_dispersion),
    }
    return d, meta


def _perf(rets: pd.Series) -> dict[str, float]:
    x = pd.to_numeric(rets, errors="coerce").dropna().astype(float)
    if x.empty:
        return {"ann_return": float("nan"), "ann_vol": float("nan"), "sharpe": float("nan"), "max_drawdown": float("nan")}
    eq = _safe_cum_equity(x)
    ann = float(np.power(float(eq.iloc[-1]), 252.0 / max(int(x.shape[0]), 1)) - 1.0)
    vol = float(x.std(ddof=0) * np.sqrt(252.0))
    return {
        "ann_return": ann,
        "ann_vol": vol,
        "sharpe": float(ann / vol) if vol > 1e-12 else float("nan"),
        "max_drawdown": float((eq / eq.cummax() - 1.0).min()),
    }


def _safe_cum_equity(simple_rets: pd.Series) -> pd.Series:
    x = pd.to_numeric(simple_rets, errors="coerce").astype(float)
    gross = (1.0 + x).clip(lower=1e-9, upper=10.0)
    log_eq = np.log(gross).cumsum()
    log_eq = pd.Series(log_eq, index=x.index).clip(lower=-50.0, upper=50.0)
    return np.exp(log_eq)


def _backtest(
    regime_df: pd.DataFrame,
    returns_wide: pd.DataFrame,
    cost_bps: float,
    max_daily_turnover: float,
    start_exposure: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if regime_df.empty:
        return pd.DataFrame(), {"error": "empty_regime_df"}

    mkt = returns_wide.mean(axis=1, skipna=True).rename("mkt_log_ret")
    d = regime_df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date"]).set_index("date")
    bt = d.join(mkt, how="inner").dropna(subset=["mkt_log_ret", "exposure"]).copy()
    if bt.empty:
        return pd.DataFrame(), {"error": "no_overlap"}

    bt["mkt_simple_ret_raw"] = np.expm1(bt["mkt_log_ret"].astype(float))
    bt["mkt_simple_ret"] = bt["mkt_simple_ret_raw"].clip(lower=-0.95, upper=2.0)
    bt["exposure_target"] = bt["exposure"].astype(float).clip(0.0, 1.0)
    target_exec = bt["exposure_target"].shift(1).fillna(float(start_exposure)).to_numpy(dtype=float)
    cap = float(max(0.0, max_daily_turnover))
    exec_exp = np.zeros_like(target_exec)
    exec_exp[0] = float(np.clip(start_exposure, 0.0, 1.0))
    for i in range(1, target_exec.shape[0]):
        desired = float(np.clip(target_exec[i], 0.0, 1.0))
        if cap <= 0:
            exec_exp[i] = desired
            continue
        step = float(np.clip(desired - exec_exp[i - 1], -cap, cap))
        exec_exp[i] = float(np.clip(exec_exp[i - 1] + step, 0.0, 1.0))
    bt["exposure_exec"] = exec_exp
    bt["turnover"] = bt["exposure_exec"].diff().abs().fillna(0.0)
    bt["cost"] = bt["turnover"] * (float(cost_bps) / 10000.0)
    bt["strategy_simple_ret_raw"] = bt["exposure_exec"] * bt["mkt_simple_ret"] - bt["cost"]
    bt["strategy_simple_ret"] = bt["strategy_simple_ret_raw"].clip(lower=-0.99, upper=2.0)
    bt["benchmark_simple_ret"] = bt["mkt_simple_ret"]
    bt["strategy_equity"] = _safe_cum_equity(bt["strategy_simple_ret"])
    bt["benchmark_equity"] = _safe_cum_equity(bt["benchmark_simple_ret"])
    s, b = _perf(bt["strategy_simple_ret"]), _perf(bt["benchmark_simple_ret"])
    n_clip_mkt = int((bt["mkt_simple_ret_raw"] != bt["mkt_simple_ret"]).sum())
    n_clip_strategy = int((bt["strategy_simple_ret_raw"] != bt["strategy_simple_ret"]).sum())
    summary = {
        "start": bt.index.min().date().isoformat(),
        "end": bt.index.max().date().isoformat(),
        "n_days": int(bt.shape[0]),
        "strategy": s,
        "benchmark": b,
        "avg_exposure": float(bt["exposure_exec"].mean()),
        "avg_turnover": float(bt["turnover"].mean()),
        "total_cost_paid": float(bt["cost"].sum()),
        "max_daily_turnover": float(cap),
        "n_clip_mkt_simple_ret": int(n_clip_mkt),
        "n_clip_strategy_simple_ret": int(n_clip_strategy),
        "regime_counts": bt["regime"].value_counts().to_dict(),
    }
    out = bt.reset_index().rename(columns={"index": "date"})
    out["date"] = out["date"].dt.date.astype(str)
    return out, summary


def _cluster_alerts(ts: pd.DataFrame, lookback: int) -> tuple[pd.DataFrame, dict[str, Any], str]:
    d = ts.copy()
    if d.empty or ("date" not in d.columns):
        return pd.DataFrame(), {"n_active": 0, "active_metrics": []}, "cluster_alerts=sem_dados"
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date"])
    d = d[~d["insufficient_universe"]]
    if d.empty:
        return pd.DataFrame(), {"n_active": 0, "active_metrics": []}, "cluster_alerts=sem_dados"
    for c in ["cluster_count", "largest_share", "entropy", "turnover_pair_frac"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["cluster_count", "largest_share", "entropy"])
    if d.empty:
        return pd.DataFrame(), {"n_active": 0, "active_metrics": []}, "cluster_alerts=sem_metricas"

    last = d.iloc[-1]
    rows: list[dict[str, Any]] = []
    rules = [("cluster_count", "low"), ("largest_share", "high"), ("entropy", "low"), ("turnover_pair_frac", "high")]
    for metric, side in rules:
        s_all = d[metric].dropna()
        if s_all.empty:
            continue
        s = s_all.tail(max(30, min(int(lookback), int(s_all.shape[0]))))
        q05, q95 = float(s.quantile(0.05)), float(s.quantile(0.95))
        val = float(last[metric])
        if side == "high":
            alert = bool(val >= q95)
            rule = f">=q95({q95:.6f})"
            sev = float((val - q95) / max(abs(q95), 1e-9))
        else:
            alert = bool(val <= q05)
            rule = f"<=q05({q05:.6f})"
            sev = float((q05 - val) / max(abs(q05), 1e-9))
        rows.append(
            {
                "date": last["date"].date().isoformat(),
                "metric": metric,
                "latest_value": val,
                "q05": q05,
                "q95": q95,
                "rule": rule,
                "severity_ratio": sev if alert else 0.0,
                "alert": alert,
            }
        )

    a = pd.DataFrame(rows)
    active = a[a["alert"]]["metric"].tolist() if not a.empty else []
    payload = {"n_active": int(len(active)), "active_metrics": active}
    return a, payload, f"cluster_alerts_active={','.join(active) if active else 'none'}"


def _era_name(date_like: Any) -> str:
    dt = pd.Timestamp(date_like)
    y = int(dt.year)
    if y <= 2019:
        return "2018_2019"
    if y == 2020:
        return "2020"
    if y <= 2022:
        return "2021_2022"
    return "2023_2026"


def _build_operational_alerts(
    ts_off: pd.DataFrame,
    regime_df: pd.DataFrame,
    robust_metrics: dict[str, float],
) -> tuple[pd.DataFrame, dict[str, Any], str]:
    if ts_off.empty or regime_df.empty:
        payload = {"latest_date": "", "latest_events": [], "n_events_total": 0, "n_events_last_60d": 0, "event_counts": {}}
        return pd.DataFrame(), payload, "operational_alerts=sem_dados"

    ts = ts_off.copy()
    ts["date"] = pd.to_datetime(ts["date"], errors="coerce")
    ts = ts.dropna(subset=["date"])
    ts = ts[~ts["insufficient_universe"]].copy()
    rg = regime_df.copy()
    rg["date"] = pd.to_datetime(rg["date"], errors="coerce")
    rg = rg.dropna(subset=["date"]).copy()
    d = rg.merge(ts[["date", "N_used", "p1", "deff"]], on="date", how="left")
    for base in ["N_used", "p1", "deff"]:
        if base in d.columns:
            continue
        for alt in [f"{base}_x", f"{base}_y", f"{base}_ts"]:
            if alt in d.columns:
                d[base] = pd.to_numeric(d[alt], errors="coerce")
                break
    d = d.sort_values("date").reset_index(drop=True)
    if d.empty:
        payload = {"latest_date": "", "latest_events": [], "n_events_total": 0, "n_events_last_60d": 0, "event_counts": {}}
        return pd.DataFrame(), payload, "operational_alerts=sem_dados"

    p1_q25 = float(pd.to_numeric(d["p1"], errors="coerce").quantile(0.25))
    p1_q80 = float(pd.to_numeric(d["p1"], errors="coerce").quantile(0.80))
    deff_q75 = float(pd.to_numeric(d["deff"], errors="coerce").quantile(0.75))
    abs_dp1_q90 = float(pd.to_numeric(d["dp1_5"], errors="coerce").abs().quantile(0.90))
    abs_dd_q90 = float(pd.to_numeric(d["ddeff_5"], errors="coerce").abs().quantile(0.90))

    events: list[dict[str, Any]] = []
    for i, r in d.iterrows():
        date_s = pd.Timestamp(r["date"]).date().isoformat()
        reg = str(r.get("regime", ""))
        p1 = float(r.get("p1", np.nan))
        deff = float(r.get("deff", np.nan))
        dp1 = abs(float(r.get("dp1_5", np.nan)))
        dd = abs(float(r.get("ddeff_5", np.nan)))
        prev_reg = str(d.iloc[i - 1]["regime"]) if i > 0 else ""

        if i > 0 and reg != prev_reg:
            events.append(
                {
                    "date": date_s,
                    "event_code": "REGIME_CHANGE",
                    "severity": 1.0,
                    "regime": reg,
                    "message": f"regime mudou de {prev_reg} para {reg}.",
                    "p1": p1,
                    "deff": deff,
                    "dp1_5": float(r.get("dp1_5", np.nan)),
                    "ddeff_5": float(r.get("ddeff_5", np.nan)),
                }
            )

        if reg == "stress" and prev_reg != "stress":
            sev = float(max(0.0, (p1 - p1_q80) / max(abs(p1_q80), 1e-9))) if np.isfinite(p1_q80) else 0.0
            events.append(
                {
                    "date": date_s,
                    "event_code": "STRESS_ENTER",
                    "severity": sev,
                    "regime": reg,
                    "message": "entrada em stress: aumentar defesa e limitar risco.",
                    "p1": p1,
                    "deff": deff,
                    "dp1_5": float(r.get("dp1_5", np.nan)),
                    "ddeff_5": float(r.get("ddeff_5", np.nan)),
                }
            )

        if i > 0 and prev_reg == "stress" and reg != "stress":
            sev = float(max(0.0, (p1_q80 - p1) / max(abs(p1_q80), 1e-9))) if np.isfinite(p1_q80) else 0.0
            events.append(
                {
                    "date": date_s,
                    "event_code": "STRESS_EXIT",
                    "severity": sev,
                    "regime": reg,
                    "message": "saida de stress: liberar risco de forma gradual.",
                    "p1": p1,
                    "deff": deff,
                    "dp1_5": float(r.get("dp1_5", np.nan)),
                    "ddeff_5": float(r.get("ddeff_5", np.nan)),
                }
            )

        if reg == "transition" and (
            (np.isfinite(dp1) and np.isfinite(abs_dp1_q90) and dp1 >= abs_dp1_q90)
            or (np.isfinite(dd) and np.isfinite(abs_dd_q90) and dd >= abs_dd_q90)
        ):
            sev_a = (dp1 / max(abs_dp1_q90, 1e-9)) if np.isfinite(abs_dp1_q90) and np.isfinite(dp1) else 0.0
            sev_b = (dd / max(abs_dd_q90, 1e-9)) if np.isfinite(abs_dd_q90) and np.isfinite(dd) else 0.0
            events.append(
                {
                    "date": date_s,
                    "event_code": "TRANSITION_SPIKE",
                    "severity": float(max(sev_a, sev_b) - 1.0),
                    "regime": reg,
                    "message": "transicao acelerando: reduzir tamanho e esperar confirmacao.",
                    "p1": p1,
                    "deff": deff,
                    "dp1_5": float(r.get("dp1_5", np.nan)),
                    "ddeff_5": float(r.get("ddeff_5", np.nan)),
                }
            )

        if reg == "dispersion" and np.isfinite(p1) and np.isfinite(deff) and np.isfinite(p1_q25) and np.isfinite(deff_q75):
            if (p1 <= p1_q25) and (deff >= deff_q75):
                sev = float(max(0.0, (deff - deff_q75) / max(abs(deff_q75), 1e-9)))
                events.append(
                    {
                        "date": date_s,
                        "event_code": "DISPERSION_CONFIRM",
                        "severity": sev,
                        "regime": reg,
                        "message": "dispersao confirmada: priorizar rotacao e selecao.",
                        "p1": p1,
                        "deff": deff,
                        "dp1_5": float(r.get("dp1_5", np.nan)),
                        "ddeff_5": float(r.get("ddeff_5", np.nan)),
                    }
                )

    latest_date = pd.Timestamp(d.iloc[-1]["date"]).date().isoformat()
    latest_jm = float(robust_metrics.get("latest_joint_majority_5", np.nan))
    jm60 = float(robust_metrics.get("joint_majority_60d", np.nan))
    if np.isfinite(latest_jm) and int(latest_jm) == 0:
        r = d.iloc[-1]
        events.append(
            {
                "date": latest_date,
                "event_code": "SIGNAL_UNCONFIRMED",
                "severity": 1.0,
                "regime": str(r.get("regime", "")),
                "message": "direcao recente ainda sem confirmacao entre janelas.",
                "p1": float(r.get("p1", np.nan)),
                "deff": float(r.get("deff", np.nan)),
                "dp1_5": float(r.get("dp1_5", np.nan)),
                "ddeff_5": float(r.get("ddeff_5", np.nan)),
            }
        )
    if np.isfinite(jm60) and (jm60 < 0.30):
        r = d.iloc[-1]
        events.append(
            {
                "date": latest_date,
                "event_code": "ROBUSTNESS_LOW",
                "severity": float(max(0.0, (0.30 - jm60) / 0.30)),
                "regime": str(r.get("regime", "")),
                "message": "consistencia 60d abaixo do alvo operacional.",
                "p1": float(r.get("p1", np.nan)),
                "deff": float(r.get("deff", np.nan)),
                "dp1_5": float(r.get("dp1_5", np.nan)),
                "ddeff_5": float(r.get("ddeff_5", np.nan)),
            }
        )

    ev = pd.DataFrame(events)
    if ev.empty:
        payload = {"latest_date": latest_date, "latest_events": [], "n_events_total": 0, "n_events_last_60d": 0, "event_counts": {}}
        return ev, payload, "operational_alerts=none"

    ev["date"] = pd.to_datetime(ev["date"], errors="coerce")
    ev = ev.dropna(subset=["date"]).sort_values(["date", "event_code"]).reset_index(drop=True)
    latest_ts = ev["date"].max()
    latest_mask = ev["date"] == latest_ts
    last60_cut = latest_ts - pd.Timedelta(days=60)
    last60 = ev[ev["date"] >= last60_cut].copy()
    latest_events = latest_mask.sum()
    payload = {
        "latest_date": latest_ts.date().isoformat(),
        "latest_events": ev.loc[latest_mask, "event_code"].astype(str).tolist(),
        "n_events_total": int(ev.shape[0]),
        "n_events_last_60d": int(last60.shape[0]),
        "event_counts": ev["event_code"].value_counts().to_dict(),
        "latest_event_rows": ev.loc[latest_mask]
        .assign(date=lambda x: x["date"].dt.date.astype(str))
        .to_dict(orient="records"),
    }
    ev["date"] = ev["date"].dt.date.astype(str)
    txt = f"operational_alerts_latest={','.join(payload['latest_events']) if payload['latest_events'] else 'none'}"
    return ev, payload, txt


def _apply_level_persistence(levels: list[str], window: int, count_needed: int) -> list[str]:
    if not levels:
        return []
    rank = {"green": 0, "yellow": 1, "red": 2}
    inv = {v: k for k, v in rank.items()}
    w = int(max(1, window))
    k = int(max(1, count_needed))
    out = [levels[0] if levels[0] in rank else "green"]
    cur = rank[out[0]]
    for i in range(1, len(levels)):
        raw = [rank.get(x, 0) for x in levels[max(0, i - w + 1) : i + 1]]
        if not raw:
            out.append(inv[cur])
            continue
        want = rank.get(levels[i], 0)
        if want > cur:
            n = int(np.sum(np.asarray(raw, dtype=int) >= want))
            if n >= k:
                cur = want
        elif want < cur:
            n = int(np.sum(np.asarray(raw, dtype=int) <= want))
            if n >= k:
                cur = want
        out.append(inv[cur])
    return out


def _build_alert_levels(
    regime_df: pd.DataFrame,
    ts_off: pd.DataFrame,
    risk_q_yellow: float,
    risk_q_red: float,
    min_conf_yellow: float,
    min_conf_red: float,
    persist_window: int,
    persist_count: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if regime_df.empty or ts_off.empty:
        return pd.DataFrame(), {}
    rg = regime_df.copy()
    rg["date"] = pd.to_datetime(rg["date"], errors="coerce")
    rg = rg.dropna(subset=["date"])
    ts = ts_off.copy()
    ts["date"] = pd.to_datetime(ts["date"], errors="coerce")
    ts = ts.dropna(subset=["date"])
    x = rg.merge(
        ts[["date", "p1", "deff", "insufficient_universe"]],
        on="date",
        how="left",
        suffixes=("", "_ts"),
    ).sort_values("date")
    if x.empty:
        return pd.DataFrame(), {}
    p1_rank = pd.to_numeric(x["p1"], errors="coerce").rank(pct=True)
    deff_rank = pd.to_numeric(x["deff"], errors="coerce").rank(pct=True)
    dp = pd.to_numeric(x["dp1_5"], errors="coerce").abs()
    dd = pd.to_numeric(x["ddeff_5"], errors="coerce").abs()
    tr = pd.to_numeric(x.get("transition_score"), errors="coerce")
    tr_rank = tr.rank(pct=True)
    x["risk_score"] = (0.45 * p1_rank + 0.30 * (1.0 - deff_rank) + 0.25 * tr_rank).fillna(0.0).clip(0.0, 1.0)

    dp_ref = float(dp.quantile(0.80))
    dd_ref = float(dd.quantile(0.80))
    turb = 0.5 * (dp / max(dp_ref, 1e-9)).clip(0.0, 2.0) + 0.5 * (dd / max(dd_ref, 1e-9)).clip(0.0, 2.0)
    conf = 0.55 + 0.20 * (x["regime_raw"] == x["regime"]).astype(float) - 0.30 * turb.fillna(1.0)
    conf -= 0.10 * pd.to_numeric(x.get("insufficient_universe", False)).astype(float)
    x["signal_confidence"] = conf.clip(0.0, 1.0)

    rq_y = float(np.clip(risk_q_yellow, 0.0, 1.0))
    rq_r = float(np.clip(risk_q_red, 0.0, 1.0))
    thr_y = float(x["risk_score"].quantile(rq_y))
    thr_r = float(x["risk_score"].quantile(rq_r))
    raw = []
    for _, r in x.iterrows():
        risk = float(r.get("risk_score", np.nan))
        c = float(r.get("signal_confidence", np.nan))
        if np.isfinite(risk) and np.isfinite(c) and (risk >= thr_r) and (c >= float(min_conf_red)):
            raw.append("red")
        elif np.isfinite(risk) and np.isfinite(c) and (risk >= thr_y) and (c >= float(min_conf_yellow)):
            raw.append("yellow")
        else:
            raw.append("green")
    x["alert_level_raw"] = raw
    x["alert_level"] = _apply_level_persistence(
        levels=raw,
        window=int(max(1, persist_window)),
        count_needed=int(max(1, persist_count)),
    )
    x["date"] = x["date"].dt.date.astype(str)
    out = x[
        [
            "date",
            "regime",
            "regime_raw",
            "risk_score",
            "signal_confidence",
            "alert_level_raw",
            "alert_level",
            "transition_score",
            "dp1_5",
            "ddeff_5",
        ]
    ].copy()
    payload = {
        "risk_threshold_yellow": thr_y,
        "risk_threshold_red": thr_r,
        "risk_q_yellow": rq_y,
        "risk_q_red": rq_r,
        "min_conf_yellow": float(min_conf_yellow),
        "min_conf_red": float(min_conf_red),
        "persist_window": int(max(1, persist_window)),
        "persist_count": int(max(1, persist_count)),
        "latest_level": str(out.iloc[-1]["alert_level"]) if not out.empty else "",
        "latest_level_raw": str(out.iloc[-1]["alert_level_raw"]) if not out.empty else "",
        "counts": out["alert_level"].value_counts().to_dict(),
    }
    return out, payload


def _normal_two_sided_p(z: float) -> float:
    if not np.isfinite(z):
        return float("nan")
    return float(math.erfc(abs(float(z)) / math.sqrt(2.0)))


def _build_significance_tables(ts_map: dict[int, pd.DataFrame], outdir: Path) -> pd.DataFrame:
    summary_rows: list[dict[str, Any]] = []
    for w, ts in sorted(ts_map.items()):
        if ts.empty:
            continue
        d = ts[~ts["insufficient_universe"]].copy()
        if d.empty:
            continue
        d["delta_p1_vs_bootstrap"] = pd.to_numeric(d["p1"], errors="coerce") - pd.to_numeric(d["p1_bootstrap"], errors="coerce")
        d["delta_deff_vs_bootstrap"] = pd.to_numeric(d["deff_bootstrap"], errors="coerce") - pd.to_numeric(d["deff"], errors="coerce")
        out_cols = ["date"]
        for col in ["delta_p1_vs_bootstrap", "delta_deff_vs_bootstrap"]:
            s = pd.to_numeric(d[col], errors="coerce")
            mu = float(s.mean(skipna=True))
            sd = float(s.std(ddof=0, skipna=True))
            n = int(s.notna().sum())
            z = s / sd if (np.isfinite(sd) and sd > 1e-12) else pd.Series(np.nan, index=s.index)
            p = z.apply(_normal_two_sided_p)
            mean_z = float(mu / (sd / math.sqrt(n))) if (n > 0 and np.isfinite(sd) and sd > 1e-12) else float("nan")
            mean_p = _normal_two_sided_p(mean_z)
            latest_valid = s.dropna()
            latest_delta = float(latest_valid.iloc[-1]) if not latest_valid.empty else float("nan")
            latest_z = float(latest_delta / sd) if (np.isfinite(latest_delta) and np.isfinite(sd) and sd > 1e-12) else float("nan")
            latest_p = _normal_two_sided_p(latest_z)
            sig_share = float((p[s.notna()] < 0.05).mean()) if n > 0 else float("nan")
            summary_rows.append(
                {
                    "window": int(w),
                    "metric": str(col),
                    "n": n,
                    "mean_delta": mu,
                    "std_delta": sd,
                    "significant_share_p_lt_0_05": sig_share,
                    "mean_z_vs_zero": mean_z,
                    "mean_pvalue_vs_zero": mean_p,
                    "latest_delta": latest_delta,
                    "latest_z": latest_z,
                    "latest_pvalue": latest_p,
                }
            )
            d[f"z_{col}"] = z
            d[f"p_{col}"] = p
            out_cols += [col, f"z_{col}", f"p_{col}"]
        d[out_cols].to_csv(outdir / f"significance_timeseries_T{int(w)}.csv", index=False)
    out = pd.DataFrame(summary_rows)
    if not out.empty:
        out.to_csv(outdir / "significance_summary_by_window.csv", index=False)
    return out


def _switch_count(x: pd.Series) -> int:
    s = x.dropna().astype(str)
    if s.empty:
        return 0
    return int(max(0, (s != s.shift(1)).sum() - 1))


def _state_from_risk(risk: float, conf: float, sw30: float) -> str:
    if (risk >= 0.80 and conf < 0.55) or (sw30 >= 4):
        return "instavel"
    if (risk >= 0.60) or (conf < 0.70) or (sw30 >= 2):
        return "transicao"
    return "estavel"


def _build_asset_sector_diagnostics(
    returns_wide: pd.DataFrame,
    universe_core: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if returns_wide.empty or universe_core.empty:
        return pd.DataFrame(), pd.DataFrame(), {}
    r = returns_wide.copy().sort_index()
    r = r.apply(pd.to_numeric, errors="coerce")
    mkt = r.mean(axis=1, skipna=True)
    rows: list[dict[str, Any]] = []
    for t in r.columns.astype(str).tolist():
        s = r[t]
        cov = float(s.notna().mean())
        vol60 = s.rolling(60, min_periods=40).std(ddof=0) * np.sqrt(252.0)
        vol252 = s.rolling(252, min_periods=120).std(ddof=0) * np.sqrt(252.0)
        corr120 = s.rolling(120, min_periods=60).corr(mkt)
        q30 = float(corr120.quantile(0.30))
        q70 = float(corr120.quantile(0.70))
        st = pd.Series(np.where(corr120 >= q70, "high", np.where(corr120 <= q30, "low", "mid")), index=corr120.index)
        sw30 = _switch_count(st.tail(30))
        sw90 = _switch_count(st.tail(90))
        sw180 = _switch_count(st.tail(180))
        latest_corr = float(pd.to_numeric(corr120, errors="coerce").dropna().iloc[-1]) if corr120.notna().any() else float("nan")
        prev5_corr = float(pd.to_numeric(corr120.shift(5), errors="coerce").dropna().iloc[-1]) if corr120.shift(5).notna().any() else float("nan")
        latest_v60 = float(pd.to_numeric(vol60, errors="coerce").dropna().iloc[-1]) if vol60.notna().any() else float("nan")
        prev5_v60 = float(pd.to_numeric(vol60.shift(5), errors="coerce").dropna().iloc[-1]) if vol60.shift(5).notna().any() else float("nan")
        rows.append(
            {
                "ticker": t,
                "coverage": cov,
                "vol60_latest": latest_v60,
                "vol252_latest": float(pd.to_numeric(vol252, errors="coerce").dropna().iloc[-1]) if vol252.notna().any() else float("nan"),
                "corr120_latest": latest_corr,
                "corr120_abs_latest": abs(latest_corr) if np.isfinite(latest_corr) else float("nan"),
                "delta_corr120_5d": (latest_corr - prev5_corr) if (np.isfinite(latest_corr) and np.isfinite(prev5_corr)) else float("nan"),
                "delta_vol60_5d": (latest_v60 - prev5_v60) if (np.isfinite(latest_v60) and np.isfinite(prev5_v60)) else float("nan"),
                "switches_30d": int(sw30),
                "switches_90d": int(sw90),
                "switches_180d": int(sw180),
            }
        )
    a = pd.DataFrame(rows)
    if a.empty:
        return a, pd.DataFrame(), {}
    a["rank_vol"] = pd.to_numeric(a["vol60_latest"], errors="coerce").rank(pct=True)
    a["rank_corr_abs"] = pd.to_numeric(a["corr120_abs_latest"], errors="coerce").rank(pct=True)
    a["rank_sw90"] = pd.to_numeric(a["switches_90d"], errors="coerce").rank(pct=True)
    a["risk_score"] = (0.45 * a["rank_vol"] + 0.35 * a["rank_corr_abs"] + 0.20 * a["rank_sw90"]).fillna(0.0).clip(0.0, 1.0)
    d_corr_rank = pd.to_numeric(a["delta_corr120_5d"], errors="coerce").abs().rank(pct=True).fillna(0.5)
    a["confidence_score"] = (1.0 - (0.60 * a["rank_sw90"].fillna(0.5) + 0.40 * d_corr_rank)).clip(0.0, 1.0)
    a["regime_asset"] = [
        _state_from_risk(float(rw["risk_score"]), float(rw["confidence_score"]), float(rw["switches_30d"])) for _, rw in a.iterrows()
    ]
    a["sensitivity_score"] = a["risk_score"].clip(0.0, 1.0)
    a["stability_score"] = (a["confidence_score"] * (1.0 - 0.50 * a["risk_score"])).clip(0.0, 1.0)

    core = universe_core[["ticker", "sector"]].copy()
    core["ticker"] = core["ticker"].astype(str)
    a = core.merge(a, on="ticker", how="left")
    a = a.sort_values(["sector", "ticker"]).reset_index(drop=True)

    s = (
        a.groupby("sector", as_index=False)
        .agg(
            n_assets=("ticker", "count"),
            risk_mean=("risk_score", "mean"),
            confidence_mean=("confidence_score", "mean"),
            pct_instavel=("regime_asset", lambda x: float((x == "instavel").mean())),
            pct_transicao=("regime_asset", lambda x: float((x == "transicao").mean())),
            switches_30d_mean=("switches_30d", "mean"),
            switches_90d_mean=("switches_90d", "mean"),
            switches_180d_mean=("switches_180d", "mean"),
            delta_corr120_5d_mean=("delta_corr120_5d", "mean"),
            delta_vol60_5d_mean=("delta_vol60_5d", "mean"),
        )
        .sort_values(["risk_mean", "pct_instavel"], ascending=[False, False])
        .reset_index(drop=True)
    )
    levels = []
    plans = []
    for _, rw in s.iterrows():
        lvl = "verde"
        if (float(rw["pct_instavel"]) >= 0.20) or (float(rw["risk_mean"]) >= 0.75):
            lvl = "vermelho"
        elif (float(rw["pct_instavel"]) >= 0.10) or (float(rw["risk_mean"]) >= 0.60):
            lvl = "amarelo"
        levels.append(lvl)
        plans.append(
            "leitura estrutural sugerida: fragilidade alta no setor"
            if lvl == "vermelho"
            else (
                "leitura estrutural sugerida: transicao em monitoramento"
                if lvl == "amarelo"
                else "leitura estrutural sugerida: estabilidade relativa"
            )
        )
    s["alerta_setor"] = levels
    s["plano_acao"] = plans

    payload = {
        "n_assets": int(a["ticker"].nunique()),
        "n_sectors": int(s.shape[0]),
        "regime_mix_assets": a["regime_asset"].value_counts(normalize=True).to_dict(),
        "top_risk_asset": str(a.sort_values(["risk_score", "switches_90d"], ascending=[False, False]).iloc[0]["ticker"]) if not a.empty else "",
        "top_risk_sector": str(s.iloc[0]["sector"]) if not s.empty else "",
    }
    return a, s, payload


def _build_era_evaluation(ts_off: pd.DataFrame, regime_df: pd.DataFrame, bt_df: pd.DataFrame) -> pd.DataFrame:
    if ts_off.empty or regime_df.empty or bt_df.empty:
        return pd.DataFrame()
    ts = ts_off.copy()
    rg = regime_df.copy()
    bt = bt_df.copy()
    for d in (ts, rg, bt):
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
    ts = ts.dropna(subset=["date"])
    rg = rg.dropna(subset=["date"])
    bt = bt.dropna(subset=["date"])
    merged = rg.merge(
        ts[["date", "N_used", "p1", "deff", "turnover_pair_frac"]],
        on="date",
        how="left",
        suffixes=("", "_ts"),
    ).merge(bt[["date", "strategy_simple_ret", "benchmark_simple_ret"]], on="date", how="left")
    merged = merged.sort_values("date").reset_index(drop=True)
    if merged.empty:
        return pd.DataFrame()
    merged["era"] = merged["date"].map(_era_name)
    rows: list[dict[str, Any]] = []
    for era in ERA_ORDER:
        x = merged[merged["era"] == era].copy()
        if x.empty:
            continue
        sret = pd.to_numeric(x["strategy_simple_ret"], errors="coerce").dropna()
        bret = pd.to_numeric(x["benchmark_simple_ret"], errors="coerce").dropna()
        sp = _perf(sret)
        bp = _perf(bret)
        row = {
            "era": era,
            "start": x["date"].min().date().isoformat(),
            "end": x["date"].max().date().isoformat(),
            "n_days": int(x.shape[0]),
            "n_used_mean": float(pd.to_numeric(x["N_used"], errors="coerce").mean()),
            "p1_mean": float(pd.to_numeric(x["p1"], errors="coerce").mean()),
            "p1_std": float(pd.to_numeric(x["p1"], errors="coerce").std(ddof=0)),
            "deff_mean": float(pd.to_numeric(x["deff"], errors="coerce").mean()),
            "deff_std": float(pd.to_numeric(x["deff"], errors="coerce").std(ddof=0)),
            "turnover_pair_mean": float(pd.to_numeric(x["turnover_pair_frac"], errors="coerce").mean()),
            "strategy_ann_return": float(sp["ann_return"]),
            "strategy_sharpe": float(sp["sharpe"]),
            "strategy_max_drawdown": float(sp["max_drawdown"]),
            "benchmark_ann_return": float(bp["ann_return"]),
            "benchmark_sharpe": float(bp["sharpe"]),
            "benchmark_max_drawdown": float(bp["max_drawdown"]),
            "alpha_ann_return": float(sp["ann_return"] - bp["ann_return"]),
            "dd_improvement": float(abs(bp["max_drawdown"]) - abs(sp["max_drawdown"])),
        }
        for reg in ["stress", "transition", "stable", "dispersion"]:
            row[f"share_{reg}"] = float((x["regime"] == reg).mean())
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["__ord"] = out["era"].map({k: i for i, k in enumerate(ERA_ORDER)})
    out = out.sort_values("__ord").drop(columns="__ord").reset_index(drop=True)
    return out


def _action_map(regime: str) -> tuple[str, str]:
    r = str(regime).lower().strip()
    if r == "stress":
        return "STRESS_PROFILE", "defensive"
    if r == "transition":
        return "TRANSITION_PROFILE", "cautious"
    if r == "dispersion":
        return "DISPERSION_PROFILE", "offensive_selective"
    return "BASELINE_PROFILE", "balanced"


def _reliability_tier(score: float) -> str:
    if not np.isfinite(score):
        return "low"
    if score >= 0.67:
        return "high"
    if score >= 0.45:
        return "medium"
    return "low"


def _build_action_playbook(
    regime_df: pd.DataFrame,
    ts_off: pd.DataFrame,
    bt_df: pd.DataFrame,
    horizon_days: int,
) -> pd.DataFrame:
    if regime_df.empty or ts_off.empty or bt_df.empty:
        return pd.DataFrame()
    rg = regime_df.copy()
    rg["date"] = pd.to_datetime(rg["date"], errors="coerce")
    rg = rg.dropna(subset=["date"])
    ts = ts_off.copy()
    ts["date"] = pd.to_datetime(ts["date"], errors="coerce")
    ts = ts.dropna(subset=["date"])
    x = rg.merge(
        ts[["date", "N_used", "p1", "deff", "structure_score", "insufficient_universe"]],
        on="date",
        how="left",
        suffixes=("", "_ts"),
    ).sort_values("date")
    if x.empty:
        return pd.DataFrame()
    dp1_ref = float(pd.to_numeric(x["dp1_5"], errors="coerce").abs().quantile(0.80))
    dd_ref = float(pd.to_numeric(x["ddeff_5"], errors="coerce").abs().quantile(0.80))

    bt = bt_df.copy()
    bt["date"] = pd.to_datetime(bt["date"], errors="coerce")
    bt = bt.dropna(subset=["date"]).set_index("date").sort_index()

    rows: list[dict[str, Any]] = []
    for _, r in x.iterrows():
        date_s = pd.Timestamp(r["date"]).date().isoformat()
        reg = str(r.get("regime", ""))
        action_code, risk_stance = _action_map(reg)
        dp1 = abs(float(r.get("dp1_5", np.nan)))
        dd = abs(float(r.get("ddeff_5", np.nan)))
        smooth_term = 0.0
        if np.isfinite(dp1) and np.isfinite(dp1_ref) and dp1_ref > 1e-9:
            smooth_term += min(1.0, dp1 / dp1_ref)
        if np.isfinite(dd) and np.isfinite(dd_ref) and dd_ref > 1e-9:
            smooth_term += min(1.0, dd / dd_ref)
        smooth_term = smooth_term / 2.0 if smooth_term > 0 else 0.5
        score = 0.25
        score += 0.20 if (not bool(r.get("insufficient_universe", False))) else 0.0
        n_used = float(r.get("N_used", np.nan))
        score += 0.20 if (np.isfinite(n_used) and n_used >= 25.0) else 0.0
        struct = float(r.get("structure_score", np.nan))
        if np.isfinite(struct):
            score += 0.20 if struct > 0 else 0.05
        score += 0.35 * (1.0 - smooth_term)
        score = float(np.clip(score, 0.0, 1.0))

        fut = _future_stats(bt_by_date=bt, anchor_date=date_s, horizon_days=int(max(1, horizon_days)))
        rows.append(
            {
                "date": date_s,
                "regime": reg,
                "exposure": float(r.get("exposure", np.nan)),
                "action_code": action_code,
                "risk_stance": risk_stance,
                "signal_reliability": score,
                "signal_tier": _reliability_tier(score),
                "N_used": float(r.get("N_used", np.nan)),
                "p1": float(r.get("p1", np.nan)),
                "deff": float(r.get("deff", np.nan)),
                "dp1_5": float(r.get("dp1_5", np.nan)),
                "ddeff_5": float(r.get("ddeff_5", np.nan)),
                "future_days_used": float(fut["future_days_used"]),
                "alpha_cum_future": float(fut["alpha_cum"]),
                "dd_improvement_future": float(fut["dd_improvement"]),
                "tradeoff_label": _honest_verdict(alpha_cum=float(fut["alpha_cum"]), dd_improvement=float(fut["dd_improvement"])),
            }
        )
    return pd.DataFrame(rows)


def _build_ui_view_model(
    run_id: str,
    ts_off: pd.DataFrame,
    regime_df: pd.DataFrame,
    robust_metrics: dict[str, float],
    gate: dict[str, Any],
    op_payload: dict[str, Any],
    case_df: pd.DataFrame,
    era_df: pd.DataFrame,
    playbook_df: pd.DataFrame,
    alert_level_payload: dict[str, Any] | None = None,
    asset_sector_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    alert_level_payload = alert_level_payload or {}
    asset_sector_payload = asset_sector_payload or {}

    latest_state: dict[str, Any] = {}
    if not ts_off.empty:
        suff = ts_off[~ts_off["insufficient_universe"]].copy()
        if not suff.empty:
            x = suff.iloc[-1]
            latest_state = {
                "date": str(x["date"]),
                "N_used": int(x["N_used"]),
                "p1": float(x["p1"]),
                "deff": float(x["deff"]),
            }

    latest_regime: dict[str, Any] = {}
    if not regime_df.empty:
        r = regime_df.iloc[-1]
        latest_regime = {"regime": str(r.get("regime", "unknown")), "exposure": float(r.get("exposure", np.nan))}

    playbook_latest = {}
    if not playbook_df.empty:
        rr = playbook_df.iloc[-1]
        playbook_latest = {
            "date": str(rr.get("date", "")),
            "regime": str(rr.get("regime", "")),
            "action_code": str(rr.get("action_code", "")),
            "risk_stance": str(rr.get("risk_stance", "")),
            "signal_tier": str(rr.get("signal_tier", "")),
            "signal_reliability": float(rr.get("signal_reliability", np.nan)),
            "tradeoff_label": str(rr.get("tradeoff_label", "")),
        }

    return {
        "schema_version": "lab_corr_view_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": str(run_id),
        "deployment_gate_blocked": bool(gate.get("blocked", True)),
        "deployment_reasons": [str(x) for x in gate.get("reasons", [])],
        "latest_state": latest_state,
        "latest_regime": latest_regime,
        "robustness": {
            "joint_majority_60d": float(robust_metrics.get("joint_majority_60d", np.nan)),
            "latest_joint_majority_5": float(robust_metrics.get("latest_joint_majority_5", np.nan)),
        },
        "alerts": {
            "latest_date": str(op_payload.get("latest_date", "")),
            "latest_events": [str(x) for x in op_payload.get("latest_events", [])],
            "n_events_last_60d": int(op_payload.get("n_events_last_60d", 0)),
        },
        "alert_levels": {
            "latest_level": str(alert_level_payload.get("latest_level", "")),
            "latest_level_raw": str(alert_level_payload.get("latest_level_raw", "")),
            "counts": alert_level_payload.get("counts", {}),
        },
        "asset_sector": {
            "n_assets": int(asset_sector_payload.get("n_assets", 0)),
            "n_sectors": int(asset_sector_payload.get("n_sectors", 0)),
            "top_risk_asset": str(asset_sector_payload.get("top_risk_asset", "")),
            "top_risk_sector": str(asset_sector_payload.get("top_risk_sector", "")),
        },
        "playbook_latest": playbook_latest,
        "case_preview": case_df.head(3).to_dict(orient="records") if not case_df.empty else [],
        "era_summary": era_df.to_dict(orient="records") if not era_df.empty else [],
    }


def _qa(
    ts_map: dict[int, pd.DataFrame],
    core_counts: pd.DataFrame,
    n_core: int,
    min_assets: int,
    official_window: int,
    max_insufficient_ratio: float,
    min_n_used_ratio: float,
    q_min: float,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    core_sum = int(core_counts["n_tickers"].sum()) if not core_counts.empty else 0
    core_min = int(core_counts["n_tickers"].min()) if not core_counts.empty else 0
    checks.append(
        {
            "check": "core_counts_match_n_core",
            "ok": core_sum == int(n_core),
            "value": core_sum,
            "expected": int(n_core),
        }
    )
    checks.append(
        {
            "check": "all_sectors_nonzero_in_core",
            "ok": bool((core_counts["n_tickers"] > 0).all()) if not core_counts.empty else False,
            "min_sector_count": core_min,
        }
    )
    min_n_used_required = max(float(min_assets), float(np.ceil(float(n_core) * float(min_n_used_ratio))))
    for w, ts in sorted(ts_map.items()):
        if ts.empty:
            checks.append({"check": f"T{w}_non_empty", "ok": False, "n_rows": 0})
            continue
        insufficient_ratio = float(ts["insufficient_universe"].mean())
        checks.append(
            {
                "check": f"T{w}_insufficient_ratio",
                "ok": insufficient_ratio <= float(max_insufficient_ratio),
                "value": insufficient_ratio,
                "max_allowed": float(max_insufficient_ratio),
            }
        )
        suff = ts[~ts["insufficient_universe"]].copy()
        if suff.empty:
            checks.append({"check": f"T{w}_sufficient_non_empty", "ok": False, "n_sufficient": 0})
            continue
        n_used_min = float(suff["N_used"].min())
        checks.append(
            {
                "check": f"T{w}_N_used_min_sufficient",
                "ok": n_used_min >= float(min_n_used_required),
                "value": n_used_min,
                "min_required": float(min_n_used_required),
            }
        )
        q_series = float(w) / suff["N_used"].astype(float).clip(lower=1.0)
        q_observed_min = float(q_series.min())
        checks.append(
            {
                "check": f"T{w}_q_min",
                "ok": q_observed_min >= float(q_min),
                "value": q_observed_min,
                "min_required": float(q_min),
            }
        )
        checks.append({"check": f"T{w}_p1_range", "ok": bool(suff["p1"].between(0.0, 1.0, inclusive="both").all())})
        checks.append({"check": f"T{w}_deff_range", "ok": bool(((suff["deff"] >= 1.0) & (suff["deff"] <= (suff["N_used"] + 1e-6))).all())})
        last20 = suff.tail(20)
        checks.append(
            {
                "check": f"T{w}_last20_complete",
                "ok": bool(last20.shape[0] == 20 and last20[["p1", "deff"]].notna().all().all()),
                "n_last20": int(last20.shape[0]),
                "required": 20,
            }
        )
    checks.append(
        {
            "check": "official_window_present",
            "ok": official_window in ts_map,
            "official_window": int(official_window),
            "available_windows": [int(x) for x in sorted(ts_map.keys())],
        }
    )
    failed_checks = [str(c.get("check", "")) for c in checks if not bool(c.get("ok", False))]
    return {"ok": bool(len(failed_checks) == 0), "checks": checks, "failed_checks": failed_checks}


def _freeze_baseline(
    baseline_dir: Path,
    outdir: Path,
    universe_core: pd.DataFrame,
    ts_official: pd.DataFrame,
    official_window: int,
    start: str,
    end: str,
    cov_core: float,
    cov_window: float,
) -> dict[str, Any]:
    baseline_dir.mkdir(parents=True, exist_ok=True)
    tickers = sorted(universe_core["ticker"].astype(str).tolist())
    uh = hashlib.sha256("|".join(tickers).encode("utf-8")).hexdigest()
    prev_meta_path = baseline_dir / "baseline_meta.json"
    prev_ts_path = baseline_dir / f"macro_timeseries_T{official_window}.csv"

    prev_meta: dict[str, Any] = {}
    if prev_meta_path.exists():
        try:
            prev_meta = json.loads(prev_meta_path.read_text(encoding="utf-8"))
        except Exception:
            prev_meta = {}
    prev_last: dict[str, float] = {}
    if prev_ts_path.exists():
        try:
            p = pd.read_csv(prev_ts_path)
            p = p[~p["insufficient_universe"]]
            if not p.empty:
                prev_last = {"p1": float(p.iloc[-1]["p1"]), "deff": float(p.iloc[-1]["deff"])}
        except Exception:
            prev_last = {}

    cur = ts_official[~ts_official["insufficient_universe"]]
    cur_last = {"p1": float(cur.iloc[-1]["p1"]), "deff": float(cur.iloc[-1]["deff"])} if not cur.empty else {}

    universe_core.to_csv(baseline_dir / "universe_core.csv", index=False)
    ts_official.to_csv(prev_ts_path, index=False)
    meta = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_from_run_dir": str(outdir),
        "official_window": int(official_window),
        "period_start": str(start),
        "period_end": str(end),
        "coverage_core_threshold": float(cov_core),
        "coverage_window_threshold": float(cov_window),
        "n_core": int(universe_core.shape[0]),
        "universe_hash": uh,
    }
    (baseline_dir / "baseline_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    cmp = {
        "baseline_dir": str(baseline_dir),
        "same_universe_vs_previous": (prev_meta.get("universe_hash", uh) == uh),
        "previous_run_dir": prev_meta.get("frozen_from_run_dir"),
        "current_run_dir": str(outdir),
        "delta_p1_vs_previous": (cur_last.get("p1", np.nan) - prev_last.get("p1", np.nan)) if prev_last else np.nan,
        "delta_deff_vs_previous": (cur_last.get("deff", np.nan) - prev_last.get("deff", np.nan)) if prev_last else np.nan,
    }
    (baseline_dir / "baseline_compare.json").write_text(json.dumps(cmp, indent=2), encoding="utf-8")
    return cmp


def _build_deployment_gate(
    qa: dict[str, Any],
    robust_metrics: dict[str, float],
    baseline: dict[str, Any],
    alerts_payload: dict[str, Any],
    min_joint_majority_60d: float,
    require_latest_majority: bool,
    max_abs_delta_p1: float,
    max_abs_delta_deff: float,
    max_active_cluster_alerts: int,
    policy_lock: dict[str, Any],
    require_policy_lock_match: bool,
) -> dict[str, Any]:
    blocked = False
    reasons: list[str] = []
    checks: dict[str, Any] = {}

    qa_ok = bool(qa.get("ok", False))
    checks["qa_ok"] = qa_ok
    if not qa_ok:
        blocked = True
        reasons.append("qa_failed")

    jm60 = float(robust_metrics.get("joint_majority_60d", np.nan))
    checks["joint_majority_60d"] = jm60
    if np.isfinite(jm60) and (jm60 < float(min_joint_majority_60d)):
        blocked = True
        reasons.append("robustness_joint_majority_below_threshold")

    latest_jm = float(robust_metrics.get("latest_joint_majority_5", np.nan))
    checks["latest_joint_majority_5"] = latest_jm
    if bool(require_latest_majority):
        if (not np.isfinite(latest_jm)) or (int(latest_jm) != 1):
            blocked = True
            reasons.append("latest_window_direction_not_confirmed")

    same_universe = bool(baseline.get("same_universe_vs_previous", True))
    checks["same_universe_vs_previous"] = same_universe
    if not same_universe:
        blocked = True
        reasons.append("baseline_universe_mismatch")

    dp1 = float(baseline.get("delta_p1_vs_previous", np.nan))
    dd = float(baseline.get("delta_deff_vs_previous", np.nan))
    checks["abs_delta_p1_vs_previous"] = abs(dp1) if np.isfinite(dp1) else np.nan
    checks["abs_delta_deff_vs_previous"] = abs(dd) if np.isfinite(dd) else np.nan
    if np.isfinite(dp1) and (abs(dp1) > float(max_abs_delta_p1)):
        blocked = True
        reasons.append("baseline_delta_p1_exceeds_threshold")
    if np.isfinite(dd) and (abs(dd) > float(max_abs_delta_deff)):
        blocked = True
        reasons.append("baseline_delta_deff_exceeds_threshold")

    n_alerts = int(alerts_payload.get("n_active", 0))
    checks["active_cluster_alerts"] = n_alerts
    if n_alerts > int(max_active_cluster_alerts):
        blocked = True
        reasons.append("too_many_cluster_alerts")

    policy_match = bool(policy_lock.get("matches_previous_release", True))
    checks["policy_lock_matches_previous_release"] = policy_match
    if bool(require_policy_lock_match) and (not policy_match):
        blocked = True
        reasons.append("policy_lock_mismatch")

    return {
        "blocked": blocked,
        "reasons": reasons,
        "checks": checks,
        "thresholds": {
            "min_joint_majority_60d": float(min_joint_majority_60d),
            "require_latest_majority": bool(require_latest_majority),
            "max_abs_delta_p1": float(max_abs_delta_p1),
            "max_abs_delta_deff": float(max_abs_delta_deff),
            "max_active_cluster_alerts": int(max_active_cluster_alerts),
            "require_policy_lock_match": bool(require_policy_lock_match),
        },
    }


def _write_compact_report(
    outdir: Path,
    run_id: str,
    period_start: str,
    period_end: str,
    n_core: int,
    official_window: int,
    ts_official: pd.DataFrame,
    regime_df: pd.DataFrame,
    robust_metrics: dict[str, float],
    baseline: dict[str, Any],
    bt_summary: dict[str, Any],
    gate: dict[str, Any],
) -> None:
    lines: list[str] = []
    lines.append(f"run_id={run_id}")
    lines.append(f"period={period_start}->{period_end}")
    lines.append(f"N_core={int(n_core)}")
    lines.append(f"official_window=T{int(official_window)}")
    if (not ts_official.empty) and ("insufficient_universe" in ts_official.columns):
        suff = ts_official[~ts_official["insufficient_universe"]].copy()
    else:
        suff = pd.DataFrame()
    if not suff.empty:
        x = suff.iloc[-1]
        lines.append(f"latest={x['date']} N_used={int(x['N_used'])} p1={float(x['p1']):.6f} deff={float(x['deff']):.6f}")
    else:
        lines.append("latest=unavailable")
    if not regime_df.empty:
        r = regime_df.iloc[-1]
        lines.append(f"regime_now={r['regime']} exposure={float(r['exposure']):.2f}")
    else:
        lines.append("regime_now=unavailable")
    lines.append(
        "robustness="
        f"joint_majority_60d={float(robust_metrics.get('joint_majority_60d', np.nan)):.3f}, "
        f"latest_joint_majority_5={int(float(robust_metrics.get('latest_joint_majority_5', np.nan))) if np.isfinite(float(robust_metrics.get('latest_joint_majority_5', np.nan))) else 'nan'}"
    )
    lines.append(
        "baseline_delta="
        f"dp1={float(baseline.get('delta_p1_vs_previous', np.nan)):.6f}, "
        f"ddeff={float(baseline.get('delta_deff_vs_previous', np.nan)):.6f}"
    )
    if bt_summary and ("strategy" in bt_summary) and ("benchmark" in bt_summary):
        s = bt_summary["strategy"]
        b = bt_summary["benchmark"]
        lines.append(
            "backtest="
            f"str_ann={float(s.get('ann_return', np.nan)):.4f}, str_mdd={float(s.get('max_drawdown', np.nan)):.4f}, "
            f"bench_ann={float(b.get('ann_return', np.nan)):.4f}, bench_mdd={float(b.get('max_drawdown', np.nan)):.4f}"
        )
    else:
        lines.append("backtest=unavailable")
    lines.append(f"deployment_gate_blocked={bool(gate.get('blocked', True))}")
    lines.append(f"deployment_gate_reasons={','.join(gate.get('reasons', [])) if gate.get('reasons') else 'none'}")
    (outdir / "summary_compact.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _dict_hash(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _build_policy_lock(
    out_base: Path,
    outdir: Path,
    effective_policy: dict[str, Any],
) -> dict[str, Any]:
    cur_hash = _dict_hash(effective_policy)
    prev_hash = ""
    prev_run = ""
    pointer = out_base / "latest_release.json"
    if pointer.exists():
        try:
            p = json.loads(pointer.read_text(encoding="utf-8"))
            prev_run = str(p.get("run_dir", "")).strip()
        except Exception:
            prev_run = ""
    if prev_run and Path(prev_run).exists() and (Path(prev_run).resolve() != outdir.resolve()):
        lock_path = Path(prev_run) / "production_policy_lock.json"
        if lock_path.exists():
            try:
                lk = json.loads(lock_path.read_text(encoding="utf-8"))
                prev_hash = str(lk.get("policy_hash", "")).strip()
            except Exception:
                prev_hash = ""
        else:
            pu = Path(prev_run) / "policy_used.json"
            if pu.exists():
                try:
                    pj = json.loads(pu.read_text(encoding="utf-8"))
                    prev_eff = (pj.get("effective", {}) or {}) if isinstance(pj, dict) else {}
                    if isinstance(prev_eff, dict) and prev_eff:
                        prev_hash = _dict_hash(prev_eff)
                except Exception:
                    prev_hash = ""
    out = {
        "policy_hash": cur_hash,
        "previous_policy_hash": prev_hash,
        "matches_previous_release": bool((not prev_hash) or (cur_hash == prev_hash)),
        "previous_release_run_dir": prev_run,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (outdir / "production_policy_lock.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def _update_release_pointer(out_base: Path, run_meta: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    pointer_path = out_base / "latest_release.json"
    decision = {
        "updated": False,
        "pointer_path": str(pointer_path),
        "blocked": bool(gate.get("blocked", True)),
        "reasons": gate.get("reasons", []),
    }
    if decision["blocked"]:
        return decision
    payload = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": Path(str(run_meta["run_dir"])).name,
        "run_dir": run_meta["run_dir"],
        "status": "ok",
    }
    pointer_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    decision["updated"] = True
    return decision


def _load_policy(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def _slug_token_policy(text: str) -> str:
    s = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(text))
    while "__" in s:
        s = s.replace("__", "_")
    s = s.strip("_")
    return s or "custom"


def _resolve_baseline_dir(
    args: argparse.Namespace,
    out_base: Path,
    policy_path: Path,
    policy: dict[str, Any],
) -> tuple[Path, str]:
    baseline_dir = Path(str(args.baseline_dir))
    default_baseline = Path(DEFAULT_BASELINE_DIR)
    if baseline_dir != default_baseline:
        return baseline_dir, "manual"

    official_policy = ROOT / "config" / "lab_corr_policy.json"
    try:
        is_official_policy = policy_path.resolve() == official_policy.resolve()
    except Exception:
        is_official_policy = str(policy_path) == str(official_policy)

    if int(args.apply_policy) != 1 or not policy or is_official_policy:
        return baseline_dir, "official_default"

    version = str((policy or {}).get("version", "")).strip()
    ns = _slug_token_policy(version if version else policy_path.stem)
    if ns in {"lab_corr_policy_v1", "lab_corr_policy", "official"}:
        return baseline_dir, "official_default"
    return out_base / f"_baseline_{ns}", ns


def _apply_policy_to_args(args: argparse.Namespace, policy: dict[str, Any]) -> None:
    if not policy:
        return
    data = policy.get("data", {}) or {}
    regime = policy.get("regime", {}) or {}
    gate = policy.get("gate", {}) or {}
    backtest = policy.get("backtest", {}) or {}
    calib = policy.get("calibration", {}) or {}
    cases = policy.get("case_studies", {}) or {}
    alerts = policy.get("alerts", {}) or {}

    for key, attr in [
        ("start", "start"),
        ("end", "end"),
        ("coverage_core", "coverage_core"),
        ("coverage_window", "coverage_window"),
        ("min_assets", "min_assets"),
        ("overlap_step", "overlap_step"),
        ("official_window", "official_window"),
        ("business_days_only", "business_days_only"),
    ]:
        if key in data:
            setattr(args, attr, data[key])

    if "hysteresis_days" in regime:
        args.hysteresis_days = regime["hysteresis_days"]
    if "threshold_mode" in regime:
        args.regime_threshold_mode = str(regime["threshold_mode"])
    if "walkforward_min_history_days" in regime:
        args.walkforward_min_history = int(regime["walkforward_min_history_days"])
    exp = regime.get("exposure", {}) or {}
    if "stress" in exp:
        args.exp_stress = exp["stress"]
    if "transition" in exp:
        args.exp_transition = exp["transition"]
    if "stable" in exp:
        args.exp_stable = exp["stable"]
    if "dispersion" in exp:
        args.exp_dispersion = exp["dispersion"]
    tw = regime.get("transition_weights", {}) or {}
    if "dp1" in tw:
        args.transition_weight_dp1 = tw["dp1"]
    if "ddeff" in tw:
        args.transition_weight_ddeff = tw["ddeff"]
    if "overlap_instability" in tw:
        args.transition_weight_overlap = tw["overlap_instability"]

    for key, attr in [
        ("min_joint_majority_60d", "min_joint_majority_60d"),
        ("require_latest_majority", "require_latest_majority"),
        ("max_abs_delta_p1", "max_abs_delta_p1"),
        ("max_abs_delta_deff", "max_abs_delta_deff"),
        ("max_active_cluster_alerts", "max_active_cluster_alerts"),
        ("require_policy_lock_match", "require_policy_lock_match"),
        ("alert_lookback", "alert_lookback"),
        ("max_insufficient_ratio", "max_insufficient_ratio"),
        ("min_n_used_ratio", "min_n_used_ratio"),
        ("q_min", "q_min"),
    ]:
        if key in gate:
            setattr(args, attr, gate[key])

    for key, attr in [("cost_bps", "cost_bps"), ("max_daily_turnover", "max_daily_turnover")]:
        if key in backtest:
            setattr(args, attr, backtest[key])

    if "enabled" in calib:
        args.calibrate_exposure_grid = int(bool(calib["enabled"]))
    if "apply_best" in calib:
        args.apply_grid_best = int(bool(calib["apply_best"]))
    if "objective" in calib:
        args.calibration_objective = str(calib["objective"])
    if "horizon_days" in cases:
        args.case_horizon_days = int(cases["horizon_days"])
    if "regimes" in cases and isinstance(cases["regimes"], list):
        args.case_regimes = ",".join([str(x) for x in cases["regimes"] if str(x).strip()])
    for key, attr in [
        ("risk_q_yellow", "risk_q_yellow"),
        ("risk_q_red", "risk_q_red"),
        ("min_conf_yellow", "min_conf_yellow"),
        ("min_conf_red", "min_conf_red"),
        ("persist_window", "alert_persist_window"),
        ("persist_count", "alert_persist_count"),
    ]:
        if key in alerts:
            setattr(args, attr, alerts[key])


def _default_exposure_candidates(
    exp_stress: float,
    exp_transition: float,
    exp_stable: float,
    exp_dispersion: float,
) -> list[dict[str, float]]:
    base = {
        "stress": float(exp_stress),
        "transition": float(exp_transition),
        "stable": float(exp_stable),
        "dispersion": float(exp_dispersion),
    }
    cands = [
        base,
        {"stress": 0.05, "transition": 0.35, "stable": 0.65, "dispersion": 0.90},
        {"stress": 0.10, "transition": 0.40, "stable": 0.70, "dispersion": 0.90},
        {"stress": 0.15, "transition": 0.45, "stable": 0.70, "dispersion": 0.90},
        {"stress": 0.10, "transition": 0.45, "stable": 0.75, "dispersion": 0.95},
        {"stress": 0.15, "transition": 0.50, "stable": 0.75, "dispersion": 0.95},
        {"stress": 0.20, "transition": 0.50, "stable": 0.80, "dispersion": 1.00},
        {"stress": 0.05, "transition": 0.30, "stable": 0.60, "dispersion": 0.85},
    ]
    out = []
    seen = set()
    for c in cands:
        t = (round(float(c["stress"]), 4), round(float(c["transition"]), 4), round(float(c["stable"]), 4), round(float(c["dispersion"]), 4))
        if t in seen:
            continue
        seen.add(t)
        out.append({k: float(np.clip(v, 0.0, 1.0)) for k, v in c.items()})
    return out


def _score_backtest(summary: dict[str, Any], objective: str) -> float:
    s = (summary or {}).get("strategy", {}) or {}
    ann = float(s.get("ann_return", np.nan))
    sharpe = float(s.get("sharpe", np.nan))
    mdd = float(s.get("max_drawdown", np.nan))
    if not (np.isfinite(ann) and np.isfinite(sharpe) and np.isfinite(mdd)):
        return float("-inf")
    dd = max(abs(mdd), 1e-6)
    if objective == "ann_over_dd":
        return float(ann / dd)
    if objective == "sharpe":
        return float(sharpe)
    return float(sharpe + 0.25 * (ann / dd))


def _exposure_grid_search(
    regime_df: pd.DataFrame,
    returns_wide: pd.DataFrame,
    cost_bps: float,
    max_daily_turnover: float,
    start_exposure: float,
    candidates: list[dict[str, float]],
    objective: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if regime_df.empty or len(candidates) == 0:
        return pd.DataFrame(), {}
    for i, c in enumerate(candidates):
        tmp = regime_df.copy()
        tmp["exposure"] = tmp["regime"].map(c).fillna(float(start_exposure)).astype(float).clip(0.0, 1.0)
        _, bt = _backtest(
            regime_df=tmp,
            returns_wide=returns_wide,
            cost_bps=float(cost_bps),
            max_daily_turnover=float(max_daily_turnover),
            start_exposure=float(start_exposure),
        )
        score = _score_backtest(bt, objective=objective)
        s = (bt or {}).get("strategy", {}) or {}
        rows.append(
            {
                "candidate_id": i,
                "exp_stress": float(c["stress"]),
                "exp_transition": float(c["transition"]),
                "exp_stable": float(c["stable"]),
                "exp_dispersion": float(c["dispersion"]),
                "score": score,
                "ann_return": float(s.get("ann_return", np.nan)),
                "sharpe": float(s.get("sharpe", np.nan)),
                "max_drawdown": float(s.get("max_drawdown", np.nan)),
                "ann_vol": float(s.get("ann_vol", np.nan)),
                "avg_turnover": float(bt.get("avg_turnover", np.nan)),
            }
        )
    df = pd.DataFrame(rows).sort_values(["score", "ann_return"], ascending=[False, False]).reset_index(drop=True)
    if df.empty:
        return df, {}
    best = df.iloc[0].to_dict()
    best_payload = {
        "objective": objective,
        "candidate_id": int(best["candidate_id"]),
        "exposure": {
            "stress": float(best["exp_stress"]),
            "transition": float(best["exp_transition"]),
            "stable": float(best["exp_stable"]),
            "dispersion": float(best["exp_dispersion"]),
        },
        "metrics": {
            "score": float(best["score"]),
            "ann_return": float(best["ann_return"]),
            "sharpe": float(best["sharpe"]),
            "max_drawdown": float(best["max_drawdown"]),
            "ann_vol": float(best["ann_vol"]),
            "avg_turnover": float(best["avg_turnover"]),
        },
        "n_candidates": int(df.shape[0]),
    }
    return df, best_payload


def _write_daily_brief(
    outdir: Path,
    regime_df: pd.DataFrame,
    ts_off: pd.DataFrame,
    robust_metrics: dict[str, float],
    gate: dict[str, Any],
    baseline: dict[str, Any],
    calibration_best: dict[str, Any],
) -> None:
    lines: list[str] = []
    lines.append("Daily Regime Brief")
    lines.append(f"run_id={outdir.name}")
    if (not ts_off.empty) and ("insufficient_universe" in ts_off.columns):
        suff = ts_off[~ts_off["insufficient_universe"]].copy()
    else:
        suff = pd.DataFrame()
    if not suff.empty:
        x = suff.iloc[-1]
        lines.append(f"date={x['date']} N_used={int(x['N_used'])} p1={float(x['p1']):.6f} deff={float(x['deff']):.6f}")
    if not regime_df.empty:
        r = regime_df.iloc[-1]
        lines.append(f"regime_now={r['regime']} exposure_now={float(r['exposure']):.2f}")
    lines.append(
        "change_signal="
        f"joint_majority_60d={float(robust_metrics.get('joint_majority_60d', np.nan)):.3f}; "
        f"latest_joint_majority_5={float(robust_metrics.get('latest_joint_majority_5', np.nan)):.0f}"
    )
    lines.append(
        "trust="
        f"gate_blocked={bool(gate.get('blocked', True))}; "
        f"baseline_delta_p1={float(baseline.get('delta_p1_vs_previous', np.nan)):.6f}; "
        f"baseline_delta_deff={float(baseline.get('delta_deff_vs_previous', np.nan)):.6f}"
    )
    if calibration_best:
        exp = calibration_best.get("exposure", {})
        m = calibration_best.get("metrics", {})
        lines.append(
            "recommended_exposure="
            f"stress={float(exp.get('stress', np.nan)):.2f}, transition={float(exp.get('transition', np.nan)):.2f}, "
            f"stable={float(exp.get('stable', np.nan)):.2f}, dispersion={float(exp.get('dispersion', np.nan)):.2f}"
        )
        lines.append(
            "calibration_score="
            f"{float(m.get('score', np.nan)):.6f}; ann={float(m.get('ann_return', np.nan)):.4f}; "
            f"mdd={float(m.get('max_drawdown', np.nan)):.4f}; sharpe={float(m.get('sharpe', np.nan)):.4f}"
        )
    (outdir / "daily_regime_brief.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_commercial_narrative(
    outdir: Path,
    regime_df: pd.DataFrame,
    ts_off: pd.DataFrame,
    bt_summary: dict[str, Any],
    calibration_best: dict[str, Any],
    gate: dict[str, Any],
) -> None:
    if (not ts_off.empty) and ("insufficient_universe" in ts_off.columns):
        suff = ts_off[~ts_off["insufficient_universe"]].copy()
    else:
        suff = pd.DataFrame()
    if suff.empty:
        return
    last = suff.iloc[-1]
    regime_now = "unknown"
    exp_now = float("nan")
    if not regime_df.empty:
        rr = regime_df.iloc[-1]
        regime_now = str(rr.get("regime", "unknown"))
        exp_now = float(rr.get("exposure", np.nan))
    s = (bt_summary or {}).get("strategy", {}) or {}
    b = (bt_summary or {}).get("benchmark", {}) or {}
    rec = (calibration_best or {}).get("exposure", {}) or {}
    lines = []
    lines.append("Bloco 1 - Estado Atual")
    lines.append(
        f"Hoje o motor identifica regime {regime_now} com p1={float(last['p1']):.4f} e deff={float(last['deff']):.2f}. "
        f"Exposicao operacional atual={exp_now:.2f}."
    )
    lines.append("Bloco 2 - Mudanca Recente")
    lines.append(
        f"O sinal estrutural segue monitorado por robustez entre janelas e o gate esta "
        f"{'liberado' if not bool(gate.get('blocked', True)) else 'bloqueado'} para publicacao."
    )
    lines.append("Bloco 3 - Interpretacao Estatistica e Risco")
    if rec:
        lines.append(
            f"Leitura estrutural sugerida pela calibracao curta: stress={float(rec.get('stress', np.nan)):.2f}, "
            f"transition={float(rec.get('transition', np.nan)):.2f}, stable={float(rec.get('stable', np.nan)):.2f}, "
            f"dispersion={float(rec.get('dispersion', np.nan)):.2f}."
        )
    lines.append(
        f"No historico atual, a estrategia regime teve max drawdown={float(s.get('max_drawdown', np.nan)):.4f} "
        f"vs benchmark={float(b.get('max_drawdown', np.nan)):.4f}."
    )
    (outdir / "commercial_narrative.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_csv_list(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _future_stats(bt_by_date: pd.DataFrame, anchor_date: str, horizon_days: int) -> dict[str, float]:
    if bt_by_date.empty:
        return {
            "future_days_used": 0.0,
            "bench_cum_return": float("nan"),
            "strategy_cum_return": float("nan"),
            "bench_max_drawdown": float("nan"),
            "strategy_max_drawdown": float("nan"),
            "alpha_cum": float("nan"),
            "dd_improvement": float("nan"),
        }
    idx = bt_by_date.index.get_indexer([pd.Timestamp(anchor_date)])
    if idx.size == 0 or int(idx[0]) < 0:
        return {
            "future_days_used": 0.0,
            "bench_cum_return": float("nan"),
            "strategy_cum_return": float("nan"),
            "bench_max_drawdown": float("nan"),
            "strategy_max_drawdown": float("nan"),
            "alpha_cum": float("nan"),
            "dd_improvement": float("nan"),
        }
    pos = int(idx[0])
    sub = bt_by_date.iloc[pos + 1 : pos + 1 + int(max(1, horizon_days))].copy()
    if sub.empty:
        return {
            "future_days_used": 0.0,
            "bench_cum_return": float("nan"),
            "strategy_cum_return": float("nan"),
            "bench_max_drawdown": float("nan"),
            "strategy_max_drawdown": float("nan"),
            "alpha_cum": float("nan"),
            "dd_improvement": float("nan"),
        }

    b = pd.to_numeric(sub["benchmark_simple_ret"], errors="coerce").dropna()
    s = pd.to_numeric(sub["strategy_simple_ret"], errors="coerce").dropna()
    n = int(min(b.shape[0], s.shape[0]))
    if n <= 0:
        return {
            "future_days_used": 0.0,
            "bench_cum_return": float("nan"),
            "strategy_cum_return": float("nan"),
            "bench_max_drawdown": float("nan"),
            "strategy_max_drawdown": float("nan"),
            "alpha_cum": float("nan"),
            "dd_improvement": float("nan"),
        }
    b = b.iloc[:n]
    s = s.iloc[:n]
    b_eq = (1.0 + b).cumprod()
    s_eq = (1.0 + s).cumprod()
    b_cum = float(b_eq.iloc[-1] - 1.0)
    s_cum = float(s_eq.iloc[-1] - 1.0)
    b_mdd = float((b_eq / b_eq.cummax() - 1.0).min())
    s_mdd = float((s_eq / s_eq.cummax() - 1.0).min())
    return {
        "future_days_used": float(n),
        "bench_cum_return": b_cum,
        "strategy_cum_return": s_cum,
        "bench_max_drawdown": b_mdd,
        "strategy_max_drawdown": s_mdd,
        "alpha_cum": float(s_cum - b_cum),
        "dd_improvement": float(abs(b_mdd) - abs(s_mdd)),
    }


def _case_score(df: pd.DataFrame, regime: str) -> pd.Series:
    if regime == "stress":
        p1 = pd.to_numeric(df.get("p1"), errors="coerce")
        deff = pd.to_numeric(df.get("deff"), errors="coerce")
        return p1.fillna(-1.0) - 0.01 * deff.fillna(1e9)
    if regime == "transition":
        dp1 = pd.to_numeric(df.get("dp1_5"), errors="coerce").abs()
        dd = pd.to_numeric(df.get("ddeff_5"), errors="coerce").abs()
        return dp1.fillna(0.0) + 0.25 * dd.fillna(0.0)
    if regime == "dispersion":
        p1 = pd.to_numeric(df.get("p1"), errors="coerce")
        deff = pd.to_numeric(df.get("deff"), errors="coerce")
        return deff.fillna(-1.0) - p1.fillna(1e9)
    return pd.Series(np.zeros(df.shape[0]), index=df.index, dtype=float)


def _choose_case_row(df: pd.DataFrame, regime: str) -> dict[str, Any]:
    x = df[df["regime"] == regime].copy()
    if x.empty:
        return {}
    x["score_case"] = _case_score(x, regime=regime)
    x["date_dt"] = pd.to_datetime(x["date"], errors="coerce")
    x = x.sort_values(["score_case", "date_dt"], ascending=[False, False]).reset_index(drop=True)
    if x.empty:
        return {}
    return x.iloc[0].to_dict()


def _honest_verdict(alpha_cum: float, dd_improvement: float) -> str:
    if (not np.isfinite(alpha_cum)) or (not np.isfinite(dd_improvement)):
        return "inconclusivo: janela futura insuficiente."
    if (alpha_cum >= 0.0) and (dd_improvement >= 0.0):
        return "sinal util: melhor retorno e menor drawdown."
    if (alpha_cum < 0.0) and (dd_improvement >= 0.0):
        return "trade-off: protegeu risco, mas sacrificou retorno."
    if (alpha_cum >= 0.0) and (dd_improvement < 0.0):
        return "trade-off: ganhou retorno, mas com pior drawdown."
    return "sinal fraco nesse episodio: piorou retorno e drawdown."


def _build_case_studies(
    regime_df: pd.DataFrame,
    ts_off: pd.DataFrame,
    bt_df: pd.DataFrame,
    target_regimes: list[str],
    horizon_days: int,
) -> pd.DataFrame:
    if regime_df.empty or ts_off.empty or bt_df.empty:
        return pd.DataFrame()
    ts = ts_off.copy()
    ts = ts[~ts["insufficient_universe"]].copy()
    ts["date"] = pd.to_datetime(ts["date"], errors="coerce")
    ts = ts.dropna(subset=["date"])

    rg = regime_df.copy()
    rg["date"] = pd.to_datetime(rg["date"], errors="coerce")
    rg = rg.dropna(subset=["date"])
    merged = rg.merge(
        ts[["date", "N_used", "p1", "deff", "lambda1", "lambda2", "top5"]],
        on="date",
        how="left",
        suffixes=("", "_ts"),
    )
    if merged.empty:
        return pd.DataFrame()
    merged["date"] = merged["date"].dt.date.astype(str)

    bt = bt_df.copy()
    bt["date"] = pd.to_datetime(bt["date"], errors="coerce")
    bt = bt.dropna(subset=["date"]).set_index("date").sort_index()

    rows: list[dict[str, Any]] = []
    for r in target_regimes:
        if r not in {"stress", "transition", "dispersion", "stable"}:
            continue
        chosen = _choose_case_row(merged, regime=r)
        if not chosen:
            continue
        date_s = str(chosen.get("date"))
        f = _future_stats(bt_by_date=bt, anchor_date=date_s, horizon_days=int(max(1, horizon_days)))
        row = {
            "case_regime": r,
            "date": date_s,
            "N_used": float(chosen.get("N_used", np.nan)),
            "p1": float(chosen.get("p1", np.nan)),
            "deff": float(chosen.get("deff", np.nan)),
            "lambda1": float(chosen.get("lambda1", np.nan)),
            "lambda2": float(chosen.get("lambda2", np.nan)),
            "top5": float(chosen.get("top5", np.nan)),
            "exposure": float(chosen.get("exposure", np.nan)),
            "horizon_days": float(horizon_days),
            "future_days_used": float(f["future_days_used"]),
            "bench_cum_return": float(f["bench_cum_return"]),
            "strategy_cum_return": float(f["strategy_cum_return"]),
            "alpha_cum": float(f["alpha_cum"]),
            "bench_max_drawdown": float(f["bench_max_drawdown"]),
            "strategy_max_drawdown": float(f["strategy_max_drawdown"]),
            "dd_improvement": float(f["dd_improvement"]),
            "honest_verdict": _honest_verdict(alpha_cum=float(f["alpha_cum"]), dd_improvement=float(f["dd_improvement"])),
        }
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    order = [r for r in target_regimes if r in out["case_regime"].tolist()]
    out["__ord"] = out["case_regime"].map({k: i for i, k in enumerate(order)})
    out = out.sort_values(["__ord", "date"]).drop(columns="__ord").reset_index(drop=True)
    return out


def _write_case_studies_demo(outdir: Path, cases_df: pd.DataFrame) -> None:
    if cases_df.empty:
        (outdir / "case_studies_demo.txt").write_text("No case studies available.\n", encoding="utf-8")
        return
    lines: list[str] = []
    lines.append("Case Studies Demo (T120)")
    for _, r in cases_df.iterrows():
        lines.append("")
        lines.append(f"[{str(r['case_regime']).upper()}] date={r['date']}")
        lines.append(
            f"state: p1={float(r['p1']):.4f}, deff={float(r['deff']):.2f}, "
            f"lambda1={float(r['lambda1']):.2f}, lambda2={float(r['lambda2']):.2f}, top5={float(r['top5']):.3f}"
        )
        lines.append(
            f"leitura_estrutural: exposure={float(r['exposure']):.2f} for next {int(r['horizon_days'])} days "
            f"(used={int(r['future_days_used'])})"
        )
        lines.append(
            f"outcome: strategy={float(r['strategy_cum_return']):.4f}, benchmark={float(r['bench_cum_return']):.4f}, "
            f"alpha={float(r['alpha_cum']):.4f}, dd_improvement={float(r['dd_improvement']):.4f}"
        )
        lines.append(f"honesty: {str(r['honest_verdict'])}")
    (outdir / "case_studies_demo.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_structural_parallel_diagnostics(
    ts_off: pd.DataFrame,
    outdir: Path,
    *,
    csd_window: int,
    train_end: str,
    file_prefix: str = "diagnostics_structural",
    universe_name: str = "global",
) -> dict[str, Any]:
    if ts_off.empty:
        return {"status": "skipped", "reason": "empty_ts"}

    d = ts_off.copy()
    d["date"] = pd.to_datetime(d.get("date"), errors="coerce")
    d["T_window"] = pd.to_numeric(d.get("T_window"), errors="coerce")
    d["N_used"] = pd.to_numeric(d.get("N_used"), errors="coerce")
    d["Q"] = pd.to_numeric(d.get("Q"), errors="coerce")
    need_q = d["Q"].isna() & d["T_window"].notna() & d["N_used"].notna()
    d.loc[need_q, "Q"] = d.loc[need_q, "T_window"] / d.loc[need_q, "N_used"].clip(lower=1.0)
    d["universe_name"] = str(universe_name)
    d["phi"] = pd.to_numeric(d.get("p1"), errors="coerce")
    d["deff"] = pd.to_numeric(d.get("deff"), errors="coerce")
    d["lambda1"] = pd.to_numeric(d.get("lambda1"), errors="coerce")
    d["topk"] = pd.to_numeric(d.get("top5"), errors="coerce")
    d["forman_mean"] = pd.to_numeric(d.get("forman_mean"), errors="coerce")
    d["forman_share_negative"] = pd.to_numeric(d.get("forman_share_negative"), errors="coerce")
    d["H"] = np.where(d["deff"] > 0.0, np.log(d["deff"]), np.nan)
    d = d.dropna(subset=["date", "phi", "deff"]).sort_values("date").reset_index(drop=True)
    if d.empty:
        return {"status": "skipped", "reason": "empty_after_filter"}

    if STRUCTURAL_OK:
        d["ac1_phi"] = pd.to_numeric(structural_rolling_ac1(d["phi"], window=int(max(3, csd_window))), errors="coerce")
    else:
        d["ac1_phi"] = np.nan

    if str(train_end).strip():
        cutoff = pd.Timestamp(str(train_end).strip())
        train_mask = d["date"] <= cutoff
        if int(train_mask.sum()) < 10:
            pos = int(max(5, min(len(d) - 1, int(0.7 * len(d)))))
            train_mask = d.index <= pos
    else:
        pos = int(max(5, min(len(d) - 1, int(0.7 * len(d)))))
        train_mask = d.index <= pos

    if STRUCTURAL_OK:
        norm = structural_fit_normalizer(
            {
                "phi": d.loc[train_mask, "phi"],
                "deff": d.loc[train_mask, "deff"],
                "ac1_phi": d.loc[train_mask, "ac1_phi"],
                "neg_kappa_mean": -d.loc[train_mask, "forman_mean"].fillna(0.0),
            }
        )
        z_phi = structural_transform(d["phi"], norm, key="phi")
        z_deff = structural_transform(d["deff"], norm, key="deff")
        z_ac1 = structural_transform(d["ac1_phi"], norm, key="ac1_phi")
        z_neg_kappa = structural_transform(-d["forman_mean"].fillna(0.0), norm, key="neg_kappa_mean")
        d["score"] = pd.to_numeric(
            structural_score_fusion(
                {
                    "phi": z_phi,
                    "deff": z_deff,
                    "ac1_phi": z_ac1,
                    "neg_kappa_mean": z_neg_kappa,
                }
            ),
            errors="coerce",
        )
    else:
        norm = {}
        d["score"] = np.nan

    base_valid_cols = ["phi", "deff", "ac1_phi"]
    optional_valid_cols: list[str] = []
    if "forman_mean" in d.columns and d["forman_mean"].replace([np.inf, -np.inf], np.nan).notna().any():
        optional_valid_cols.append("forman_mean")
    valid_cols = base_valid_cols + optional_valid_cols
    d["flags_valid"] = d[valid_cols].replace([np.inf, -np.inf], np.nan).notna().all(axis=1).astype(int)
    d["date"] = pd.DatetimeIndex(d["date"]).strftime("%Y-%m-%d")

    diag_cols = [
        "date",
        "universe_name",
        "T_window",
        "N_used",
        "Q",
        "phi",
        "H",
        "deff",
        "lambda1",
        "topk",
        "ac1_phi",
        "forman_mean",
        "forman_share_negative",
        "flags_valid",
    ]
    score_cols = [
        "date",
        "universe_name",
        "T_window",
        "N_used",
        "Q",
        "score",
        "phi",
        "deff",
        "ac1_phi",
        "forman_mean",
        "flags_valid",
    ]
    for c in diag_cols + score_cols:
        if c not in d.columns:
            d[c] = np.nan

    pref = str(file_prefix).strip() or "diagnostics_structural"
    diag_path = outdir / f"{pref}_daily.csv"
    score_path = outdir / f"{pref}_score_daily.csv"
    meta_path = outdir / f"{pref}_meta.json"
    d[diag_cols].to_csv(diag_path, index=False)
    d[score_cols].to_csv(score_path, index=False)
    meta = {
        "status": "ok",
        "universe_name": str(universe_name),
        "structural_ok": bool(STRUCTURAL_OK),
        "rows": int(d.shape[0]),
        "csd_window": int(csd_window),
        "train_end": str(train_end),
        "normalizer": norm,
        "files": {
            "diagnostics_daily_csv": str(diag_path),
            "diagnostics_score_daily_csv": str(score_path),
            "diagnostics_meta_json": str(meta_path),
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def _slug_token(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", str(text).strip().lower()).strip("_")
    return s or "unknown"


def _write_vectors(v1_df: pd.DataFrame, out_base: Path) -> str:
    if v1_df.empty:
        return ""
    out_base.parent.mkdir(parents=True, exist_ok=True)
    parquet_path = out_base.with_suffix(".parquet")
    try:
        v1_df.to_parquet(parquet_path, index=False)
        return str(parquet_path)
    except Exception:
        csv_path = out_base.with_suffix(".csv")
        v1_df.to_csv(csv_path, index=False)
        return str(csv_path)


def _load_hierarchical_metadata(universe_core: pd.DataFrame, metadata_path: Path) -> pd.DataFrame:
    try:
        from engine.ops.metadata import load_asset_metadata
    except Exception:
        load_asset_metadata = None

    if load_asset_metadata is not None and metadata_path.exists():
        md = load_asset_metadata(metadata_path)
    else:
        md = pd.DataFrame(columns=["asset_id", "ticker", "sector_gics", "sector_internal", "liquidity_proxy"])

    core = universe_core.copy()
    core["ticker"] = core["ticker"].astype(str)
    core["sector"] = core["sector"].astype(str)
    core["coverage"] = pd.to_numeric(core.get("coverage"), errors="coerce")

    if md.empty:
        out = core.copy()
        out["asset_id"] = out["ticker"]
        out["sector_gics"] = out["sector"]
        out["sector_internal"] = out["sector"]
        out["liquidity_proxy"] = out["coverage"]
        return out[["asset_id", "ticker", "sector_gics", "sector_internal", "liquidity_proxy"]].copy()

    out = core.merge(md, on="ticker", how="left", suffixes=("", "_md"))
    out["asset_id"] = out["asset_id"].fillna(out["ticker"]).astype(str)
    out["sector_gics"] = out["sector_gics"].fillna(out["sector"]).astype(str)
    out["sector_internal"] = out["sector_internal"].fillna(out["sector_gics"]).astype(str)
    out["liquidity_proxy"] = pd.to_numeric(out["liquidity_proxy"], errors="coerce").fillna(out["coverage"])
    out = out[["asset_id", "ticker", "sector_gics", "sector_internal", "liquidity_proxy"]].drop_duplicates(subset=["asset_id"], keep="first")
    return out.sort_values("asset_id").reset_index(drop=True)


def _compute_cross_daily(global_v1: pd.DataFrame, sector_v1_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    cols = ["date", "sector", "loading_sector_on_global", "overlap_sector_global", "n_assets_sector"]
    if global_v1.empty or (not sector_v1_map):
        return pd.DataFrame(columns=cols)
    g = global_v1.copy()
    g["date"] = g["date"].astype(str)
    g["asset_id"] = g["asset_id"].astype(str)
    g["weight"] = pd.to_numeric(g["weight"], errors="coerce")
    g = g.dropna(subset=["date", "asset_id", "weight"])
    if g.empty:
        return pd.DataFrame(columns=cols)
    g_by_date = {str(d): x for d, x in g.groupby("date")}
    rows: list[dict[str, Any]] = []
    for sector_name, sraw in sector_v1_map.items():
        if sraw.empty:
            continue
        s = sraw.copy()
        s["date"] = s["date"].astype(str)
        s["asset_id"] = s["asset_id"].astype(str)
        s["weight"] = pd.to_numeric(s["weight"], errors="coerce")
        s = s.dropna(subset=["date", "asset_id", "weight"])
        if s.empty:
            continue
        for d, ss in s.groupby("date"):
            gg = g_by_date.get(str(d))
            if gg is None or gg.empty:
                continue
            g_map = gg.set_index("asset_id")["weight"].to_dict()
            s_map = ss.set_index("asset_id")["weight"].to_dict()
            assets = sorted(set(s_map.keys()))
            if len(assets) < 2:
                continue
            gv = np.asarray([float(g_map.get(a, 0.0)) for a in assets], dtype=float)
            sv = np.asarray([float(s_map.get(a, 0.0)) for a in assets], dtype=float)
            loading = float(np.sum(np.square(gv)))
            ng = float(np.linalg.norm(gv))
            ns = float(np.linalg.norm(sv))
            overlap = float("nan")
            if ng > 1e-12 and ns > 1e-12:
                overlap = float(abs(np.dot(sv / ns, gv / ng)))
            rows.append(
                {
                    "date": str(d),
                    "sector": str(sector_name),
                    "loading_sector_on_global": loading,
                    "overlap_sector_global": overlap,
                    "n_assets_sector": int(len(assets)),
                }
            )
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows).sort_values(["date", "sector"]).reset_index(drop=True)


def _write_hierarchical_state(
    outdir: Path,
    global_score_df: pd.DataFrame,
    sector_score_map_gics: dict[str, pd.DataFrame],
    sector_score_map_internal: dict[str, pd.DataFrame],
    cross_gics: pd.DataFrame,
    cross_internal: pd.DataFrame,
) -> tuple[Path, Path]:
    g = global_score_df.copy()
    if g.empty or ("date" not in g.columns) or ("score" not in g.columns):
        daily = outdir / "hierarchical_state_daily.jsonl"
        latest = outdir / "hierarchical_state_latest.json"
        daily.write_text("", encoding="utf-8")
        latest.write_text(json.dumps({"status": "empty"}, ensure_ascii=False, indent=2), encoding="utf-8")
        return daily, latest
    g["date"] = g["date"].astype(str)
    g["score"] = pd.to_numeric(g["score"], errors="coerce")
    g = g.dropna(subset=["date"]).sort_values("date")
    if g.empty:
        daily = outdir / "hierarchical_state_daily.jsonl"
        latest = outdir / "hierarchical_state_latest.json"
        daily.write_text("", encoding="utf-8")
        latest.write_text(json.dumps({"status": "empty"}, ensure_ascii=False, indent=2), encoding="utf-8")
        return daily, latest

    def _stack_sector_scores(m: dict[str, pd.DataFrame], kind: str) -> pd.DataFrame:
        rows: list[pd.DataFrame] = []
        for sec, df in m.items():
            if df.empty:
                continue
            x = df.copy()
            x["date"] = x["date"].astype(str)
            x["score"] = pd.to_numeric(x["score"], errors="coerce")
            x["sector"] = str(sec)
            x["kind"] = str(kind)
            rows.append(x[["date", "score", "sector", "kind"]])
        if not rows:
            return pd.DataFrame(columns=["date", "score", "sector", "kind"])
        return pd.concat(rows, ignore_index=True)

    sec_all = pd.concat(
        [
            _stack_sector_scores(sector_score_map_gics, "gics"),
            _stack_sector_scores(sector_score_map_internal, "internal"),
        ],
        ignore_index=True,
    )
    cross_all = pd.concat(
        [
            cross_gics.assign(kind="gics") if not cross_gics.empty else pd.DataFrame(),
            cross_internal.assign(kind="internal") if not cross_internal.empty else pd.DataFrame(),
        ],
        ignore_index=True,
    )
    if not cross_all.empty:
        cross_all["date"] = cross_all["date"].astype(str)
        cross_all["loading_sector_on_global"] = pd.to_numeric(cross_all["loading_sector_on_global"], errors="coerce")
        cross_all["overlap_sector_global"] = pd.to_numeric(cross_all["overlap_sector_global"], errors="coerce")

    lines: list[str] = []
    latest_payload: dict[str, Any] = {"status": "empty"}
    for d in sorted(g["date"].unique().tolist()):
        g_row = g[g["date"] == d].sort_values("date").tail(1)
        global_score = float(g_row["score"].iloc[-1]) if (not g_row.empty) else float("nan")

        sec_d = sec_all[sec_all["date"] == d].copy() if not sec_all.empty else pd.DataFrame()
        sec_top = (
            sec_d.sort_values("score", ascending=False)[["sector", "kind", "score"]].head(5).to_dict(orient="records")
            if not sec_d.empty
            else []
        )
        cr_d = cross_all[cross_all["date"] == d].copy() if not cross_all.empty else pd.DataFrame()
        top_load = (
            cr_d.sort_values("loading_sector_on_global", ascending=False)[["sector", "kind", "loading_sector_on_global"]]
            .head(5)
            .to_dict(orient="records")
            if not cr_d.empty
            else []
        )
        top_overlap = (
            cr_d.sort_values("overlap_sector_global", ascending=False)[["sector", "kind", "overlap_sector_global"]]
            .head(5)
            .to_dict(orient="records")
            if not cr_d.empty
            else []
        )
        payload = {
            "date": str(d),
            "global_score": global_score,
            "top_sectors_by_score": sec_top,
            "top_sectors_by_loading": top_load,
            "top_sectors_by_overlap": top_overlap,
        }
        lines.append(json.dumps(payload, ensure_ascii=False))
        latest_payload = payload

    daily_path = outdir / "hierarchical_state_daily.jsonl"
    latest_path = outdir / "hierarchical_state_latest.json"
    daily_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    latest_path.write_text(json.dumps(latest_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return daily_path, latest_path


def _run_hierarchical_diagnostics(
    *,
    outdir: Path,
    returns_core: pd.DataFrame,
    universe_core: pd.DataFrame,
    official_window: int,
    min_assets: int,
    coverage_window: float,
    overlap_step: int,
    noise_step: int,
    bootstrap_block: int,
    seed: int,
    compute_forman: bool,
    forman_topk: int,
    csd_window: int,
    train_end: str,
    metadata_path: Path,
    n_global: int,
    n_sector: int,
    min_coverage_global: float,
    min_coverage_sector: float,
    enable_internal: bool,
    save_v1: bool,
) -> dict[str, Any]:
    try:
        from engine.core.universe import select_global_universe, select_sector_universe
    except Exception as exc:
        return {"status": "fail", "reason": f"engine.core.universe import failed: {exc}"}

    md = _load_hierarchical_metadata(universe_core=universe_core, metadata_path=metadata_path)
    if md.empty:
        return {"status": "fail", "reason": "metadata empty for core universe"}
    md["coverage"] = [float(returns_core[t].notna().mean()) if t in returns_core.columns else np.nan for t in md["ticker"].astype(str)]

    hier_dir = outdir / "hierarchical"
    uni_dir = hier_dir / "universes"
    vec_dir = hier_dir / "vectors"
    uni_dir.mkdir(parents=True, exist_ok=True)
    vec_dir.mkdir(parents=True, exist_ok=True)
    md.to_csv(hier_dir / "asset_metadata_used.csv", index=False)

    g_assets = select_global_universe(
        returns_df=returns_core,
        metadata=md,
        n_global=int(n_global),
        min_coverage=float(min_coverage_global),
    )
    if len(g_assets) < int(min_assets):
        return {"status": "fail", "reason": f"global universe too small after selection: {len(g_assets)} < {int(min_assets)}"}
    g_meta = md[md["asset_id"].isin(g_assets)].copy().sort_values("asset_id")
    g_meta.to_csv(uni_dir / "global_universe.csv", index=False)

    universe_score_map_gics: dict[str, pd.DataFrame] = {}
    universe_score_map_internal: dict[str, pd.DataFrame] = {}
    universe_v1_map_gics: dict[str, pd.DataFrame] = {}
    universe_v1_map_internal: dict[str, pd.DataFrame] = {}
    sector_universe_rows: list[dict[str, Any]] = []

    g_returns = returns_core[g_assets].copy()
    g_sector_map = md.set_index("asset_id")["sector_gics"].astype(str).to_dict()
    g_ts, _, _, g_v1 = _process_window(
        returns_wide=g_returns,
        sector_by_ticker={t: g_sector_map.get(t, "unknown") for t in g_assets},
        window=int(official_window),
        cov_window=float(coverage_window),
        min_assets=int(min_assets),
        noise_step=int(noise_step),
        bootstrap_block=int(bootstrap_block),
        overlap_step=int(overlap_step),
        seed=int(seed),
        compute_forman=bool(compute_forman),
        forman_topk=int(forman_topk),
        capture_v1=bool(save_v1),
        universe_name="global",
    )
    g_meta_struct = _build_structural_parallel_diagnostics(
        ts_off=g_ts,
        outdir=hier_dir,
        csd_window=int(csd_window),
        train_end=str(train_end),
        file_prefix="diagnostics_global",
        universe_name="global",
    )
    g_score = pd.read_csv(hier_dir / "diagnostics_global_score_daily.csv") if (hier_dir / "diagnostics_global_score_daily.csv").exists() else pd.DataFrame()
    g_v1_path = ""
    if bool(save_v1) and (not g_v1.empty):
        g_v1_path = _write_vectors(g_v1, vec_dir / "v1_global")

    for sec in sorted(md["sector_gics"].dropna().astype(str).unique().tolist()):
        sec_assets = select_sector_universe(
            returns_df=returns_core,
            metadata=md,
            sector_name=str(sec),
            n_sector=int(n_sector),
            min_coverage=float(min_coverage_sector),
            sector_col="sector_gics",
        )
        if len(sec_assets) < int(min_assets):
            continue
        sec_slug = _slug_token(sec)
        md[md["asset_id"].isin(sec_assets)].copy().sort_values("asset_id").to_csv(uni_dir / f"sector_universe_gics_{sec_slug}.csv", index=False)
        sec_returns = returns_core[sec_assets].copy()
        sec_ts, _, _, sec_v1 = _process_window(
            returns_wide=sec_returns,
            sector_by_ticker={t: str(sec) for t in sec_assets},
            window=int(official_window),
            cov_window=float(coverage_window),
            min_assets=int(min_assets),
            noise_step=int(noise_step),
            bootstrap_block=int(bootstrap_block),
            overlap_step=int(overlap_step),
            seed=int(seed),
            compute_forman=bool(compute_forman),
            forman_topk=int(forman_topk),
            capture_v1=bool(save_v1),
            universe_name=f"gics:{sec}",
        )
        _build_structural_parallel_diagnostics(
            ts_off=sec_ts,
            outdir=hier_dir,
            csd_window=int(csd_window),
            train_end=str(train_end),
            file_prefix=f"diagnostics_sector_gics_{sec_slug}",
            universe_name=f"gics:{sec}",
        )
        score_path = hier_dir / f"diagnostics_sector_gics_{sec_slug}_score_daily.csv"
        if score_path.exists():
            universe_score_map_gics[str(sec)] = pd.read_csv(score_path)
        if bool(save_v1) and (not sec_v1.empty):
            _write_vectors(sec_v1, vec_dir / f"v1_gics_{sec_slug}")
            universe_v1_map_gics[str(sec)] = sec_v1.copy()
        sector_universe_rows.append({"kind": "gics", "sector": str(sec), "n_assets": int(len(sec_assets)), "slug": sec_slug})

    if bool(enable_internal):
        for sec in sorted(md["sector_internal"].dropna().astype(str).unique().tolist()):
            sec_assets = select_sector_universe(
                returns_df=returns_core,
                metadata=md,
                sector_name=str(sec),
                n_sector=int(n_sector),
                min_coverage=float(min_coverage_sector),
                sector_col="sector_internal",
            )
            if len(sec_assets) < int(min_assets):
                continue
            sec_slug = _slug_token(sec)
            md[md["asset_id"].isin(sec_assets)].copy().sort_values("asset_id").to_csv(uni_dir / f"sector_universe_internal_{sec_slug}.csv", index=False)
            sec_returns = returns_core[sec_assets].copy()
            sec_ts, _, _, sec_v1 = _process_window(
                returns_wide=sec_returns,
                sector_by_ticker={t: str(sec) for t in sec_assets},
                window=int(official_window),
                cov_window=float(coverage_window),
                min_assets=int(min_assets),
                noise_step=int(noise_step),
                bootstrap_block=int(bootstrap_block),
                overlap_step=int(overlap_step),
                seed=int(seed),
                compute_forman=bool(compute_forman),
                forman_topk=int(forman_topk),
                capture_v1=bool(save_v1),
                universe_name=f"internal:{sec}",
            )
            _build_structural_parallel_diagnostics(
                ts_off=sec_ts,
                outdir=hier_dir,
                csd_window=int(csd_window),
                train_end=str(train_end),
                file_prefix=f"diagnostics_sector_internal_{sec_slug}",
                universe_name=f"internal:{sec}",
            )
            score_path = hier_dir / f"diagnostics_sector_internal_{sec_slug}_score_daily.csv"
            if score_path.exists():
                universe_score_map_internal[str(sec)] = pd.read_csv(score_path)
            if bool(save_v1) and (not sec_v1.empty):
                _write_vectors(sec_v1, vec_dir / f"v1_internal_{sec_slug}")
                universe_v1_map_internal[str(sec)] = sec_v1.copy()
            sector_universe_rows.append({"kind": "internal", "sector": str(sec), "n_assets": int(len(sec_assets)), "slug": sec_slug})

    sector_universe_df = pd.DataFrame(sector_universe_rows)
    if not sector_universe_df.empty:
        sector_universe_df.to_csv(uni_dir / "sector_universe_index.csv", index=False)

    cross_gics = _compute_cross_daily(g_v1, universe_v1_map_gics) if bool(save_v1) else pd.DataFrame()
    cross_internal = _compute_cross_daily(g_v1, universe_v1_map_internal) if bool(save_v1) else pd.DataFrame()
    cross_gics_path = hier_dir / "cross_sector_global_gics_daily.csv"
    cross_internal_path = hier_dir / "cross_sector_global_internal_daily.csv"
    cross_gics.to_csv(cross_gics_path, index=False)
    cross_internal.to_csv(cross_internal_path, index=False)

    state_daily, state_latest = _write_hierarchical_state(
        outdir=hier_dir,
        global_score_df=g_score,
        sector_score_map_gics=universe_score_map_gics,
        sector_score_map_internal=universe_score_map_internal,
        cross_gics=cross_gics,
        cross_internal=cross_internal,
    )

    latest_ptr = outdir / "platform" / "latest_hierarchical_state.json"
    latest_ptr.parent.mkdir(parents=True, exist_ok=True)
    latest_ptr.write_text(
        json.dumps(
            {
                "status": "ok",
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "hierarchical_dir": str(hier_dir),
                "daily_jsonl": str(state_daily),
                "latest_json": str(state_latest),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return {
        "status": "ok",
        "metadata_path": str(metadata_path),
        "asset_metadata_used_csv": str(hier_dir / "asset_metadata_used.csv"),
        "global_universe_size": int(len(g_assets)),
        "sector_universes_count": int(sector_universe_df.shape[0]) if not sector_universe_df.empty else 0,
        "gics_universes_count": int(sector_universe_df[sector_universe_df["kind"] == "gics"].shape[0]) if not sector_universe_df.empty else 0,
        "internal_universes_count": int(sector_universe_df[sector_universe_df["kind"] == "internal"].shape[0]) if not sector_universe_df.empty else 0,
        "save_v1": bool(save_v1),
        "global_v1_file": str(g_v1_path),
        "cross_gics_csv": str(cross_gics_path),
        "cross_internal_csv": str(cross_internal_path),
        "hierarchical_state_daily_jsonl": str(state_daily),
        "hierarchical_state_latest_json": str(state_latest),
        "latest_hierarchical_state_json": str(latest_ptr),
        "diagnostics_global_score_csv": str(hier_dir / "diagnostics_global_score_daily.csv"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Macro corr lab offline (no API).")
    ap.add_argument("--policy-path", type=str, default=str(ROOT / "config" / "lab_corr_policy.json"))
    ap.add_argument("--apply-policy", type=int, default=1)
    ap.add_argument("--panel-path", type=str, default="")
    ap.add_argument("--universe-path", type=str, default="")
    ap.add_argument("--out-base", type=str, default=str(DEFAULT_OUT_BASE))
    ap.add_argument("--baseline-dir", type=str, default=str(DEFAULT_BASELINE_DIR))
    ap.add_argument("--start", type=str, default="2018-01-01")
    ap.add_argument("--end", type=str, default="2026-02-12")
    ap.add_argument("--coverage-core", type=float, default=0.95)
    ap.add_argument("--coverage-window", type=float, default=0.98)
    ap.add_argument("--min-assets", type=int, default=25)
    ap.add_argument("--max-core-assets", type=int, default=0, help="Cap core universe size after coverage filter (0 = no cap).")
    ap.add_argument("--windows", type=str, default="60,120,252", help="Rolling windows to run, comma-separated.")
    ap.add_argument(
        "--business-days-only",
        type=int,
        default=1,
        help="Keep only Monday-Friday rows before building windows.",
    )
    ap.add_argument("--noise-step", type=int, default=5, help="Compute shuffle/bootstrap baseline every N days.")
    ap.add_argument("--bootstrap-block", type=int, default=10)
    ap.add_argument("--overlap-step", type=int, default=5, help="Compute eigvec overlap every N days.")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--enable-structural-v1", type=int, default=0, help="Write parallel structural diagnostics files.")
    ap.add_argument("--structural-topk", type=int, default=10)
    ap.add_argument("--structural-csd-window", type=int, default=60)
    ap.add_argument("--structural-train-end", type=str, default="2019-12-31")
    ap.add_argument("--enable-hierarchical", type=int, default=0, help="Run global + sector structural diagnostics blocks.")
    ap.add_argument("--asset-metadata-path", type=str, default="data/asset_metadata.csv")
    ap.add_argument("--n-global", type=int, default=250)
    ap.add_argument("--n-sector", type=int, default=60)
    ap.add_argument("--min-coverage-global", type=float, default=0.98)
    ap.add_argument("--min-coverage-sector", type=float, default=0.95)
    ap.add_argument("--enable-internal-sectors", type=int, default=1)
    ap.add_argument("--save-v1", type=int, default=0, help="Persist dominant eigenvector weights by universe/day for cross metrics.")
    ap.add_argument("--official-window", type=int, default=120)
    ap.add_argument("--freeze-baseline", type=int, default=1)
    ap.add_argument("--strict-checks", type=int, default=1)
    ap.add_argument("--cost-bps", type=float, default=5.0)
    ap.add_argument("--max-daily-turnover", type=float, default=0.10)
    ap.add_argument("--hysteresis-days", type=int, default=3)
    ap.add_argument(
        "--regime-threshold-mode",
        type=str,
        default="walk_forward",
        choices=["walk_forward", "full_sample"],
        help="walk_forward calibrates thresholds with data up to t-1.",
    )
    ap.add_argument("--walkforward-min-history", type=int, default=252)
    ap.add_argument("--transition-weight-dp1", type=float, default=0.50)
    ap.add_argument("--transition-weight-ddeff", type=float, default=0.50)
    ap.add_argument("--transition-weight-overlap", type=float, default=0.00)
    ap.add_argument("--exp-stress", type=float, default=0.10)
    ap.add_argument("--exp-transition", type=float, default=0.40)
    ap.add_argument("--exp-stable", type=float, default=0.70)
    ap.add_argument("--exp-dispersion", type=float, default=0.90)
    ap.add_argument("--min-joint-majority-60d", type=float, default=0.30)
    ap.add_argument("--require-latest-majority", type=int, default=0)
    ap.add_argument("--max-abs-delta-p1", type=float, default=0.01)
    ap.add_argument("--max-abs-delta-deff", type=float, default=1.00)
    ap.add_argument("--max-active-cluster-alerts", type=int, default=1)
    ap.add_argument("--require-policy-lock-match", type=int, default=0)
    ap.add_argument("--alert-lookback", type=int, default=252)
    ap.add_argument("--risk-q-yellow", type=float, default=0.70)
    ap.add_argument("--risk-q-red", type=float, default=0.90)
    ap.add_argument("--min-conf-yellow", type=float, default=0.35)
    ap.add_argument("--min-conf-red", type=float, default=0.45)
    ap.add_argument("--alert-persist-window", type=int, default=3)
    ap.add_argument("--alert-persist-count", type=int, default=2)
    ap.add_argument("--max-insufficient-ratio", type=float, default=0.05)
    ap.add_argument("--min-n-used-ratio", type=float, default=0.90)
    ap.add_argument("--q-min", type=float, default=0.12)
    ap.add_argument("--case-horizon-days", type=int, default=20)
    ap.add_argument("--case-regimes", type=str, default="stress,transition,dispersion")
    ap.add_argument("--calibrate-exposure-grid", type=int, default=1)
    ap.add_argument("--apply-grid-best", type=int, default=0)
    ap.add_argument("--calibration-objective", type=str, default="composite", choices=["composite", "ann_over_dd", "sharpe"])
    ap.add_argument("--update-release-pointer", type=int, default=1)
    args = ap.parse_args()

    policy_path = Path(args.policy_path)
    policy = _load_policy(policy_path)
    if int(args.apply_policy) == 1 and policy:
        _apply_policy_to_args(args, policy)

    if args.panel_path:
        panel_path = Path(args.panel_path)
        universe_path = Path(args.universe_path) if args.universe_path else None
    else:
        latest = _find_latest_finance_run()
        panel_path = latest / "panel_long_sector.csv"
        universe_path = latest / "universe_fixed.csv"
    if not panel_path.exists():
        raise SystemExit(f"panel path not found: {panel_path}")
    if universe_path is not None and (not universe_path.exists()):
        raise SystemExit(f"universe path not found: {universe_path}")

    out_base = Path(args.out_base)
    baseline_dir_resolved, baseline_namespace = _resolve_baseline_dir(
        args=args,
        out_base=out_base,
        policy_path=policy_path,
        policy=policy,
    )
    args.baseline_dir = str(baseline_dir_resolved)
    outdir = out_base / _run_id()
    outdir.mkdir(parents=True, exist_ok=True)
    effective_cfg = {
        "start": args.start,
        "end": args.end,
        "coverage_core": float(args.coverage_core),
        "coverage_window": float(args.coverage_window),
        "max_core_assets": int(args.max_core_assets),
        "windows": sorted(set([int(args.official_window)] + [int(x) for x in _parse_windows(args.windows)])),
        "overlap_step": int(args.overlap_step),
        "business_days_only": bool(int(args.business_days_only)),
        "enable_structural_v1": bool(int(args.enable_structural_v1)),
        "structural_topk": int(args.structural_topk),
        "structural_csd_window": int(args.structural_csd_window),
        "structural_train_end": str(args.structural_train_end),
        "enable_hierarchical": bool(int(args.enable_hierarchical)),
        "asset_metadata_path": str(args.asset_metadata_path),
        "n_global": int(args.n_global),
        "n_sector": int(args.n_sector),
        "min_coverage_global": float(args.min_coverage_global),
        "min_coverage_sector": float(args.min_coverage_sector),
        "enable_internal_sectors": bool(int(args.enable_internal_sectors)),
        "save_v1": bool(int(args.save_v1)),
        "official_window": int(args.official_window),
        "hysteresis_days": int(args.hysteresis_days),
        "regime_threshold_mode": str(args.regime_threshold_mode),
        "walkforward_min_history": int(args.walkforward_min_history),
        "transition_weight_dp1": float(args.transition_weight_dp1),
        "transition_weight_ddeff": float(args.transition_weight_ddeff),
        "transition_weight_overlap": float(args.transition_weight_overlap),
        "exp_stress": float(args.exp_stress),
        "exp_transition": float(args.exp_transition),
        "exp_stable": float(args.exp_stable),
        "exp_dispersion": float(args.exp_dispersion),
        "min_joint_majority_60d": float(args.min_joint_majority_60d),
        "require_latest_majority": bool(int(args.require_latest_majority)),
        "require_policy_lock_match": bool(int(args.require_policy_lock_match)),
        "max_abs_delta_p1": float(args.max_abs_delta_p1),
        "max_abs_delta_deff": float(args.max_abs_delta_deff),
        "max_active_cluster_alerts": int(args.max_active_cluster_alerts),
        "risk_q_yellow": float(args.risk_q_yellow),
        "risk_q_red": float(args.risk_q_red),
        "min_conf_yellow": float(args.min_conf_yellow),
        "min_conf_red": float(args.min_conf_red),
        "alert_persist_window": int(args.alert_persist_window),
        "alert_persist_count": int(args.alert_persist_count),
        "max_insufficient_ratio": float(args.max_insufficient_ratio),
        "min_n_used_ratio": float(args.min_n_used_ratio),
        "q_min": float(args.q_min),
        "cost_bps": float(args.cost_bps),
        "max_daily_turnover": float(args.max_daily_turnover),
        "calibrate_exposure_grid": bool(int(args.calibrate_exposure_grid)),
        "apply_grid_best": bool(int(args.apply_grid_best)),
        "calibration_objective": str(args.calibration_objective),
        "case_horizon_days": int(args.case_horizon_days),
        "case_regimes": str(args.case_regimes),
        "baseline_dir": str(args.baseline_dir),
        "baseline_namespace": str(baseline_namespace),
    }
    (outdir / "policy_used.json").write_text(
        json.dumps(
            {
                "policy_path": str(policy_path),
                "policy_loaded": bool(policy),
                "policy": policy,
                "effective": effective_cfg,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    policy_lock = _build_policy_lock(
        out_base=out_base,
        outdir=outdir,
        effective_policy=effective_cfg,
    )

    panel = pd.read_csv(panel_path)
    _ensure_cols(panel, ["date", "ticker", "sector", "r"], "panel")
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    panel["r"] = pd.to_numeric(panel["r"], errors="coerce")
    panel = panel.dropna(subset=["date", "ticker", "sector", "r"]).copy()
    panel = panel[(panel["date"] >= pd.Timestamp(args.start)) & (panel["date"] <= pd.Timestamp(args.end))].sort_values(["date", "ticker"]).reset_index(drop=True)
    if bool(int(args.business_days_only)):
        panel = panel[panel["date"].dt.dayofweek < 5].copy()
        panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)
    if panel.empty:
        raise SystemExit("No rows in target period after filtering.")

    trading_days = pd.DatetimeIndex(sorted(panel["date"].drop_duplicates().to_list()))
    returns_wide = panel.pivot_table(index="date", columns="ticker", values="r", aggfunc="last").reindex(trading_days).sort_index()
    sector_map = panel[["ticker", "sector"]].drop_duplicates(subset=["ticker"], keep="last").set_index("ticker")["sector"].to_dict()

    coverage = returns_wide.notna().mean(axis=0)
    core_candidates = coverage[coverage >= float(args.coverage_core)].index.astype(str).tolist()
    core_df = pd.DataFrame(
        {
            "ticker": core_candidates,
            "sector": [sector_map.get(t, "unknown") for t in core_candidates],
            "coverage": [float(coverage.loc[t]) for t in core_candidates],
        }
    )
    core_df = core_df.sort_values(["coverage", "ticker"], ascending=[False, True]).reset_index(drop=True)
    max_core_assets = int(max(0, int(args.max_core_assets)))
    if max_core_assets > 0:
        core_df = core_df.head(max_core_assets).copy()
    core = sorted(core_df["ticker"].astype(str).tolist())
    if len(core) < int(args.min_assets):
        raise SystemExit(f"core universe too small: {len(core)} < min_assets={int(args.min_assets)}")

    universe_core = core_df.sort_values(["sector", "ticker"]).reset_index(drop=True)
    universe_core.to_csv(outdir / "universe_core.csv", index=False)
    core_counts = universe_core.groupby("sector", as_index=False).agg(n_tickers=("ticker", "count"))
    core_counts.to_csv(outdir / "universe_core_by_sector.csv", index=False)

    R = returns_wide[core].copy()
    R.to_csv(outdir / "returns_wide_core.csv", index=True, index_label="date")
    windows = _parse_windows(args.windows)
    if int(args.official_window) not in windows:
        windows = sorted(set(windows + [int(args.official_window)]))

    summary = [
        "Macro Correlation Offline Lab",
        f"run_dir: {outdir}",
        f"panel_path: {panel_path}",
        f"period: {args.start} -> {args.end}",
        f"windows: {windows}",
        f"coverage_core_threshold: {args.coverage_core}",
        f"coverage_window_threshold: {args.coverage_window}",
        f"business_days_only: {bool(int(args.business_days_only))}",
        f"max_core_assets: {int(max(0, int(args.max_core_assets)))}",
        f"N_core: {len(core)}",
        "N_core_by_sector:",
    ]
    for _, r in core_counts.sort_values("sector").iterrows():
        summary.append(f"  - {r['sector']}: {int(r['n_tickers'])}")
    summary.append(f"scipy_enabled_for_clustering: {SCIPY_OK}")

    ts_map: dict[int, pd.DataFrame] = {}
    for w in windows:
        ts, snap, sec, _ = _process_window(
            returns_wide=R,
            sector_by_ticker={t: sector_map.get(t, "unknown") for t in core},
            window=w,
            cov_window=float(args.coverage_window),
            min_assets=int(args.min_assets),
            noise_step=int(args.noise_step),
            bootstrap_block=int(args.bootstrap_block),
            overlap_step=int(args.overlap_step),
            seed=int(args.seed),
            compute_forman=bool(int(args.enable_structural_v1)),
            forman_topk=int(args.structural_topk),
            capture_v1=False,
            universe_name="official",
        )
        ts_map[w] = ts.copy()
        cols = [
            "date",
            "T_window",
            "N_used",
            "Q",
            "p1",
            "deff",
            "lambda1",
            "lambda2",
            "lambda3",
            "lambda4",
            "lambda5",
            "lambda6",
            "lambda7",
            "lambda8",
            "lambda9",
            "lambda10",
            "top5",
            "cluster_count",
            "largest_share",
            "entropy",
            "turnover_pair_frac",
            "p1_shuffle",
            "deff_shuffle",
            "structure_score",
            "p1_bootstrap",
            "deff_bootstrap",
            "structure_score_bootstrap",
            "eigvec_overlap_1d",
            "eigvec_instability_1d",
            "forman_mean",
            "forman_share_negative",
            "insufficient_universe",
        ]
        for c in cols:
            if c not in ts.columns:
                ts[c] = np.nan
        ts = ts[cols]
        ts.to_csv(outdir / f"macro_timeseries_T{w}.csv", index=False)
        if not snap.empty:
            snap = snap.sort_values(["date", "ticker"]).reset_index(drop=True)
        snap.to_csv(outdir / f"clusters_snapshots_T{w}.csv", index=False)
        summary += ["", _summary_block(ts=ts, sector_daily=sec, window=w, outdir=outdir)]

    robust_df, robust_metrics, robust_txt = _build_robustness(ts_map=ts_map, outdir=outdir)
    summary += ["", "[ROBUSTNESS]", robust_txt]
    signif_df = _build_significance_tables(ts_map=ts_map, outdir=outdir)
    if not signif_df.empty:
        summary += ["", "[SIGNIFICANCE_BOOTSTRAP]"]
        for w in sorted(signif_df["window"].dropna().astype(int).unique().tolist()):
            x = signif_df[signif_df["window"] == int(w)]
            if x.empty:
                continue
            p1r = x[x["metric"] == "delta_p1_vs_bootstrap"]
            dr = x[x["metric"] == "delta_deff_vs_bootstrap"]
            if (not p1r.empty) and (not dr.empty):
                a = p1r.iloc[0]
                b = dr.iloc[0]
                summary += [
                    f"T{w}: p1_sig_share={float(a['significant_share_p_lt_0_05']):.3f}, deff_sig_share={float(b['significant_share_p_lt_0_05']):.3f}, p1_latest_p={float(a['latest_pvalue']):.4f}, deff_latest_p={float(b['latest_pvalue']):.4f}"
                ]
    else:
        summary += ["", "[SIGNIFICANCE_BOOTSTRAP]", "unavailable"]

    if int(args.official_window) not in ts_map:
        raise SystemExit(f"official window {int(args.official_window)} missing")
    ts_off = ts_map[int(args.official_window)].copy()
    regime_df, reg_thr = _classify_regime(
        ts=ts_off,
        hysteresis_days=int(args.hysteresis_days),
        exp_stress=float(args.exp_stress),
        exp_transition=float(args.exp_transition),
        exp_stable=float(args.exp_stable),
        exp_dispersion=float(args.exp_dispersion),
        w_dp1=float(args.transition_weight_dp1),
        w_ddeff=float(args.transition_weight_ddeff),
        w_overlap=float(args.transition_weight_overlap),
        threshold_mode=str(args.regime_threshold_mode),
        walkforward_min_history=int(args.walkforward_min_history),
    )
    regime_df.to_csv(outdir / f"regime_series_T{int(args.official_window)}.csv", index=False)
    (outdir / f"regime_thresholds_T{int(args.official_window)}.json").write_text(json.dumps(reg_thr, indent=2), encoding="utf-8")

    calibration_best: dict[str, Any] = {}
    calib_df = pd.DataFrame()
    if int(args.calibrate_exposure_grid) == 1 and (not regime_df.empty):
        candidates = _default_exposure_candidates(
            exp_stress=float(args.exp_stress),
            exp_transition=float(args.exp_transition),
            exp_stable=float(args.exp_stable),
            exp_dispersion=float(args.exp_dispersion),
        )
        policy_cands = (((policy.get("calibration", {}) or {}).get("candidate_exposures")) if policy else None)
        if isinstance(policy_cands, list) and len(policy_cands) > 0:
            c2 = []
            for c in policy_cands:
                if not isinstance(c, dict):
                    continue
                if not all(k in c for k in ("stress", "transition", "stable", "dispersion")):
                    continue
                c2.append(
                    {
                        "stress": float(np.clip(float(c["stress"]), 0.0, 1.0)),
                        "transition": float(np.clip(float(c["transition"]), 0.0, 1.0)),
                        "stable": float(np.clip(float(c["stable"]), 0.0, 1.0)),
                        "dispersion": float(np.clip(float(c["dispersion"]), 0.0, 1.0)),
                    }
                )
            if c2:
                candidates = c2
        calib_df, calibration_best = _exposure_grid_search(
            regime_df=regime_df,
            returns_wide=R,
            cost_bps=float(args.cost_bps),
            max_daily_turnover=float(args.max_daily_turnover),
            start_exposure=float(args.exp_stable),
            candidates=candidates,
            objective=str(args.calibration_objective),
        )
        calib_df.to_csv(outdir / f"exposure_grid_T{int(args.official_window)}.csv", index=False)
        (outdir / f"exposure_recommendation_T{int(args.official_window)}.json").write_text(
            json.dumps(calibration_best, indent=2),
            encoding="utf-8",
        )
        if int(args.apply_grid_best) == 1 and calibration_best:
            exp = calibration_best.get("exposure", {})
            regime_df["exposure"] = regime_df["regime"].map(exp).fillna(float(args.exp_stable)).astype(float).clip(0.0, 1.0)
            regime_df.to_csv(outdir / f"regime_series_T{int(args.official_window)}.csv", index=False)

    bt_df, bt_sum = _backtest(
        regime_df=regime_df,
        returns_wide=R,
        cost_bps=float(args.cost_bps),
        max_daily_turnover=float(args.max_daily_turnover),
        start_exposure=float(args.exp_stable),
    )
    bt_df.to_csv(outdir / f"backtest_regime_T{int(args.official_window)}.csv", index=False)
    (outdir / f"backtest_summary_T{int(args.official_window)}.json").write_text(json.dumps(bt_sum, indent=2), encoding="utf-8")

    era_df = _build_era_evaluation(ts_off=ts_off, regime_df=regime_df, bt_df=bt_df)
    era_df.to_csv(outdir / f"era_evaluation_T{int(args.official_window)}.csv", index=False)
    (outdir / f"era_evaluation_T{int(args.official_window)}.json").write_text(
        json.dumps(era_df.to_dict(orient="records"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    case_regimes = [x.lower() for x in _parse_csv_list(args.case_regimes)]
    case_df = _build_case_studies(
        regime_df=regime_df,
        ts_off=ts_off,
        bt_df=bt_df,
        target_regimes=case_regimes,
        horizon_days=int(max(1, args.case_horizon_days)),
    )
    case_df.to_csv(outdir / f"case_studies_T{int(args.official_window)}.csv", index=False)
    (outdir / f"case_studies_T{int(args.official_window)}.json").write_text(
        json.dumps(case_df.to_dict(orient="records"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_case_studies_demo(outdir=outdir, cases_df=case_df)

    playbook_df = _build_action_playbook(
        regime_df=regime_df,
        ts_off=ts_off,
        bt_df=bt_df,
        horizon_days=int(max(1, args.case_horizon_days)),
    )
    playbook_df.to_csv(outdir / f"action_playbook_T{int(args.official_window)}.csv", index=False)
    (outdir / f"action_playbook_T{int(args.official_window)}.json").write_text(
        json.dumps(playbook_df.to_dict(orient="records"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    asset_diag_df, sector_diag_df, asset_sector_payload = _build_asset_sector_diagnostics(
        returns_wide=R,
        universe_core=universe_core,
    )
    asset_diag_df.to_csv(outdir / "asset_regime_diagnostics.csv", index=False)
    sector_diag_df.to_csv(outdir / "sector_regime_diagnostics.csv", index=False)
    if not asset_diag_df.empty:
        asset_diag_df.sort_values(["sensitivity_score", "risk_score"], ascending=[False, False]).head(50).to_csv(
            outdir / "asset_top_sensitive.csv",
            index=False,
        )
        asset_diag_df.sort_values(["stability_score", "confidence_score"], ascending=[False, False]).head(50).to_csv(
            outdir / "asset_top_stable.csv",
            index=False,
        )
    (outdir / "asset_sector_summary.json").write_text(
        json.dumps(asset_sector_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    alerts, alerts_payload, alerts_txt = _cluster_alerts(ts=ts_off, lookback=int(args.alert_lookback))
    alerts.to_csv(outdir / f"cluster_alerts_T{int(args.official_window)}.csv", index=False)
    op_alerts, op_payload, op_txt = _build_operational_alerts(
        ts_off=ts_off,
        regime_df=regime_df,
        robust_metrics=robust_metrics,
    )
    op_alerts.to_csv(outdir / f"operational_alerts_T{int(args.official_window)}.csv", index=False)
    (outdir / f"operational_alerts_T{int(args.official_window)}.json").write_text(
        json.dumps(op_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    level_df, level_payload = _build_alert_levels(
        regime_df=regime_df,
        ts_off=ts_off,
        risk_q_yellow=float(args.risk_q_yellow),
        risk_q_red=float(args.risk_q_red),
        min_conf_yellow=float(args.min_conf_yellow),
        min_conf_red=float(args.min_conf_red),
        persist_window=int(args.alert_persist_window),
        persist_count=int(args.alert_persist_count),
    )
    level_df.to_csv(outdir / f"alert_levels_T{int(args.official_window)}.csv", index=False)
    (outdir / f"alert_levels_T{int(args.official_window)}.json").write_text(
        json.dumps(level_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    structural_meta: dict[str, Any] = {}
    if bool(int(args.enable_structural_v1)):
        structural_meta = _build_structural_parallel_diagnostics(
            ts_off=ts_off,
            outdir=outdir,
            csd_window=int(args.structural_csd_window),
            train_end=str(args.structural_train_end),
        )
        if str(structural_meta.get("status", "")).lower() == "ok":
            summary += [
                "",
                "[STRUCTURAL_V1]",
                f"rows={int(structural_meta.get('rows', 0))}; csd_window={int(args.structural_csd_window)}; structural_ok={bool(structural_meta.get('structural_ok', False))}",
            ]
        else:
            summary += [
                "",
                "[STRUCTURAL_V1]",
                f"status={str(structural_meta.get('status', 'unknown'))}; reason={str(structural_meta.get('reason', 'n/a'))}",
            ]
    hierarchical_meta: dict[str, Any] = {}
    if bool(int(args.enable_hierarchical)):
        metadata_path = Path(str(args.asset_metadata_path))
        if not metadata_path.is_absolute():
            metadata_path = ROOT / str(args.asset_metadata_path)
        hierarchical_meta = _run_hierarchical_diagnostics(
            outdir=outdir,
            returns_core=R,
            universe_core=universe_core,
            official_window=int(args.official_window),
            min_assets=int(args.min_assets),
            coverage_window=float(args.coverage_window),
            overlap_step=int(args.overlap_step),
            noise_step=int(args.noise_step),
            bootstrap_block=int(args.bootstrap_block),
            seed=int(args.seed),
            compute_forman=bool(int(args.enable_structural_v1)),
            forman_topk=int(args.structural_topk),
            csd_window=int(args.structural_csd_window),
            train_end=str(args.structural_train_end),
            metadata_path=metadata_path,
            n_global=int(args.n_global),
            n_sector=int(args.n_sector),
            min_coverage_global=float(args.min_coverage_global),
            min_coverage_sector=float(args.min_coverage_sector),
            enable_internal=bool(int(args.enable_internal_sectors)),
            save_v1=bool(int(args.save_v1)),
        )
        if str(hierarchical_meta.get("status", "")).lower() == "ok":
            summary += [
                "",
                "[HIERARCHICAL]",
                f"global_universe={int(hierarchical_meta.get('global_universe_size', 0))}; sector_universes={int(hierarchical_meta.get('sector_universes_count', 0))}; save_v1={bool(hierarchical_meta.get('save_v1', False))}",
            ]
        else:
            summary += [
                "",
                "[HIERARCHICAL]",
                f"status={str(hierarchical_meta.get('status', 'fail'))}; reason={str(hierarchical_meta.get('reason', 'n/a'))}",
            ]
    summary += ["", f"[OFFICIAL_T{int(args.official_window)}]"]
    if bt_df.empty:
        summary += ["backtest: unavailable"]
    else:
        s, b = bt_sum["strategy"], bt_sum["benchmark"]
        summary += [
            f"backtest: strategy_ann={float(s.get('ann_return', np.nan)):.6f}, strategy_sharpe={float(s.get('sharpe', np.nan)):.4f}, strategy_mdd={float(s.get('max_drawdown', np.nan)):.4f}; bench_ann={float(b.get('ann_return', np.nan)):.6f}, bench_sharpe={float(b.get('sharpe', np.nan)):.4f}, bench_mdd={float(b.get('max_drawdown', np.nan)):.4f}; avg_exposure={float(bt_sum.get('avg_exposure', np.nan)):.4f}, avg_turnover={float(bt_sum.get('avg_turnover', np.nan)):.6f}"
        ]
    if not regime_df.empty:
        rc = regime_df["regime"].value_counts(normalize=True).to_dict()
        summary += [
            f"regime_mix: stress={rc.get('stress',0.0):.3f}, transition={rc.get('transition',0.0):.3f}, stable={rc.get('stable',0.0):.3f}, dispersion={rc.get('dispersion',0.0):.3f}"
        ]
    else:
        summary += ["regime_mix: unavailable"]
    summary += [f"alerts: {alerts_txt}"]
    summary += [f"operational_alerts: {op_txt}"]
    if level_payload:
        summary += [
            "alert_levels: "
            f"latest={level_payload.get('latest_level','')}; "
            f"counts={json.dumps(level_payload.get('counts', {}), ensure_ascii=False)}"
        ]
    if calibration_best:
        exp = calibration_best.get("exposure", {})
        m = calibration_best.get("metrics", {})
        summary += [
            "",
            "[EXPOSURE_CALIBRATION]",
            f"objective={calibration_best.get('objective')}; n_candidates={calibration_best.get('n_candidates')}; apply_grid_best={bool(int(args.apply_grid_best))}",
            f"recommended: stress={float(exp.get('stress', np.nan)):.2f}, transition={float(exp.get('transition', np.nan)):.2f}, stable={float(exp.get('stable', np.nan)):.2f}, dispersion={float(exp.get('dispersion', np.nan)):.2f}",
            f"recommended_metrics: score={float(m.get('score', np.nan)):.6f}, ann_return={float(m.get('ann_return', np.nan)):.6f}, sharpe={float(m.get('sharpe', np.nan)):.4f}, max_drawdown={float(m.get('max_drawdown', np.nan)):.4f}",
        ]
    if not case_df.empty:
        summary += ["", "[CASE_STUDIES]"]
        for _, cr in case_df.iterrows():
            summary += [
                f"{str(cr['case_regime'])}: date={str(cr['date'])}, exposure={float(cr['exposure']):.2f}, "
                f"alpha_{int(cr['horizon_days'])}d={float(cr['alpha_cum']):.4f}, "
                f"dd_improvement={float(cr['dd_improvement']):.4f}, honesty={str(cr['honest_verdict'])}"
            ]
    else:
        summary += ["", "[CASE_STUDIES]", "none"]
    if not era_df.empty:
        summary += ["", "[ERA_EVALUATION]"]
        for _, er in era_df.iterrows():
            summary += [
                f"{str(er['era'])}: n_days={int(er['n_days'])}, p1_mean={float(er['p1_mean']):.4f}, "
                f"deff_mean={float(er['deff_mean']):.2f}, alpha_ann={float(er['alpha_ann_return']):.4f}, "
                f"dd_improvement={float(er['dd_improvement']):.4f}"
            ]
    else:
        summary += ["", "[ERA_EVALUATION]", "none"]
    if not playbook_df.empty:
        x = playbook_df.iloc[-1]
        summary += [
            "",
            "[ACTION_PLAYBOOK]",
            f"latest_date={x['date']}; regime={x['regime']}; action={x['action_code']}; "
            f"signal={x['signal_tier']}({float(x['signal_reliability']):.3f}); tradeoff={x['tradeoff_label']}",
        ]
    else:
        summary += ["", "[ACTION_PLAYBOOK]", "none"]
    if not asset_diag_df.empty and not sector_diag_df.empty:
        top_a = asset_diag_df.sort_values(["risk_score", "switches_90d"], ascending=[False, False]).iloc[0]
        top_s = sector_diag_df.iloc[0]
        summary += [
            "",
            "[ASSET_SECTOR_DIAGNOSTICS]",
            f"assets={int(asset_diag_df['ticker'].nunique())}; sectors={int(sector_diag_df.shape[0])}; top_risk_asset={top_a['ticker']}; top_risk_sector={top_s['sector']}",
            f"top_sector_metrics: risk_mean={float(top_s['risk_mean']):.4f}, confidence_mean={float(top_s['confidence_mean']):.4f}, pct_instavel={float(top_s['pct_instavel']):.4f}",
        ]
    else:
        summary += ["", "[ASSET_SECTOR_DIAGNOSTICS]", "unavailable"]

    qa = _qa(
        ts_map=ts_map,
        core_counts=core_counts,
        n_core=int(len(core)),
        min_assets=int(args.min_assets),
        official_window=int(args.official_window),
        max_insufficient_ratio=float(args.max_insufficient_ratio),
        min_n_used_ratio=float(args.min_n_used_ratio),
        q_min=float(args.q_min),
    )
    (outdir / "qa_checks.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    summary += ["", "[QA]", f"qa_ok={qa['ok']}; checks={len(qa['checks'])}"]
    if not qa["ok"]:
        failed = [str(x) for x in qa.get("failed_checks", [])]
        summary += [f"qa_failed={','.join(failed)}"]
        for c in qa["checks"]:
            if bool(c.get("ok", False)):
                continue
            detail_keys = ["value", "max_allowed", "min_required", "expected", "n_rows", "n_sufficient", "n_last20", "required"]
            details = [f"{k}={c[k]}" for k in detail_keys if k in c]
            if details:
                summary += [f"qa_detail_{c['check']}={','.join(details)}"]

    baseline: dict[str, Any] = {}
    if int(args.freeze_baseline) == 1:
        baseline = _freeze_baseline(
            baseline_dir=Path(args.baseline_dir),
            outdir=outdir,
            universe_core=universe_core,
            ts_official=ts_off,
            official_window=int(args.official_window),
            start=str(args.start),
            end=str(args.end),
            cov_core=float(args.coverage_core),
            cov_window=float(args.coverage_window),
        )
        summary += [
            "",
            "[BASELINE]",
            f"baseline_dir={baseline['baseline_dir']}; same_universe_vs_previous={baseline['same_universe_vs_previous']}; delta_p1_vs_previous={baseline['delta_p1_vs_previous']}; delta_deff_vs_previous={baseline['delta_deff_vs_previous']}",
        ]
    else:
        baseline = {"same_universe_vs_previous": True, "delta_p1_vs_previous": np.nan, "delta_deff_vs_previous": np.nan}
    summary += [
        "",
        "[POLICY_LOCK]",
        f"matches_previous_release={bool(policy_lock.get('matches_previous_release', True))}; previous_hash_present={bool(str(policy_lock.get('previous_policy_hash', '')).strip())}",
    ]

    gate = _build_deployment_gate(
        qa=qa,
        robust_metrics=robust_metrics,
        baseline=baseline,
        alerts_payload=alerts_payload,
        min_joint_majority_60d=float(args.min_joint_majority_60d),
        require_latest_majority=bool(int(args.require_latest_majority)),
        max_abs_delta_p1=float(args.max_abs_delta_p1),
        max_abs_delta_deff=float(args.max_abs_delta_deff),
        max_active_cluster_alerts=int(args.max_active_cluster_alerts),
        policy_lock=policy_lock,
        require_policy_lock_match=bool(int(args.require_policy_lock_match)),
    )
    (outdir / "deployment_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    summary += ["", "[DEPLOYMENT_GATE]", f"blocked={gate['blocked']}; reasons={','.join(gate['reasons']) if gate['reasons'] else 'none'}"]

    view_model = _build_ui_view_model(
        run_id=outdir.name,
        ts_off=ts_off,
        regime_df=regime_df,
        robust_metrics=robust_metrics,
        gate=gate,
        op_payload=op_payload,
        alert_level_payload=level_payload,
        asset_sector_payload=asset_sector_payload,
        case_df=case_df,
        era_df=era_df,
        playbook_df=playbook_df,
    )
    (outdir / f"ui_view_model_T{int(args.official_window)}.json").write_text(
        json.dumps(view_model, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    (outdir / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")

    structural_files = structural_meta.get("files", {}) if isinstance(structural_meta, dict) else {}
    hierarchical_enabled = bool(int(args.enable_hierarchical))
    summary_json = {
        "status": "ok" if (not gate["blocked"]) else "fail",
        "run_id": outdir.name,
        "run_dir": str(outdir),
        "period_start": str(args.start),
        "period_end": str(args.end),
        "n_core": int(len(core)),
        "official_window": int(args.official_window),
        "policy_path": str(policy_path),
        "policy_loaded": bool(policy),
        "deployment_gate": gate,
        "policy_lock": policy_lock,
        "checks": {"qa_ok": bool(qa["ok"]), "scipy_clustering": bool(SCIPY_OK)},
        "scores": {
            "joint_majority_60d": float(robust_metrics.get("joint_majority_60d", np.nan)),
            "latest_joint_majority_5": float(robust_metrics.get("latest_joint_majority_5", np.nan)),
            "active_cluster_alerts": int(alerts_payload.get("n_active", 0)),
        },
        "calibration": calibration_best if calibration_best else {},
        "operational_alerts": op_payload,
        "alert_levels": level_payload if level_payload else {},
        "significance": {
            "count": int(signif_df.shape[0]) if not signif_df.empty else 0,
            "file_csv": str(outdir / "significance_summary_by_window.csv"),
        },
        "structural_v1": {
            "enabled": bool(int(args.enable_structural_v1)),
            "status": str(structural_meta.get("status", "disabled")) if bool(int(args.enable_structural_v1)) else "disabled",
            "file_csv": str(structural_files.get("diagnostics_daily_csv", outdir / "diagnostics_structural_daily.csv")),
            "file_score_csv": str(structural_files.get("diagnostics_score_daily_csv", outdir / "diagnostics_structural_score_daily.csv")),
            "file_meta_json": str(structural_files.get("diagnostics_meta_json", outdir / "diagnostics_structural_meta.json")),
        },
        "hierarchical": {
            "enabled": hierarchical_enabled,
            "status": str(hierarchical_meta.get("status", "disabled")) if hierarchical_enabled else "disabled",
            "global_universe_size": int(hierarchical_meta.get("global_universe_size", 0)) if hierarchical_enabled else 0,
            "sector_universes_count": int(hierarchical_meta.get("sector_universes_count", 0)) if hierarchical_enabled else 0,
            "gics_universes_count": int(hierarchical_meta.get("gics_universes_count", 0)) if hierarchical_enabled else 0,
            "internal_universes_count": int(hierarchical_meta.get("internal_universes_count", 0)) if hierarchical_enabled else 0,
            "save_v1": bool(hierarchical_meta.get("save_v1", False)) if hierarchical_enabled else False,
            "files": {
                "asset_metadata_used_csv": str(hierarchical_meta.get("asset_metadata_used_csv", "")),
                "cross_gics_csv": str(hierarchical_meta.get("cross_gics_csv", "")),
                "cross_internal_csv": str(hierarchical_meta.get("cross_internal_csv", "")),
                "hierarchical_state_daily_jsonl": str(hierarchical_meta.get("hierarchical_state_daily_jsonl", "")),
                "hierarchical_state_latest_json": str(hierarchical_meta.get("hierarchical_state_latest_json", "")),
                "latest_hierarchical_state_json": str(hierarchical_meta.get("latest_hierarchical_state_json", "")),
                "diagnostics_global_score_csv": str(hierarchical_meta.get("diagnostics_global_score_csv", "")),
            },
        },
        "era_evaluation": {
            "count": int(era_df.shape[0]) if not era_df.empty else 0,
            "file_csv": str(outdir / f"era_evaluation_T{int(args.official_window)}.csv"),
            "file_json": str(outdir / f"era_evaluation_T{int(args.official_window)}.json"),
        },
        "action_playbook": {
            "count": int(playbook_df.shape[0]) if not playbook_df.empty else 0,
            "horizon_days": int(max(1, args.case_horizon_days)),
            "file_csv": str(outdir / f"action_playbook_T{int(args.official_window)}.csv"),
            "file_json": str(outdir / f"action_playbook_T{int(args.official_window)}.json"),
        },
        "asset_sector_diagnostics": {
            "assets_count": int(asset_diag_df["ticker"].nunique()) if not asset_diag_df.empty else 0,
            "sectors_count": int(sector_diag_df.shape[0]) if not sector_diag_df.empty else 0,
            "files": {
                "assets_csv": str(outdir / "asset_regime_diagnostics.csv"),
                "sectors_csv": str(outdir / "sector_regime_diagnostics.csv"),
                "top_sensitive_csv": str(outdir / "asset_top_sensitive.csv"),
                "top_stable_csv": str(outdir / "asset_top_stable.csv"),
                "summary_json": str(outdir / "asset_sector_summary.json"),
            },
        },
        "case_studies": {
            "count": int(case_df.shape[0]) if not case_df.empty else 0,
            "regimes_requested": case_regimes,
            "horizon_days": int(max(1, args.case_horizon_days)),
            "files": {
                "csv": str(outdir / f"case_studies_T{int(args.official_window)}.csv"),
                "json": str(outdir / f"case_studies_T{int(args.official_window)}.json"),
                "demo_txt": str(outdir / "case_studies_demo.txt"),
            },
        },
        "ui_view_model": {
            "schema_version": str(view_model.get("schema_version", "")),
            "file_json": str(outdir / f"ui_view_model_T{int(args.official_window)}.json"),
        },
    }
    (outdir / "summary.json").write_text(json.dumps(summary_json, indent=2, ensure_ascii=False), encoding="utf-8")

    _write_compact_report(
        outdir=outdir,
        run_id=outdir.name,
        period_start=str(args.start),
        period_end=str(args.end),
        n_core=int(len(core)),
        official_window=int(args.official_window),
        ts_official=ts_off,
        regime_df=regime_df,
        robust_metrics=robust_metrics,
        baseline=baseline,
        bt_summary=bt_sum,
        gate=gate,
    )
    _write_daily_brief(
        outdir=outdir,
        regime_df=regime_df,
        ts_off=ts_off,
        robust_metrics=robust_metrics,
        gate=gate,
        baseline=baseline,
        calibration_best=calibration_best,
    )
    _write_commercial_narrative(
        outdir=outdir,
        regime_df=regime_df,
        ts_off=ts_off,
        bt_summary=bt_sum,
        calibration_best=calibration_best,
        gate=gate,
    )

    meta = {
        "run_dir": str(outdir),
        "panel_path": str(panel_path),
        "period_start": str(args.start),
        "period_end": str(args.end),
        "N_core": int(len(core)),
        "max_core_assets": int(max(0, int(args.max_core_assets))),
        "coverage_core": float(args.coverage_core),
        "windows": [int(x) for x in windows],
        "official_window": int(args.official_window),
        "policy_path": str(policy_path),
        "policy_loaded": bool(policy),
        "baseline_dir": str(args.baseline_dir),
        "baseline_namespace": str(baseline_namespace),
        "freeze_baseline": int(args.freeze_baseline),
        "strict_checks": int(args.strict_checks),
        "qa_ok": bool(qa["ok"]),
        "deployment_blocked": bool(gate["blocked"]),
        "policy_lock_match": bool(policy_lock.get("matches_previous_release", True)),
        "robustness_rows": int(robust_df.shape[0]) if not robust_df.empty else 0,
        "significance_rows": int(signif_df.shape[0]) if not signif_df.empty else 0,
        "case_studies_count": int(case_df.shape[0]) if not case_df.empty else 0,
        "era_rows": int(era_df.shape[0]) if not era_df.empty else 0,
        "playbook_rows": int(playbook_df.shape[0]) if not playbook_df.empty else 0,
        "asset_diagnostics_rows": int(asset_diag_df.shape[0]) if not asset_diag_df.empty else 0,
        "sector_diagnostics_rows": int(sector_diag_df.shape[0]) if not sector_diag_df.empty else 0,
        "operational_alerts_latest": int(len(op_payload.get("latest_events", []))) if isinstance(op_payload, dict) else 0,
        "structural_v1_enabled": bool(int(args.enable_structural_v1)),
        "structural_v1_status": str(structural_meta.get("status", "disabled")) if bool(int(args.enable_structural_v1)) else "disabled",
        "structural_v1_rows": int(structural_meta.get("rows", 0)) if bool(int(args.enable_structural_v1)) else 0,
        "hierarchical_enabled": bool(int(args.enable_hierarchical)),
        "hierarchical_status": str(hierarchical_meta.get("status", "disabled")) if bool(int(args.enable_hierarchical)) else "disabled",
        "hierarchical_global_universe_size": int(hierarchical_meta.get("global_universe_size", 0)) if bool(int(args.enable_hierarchical)) else 0,
        "hierarchical_sector_universes_count": int(hierarchical_meta.get("sector_universes_count", 0)) if bool(int(args.enable_hierarchical)) else 0,
        "hierarchical_state_latest_json": str(hierarchical_meta.get("hierarchical_state_latest_json", "")) if bool(int(args.enable_hierarchical)) else "",
        "hierarchical_cross_gics_csv": str(hierarchical_meta.get("cross_gics_csv", "")) if bool(int(args.enable_hierarchical)) else "",
        "hierarchical_cross_internal_csv": str(hierarchical_meta.get("cross_internal_csv", "")) if bool(int(args.enable_hierarchical)) else "",
    }

    release_decision = {"updated": False}
    if int(args.update_release_pointer) == 1:
        release_decision = _update_release_pointer(out_base=out_base, run_meta=meta, gate=gate)
        (outdir / "release_pointer_update.json").write_text(json.dumps(release_decision, indent=2), encoding="utf-8")
    meta["release_pointer_updated"] = bool(release_decision.get("updated", False))

    (outdir / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))

    if int(args.strict_checks) == 1 and bool(gate["blocked"]):
        raise SystemExit(f"Deployment gate blocked: {', '.join(gate['reasons']) if gate['reasons'] else 'unspecified'}")


if __name__ == "__main__":
    main()
