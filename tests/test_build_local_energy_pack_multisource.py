from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_build_local_energy_pack_multisource(tmp_path):
    raw_root = tmp_path / "raw"
    carga_dir = raw_root / "ons_carga_diaria"
    ear_dir = raw_root / "ons_ear_subsistema_di"
    cmo_dir = raw_root / "ons_cmo_semanal"
    out_results = tmp_path / "results"

    _write_csv(
        carga_dir / "CARGA_ENERGIA_2025.csv",
        [
            {"id_subsistema": "N", "nom_subsistema": "NORTE", "din_instante": "2025-01-01", "val_cargaenergiamwmed": 1000},
            {"id_subsistema": "N", "nom_subsistema": "NORTE", "din_instante": "2025-01-02", "val_cargaenergiamwmed": 1020},
            {"id_subsistema": "SE", "nom_subsistema": "SUDESTE", "din_instante": "2025-01-01", "val_cargaenergiamwmed": 2000},
            {"id_subsistema": "SE", "nom_subsistema": "SUDESTE", "din_instante": "2025-01-02", "val_cargaenergiamwmed": 1980},
        ],
    )
    _write_csv(
        ear_dir / "EAR_DIARIO_SUBSISTEMA_2025.csv",
        [
            {"id_subsistema": "N", "nom_subsistema": "NORTE", "din_instante": "2025-01-01", "val_ear_verificada_percentual": 64.2},
            {"id_subsistema": "N", "nom_subsistema": "NORTE", "din_instante": "2025-01-02", "val_ear_verificada_percentual": 65.1},
            {"id_subsistema": "SE", "nom_subsistema": "SUDESTE", "din_instante": "2025-01-01", "val_ear_verificada_percentual": 59.1},
            {"id_subsistema": "SE", "nom_subsistema": "SUDESTE", "din_instante": "2025-01-02", "val_ear_verificada_percentual": 59.4},
        ],
    )
    _write_csv(
        cmo_dir / "CMO_SEMANAL_2025.csv",
        [
            {"id_subsistema": "N", "nom_subsistema": "NORTE", "din_instante": "2025-01-01", "val_cmo": 212.0, "num_patamar": 1},
            {"id_subsistema": "N", "nom_subsistema": "NORTE", "din_instante": "2025-01-01", "val_cmo": 230.0, "num_patamar": 2},
            {"id_subsistema": "N", "nom_subsistema": "NORTE", "din_instante": "2025-01-08", "val_cmo": 218.0, "num_patamar": 1},
            {"id_subsistema": "SE", "nom_subsistema": "SUDESTE", "din_instante": "2025-01-01", "val_cmo": 205.0, "num_patamar": 1},
            {"id_subsistema": "SE", "nom_subsistema": "SUDESTE", "din_instante": "2025-01-08", "val_cmo": 209.0, "num_patamar": 1},
        ],
    )

    cmd = [
        sys.executable,
        "scripts/data/build_local_energy_pack.py",
        "--raw-dir",
        str(carga_dir),
        "--ear-dir",
        str(ear_dir),
        "--cmo-dir",
        str(cmo_dir),
        "--results-dir",
        str(out_results),
        "--min-rows",
        "1",
        "--write-canonical-raw",
        "0",
        "--start",
        "2025-01-01",
    ]
    subprocess.run(cmd, cwd=str(ROOT), check=True)  # noqa: S603

    packs = sorted([p for p in out_results.iterdir() if p.is_dir() and p.name.startswith("local_pack_")], reverse=True)
    assert packs, "expected local_pack output"
    latest = packs[0]
    panel = pd.read_csv(latest / "panel_long_sector.csv")
    universe = pd.read_csv(latest / "universe_fixed.csv")

    tickers = set(panel["ticker"].astype(str).tolist())
    assert "ONS_BR" in tickers
    assert "ONS_N" in tickers
    assert "EAR_N" in tickers
    assert "CMO_N" in tickers

    sectors = set(universe["sector"].astype(str).tolist())
    assert "energy_load" in sectors
    assert "energy_storage" in sectors
    assert "energy_price" in sectors
