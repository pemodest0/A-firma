# Energia: ingestao canonica (one-shot)

Objetivo: baixar dados oficiais de energia uma unica vez (quando necessario), normalizar no formato canonico do motor e validar adequacy.

## Comando unico

```bash
bash scripts/ops/sync_energy_data.sh
```

Saidas principais:

- `results/energy_download/energy_sync_<timestamp>/sync_manifest.json`
- `results/energy_download/local_pack_<timestamp>/panel_long_sector.csv`
- `results/energy_download/local_pack_<timestamp>/universe_fixed.csv`
- `data/raw/energy/ons/*.csv`
- `results/validation/data_adequacy_energy_sync_<timestamp>/summary.json`

## Modo sem download (usa somente local)

```bash
bash scripts/ops/sync_energy_data.sh --download 0
```

## Janela de anos especifica

```bash
bash scripts/ops/sync_energy_data.sh --from-year 2018 --to-year 2026
```

## Forcar redownload

```bash
bash scripts/ops/sync_energy_data.sh --force 1
```

Notas:

- O script e idempotente por padrao (`--force 0`): arquivos existentes nao sao baixados novamente.
- O build canonico usa `scripts/data/build_local_energy_pack.py`.
- O adequacy gate roda ao final para registrar status operacional.
