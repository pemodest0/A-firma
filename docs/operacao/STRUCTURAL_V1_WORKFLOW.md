# Structural v1 Workflow (Etapas 0-6)

Implementacao v1 dos pilares estruturais sem quebrar o pipeline atual.

## Governanca aplicada

- Nao altera pipeline padrao por default.
- Seed fixa em scripts demo.
- Contratos em `contracts/features.yaml` e `contracts/output_schema.json`.
- Cada script gera `RUN_MANIFEST.json`.
- Saida reprodutivel em `results/<timestamp>/`.

## Etapa 0 (infra + manifesto)

```bash
python3 scripts/structural/run_manifest_smoke.py
```

## Etapa 1 (RMT)

```bash
python3 scripts/structural/run_rmt_demo.py --seed 23
```

## Etapa 2 (spectral pack)

```bash
python3 scripts/structural/run_spectral_pack_demo.py --seed 23
```

Integracao nao invasiva no offline:

```bash
python3 scripts/lab/run_corr_macro_offline.py --apply-policy 0 --enable-structural-v1 1 --update-release-pointer 0 --strict-checks 0
```

Gera arquivo paralelo no run:

- `diagnostics_structural_daily.csv`

## Etapa 3 (CSD em phi)

```bash
python3 scripts/structural/run_csd_on_phi.py --window 60
```

## Etapa 4 (grafo + Forman)

```bash
python3 scripts/structural/run_forman_on_corr.py --method topk --k 10 --seed 23
```

## Etapa 5 (score de fusao)

```bash
python3 scripts/structural/run_structural_score_demo.py --official-window 120 --csd-window 60 --seed 23
```

No run do offline com flag:

- `diagnostics_structural_score_daily.csv`

## Etapa 6 (knowledge pack IA)

- `docs/KNOWLEDGE_ASSYNTRAX.md`
- `docs/AI_SYSTEM_PROMPT.md`

## Ground truth operacional

```bash
python3 scripts/structural/run_ground_truth_tests.py --horizons 5,10,20 --drawdown-threshold 0.05 --score-quantile 0.85 --train-end 2024-12-31
```

Saidas:

- `results/structural_ground_truth_<timestamp>/ground_truth_summary.json`
- `results/structural_ground_truth_<timestamp>/ground_truth_daily.csv`
- `results/ops/ai_knowledge/latest_ground_truth.json`
