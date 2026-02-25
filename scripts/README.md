# Scripts Registry (Assyntrax / Eigen Engine)

Status: ativo  
Atualizado em: 2026-02-25

## Objetivo

Deixar claro o que e fluxo oficial e o que e pesquisa para evitar dead-ends e scripts sem dono.

## Pastas e papel

- `scripts/ops/`
  Execucao operacional diaria, healthcheck, publicacao e governanca.
- `scripts/structural/`
  Ground truth, diagnosticos e validacoes estruturais do Eigen Engine.
- `scripts/data/`
  Ingestao e preparo de dados (one-shot, packs e normalizacao).
- `scripts/lab/`
  Nucleo de laboratorio; inclui `run_corr_macro_offline.py` (core atual).
- `scripts/realestate/`
  Ingestao/processamento do dominio imobiliario.
- `scripts/bench/`
  Benchmarks e comparacoes de pesquisa (review periodico).
- `scripts/sim/`
  Simulacoes e sinteticos (research).
- `scripts/engine/`
  Entrypoints de experimentacao de engine legado/transicao (research).
- `scripts/research/`
  Scripts exploratorios avulsos que nao entram no fluxo oficial.
- `scripts/maintenance/`
  Utilitarios de manutencao local (sempre com modo dry-run padrao).

## Regra de governanca

1. Nao adicionar script novo sem declarar pasta-alvo e objetivo.
2. Script que toca pipeline oficial deve ter teste ou check de sanidade.
3. Script de manutencao deve ser seguro por padrao (sem delecao imediata).
4. Script sem uso em 14 dias entra em `review` para remover ou consolidar.

## Inventario automatico

Gerar inventario atual:

```bash
python3 scripts/ops/build_scripts_inventory.py
```

Saidas:

- `results/ops/scripts_inventory/<timestamp>/scripts_inventory.csv`
- `results/ops/scripts_inventory/<timestamp>/scripts_inventory.md`
- `results/ops/scripts_inventory/latest_inventory.json`
