# Plano de Limpeza do Repo (2 pessoas)

Data: 2026-02-25  
Status: em execucao

## Objetivo

Deixar o repositorio simples, auditavel e sem dead-end:

- sem fallback oculto,
- sem logica duplicada,
- sem pasta "legado" crescendo sem criterio.

## Regra de decisao

Para cada pasta/arquivo:

1. `KEEP`: usada no fluxo oficial diario/site.
2. `REVIEW`: pode ficar, mas precisa dono + motivo.
3. `REMOVE/MOVE`: sem uso comprovado no fluxo oficial.

Se nao houver dono e uso em 14 dias, vira candidato a remocao.

## Escopo KEEP (nucleo oficial)

- `engine/`
- `scripts/ops/`
- `scripts/structural/` (ground truth, impacto, diagnostico)
- `scripts/lab/run_corr_macro_offline.py`
- `config/`
- `contracts/`
- `website-ui/`
- `docs/operacao/` e `docs/motor/` ativos
- `tests/`

## Escopo REVIEW imediato

- `scripts/sim/`
- `scripts/engine/`
- `scripts/bench/` (manter apenas o que alimenta decisao atual)
- docs antigas fora de `docs/operacao` e `docs/motor`

## Escopo REMOVE/MOVE (prioridade alta)

- scripts com import quebrado/legado sem dono.
- cleanup scripts destrutivos sem granularidade.
- referencias antigas de naming que conflitam com `Eigen Engine`.

## Trilha de dados canonica

Ver `docs/operacao/DATA_LAYOUT_CANONICO.md`.

Camadas:

- `data/download/`
- `data/clean/`
- `data/processed/`
- `data/validated/`

## Sprint curta (executavel em blocos de 30-60 min)

1. Congelar naming e fluxo CI (feito).
2. Catalogar scripts por uso real (ops/site/research):
   - `python3 scripts/ops/build_repo_cleanup_inventory.py`
3. Marcar e remover dead-end em lotes pequenos.
4. Garantir teste/healthcheck verde a cada lote.
5. Atualizar indice de docs ao fim de cada lote.

## Gate de merge para limpeza

- `bash ./scripts/ops/run_repo_healthcheck.sh` = 0 falhas.
- `python3 -m pytest -q` verde.
- sem introduzir novo path fora da trilha de dados canonica.
- sem criar fallback silencioso.
