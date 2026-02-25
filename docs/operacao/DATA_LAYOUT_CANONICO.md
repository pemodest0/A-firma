# Data Layout Canonico (Assyntrax / Eigen Engine)

Data: 2026-02-25  
Status: ativo

## Objetivo

Padronizar dados em quatro camadas para reduzir erro de logica, facilitar auditoria e evitar duplicacao confusa.

## Camadas oficiais

- `data/download/`
  Dados baixados da fonte externa, sem transformacao de negocio.
- `data/clean/`
  Dados normalizados (tipos, timezone, schema minimo, deduplicacao).
- `data/processed/`
  Dados prontos para calculo de metricas/modelos (features e agregacoes).
- `data/validated/`
  Dados aprovados por checks de qualidade para uso operacional.

## Regras

1. Nada entra direto em `processed` sem passar por `clean`.
2. Nada entra em `validated` sem checagem explicita de schema/qualidade.
3. Scripts devem declarar explicitamente qual camada leem e qual camada escrevem.
4. Artefatos de execucao continuam em `results/` (nao misturar com dados-base).

## Mapeamento de legado atual (fase de transicao)

- `data/raw/**` -> migrar gradualmente para `data/download/**`
- `data/realestate/core/**` -> manter durante transicao e convergir para `data/clean/realestate/**`
- `data/realestate/normalized/**` -> convergir para `data/processed/realestate/**`

Compatibilidade:

- Nao mover paths antigos em bloco sem validacao.
- Migracao deve ser por script, com dry-run e log de diff.

## Script de bootstrap

Criar estrutura base:

```bash
bash scripts/ops/bootstrap_data_layout.sh
```

Migrar dados legados para as camadas canonicas (sem apagar origem):

```bash
python3 scripts/ops/migrate_data_layout.py --domain realestate --domain energy --apply
```

Obs:

- por padrao roda em dry-run.
- gera manifesto em `results/ops/data_layout/<timestamp>/`.

## Check rapido recomendado

```bash
find data -maxdepth 2 -type d | sort
```
