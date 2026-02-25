# Guia do Eigen Engine

Guia curto da API/contrato oficial do motor para desenvolvimento e operacao.

## Nome oficial
- Empresa/plataforma: `Assyntrax`
- Motor: `Eigen Engine`

## Links
- App oficial (Vercel): `https://assyntrax.vercel.app`
- Licenca: MIT (`LICENSE`)

## Pacote oficial de codigo
Use `engine/` e `scripts/lab/run_corr_macro_offline.py` para novos desenvolvimentos.

## Contrato minimo por ativo
- `run_id`
- `asset`
- `domain`
- `status` (`validated|watch|inconclusive`)
- `regime`
- `confidence`
- `quality`
- `reason`
- `data_adequacy`

## Consumo esperado
- Scripts de operacao escrevem snapshots em `results/`.
- API do site (no diretorio tecnico `website-ui/app/api/**`) consome ultimo run valido.
- UI mostra estados operacionais sem prometer retorno.

## Mapa tecnico por arquivo/funcoes
- `docs/motor/EIGEN_ENGINE_FILE_FUNCTION_MAP.md`
- `docs/motor/MANUAL_MESTRE_ASSYNTRAX.md`

## Compatibilidade
Paths legados com `ASSYNTRAX` podem permanecer por compatibilidade historica.
