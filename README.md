# Assyntrax | Eigen Engine

<p align="center">
  <img src="website-ui/public/assets/brand/assyntrax-mark.svg" alt="Logo Assyntrax" width="120" />
</p>

Repositório canônico da **Assyntrax**, com foco em **finanças**, **cripto** e no **Eigen Engine**, o motor de diagnóstico estrutural e alocação com controle de risco.

## O que é a Assyntrax
- **Assyntrax**: plataforma de diagnóstico quantitativo, pesquisa de alpha e operação assistida.
- **Eigen Engine**: motor estrutural que lê matriz de correlação, espectro, regime e pressão sistêmica.
- **Copiloto**: camada de explicação operacional, leitura de contexto e apoio de decisão.

## Foco atual do produto
- **Finanças globais**: leitura de regime, concentração, breadth e risco estrutural.
- **Cripto líquido**: sleeves agressivos, meta-switch, shadow e comparação contra benchmark correto.
- **Execução controlada**: guardrails, shadow, scorecard, stress test e governança de publicação.

## O que o motor faz
- Estima correlação e covariância robusta entre ativos.
- Extrai sinal estrutural via espectro, autovalores, autovetores e métricas de regime.
- Combina isso com ranking, sleeves, meta-switch e regras de proteção de drawdown.
- Publica somente quando os artefatos mínimos e os gates passam.

## O que existe no repositório
- `engine/structural/`: espectro, covariância robusta, limpeza RMT, estabilidade.
- `engine/portfolio/`: Monte Carlo por regime, HRP, challenger HMM, camadas auxiliares de risco.
- `execution/`: retornos, custos, premissas líquidas e avaliação operacional.
- `scripts/lab/`: pipeline estrutural principal.
- `scripts/ops/`: shadow, snapshots, registry, scorecards e automação.
- `scripts/bench/validation/`: baterias de validação, yearbooks, stress, comparativos e pesquisa.
- `website-ui/`: site e app do produto.
- `results/`: artefatos auditáveis de execução, pesquisa e publicação.

## Artefatos centrais
- `results/ops/finance_product_ready/latest_finance_product_ready.json`
- `results/ops/profit_research/latest_registry.json`
- `results/ops/profit_research/latest_patterns.json`
- `results/ops/site_data/latest_site_snapshot.json`
- `results/platform/latest_db_snapshot.json`

## Fluxo operacional
1. Atualizar a `main`
2. Rodar suites e pipelines necessários
3. Validar gates e artefatos
4. Atualizar snapshot/site/copiloto
5. Fazer commit pequeno e defensável
6. Publicar na `main`

## Links
- Repositório: `https://github.com/pemodest0/Assyntrax`
- App: `https://assyntrax.vercel.app`
- Licença: MIT

## Observação importante
O foco visível do produto hoje é **finanças e cripto**. Verticais legadas podem ainda existir no código e em artefatos históricos, mas não são a frente principal da plataforma.

## Licença
Este projeto está licenciado sob MIT. Consulte `LICENSE`.
