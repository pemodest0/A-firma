# Mercado Bitcoin Live

## Credenciais

Copie `config/live_mercado_bitcoin.env.example` para o seu ambiente e exporte:

- `MB_API_KEY`
- `MB_API_PASSWORD`
- `MB_ACCOUNT_ID` opcional

## Fluxo

1. `python3 scripts/ops/run_mercado_bitcoin_account_sync.py`
2. `python3 scripts/ops/run_live_execution_plan.py`
3. revisar `results/ops/execution_live/latest_mercado_bitcoin_order_preview.json`
4. envio manual assistido:
   `python3 scripts/ops/run_mercado_bitcoin_submit_orders.py --submit --confirm MB_SUBMIT`
5. reconciliar:
   `python3 scripts/ops/run_live_execution_reconciliation.py`

## Observações

- o planner tenta sincronizar a conta automaticamente antes de planejar
- sem credenciais, o fluxo cai para `portfolio_state.json`
- `submit_enabled` continua `false` por padrão em `config/live_execution_profile.json`
- libere submit só depois de validar saldo, resposta da API e payload real
