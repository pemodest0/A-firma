# Real-World Execution And Regulatory Audit

## Scope

This project now models a more realistic net path for Brazilian-resident research:

- cash in BRL earns a SELIC proxy when the strategy stays defensive
- Brazilian local equities use a monthly tax proxy with:
  - monthly sales exemption proxy at BRL 20,000
  - local loss-compensation proxy
  - small sell-side withholding proxy (`dedo-duro`) that offsets tax due
- crypto uses a monthly tax proxy with:
  - monthly sales exemption proxy at BRL 35,000
  - progressive capital-gains brackets above the exemption threshold
  - loss-compensation proxy
- foreign financial investments keep a conservative annual positive-gain proxy at 15%

## What Is Modeled

- transaction costs in basis points
- FX spread proxy
- cash carry via SELIC proxy
- tax timing proxies
- exemption thresholds via estimated monthly sales notional
- compensable withholding proxy for local equities

## What Is Still A Proxy

- no lot-by-lot tax ledger
- no broker note reconciliation
- no exact B3 fee schedule by market, product and venue
- no exact IRRF bookkeeping by CPF/CNPJ
- no exact compensation waterfall by tax bucket
- no exact crypto venue-by-venue compliance treatment

## Regulatory Guardrails

These outputs are safe for research, simulation and proprietary use.

They are not enough, by themselves, to justify:

- individualized investment advice for clients
- public recommendation business without proper CVM framing
- discretionary portfolio management for third parties
- live copytrade for third parties

Before public monetization or client-facing advisory use, review the current CVM framing for:

- consultoria de valores mobiliarios
- analista de valores mobiliarios
- administracao de carteira de valores mobiliarios
- copytrade and public recommendation practices

## Implementation Pointers

- core net modeling: `execution/net_assumptions.py`
- research assumptions: `config/profit_net_assumptions.json`
- official promoted mode path: `scripts/bench/validation/run_profit_alpha_hardening_suite.py`
- blended regime profile: `scripts/bench/validation/run_profit_regime_simulation_suite.py`

## Operational Checklist

1. Keep the published UI labeled as research/simulation unless legal framing changes.
2. Do not claim tax exactness; describe the tax layer as a proxy.
3. Do not claim guaranteed returns; show scenario history, not certainty.
4. If client-specific recommendations are added, get legal and accounting review first.
