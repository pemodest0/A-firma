## Agentes Diários

Este diretório descreve a função de cada agente operacional do produto.

Objetivo geral:
- manter a base de preços fresca
- corrigir lacunas antes do motor rodar
- recalcular a operação diária
- auditar qualidade, consistência e publicação
- testar o site publicado
- reagir automaticamente quando algum elo falhar

Ordem operacional esperada:
1. `daily-ingestion-agent`
2. `daily-backfill-agent`
3. `daily-operation-agent`
4. `daily-shadow-gods-agent`
5. `daily-vigilance-agent`
6. `daily-data-quality-agent`
7. `daily-publish`
8. `daily-smoke-test-agent`
9. `daily-watchdog-agent`

Regras globais:
- nenhum agente deve mascarar falha crítica como sucesso
- artefatos publicados devem apontar para a mesma data-base
- agentes de reparo podem tentar retry controlado, nunca loop infinito
- revisão de universo deve separar `crítico`, `núcleo` e `periferia`
- qualquer estado `fail` precisa gerar artefato explícito e deixar rastro
