# Nomenclatura Oficial: Assyntrax + Eigen Engine

Data: 2026-02-25
Status: ativo

## 1. Definicao canonica

- `Assyntrax`: nome da empresa, marca e plataforma/site.
- `Eigen Engine`: nome oficial do motor estrutural.
- Site oficial unico em producao: `https://assyntrax.vercel.app`.

## 2. Regra de escrita

- Em texto institucional/comercial:
  - usar `Assyntrax` para empresa/plataforma.
  - usar `Eigen Engine` para o motor.
- Em frases completas, preferir:
  - `Eigen Engine da Assyntrax`.

## 3. Compatibilidade retroativa

- Arquivos legados podem manter `ASSYNTRAX` no nome do arquivo para evitar quebra de links e scripts.
- Esta compatibilidade vale para paths e nao para copy principal de produto.

## 4. Mapeamento de termos (de -> para)

- `motor Assyntrax` -> `Eigen Engine`
- `Assyntrax Motor de Regime` -> `Eigen Engine (Assyntrax)`
- `Assyntrax Engine` (quando se referir ao motor) -> `Eigen Engine`

## 5. Fontes que devem seguir a regra

- Documentacao de produto e operacao (`README.md`, `docs/**`).
- Copy e metadata do site (`website-ui/app/**`, `website-ui/components/**`).
- Artefatos visuais e marketing (`website-ui/public/assets/**`).
- Prompts e documentos de IA (`docs/AI_SYSTEM_PROMPT.md`, `docs/KNOWLEDGE_ASSYNTRAX.md`).

## 6. Excecoes permitidas

- Nome do repositorio GitHub e URLs existentes.
- Nomes de arquivos legados ainda referenciados por scripts.
- Nome de pasta tecnica local `website-ui/` (nao representa marca/produto).
- Dados historicos em `results/` ja gerados.

## 7. Gate de consistencia recomendado

Rodar periodicamente:

```bash
rg -n --glob '!node_modules/**' \
  -e 'motor Assyntrax|Assyntrax Motor de Regime|Assyntrax Engine' \
  README.md docs website-ui
```

Qualquer ocorrencia nova deve ser tratada antes de release.
