# Assyntrax Site e App

Frontend e backend web da Assyntrax.

## Escopo atual
- Site público da plataforma
- App operacional do Eigen Engine
- Copiloto de uso
- Painéis de finanças e cripto
- Evidências, teoria e snapshots publicados

## Identidade de produto
- **Marca**: Assyntrax
- **Motor**: Eigen Engine
- **Copiloto**: Eigen Engine Assistant
- **Frente principal**: finanças e cripto

## Rotas principais
- Site:
  - `/`
  - `/financas`
  - `/cripto`
  - `/evidencias`
  - `/methods`
  - `/product`
- App:
  - `/app/dashboard`
  - `/app/aplicacoes`
  - `/app/financas`
  - `/app/cripto`
  - `/app/copiloto`
  - `/app/teoria`

## APIs principais
- `/api/platform/latest`
- `/api/assets`
- `/api/copilot`
- `/api/invest/advisory`
- `/api/invest/shadow`

## Desenvolvimento local
```bash
cd website-ui
npm install
npm run dev
```

## Qualidade
```bash
cd website-ui
npm run build
```

## Observações
- O frontend depende de snapshots e artefatos publicados pelo pipeline do Eigen Engine.
- Quando o dado publicado não existe, a UI deve degradar com clareza, sem inventar número.
- O foco do produto não é mais agro/energia como frente pública principal.
