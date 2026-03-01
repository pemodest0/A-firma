# Assyntrax Site (Diretório Técnico)

Frontend operacional e APIs web da plataforma Assyntrax.

## Links
- Site oficial (Vercel): `https://assyntrax.vercel.app`
- Repositório: `https://github.com/pemodest0/Assyntrax`
- Licença: MIT (`../LICENSE`)

## Identidade de produto
- Marca/empresa: **Assyntrax**
- Motor: **Eigen Engine**
- Copiloto: **Eigen Engine Assistant**
- Domínios ativos: **finanças**, **energia** e **agro**

## Rotas principais
- Site institucional:
  - `/`
  - `/methods` (Eigen Engine)
  - `/research/methodology` (metodologia técnica)
- App operacional:
  - `/app/dashboard`
  - `/app/universo-observavel`
  - `/app/setores`
  - `/app/operacao`
  - `/app/venda`

## APIs principais
- `/api/lab/corr/latest`
- `/api/validation/latest`
- `/api/pilot/latest`
- `/api/copilot`

## Desenvolvimento local
```bash
cd website-ui
npm install
npm run dev
```

## Qualidade
```bash
cd website-ui
npm run lint
npm run typecheck
npm run build
```

## Deploy
```bash
cd website-ui
npx vercel --prod --yes
```

## Observacoes
- O nome da pasta local `website-ui/` é apenas técnico; a marca e o deploy oficial são `Assyntrax`.
- O frontend depende de artefatos gerados pelo pipeline do Eigen Engine.
- O chat flutuante oficial é o `Eigen Engine Assistant` (`/api/copilot`).
- Se a API retornar estado inconclusivo/sem dados, a UI exibe fallback operacional sem quebrar rota.
