# Assyntrax Site (Diretorio Tecnico)

Frontend operacional e APIs web da plataforma Assyntrax.

## Links
- Site oficial (Vercel): `https://assyntrax.vercel.app`
- Repositorio: `https://github.com/pemodest0/Assyntrax`
- Licenca: MIT (`../LICENSE`)

## Rotas principais
- Site institucional:
  - `/`
  - `/methods` (Eigen Engine)
  - `/guia` (Leonardo)
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
- O nome da pasta local `website-ui/` e apenas tecnico; a marca e o deploy oficial sao `Assyntrax`.
- O frontend depende de artefatos gerados pelo pipeline do Eigen Engine.
- Se a API retornar estado inconclusivo/sem dados, a UI exibe fallback operacional (sem quebrar rota).
