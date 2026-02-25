# GitHub Canônico (Multi-Máquina)

Objetivo: manter `origin/main` como fonte unica de verdade entre Windows, Mac e execucoes do Codex.

## Regra principal
- Em caso de divergencia entre remoto e local, **vence o remoto (`origin/main`)**.
- O local deve ser sobrescrito quando necessario para voltar ao estado canonico.

## Fluxo padrao por sessao

1. Sincronizar antes de qualquer edicao:

```bash
git fetch origin --prune
git pull --ff-only origin main
```

Se precisar sobrescrever local pelo remoto (modo forçado/canonico):

```bash
./scripts/ops/git_sync_canonical.sh
```

No Windows PowerShell:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\ops\git_sync_canonical.ps1
```

2. Rodar healthcheck completo:

```bash
./scripts/ops/run_repo_healthcheck.sh
```

3. Commitar apenas arquivos do escopo:

```bash
git add <arquivos_do_escopo>
git commit -m "mensagem objetiva"
```

4. Publicar no remoto canonico:

```bash
git push origin main
```

## Regras de governanca
- O remoto canonico pode sobrescrever local quando houver conflito de estado.
- Use os scripts de sync canônico para resetar local com rastreabilidade.
- Nao misturar mudancas de features diferentes no mesmo commit.
- Sempre validar com healthcheck antes de push.
- Se o `pull --ff-only` falhar, resolver conflito localmente e repetir healthcheck.

## Verificacoes rapidas

```bash
git status -sb
git rev-parse --short HEAD
git rev-parse --short origin/main
```

## GitHub Actions e notificacoes

Se o GitHub Actions estiver bloqueado por billing, cada push em `main` gera notificacao de falha.

Politica atual (2026-02-25):

- workflow `Repo Healthcheck` em modo **manual** (`workflow_dispatch`).
- sem trigger automatico em `push/pull_request` ate regularizacao de billing.

Quando billing normalizar:

1. Reativar triggers de `push/pull_request` em `.github/workflows/repo-healthcheck.yml`.
2. Fazer um commit unico de reativacao.
3. Confirmar uma execucao verde.
