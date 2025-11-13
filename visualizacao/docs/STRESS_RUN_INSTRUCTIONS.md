## Stress Test â€“ Guia RÃ¡pido (macOS)

### 1. PrÃ©-requisitos
- macOS com Python 3.10+ instalado (`python3 --version` para conferir).
- git (opcional) caso vÃ¡ clonar o repositÃ³rio.
- Recomendado: usar `venv` para isolar dependÃªncias.

### 2. Clonar ou copiar o projeto
```bash
git clone <seu-repo> quantum_walk_project
cd quantum_walk_project
```
> Se jÃ¡ transferiu a pasta, apenas abra um terminal em `quantum_walk_project`.

### 3. Criar ambiente virtual (opcional, mas recomendado)
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Instalar dependÃªncias
```bash
pip install -r requirements.txt
```
> Caso o arquivo `requirements.txt` esteja desatualizado, instale manualmente:  
`pip install numpy pandas matplotlib yfinance`

### 5. Rodar a bateria de stress
O script `scripts/estudos/stress_pipeline.py` automatiza todos os testes descritos:
```bash
python scripts/estudos/stress_pipeline.py --domains all --noise-levels 0.0,0.01,0.02,0.05
```
- Resultados sÃ£o salvos em `results_stress/`:
  - `stress_summary.csv`: tabela com Î”MAE, Î”dir, Î”Î±, etc.
  - `stress_performance.png`: grÃ¡ficos de Î”MAE/Î”Î± versus ruÃ­do.
  - `stress_report.md`: resumo textual.
  - Subpastas (`robustness`, `financial`, `noise`, `shuffle`) por domÃ­nio.

### 6. Verificar hipÃ³teses de falha
Abra `results_stress/stress_report.md` para ver:
- Robustez (percentual de combinaÃ§Ãµes em que Hadamard vence).
- InversÃµes de Î”Î±.
- ComparaÃ§Ã£o com p-valores DM (carregados automaticamente de `scripts/estudos/make_report.py`).

### 7. RecomendaÃ§Ãµes
- Para repetir algum domÃ­nio isolado:  
  `python scripts/estudos/stress_pipeline.py --domains SPY`
- Ajustar nÃ­veis de ruÃ­do:  
  `--noise-levels 0.0,0.01,0.03,0.05,0.1`
- Se precisar rodar apenas a grade de robustez:  
  `python run_robustness_grid.py --symbol SPY --bins-list ... --window-list ... --noise-list ... --outdir results_custom/SPY`

### 8. Dicas macOS
- Use `python3`/`pip3` se o alias `python` apontar para Python 2.
- Para abrir os grÃ¡ficos: `open results_stress/stress_performance.png`.
- Para acompanhar logs em tempo real: `tail -f results_stress/<domÃ­nio>/financial/run.log` (se desejar salvar stdout).

### 9. Limpeza
- Para rodar novamente do zero:  
  `rm -rf results_stress tmp`
- Reative o ambiente virtual sempre que abrir nova sessÃ£o:  
  `source .venv/bin/activate`

Pronto! Com esses passos vocÃª consegue reproduzir a bateria completa no macOS e validar onde o modelo mantÃ©m (ou perde) vantagem.
