# Mac Test Env: pytest + DNS

Guia rapido para destravar ambiente de testes no Mac.

## 1) Instalar pytest

```bash
python3 -m pip install --user pytest
python3 -m pytest --version
```

Se o comando `pytest` nao estiver no PATH:

```bash
export PATH="$HOME/Library/Python/3.14/bin:$PATH"
```

## 2) Diagnosticar DNS

```bash
scutil --dns
python3 - <<'PY'
import socket
for h in ["pypi.org","files.pythonhosted.org","github.com"]:
    try:
        print(h, socket.gethostbyname(h))
    except Exception as e:
        print(h, "ERR", e)
PY
```

## 3) Forcar DNS estavel no Wi-Fi

```bash
networksetup -setdnsservers Wi-Fi 1.1.1.1 8.8.8.8
networksetup -getdnsservers Wi-Fi
```

Opcional para voltar ao automatico por DHCP:

```bash
networksetup -setdnsservers Wi-Fi Empty
```

## 4) Validar suite estrutural

```bash
python3 -m pytest -q tests/test_ground_truth_structural.py tests/test_run_manifest.py tests/test_rmt.py tests/test_spectral_structural.py tests/test_csd.py tests/test_graph_structural.py tests/test_forman.py tests/test_score.py
```
