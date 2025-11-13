# Modelos de Caminhada Recém-Adicionados

## 1. Caminhada Quântica em Tempo Contínuo (`continuous_time_quantum_walk.py`)
- Evolução unitária via hamiltoniano tipo Laplaciano \(H = \gamma (D - A)\).
- Integração espectral: decomposição própria, atualizando coeficientes faseados por passo.
- Saídas alinhadas com versões clássicas/discretas: distribuições, entropia de Shannon, hitting time.
- Útil para ligar com o formalismo CTQW do artigo *Quantum Walk Computing* (Qiang et al., 2024), com \(\gamma\) controlando o regime de propagação.

## 2. Caminhada Discreta com Ruído / Decoerência (`quantum_walk_noise.py`)
- Usa Qiskit (`DensityMatrix`) para aplicar moedas de Hadamard/Grover e canais de ruído via operadores de Kraus.
- `noise_profile` aceita lista de dicionários `{"type": "...", "strength": ..., "target": ...}`:
  - Tipos: `phase`, `phase_flip`, `bit_flip`, `amplitude`, `depolarizing`.
  - Alvos: `coin`, `position`, `all`, ou `q{i}` para qubits específicos.
- Permite medir impacto de canais não unitários (interpretação Lindblad/Open QW).
- Distribuições e métricas compatíveis com pipeline existente (`compute_hitting_time`, entropias).

## Referências Principais
- Qiang et al., *Quantum Walk Computing: Theory, Implementation, and Application* (2024).  
  Fornece base formal para CTQW, coins adaptativas e ruído tipo Lindblad — espelhado nos dois módulos acima.
