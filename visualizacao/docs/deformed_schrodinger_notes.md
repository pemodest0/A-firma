# Schrödinger Deformado — Notas Técnicas

## 1. Equação Utilizada
Adotamos a versão 1D com constante de Planck reduzida e massa iguais a 1:

\[
i\,\partial_t \psi(x,t) \;=\; \big[-\tfrac{1}{2}\,\partial_{xx} + V(x)\big]\psi(x,t)
 \;+\; \varepsilon\,F(x,t)\,\psi(x,t) \;-\; i \gamma\,\psi(x,t),
\]

em que:
- \(V(x)\) é o potencial base (livre, harmônico, barreira ou duplo poço);
- \(\varepsilon\,F(x,t)\) é o termo de deformação (pilotado por \(\varepsilon\) e frequência escolhida);
- \(\gamma\) modela dissipação/controle (coin\_risk no pipeline de walks);
- \(\psi\) é normalizada em cada passo.

### Termo de Drive
\[
F(x,t) = \sin(\pi x)\cos(\omega_{\text{drive}} t).
\]
Esse formato garante simetria ímpar e permite observar interferências facilmente.

## 2. Discretização
- Malha espacial uniforme \([x_{\min}, x_{\max}] \subset \mathbb{R}\) com \(N\) pontos.
- Laplaciano: diferenças finitas de segunda ordem.
- Integração temporal: Crank–Nicolson com passo \(\Delta t\), garantindo estabilidade para passos pequenos.
- O efeito de \(\gamma\) entra como termo real adicional (diagonal) nos sistemas lineares do método.

## 3. Observáveis Calculados
1. **Distribuição de probabilidade** \(P(x,t)=|\psi|^2\).
2. **Expectação de posição e momento**:
   \[
   \langle x \rangle = \int x\,P(x,t)\,\mathrm{d}x,\qquad
   \langle p \rangle = \operatorname{Re}\int \psi^\ast (-i\,\partial_x \psi)\,\mathrm{d}x.
   \]
3. **Probabilidade acumulada em uma região alvo** (padrão \(x\ge 0\)).
4. **Hitting time**: primeiro instante em que a probabilidade acumulada ultrapassa um limiar (default 0,1). Quando não alcança o limiar, exibimos aviso.

## 4. Variáveis Controláveis na Interface
| Item | Descrição | Impacto |
|------|-----------|---------|
| Potencial base | livre, harmônico, barreira, duplo poço | Define cenários: confinamento, tunelamento, bistabilidade |
| ε (força de deformação) | Intensidade do termo \(F(x,t)\) | Ajuda a simular regimes controlados/case "quantum walk blend" |
| γ (amortecimento) | Dissipa norma, aproxima abertura de sistema | Conecta com coin\_risk/adaptação no pipeline |
| Frequência do drive | Oscilações da deformação | Influencia ressonâncias/hitting time |
| Centro, largura, momento iniciais | Define o wavepacket | Permite estudar tuning de hitting e interferência |
| Passos e Δt | Resolução temporal | Trocando esses valores ajusta precisão vs tempo de execução |
| Pontos na malha | Resolução espacial | Afeta custo e fidelidade |

## 5. Métricas Relevantes
- **Probabilidade acumulada**: indica o quão rápido o pacote atinge a região de interesse.
- **Hitting time**: métrica direta para alertas — análogo ao hitting nos walks.
- **Entropia visual**: o mapa de calor mostra interferências e colisões de frontes.
- **Expectações**: acompanham deslocamento e momento para comparar com trajetórias clássicas.

## 6. Interpretação no Contexto do Projeto
- Termos \(\varepsilon\) e \(\gamma\) correspondem à deformação usada no blend quântico/clássico (coin adaptativa + dissipação controlada).
- Hitting time aqui fornece uma analogia contínua com o hitting do walk — ideal para ilustrar a tese de que os processos discretos convergem para uma equação “Schrödinger deformada”.
- Ajustar o drive/frequência permite estudar regimes equivalentes a “estratégias” de coin, mostrando como mexer em \(\varepsilon\) altera o tempo de resposta.

## 7. Possíveis Extensões
1. **Modos quânticos ruidosos**: incluir ruído aditivo conforme os canais do módulo `quantum_walk_noise`.
2. **Integração com dados reais**: usar o drive \(F(x,t)\) como função de features empíricas (ex.: vol ratio → deformação).
3. **Exportar vídeos/GIFs** para apresentações usando `matplotlib.animation`.
4. **Comparar com walk discreto**: plotar lado a lado a distribuição do walk (via módulo existente) e a solução do simulador.
5. **Adicionar métricas estatísticas** (variância, entropia de von Neumann) para análises de estabilidade.

Com essa interface, fica fácil demonstrar como pequenas deformações no Hamiltoniano induzem mudanças claras na propagação da onda, ligando de forma didática o arcabouço de walks de quantum finance ao formalismo contínuo que será abordado na dissertação. Use os sliders para explorar regimes extremos (ε alto, γ alto) e coletar figuras/insights para o texto principal.
