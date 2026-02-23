# Fundamentacao Teorica do Assyntrax (versao estruturada)

Nota: a versao teorica consolidada e transversal do motor esta em `docs/motor/THEORY_ASSYNTRAX.md`.

Este documento consolida a base matematica e estatistica do Assyntrax em linguagem tecnica, com notacao em LaTeX.
O objetivo do motor e diagnosticar **estado estrutural de risco** (estabilidade, transicao, estresse), nao gerar recomendacao direcional de compra/venda.

## 1) Espaco de probabilidade e causalidade

Mercado modelado em:

$$
(\Omega,\mathcal{F},\mathbb{P})
$$

com filtracao:

$$
\{\mathcal{F}_t\}_{t\ge0}, \quad \mathcal{F}_s \subseteq \mathcal{F}_t \text{ para } s \le t.
$$

Vetor de precos:

$$
\mathbf{S}_t = (S_{1,t}, \ldots, S_{N,t})^\top.
$$

Hipotese operacional: a matriz latente populacional $\Sigma_t$ e inobservavel; observamos uma realizacao finita $C_t$ (correlacao empirica), contendo sinal estrutural + ruido amostral.

## 2) Retornos e padronizacao

Retorno logaritmico:

$$
r_{i,t} = \ln\left(\frac{S_{i,t}}{S_{i,t-1}}\right).
$$

Padronizacao em janela:

$$
z_{i,t} = \frac{r_{i,t} - \bar r_i}{\sigma_i},
\quad
\bar r_i = \frac{1}{T}\sum_{\tau=t-T+1}^{t} r_{i,\tau}.
$$

Com isso, a analise espectral fica menos sensivel a escala nominal dos ativos.

## 3) Correlacao dinamica por EWMA

Atualizacao recursiva:

$$
C_t = (1-\lambda)\mathbf{z}_t\mathbf{z}_t^\top + \lambda C_{t-1}, \quad \lambda \in (0,1).
$$

Memoria efetiva aproximada:

$$
T_{eff} = \frac{1}{1-\lambda}.
$$

Regra pratica de validade: manter $T_{eff}$ em ordem comparavel ou superior a $N$ para evitar degradacao numerica.

## 4) Decomposicao espectral

Como $C_t$ e simetrica real:

$$
C_t = V_t \Lambda_t V_t^\top
= \sum_{k=1}^N \lambda_{k,t}\mathbf{v}_{k,t}\mathbf{v}_{k,t}^\top.
$$

Com retornos padronizados:

$$
\operatorname{Tr}(C_t)=\sum_{k=1}^N \lambda_{k,t}=N.
$$

## 5) Teoria de Matrizes Aleatorias (RMT) e limite de ruido

Sob hipotese nula i.i.d., o espectro converge para Marcenko-Pastur:

$$
f(\lambda)=\frac{Q}{2\pi \sigma^2 \lambda}
\sqrt{(\lambda_+ - \lambda)(\lambda-\lambda_-)},
\quad Q=\frac{T}{N}.
$$

Limites:

$$
\lambda_{\pm}=\sigma^2\left(1\pm\sqrt{\frac{1}{Q}}\right)^2.
$$

Leitura operacional:
- bulk dentro de $[\lambda_-,\lambda_+]$ tende a ruido;
- autovalores acima de $\lambda_+$ sao candidatos a estrutura;
- $\lambda_1$ elevado sinaliza modo sistemico dominante.

## 6) Metricas centrais do Assyntrax

### 6.1 Absorption Ratio

$$
AR_n(t)=\frac{\sum_{k=1}^{n}\lambda_{k,t}}{\sum_{k=1}^{N}\lambda_{k,t}}.
$$

### 6.2 Entropia espectral e dimensao efetiva

$$
p_{k,t}=\frac{\lambda_{k,t}}{\sum_{j=1}^{N}\lambda_{j,t}},
\quad
H_t=-\sum_{k=1}^{N} p_{k,t}\ln p_{k,t},
\quad
ED_t=e^{H_t}.
$$

### 6.3 Inverse Participation Ratio (IPR)

$$
IPR_{k,t}=\sum_{i=1}^{N}(v_{k,t}^{(i)})^4.
$$

## 7) Estabilidade temporal (overlap de autovetores)

$$
O_{ij}(t_1,t_2)=\left|\left\langle \mathbf{v}_{i,t_1}, \mathbf{v}_{j,t_2}\right\rangle\right|^2.
$$

Queda persistente em $O_{11}$ indica rotacao do modo dominante e potencial mudanca estrutural.

## 8) Score e classificador com histerese

Mapeamento escalar:

$$
\Phi_t = -\ln(ED_t),
\quad
S_t = \frac{\Phi_t-\mu_\Phi}{\sigma_\Phi}.
$$

Classificacao com duas barreiras:

$$
R_t=
\begin{cases}
1, & S_t \ge \theta_{up} \\
0, & S_t \le \theta_{down} \\
R_{t-1}, & \theta_{down} < S_t < \theta_{up}
\end{cases}
$$

com $\theta_{up}>\theta_{down}$ e permanencia minima para reduzir chattering.

## 9) Validacao causal

1. **Walk-forward** para calibracao sem look-ahead.
2. **Stationary/Block Bootstrap** para inferencia com dependencia temporal.
3. **Event study** com foco em:

$$
\text{Recall}=\frac{TP}{TP+FN},
\quad
\text{Precision}=\frac{TP}{TP+FP}.
$$

Lead-time:

$$
\tau_{lead}=t_0-\inf\{t<t_0: R_t=1\}.
$$

## 10) Fronteiras de validade

1. Se $Q=T/N<1$, surgem problemas de posto e autovalores espurios proximos de zero.
2. Sem warmup adequado no EWMA, estados iniciais ficam distorcidos.
3. Caudas pesadas podem inflar autovalores e confundir separacao sinal/ruido.
4. O motor mede **fragilidade estrutural**, nao garante direcao de preco nem tempo exato de evento.

## 11) O que fica fora desta prova

Nao estao totalmente desenvolvidas aqui:
1. Prova completa por transformada de Stieltjes dos limites MP.
2. Derivacao fechada universal dos hiperparametros otimos de histerese ($\theta_{up},\theta_{down}$) para todos os universos.

## Referencias (ABNT resumido)

1. LALOUX, L. et al. *Noise dressing of financial correlation matrices*. Physical Review Letters, 1999.
2. BOUCHAUD, J.-P.; POTTERS, M. *Financial Applications of Random Matrix Theory*. 2005.
3. KRITZMAN, M. et al. *Principal Components as a Measure of Systemic Risk*. Journal of Portfolio Management, 2011.
4. POLITIS, D.; ROMANO, J. *The Stationary Bootstrap*. JASA, 1994.
5. DEL GIUDICE, M. *Effective dimensionality: A tutorial*. Multivariate Behavioral Research, 2020.
6. FEDERAL RESERVE. *SR 11-7: Guidance on Model Risk Management*.
7. BIS. *Basel III: Liquidity Coverage Ratio*.

## Observacao regulatoria

Uso do Assyntrax: suporte quantitativo para risco, governanca e monitoramento estrutural.
Nao constitui recomendacao de compra ou venda.
