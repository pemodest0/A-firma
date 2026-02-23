# Assyntrax - Structural Stability Engine for Complex Systems

Status: nucleo teorico canonico v1  
Atualizado em: 2026-02-23

Este documento consolida a base matematica existente do Assyntrax com o enquadramento transversal de estabilidade (financas, energia e operacao).  
Ele separa, de forma explicita, o que ja esta operacional em v1 do que e trilha de pesquisa v2.

## 1. Objetivo teorico

O Assyntrax modela sistemas complexos multivariados como sistemas dinamicos de alta dimensionalidade sujeitos a transicoes criticas.

Hipotese central:

Sistemas complexos podem apresentar perda de resiliencia mensuravel antes de transicoes criticas, detectavel por:

- concentracao espectral global,
- sinais de critical slowing down (CSD),
- fragilidade topologica emergente.

Objetivos de estimacao:

- perda de resiliencia local,
- concentracao estrutural global,
- fragilidade topologica,
- proximidade de transicao critica.

## 2. Representacao formal

Seja:

\[
X(t) \in \mathbb{R}^{N}
\]

onde cada componente representa uma variavel observavel (retorno de ativo, fluxo de carga, fluxo operacional etc.).

Para janela movel \(T\):

\[
C(t) = \mathrm{Corr}(X_{t-T+1:t})
\]

com \(C(t) \in \mathbb{R}^{N \times N}\), a matriz de correlacao empirica dinamica.

Restricao de causalidade:

- tudo em \(t\) usa apenas informacao ate \(t\),
- sem smoothing centrado,
- sem calibracao com dado futuro.

## 3. Pilar I - teoria espectral e estrutura coletiva

### 3.1 Referencia de ruido via RMT

Defina:

\[
Q = \frac{T}{N}
\]

Sob hipotese nula i.i.d., os limites de Marcenko-Pastur sao:

\[
\lambda_{\pm} = \sigma^2 \left(1 \pm \sqrt{\frac{1}{Q}}\right)^2
\]

Autovalores acima de \(\lambda_+\) sao interpretados como modos estruturais alem de ruido amostral.

### 3.2 Parametro de ordem estrutural

\[
\phi(t) = \frac{\lambda_1(t)}{\sum_{i=1}^{N}\lambda_i(t)}
\]

Leitura:

- \(\phi\) baixo: estrutura difusa,
- \(\phi\) alto: sincronizacao coletiva,
- crescimento rapido de \(\phi\): concentracao sistemica.

No Assyntrax v1, \(\phi\) e operacionalizado por `p1`.

### 3.3 Entropia espectral e dimensao efetiva

Normalizacao:

\[
p_i(t) = \frac{\lambda_i(t)}{\sum_j \lambda_j(t)}
\]

Entropia espectral:

\[
H(t) = -\sum_i p_i(t)\ln p_i(t)
\]

Dimensao efetiva:

\[
D_{\text{eff}}(t) = \exp(H(t))
\]

Queda de \(D_{\text{eff}}\) indica colapso dimensional e menor resiliencia estrutural.

No Assyntrax v1, isso e medido por `deff`.

### 3.4 Estabilidade temporal do modo dominante

Overlap do autovetor principal:

\[
O_{11}(t, t-\Delta) = |\langle v_1(t), v_1(t-\Delta)\rangle|^2
\]

Queda persistente de overlap indica rotacao estrutural do modo coletivo dominante.

No Assyntrax v1:

- `eigvec_overlap_1d`,
- `eigvec_instability_1d`.

## 4. Pilar II - critical slowing down (CSD)

Perto de bifurcacao, dinamica local pode ser aproximada por:

\[
dX = f(X,\mu)\,dt + \sigma\,dW_t
\]

Autovalor local dominante:

\[
\lambda = \frac{\partial f}{\partial X}
\]

Quando \(\lambda \to 0\), a recuperacao desacelera e a resiliencia cai.

Indicadores operacionais de CSD:

- aumento de variancia,
- aumento de autocorrelacao lag-1,
- aumento da correlacao media cruzada,
- menor taxa de recuperacao pos-choque.

No Assyntrax v1, proxies dinamicos centrais:

- `|dp1_5|`,
- `|ddeff_5|`,
- instabilidade de overlap.

## 5. Pilar III - fragilidade geometrica da rede

Interprete \(C(t)\) como grafo ponderado.

Para aresta \(e=(u,v)\), curvatura de Forman-Ricci:

\[
\kappa_F(e) =
w_e\left(
\frac{w_u+w_v}{w_e}
- \sum_{e'\sim u}\frac{w_u}{\sqrt{w_e w_{e'}}}
- \sum_{e'\sim v}\frac{w_v}{\sqrt{w_e w_{e'}}}
\right)
\]

Leitura:

- curvatura positiva: redundancia local e maior robustez,
- curvatura negativa: gargalos e propensao a propagacao de choque,
- queda abrupta da curvatura media: pre-cascata.

Status de implementacao:

- trilha de pesquisa para v2,
- ainda fora do publish gate de producao v1.

## 6. Dinamica integrada de transicao critica

O Assyntrax integra tres mecanismos:

- concentracao espectral (\(\phi\), \(D_{\text{eff}}\)),
- perda dinamica de resiliencia (CSD),
- fragilidade topologica (curvatura, v2).

A inferencia de transicao critica ganha forca quando ha convergencia de sinais:

- crescimento sustentado de \(\phi\),
- queda relevante de \(D_{\text{eff}}\),
- aumento de proxies de variancia/autocorrelacao,
- reducao de curvatura media (quando ativada).

A convergencia multi-sinal reduz falso positivo isolado.

## 7. Validacao causal e protocolo anti-leakage

Regras obrigatorias:

- split cronologico fixo treino/teste,
- calibracao walk-forward de limiares,
- block bootstrap para dependencia temporal,
- proibicao de leakage de eventos futuros,
- manifesto reprodutivel por run (parametros, commit, artefatos).

No v1, isso e reforcado por healthcheck e validadores operacionais.

## 8. Dominio de validade e fronteiras

Premissas:

- sistema observavel via serie temporal multivariada,
- dependencia principal capturavel por estrutura linear no v1,
- razao amostral adequada \(Q=T/N\).

Limites:

- choques exogenos abruptos podem escapar de precursores estruturais,
- sensibilidade a janela e hiperparametros,
- interacoes fortemente nao lineares ainda sao parcialmente capturadas.

## 9. Universalidade transversal

O formalismo matematico e invariavel; muda apenas \(X(t)\) por dominio.

Conjunto minimo v1 por setor:

- Financas: retornos log, volume, vol realizada, proxy de liquidez.
- Energia: carga agregada, fluxos regionais, preco spot, variabilidade intra (ou agregacao diaria).
- Saude (v2): ocupacao, filas/fluxo, uso de recursos.
- Logistica (v2): estoque, fluxo de transporte, demanda.

## 10. Natureza da engine

O Assyntrax nao tenta prever preco-alvo ou timestamp exato de evento.

Ele estima:

\[
R(t) = \text{resiliencia estrutural dinamica}
\]

\[
P_{\text{crit}}(t) = \text{proximidade de transicao critica}
\]

Mapeado no v1 para saidas operacionais:

- `regime` (`stable`, `transition`, `stress`, `dispersion`),
- `transition_score`,
- `instability_score`,
- decisao de publish gate.

## 11. Mapa de implementacao v1

Nucleo de producao:

- `scripts/lab/run_corr_macro_offline.py`
- `config/lab_corr_policy.json`
- pipelines diarios em `scripts/ops/`

Ja operacional em v1:

- correlacao rolling,
- metricas espectrais (`p1`, `deff`, familia de entropia),
- instabilidade temporal (overlap e derivadas),
- bootstrap de significancia e gate de publicacao.

Trilha v2:

- camada geometrica de curvatura em producao,
- blocos nao lineares adicionais,
- inferencia adaptativa hibrida fisica + IA por setor.

## 12. Referencias

1. Laloux, Cizeau, Bouchaud, Potters (1999), Noise dressing of financial correlation matrices.  
2. Bouchaud, Potters (2005), Financial applications of Random Matrix Theory.  
3. Kritzman et al. (2011), Principal Components as a Measure of Systemic Risk.  
4. Scheffer et al. (2009), Early-warning signals for critical transitions.  
5. Politis, Romano (1994), The Stationary Bootstrap.  
6. Del Giudice (2020), Effective dimensionality tutorial.  
7. Forman (2003), Bochner's method for cell complexes and combinatorial Ricci curvature.
