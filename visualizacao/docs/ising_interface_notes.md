# Interface Interativa – Cristal de Ferro (Ising 2D)

## O que o aplicativo faz
`app/ising_interface.py` mostra um cristal 2D de ferro em versão didática. Cada quadradinho do tabuleiro é um spin que aponta **para cima** (+1, vermelho) ou **para baixo** (−1, azul). Ao ajustar a temperatura e os parâmetros mecânicos da rede, a animação revela como os spins trocam informações, formando ou desfazendo domínios magnéticos.

Para manter tudo palpável durante a aula, o controle de temperatura vai de **0,1 K a 100 K**, onde **100 K representa o ponto de Curie real do ferro (~1043 K)**. Assim conseguimos atravessar todos os regimes (ferromagnético, crítico e paramagnético) sem precisar digitar números de quatro dígitos.

## Controles disponíveis
- **Temperatura alvo (K\*)** – a escala vai de 0,1 a 100. O código converte automaticamente para temperatura reduzida (`T/T_c`). Valores baixos deixam os spins “grudados”; perto de 100 K\* surgem flutuações fortes.
- **Força de acoplamento magnético J (0,1–2,0)** – J maior fortalece o alinhamento entre vizinhos.
- **Distância entre átomos (Å) (0,5–3,0)** – define o tamanho mostrado na célula BCC 3D. É puramente visual, mas ajuda na conversa sobre parâmetros cristalográficos.
- **Rigidez das ligações (N/m) (0,1–100)** – interpretações didáticas: valores baixos simulam uma rede vibrante; valores altos deixam o cristal mais “duro”.
- **Número de átomos por lado (1–100)** – controla o tamanho do tabuleiro (`N×N`).
- **Tentativas de flip por cena (1–200)** – número de atualizações de Monte Carlo antes de salvar um novo quadro. Nome amigável para o antigo “MCS por quadro”.
- **Cenas por rodada (1–300)** – quantos snapshots são adicionados a cada clique em **Rodar cenas**.
- **Comece assim** – estado inicial (`aleatório`, `tudo para cima`, `faixa metade/metade`, `xadrez`).
- **Semente** – garante reprodutibilidade.
- Botões: **Reiniciar cristal** zera a rede com os parâmetros atuais. **Rodar cenas** avança a simulação e cria novos quadros.

## O que aparece na tela principal
1. **Mapa de spins** – o heatmap colorido com contornos cinza mostra as fronteiras de domínio (onde vermelho e azul se encontram). É a peça central para discutir crescimento de fases.
2. **Painel rápido** – métricas explicadas em português simples:
   - Magnetização e energia atuais.
   - Porcentagem de spins +1 e −1.
   - Temperatura indicada (em K\*) e a equivalente real em kelvin.
   - Texto curto dizendo se estamos em “ferro alinhado”, “perto do ponto crítico” ou “desalinhado”.
3. **Gráfico de evolução** – linhas da magnetização e da energia mostrando como o cristal foi se organizando quadro a quadro.
4. **Célula BCC em 3D** – a célula unitária do ferro com escala definida pelo slider de distância. A cor dominante acompanha o sinal da magnetização.
5. **Resumo físico** – parágrafo automático que sugere uma leitura pedagógica (ex.: “temperatura superfria lembra supercondutividade”, “molas rígidas deixam a parede de domínio lenta”, etc.).

## Leituras rápidas dos regimes
- **Temperatura < 10 K\*** – spins quase congelados. Útil para falar de supercondutividade ou materiais em banho de hélio líquido.
- **Entre 10 e 70 K\*** – ferromagnetismo robusto; os domínios crescem e um lado vence. Combina com histórias de ímãs comuns.
- **Perto de 100 K\*** – tempestade de domínios; perfeito para explicar por que o Curie é um ponto especial.

## Atividades sugeridas em aula
1. **Parede de domínio ambulante** – comece com “faixa metade/metade”, mantenha 20–40 K\*, clique em **Rodar cenas** algumas vezes e peça para os alunos contarem quantos quadros a parede leva para cruzar o tabuleiro.
2. **Caça ao ponto crítico** – aumente a temperatura devagar até notar magnetização próxima de zero e domínios de vários tamanhos. Relacione com a definição formal de `T_c`.
3. **Superfrio vs. rede mole** – deixe a temperatura em 5 K\*, teste rigidez baixa (molas frouxas) e depois alta (molas rígidas) para ver o efeito nas fronteiras.

## Como amarrar com teoria
- Explique que o algoritmo usa Metropolis Monte Carlo: cada cena tenta virar `N×N` spins, aceitando viradas que reduzem energia ou, às vezes, que a aumentam (probabilidade `exp(-ΔE/k_BT)`).
- Mostre que a temperatura reduzida `k_B T / J` está escondida atrás do slider de K\*. É assim que construímos a relação com o ponto de Curie real.
- Use o gráfico de magnetização como parâmetro de ordem. Peça para os alunos preverem o valor final antes de clicar em **Rodar cenas**.

## Próximos passos possíveis
- Acrescentar campo magnético externo para demonstrar histerese.
- Gravar GIFs automáticos das cenas geradas.
- Exportar as séries de magnetização/energia para CSV e usar em atividades de análise de dados.

Com esse painel você tem um “mini laboratório” para apresentar o Modelo de Ising 2D de forma visual, direta e divertida – ótimo para turmas de graduação, para a aula de estado sólido ou mesmo como apoio ao projeto de pesquisa sobre equações de Schrödinger deformadas.
