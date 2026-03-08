import Link from "next/link";
import EngineStoryDeck from "@/components/visuals/EngineStoryDeck";
import PipelineFlow from "@/components/visuals/PipelineFlow";
import HelpHint from "@/components/ui/HelpHint";

const pilares = [
  {
    titulo: "Causalidade walk-forward",
    texto:
      "Os limiares e a leitura do regime usam apenas o histórico disponível até cada data. O motor não reescreve o passado quando vai avaliar o futuro.",
  },
  {
    titulo: "Winsorização de outliers",
    texto:
      "Choques extremos são tratados em janela móvel de 252 dias, entre 0,5% e 99,5%, para não deixar um dia aberrante destruir a leitura da matriz.",
  },
  {
    titulo: "Janela oficial T120",
    texto:
      "A produção roda em T120. T60 e T252 continuam no laboratório como comparação de sensibilidade e robustez, não como fonte de confusão operacional.",
  },
  {
    titulo: "Gate de publicação",
    texto:
      "Cada run passa por checagens mínimas de cobertura, universo, consistência e integridade. Se algo quebra, a publicação é bloqueada automaticamente.",
  },
  {
    titulo: "Robustez quantitativa",
    texto:
      "O motor roda bootstrap em blocos, subamostragem e testes de sensibilidade para medir se o resultado parece método ou sorte curta.",
  },
  {
    titulo: "Auditoria por artefatos",
    texto:
      "Tudo o que vai para a interface pode ser rastreado em artefatos versionados de run: séries, rankings, gate, QA, shadow e pesquisa de alpha.",
  },
];

const guias = [
  {
    titulo: "Guia operacional diário",
    detalhe: "Checklist único de execução, gate e publicação do Eigen Engine para uso diário da plataforma.",
    href: "https://github.com/pemodest0/Assyntrax/blob/main/docs/operacao/CHECKLIST_OPERACAO_EIGEN_ENGINE.md",
  },
  {
    titulo: "API de piloto",
    detalhe: "Contrato técnico para integração externa com chave, payload e política de governança.",
    href: "https://github.com/pemodest0/Assyntrax/blob/main/docs/venda/API_PILOTO_EXTERNA.md",
  },
  {
    titulo: "Pacote de piloto",
    detalhe: "Material executivo e técnico consolidado para um piloto de 30 dias com trilha auditável.",
    href: "https://github.com/pemodest0/Assyntrax/blob/main/docs/venda/PACOTE_PILOTO_ASSYNTRAX_EIGEN_ENGINE.md",
  },
];

const camadas = [
  {
    titulo: "Regime estrutural em finanças globais",
    dados: "Preços diários, matriz de correlação, impacto setorial, breadth e leitura de concentração de risco.",
    janelas: "T60, T120 e T252, com produção em T120.",
    validacao: "Walk-forward por blocos, baseline aleatório na mesma taxa de alerta e lead time operacional.",
    artefatos: "diagnostics_global_daily.csv, rankings_latest.json, latest_finance_product_ready.json",
  },
  {
    titulo: "Sleeve cripto líquido",
    dados: "Moedas líquidas, ranking por tração, filtros de risco-on e comparação contra benchmark em BTC.",
    janelas: "Lookbacks curtos e médios, com rebalance diário, semanal ou mensal conforme a suíte.",
    validacao: "Busca causal, delay de execução, custo líquido proxy e OOS por blocos de mercado.",
    artefatos: "latest_registry.json, profit_frontier_expansion_suite, profit_10x_rule_search_crypto_plus",
  },
  {
    titulo: "Meta-switch e execução",
    dados: "Combinação entre regime, sleeves, ranking, shadow e regras de proteção de drawdown.",
    janelas: "Camada estrutural diária com overlays semanais e mensais na execução.",
    validacao: "Torneio robusto, walk-forward congelado, scorecard semanal e paper trading shadow.",
    artefatos: "profit_layered_engine_suite, profit_drawdown_control_suite, latest_patterns.json",
  },
];

const limites = [
  "Sem promessa de retorno e sem recomendação de compra ou venda.",
  "O foco do produto é diagnóstico de risco estrutural, alocação e governança quantitativa.",
  "A estratégia pode passar períodos abaixo do benchmark e precisa ser lida em contexto.",
  "Choques exógenos podem reduzir a antecedência de alerta.",
  "Resultados históricos não garantem desempenho futuro.",
];

const fontes = [
  {
    nome: "Preços financeiros",
    detalhe:
      "Séries de fechamento diário por ativo, usadas no cálculo de retornos, correlação, estrutura e sleeves.",
  },
  {
    nome: "Mapa de universo e grupos",
    detalhe:
      "Arquivos de universo fixo, grupos, sleeves e classificação, para manter consistência de cobertura histórica.",
  },
  {
    nome: "Artefatos do Eigen Engine",
    detalhe:
      "Saídas versionadas de cada run, com séries temporais, gate, QA, rankings, diagnósticos e relatórios operacionais.",
  },
  {
    nome: "Pesquisa de alpha e shadow",
    detalhe:
      "Registry consolidado de candidatos, paper trading e padrões recentes para comparar o que realmente está melhorando.",
  },
];

const referencias = [
  { id: "M1", titulo: "Financial Applications of Random Matrix Theory", href: "https://arxiv.org/abs/0910.1205" },
  { id: "M2", titulo: "Principal Components as a Measure of Systemic Risk", href: "https://web.mit.edu/~finlunch/Fall10/PCASystemicRisk.pdf" },
  { id: "M3", titulo: "Estimation of Large Financial Covariances", href: "https://arxiv.org/abs/1909.12064" },
  { id: "M4", titulo: "Walk-forward optimization", href: "https://en.wikipedia.org/wiki/Walk_forward_optimization" },
  { id: "M5", titulo: "Block bootstrapping technique", href: "https://scores.readthedocs.io/en/stable/tutorials/Block_Bootstrapping.html" },
  { id: "M6", titulo: "Basel III LCR", href: "https://www.bis.org/publ/bcbs238.htm" },
  { id: "M7", titulo: "SR 11-7 Model Risk Management", href: "https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm" },
  { id: "M8", titulo: "FRTB revised market risk framework", href: "https://www.bis.org/bcbs/publ/d305.htm" },
];

export default function MethodsPageClient() {
  return (
    <div className="space-y-10">
      <section className="space-y-4">
        <div className="text-xs uppercase tracking-[0.3em] text-cyan-300/80">Eigen Engine</div>
        <div className="grid gap-6 lg:grid-cols-[1.05fr_0.95fr]">
          <div className="space-y-4 rounded-[32px] border border-zinc-800 bg-zinc-950/60 p-8 md:p-10">
            <h1 className="text-4xl font-semibold tracking-tight text-zinc-100 md:text-5xl">
              Metodologia, operação e prova visual do motor
            </h1>
            <p className="max-w-2xl text-base leading-8 text-zinc-300">
              O Eigen Engine trata o mercado como um sistema coletivo. Em vez de tentar adivinhar um ativo isolado,
              ele mede quando a estrutura do mercado fica mais concentrada, mais frágil ou mais saudável, e transforma
              isso em orçamento de risco, escolha de sleeves e publicação auditável.
            </p>
            <div className="grid gap-3 sm:grid-cols-2">
              <div className="rounded-2xl border border-zinc-800 bg-zinc-950/60 p-4">
                <div className="flex items-center gap-2 text-[11px] uppercase tracking-[0.16em] text-zinc-500">
                  <span>Sem look-ahead</span>
                  <HelpHint text="O motor usa apenas o que já era observável até cada data. Não usa dado futuro para classificar o presente." />
                </div>
                <div className="mt-2 text-sm text-zinc-300">
                  A causalidade é tratada como regra dura, não como detalhe de marketing.
                </div>
              </div>
              <div className="rounded-2xl border border-zinc-800 bg-zinc-950/60 p-4">
                <div className="flex items-center gap-2 text-[11px] uppercase tracking-[0.16em] text-zinc-500">
                  <span>Gate auditável</span>
                  <HelpHint text="Se cobertura, universo ou checks mínimos falham, a publicação trava automaticamente." />
                </div>
                <div className="mt-2 text-sm text-zinc-300">
                  Se a trilha fica fraca, o produto entra em modo diagnóstico em vez de fingir convicção.
                </div>
              </div>
            </div>
          </div>
          <aside className="rounded-[32px] border border-emerald-900/40 bg-emerald-950/10 p-8">
            <div className="text-xs uppercase tracking-[0.24em] text-emerald-300/80">Leitura visual</div>
            <h2 className="mt-4 text-2xl font-semibold tracking-tight text-zinc-100">
              A animação mostra como a correlação vira decisão prática
            </h2>
            <div className="mt-5 space-y-3 text-sm leading-7 text-zinc-300">
              <p>Primeiro entram as séries limpas.</p>
              <p>Depois a matriz evolui e o espectro mostra quando poucos fatores passam a mandar no mercado.</p>
              <p>Na sequência, o regime estrutural define o orçamento de risco e abre caminho para os insights.</p>
              <p>O objetivo não é enfeitar a página. É mostrar por que a matemática leva a uma decisão legível.</p>
            </div>
          </aside>
        </div>
      </section>

      <EngineStoryDeck />

      <PipelineFlow />

      <section className="grid gap-4 lg:grid-cols-[1.05fr_0.95fr]">
        <div className="rounded-3xl border border-zinc-800 bg-zinc-950/60 p-6">
          <div className="text-xs uppercase tracking-[0.3em] text-zinc-500">Pontos auditados</div>
          <div className="mt-5 grid gap-4 md:grid-cols-2">
            {pilares.map((item) => (
              <article key={item.titulo} className="rounded-2xl border border-zinc-800 bg-black/20 p-4">
                <h2 className="text-sm font-semibold text-zinc-100">{item.titulo}</h2>
                <p className="mt-2 text-sm leading-6 text-zinc-300">{item.texto}</p>
              </article>
            ))}
          </div>
        </div>

        <div className="rounded-3xl border border-emerald-900/40 bg-emerald-950/10 p-6">
          <div className="text-xs uppercase tracking-[0.3em] text-emerald-300/80">Garantias técnicas</div>
          <div className="mt-4 space-y-3 text-sm leading-7 text-zinc-200">
            <p>
              <span className="text-emerald-300">1. Causalidade:</span> o regime e os limiares são calculados apenas
              com histórico anterior à data da decisão.
            </p>
            <p>
              <span className="text-emerald-300">2. Auditabilidade:</span> cada run gera artefatos completos para
              reconstruir a leitura, o gate e o que foi mostrado na interface.
            </p>
            <p>
              <span className="text-emerald-300">3. Bloqueio automático:</span> publicação ruim não vira vitrine limpa
              por conveniência comercial.
            </p>
            <p>
              <span className="text-emerald-300">4. Incerteza declarada:</span> o produto mostra confiança, frescor e
              limites de uso em vez de vender certeza onde não existe.
            </p>
          </div>
          <div className="mt-6 rounded-2xl border border-zinc-800 bg-black/20 p-4 text-sm text-zinc-300">
            Em finanças, a matemática serve para medir organização coletiva do mercado. Em cripto, ela ajuda a
            identificar quando o sleeve agressivo está com breadth e tração reais, e quando está só surfando ruído.
          </div>
        </div>
      </section>

      <section className="rounded-3xl border border-zinc-800 bg-zinc-950/60 p-6">
        <div className="flex items-center gap-2 text-xs uppercase tracking-[0.3em] text-zinc-500">
          <span>Métodos por camada</span>
          <HelpHint text="Cada camada responde a uma pergunta diferente: contexto, sleeve, execução e proteção." />
        </div>
        <div className="mt-5 grid gap-4 lg:grid-cols-3">
          {camadas.map((item) => (
            <article key={item.titulo} className="rounded-2xl border border-zinc-800 bg-black/20 p-5">
              <h3 className="text-lg font-semibold text-zinc-100">{item.titulo}</h3>
              <div className="mt-4 space-y-3 text-sm leading-6 text-zinc-300">
                <p>
                  <span className="text-zinc-400">Dados:</span> {item.dados}
                </p>
                <p>
                  <span className="text-zinc-400">Janelas:</span> {item.janelas}
                </p>
                <p>
                  <span className="text-zinc-400">Validação:</span> {item.validacao}
                </p>
                <p>
                  <span className="text-zinc-400">Artefatos:</span> {item.artefatos}
                </p>
              </div>
            </article>
          ))}
        </div>
      </section>

      <section className="grid gap-4 lg:grid-cols-[0.95fr_1.05fr]">
        <div className="rounded-3xl border border-zinc-800 bg-zinc-950/60 p-6">
          <div className="text-xs uppercase tracking-[0.3em] text-zinc-500">Guias ativos</div>
          <div className="mt-5 space-y-4">
            {guias.map((item) => (
              <a
                key={item.titulo}
                href={item.href}
                target="_blank"
                rel="noreferrer"
                className="block rounded-2xl border border-zinc-800 bg-black/20 p-4 transition hover:border-zinc-600"
              >
                <h2 className="text-sm font-semibold text-zinc-100">{item.titulo}</h2>
                <p className="mt-2 text-sm leading-6 text-zinc-300">{item.detalhe}</p>
              </a>
            ))}
          </div>
        </div>

        <div className="rounded-3xl border border-zinc-800 bg-zinc-950/60 p-6">
          <div className="text-xs uppercase tracking-[0.3em] text-zinc-500">Fontes de dados usadas</div>
          <div className="mt-5 grid gap-4 md:grid-cols-2">
            {fontes.map((item) => (
              <article key={item.nome} className="rounded-2xl border border-zinc-800 bg-black/20 p-4">
                <h3 className="text-sm font-semibold text-zinc-100">{item.nome}</h3>
                <p className="mt-2 text-sm leading-6 text-zinc-300">{item.detalhe}</p>
              </article>
            ))}
          </div>
        </div>
      </section>

      <section className="grid gap-4 lg:grid-cols-[0.85fr_1.15fr]">
        <div className="rounded-3xl border border-zinc-800 bg-zinc-950/60 p-6">
          <div className="text-xs uppercase tracking-[0.3em] text-zinc-500">Limites de uso</div>
          <ul className="mt-5 space-y-2 text-sm leading-6 text-zinc-300">
            {limites.map((item) => (
              <li key={item} className="rounded-2xl border border-zinc-800 bg-black/20 px-4 py-3">
                {item}
              </li>
            ))}
          </ul>
          <div className="mt-5 text-sm text-zinc-400">
            A função do Eigen Engine é organizar incerteza. Quando o produto parece “mais vendável”, o teste correto é
            verificar se ele continua auditável, não se o texto ficou mais bonito.
          </div>
        </div>

        <div className="rounded-3xl border border-zinc-800 bg-zinc-950/60 p-6">
          <div className="text-xs uppercase tracking-[0.3em] text-zinc-500">Referências metodológicas</div>
          <div className="mt-5 grid gap-3 md:grid-cols-2">
            {referencias.map((item) => (
              <Link
                key={item.id}
                href={item.href}
                target="_blank"
                rel="noreferrer"
                className="rounded-2xl border border-zinc-800 bg-black/20 p-4 transition hover:border-zinc-600"
              >
                <div className="text-[10px] uppercase tracking-[0.16em] text-zinc-500">{item.id}</div>
                <div className="mt-2 text-sm text-zinc-200">{item.titulo}</div>
              </Link>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}
