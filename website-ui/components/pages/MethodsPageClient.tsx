import Link from "next/link";
import MethodLattice from "@/components/visuals/MethodLattice";
import PipelineFlow from "@/components/visuals/PipelineFlow";

const pilares = [
  {
    titulo: "Análise Causal Walk-Forward",
    texto:
      "Os limiares de classificação são recalibrados usando apenas dados anteriores a cada data. Não há uso de informação futura no cálculo do regime.",
  },
  {
    titulo: "Winsorização de Outliers",
    texto:
      "Os retornos são winsorizados em janela de 252 dias, com cortes em 0,5% e 99,5%, para reduzir distorções de choques extremos.",
  },
  {
    titulo: "Janela Oficial T120",
    texto:
      "A janela principal de produção é T120. As janelas T60 e T252 ficam como apoio para testes de robustez e comparação de sensibilidade.",
  },
  {
    titulo: "Gate de Publicação",
    texto:
      "Cada run passa por checks mínimos de cobertura, tamanho de universo e qualidade. Se algum critério falhar, a publicação é bloqueada automaticamente.",
  },
  {
    titulo: "Robustez Quantitativa",
    texto:
      "O Eigen Engine roda testes de subamostragem de 10%, sensibilidade de parâmetros e bootstrap em blocos para intervalos de confiança das métricas.",
  },
  {
    titulo: "Modo Mínimo Vendável",
    texto:
      "A interface pública opera em modo simplificado com foco em Finanças. Módulos avançados setoriais permanecem no código e podem ser reativados por feature flag.",
  },
  {
    titulo: "Garantias Técnicas",
    texto:
      "O produto mantém três garantias de base: causalidade (sem look-ahead), auditabilidade por artefatos completos de run e bloqueio de publicação em caso de falha.",
  },
];

const limites = [
  "Sem promessa de retorno e sem recomendação de compra ou venda.",
  "Uso focado em diagnóstico de risco estrutural e governança quantitativa.",
  "Choques exógenos podem reduzir antecedência de alerta.",
  "Resultados históricos não garantem desempenho futuro.",
];

const roadmap = [
  {
    etapa: "Fase 1 (já ativo)",
    foco: "Causalidade, gate e trilha auditável",
    entrega:
      "Eigen Engine com classificação de regime causal, bloqueio de publicação automático e artefatos versionados por run.",
  },
  {
    etapa: "Fase 2 (em andamento)",
    foco: "Robustez e estabilidade",
    entrega:
      "Mais testes de sensibilidade de parâmetros, ajuste de histerese e redução de falso alerta em cenários reais.",
  },
  {
    etapa: "Fase 3 (próxima)",
    foco: "Explicabilidade por ativo e setor",
    entrega:
      "Leituras mais simples por ativo, resumo setorial acionável e histórico comparável de mudanças de regime.",
  },
  {
    etapa: "Fase 4 (produto pleno)",
    foco: "Operação diária institucional",
    entrega:
      "Rotina diária consolidada com comparação dia a dia, métricas de acerto e documentação pronta para auditoria.",
  },
];

const dataSources = [
  {
    nome: "Preços financeiros",
    detalhe:
      "Séries de fechamento diário por ativo (CSV local no repositório), usadas para cálculo de retornos e estrutura de correlação.",
  },
  {
    nome: "Mapa de universo e setores",
    detalhe:
      "Arquivos de universo fixo e classificação setorial para manter consistência de cobertura e comparação histórica.",
  },
  {
    nome: "Artefatos do Eigen Engine",
    detalhe:
      "Saídas versionadas em cada run (timeseries, regime series, playbook, QA, gate e diagnósticos por ativo/setor).",
  },
  {
    nome: "Verificação de robustez",
    detalhe:
      "Resultados de bootstrap, subamostragem e comparação de janelas, publicados como artefatos de evidência.",
  },
];

const methodReferences = [
  {
    id: "M1",
    title: "Financial Applications of Random Matrix Theory (arXiv:0910.1205)",
    href: "https://arxiv.org/abs/0910.1205",
  },
  {
    id: "M2",
    title: "Principal Components as a Measure of Systemic Risk (MIT)",
    href: "https://web.mit.edu/~finlunch/Fall10/PCASystemicRisk.pdf",
  },
  {
    id: "M3",
    title: "Estimation of Large Financial Covariances (arXiv)",
    href: "https://arxiv.org/abs/1909.12064",
  },
  {
    id: "M4",
    title: "Walk-forward optimization (conceito)",
    href: "https://en.wikipedia.org/wiki/Walk_forward_optimization",
  },
  {
    id: "M5",
    title: "Block bootstrapping technique (scores docs)",
    href: "https://scores.readthedocs.io/en/stable/tutorials/Block_Bootstrapping.html",
  },
  {
    id: "M6",
    title: "Basel III LCR (BIS)",
    href: "https://www.bis.org/publ/bcbs238.htm",
  },
  {
    id: "M7",
    title: "SR 11-7 Model Risk Management (Federal Reserve)",
    href: "https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm",
  },
  {
    id: "M8",
    title: "FRTB revised market risk framework (BIS)",
    href: "https://www.bis.org/bcbs/publ/d305.htm",
  },
];

const guides = [
  {
    titulo: "Guia operacional diário",
    detalhe: "Checklist único de execução, gate e publicação para o Eigen Engine.",
    href: "https://github.com/pemodest0/Assyntrax/blob/main/docs/operacao/CHECKLIST_OPERACAO_EIGEN_ENGINE.md",
  },
  {
    titulo: "API externa de piloto",
    detalhe: "Contrato de integração técnica para clientes em piloto (chave + payload v1).",
    href: "https://github.com/pemodest0/Assyntrax/blob/main/docs/venda/API_PILOTO_EXTERNA.md",
  },
  {
    titulo: "Pacote de piloto",
    detalhe: "Material executivo/técnico consolidado para piloto de 30 dias.",
    href: "https://github.com/pemodest0/Assyntrax/blob/main/docs/venda/PACOTE_PILOTO_ASSYNTRAX_EIGEN_ENGINE.md",
  },
];

export default function MethodsPageClient() {
  return (
    <div className="space-y-10">
      <section className="grid grid-cols-1 lg:grid-cols-[1.05fr_0.95fr] gap-10 items-center">
        <div className="space-y-3">
          <div className="text-xs uppercase tracking-[0.3em] text-zinc-400">Eigen Engine</div>
          <h1 className="text-4xl md:text-5xl font-semibold tracking-tight">
            Eigen Engine: metodologia, operação e guias
          </h1>
          <p className="text-zinc-300 max-w-3xl text-lg">
            O Eigen Engine da Assyntrax aplica análise espectral de correlações para identificar mudança estrutural do mercado.
            A arquitetura prioriza causalidade, robustez estatística e controle de publicação.
          </p>
        </div>
        <div className="overflow-hidden rounded-[28px]">
          <MethodLattice />
        </div>
      </section>

      <PipelineFlow />

      <section className="rounded-2xl border border-zinc-800 bg-zinc-950/60 p-6">
        <div className="text-xs uppercase tracking-[0.3em] text-zinc-500">Guias ativos</div>
        <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
          {guides.map((item) => {
            const isInternal = item.href.startsWith("/");
            if (isInternal) {
              return (
                <Link key={item.titulo} href={item.href} className="rounded-2xl border border-zinc-800 bg-zinc-900/40 p-5 hover:border-zinc-600 transition">
                  <h2 className="text-sm font-semibold tracking-wide text-zinc-100">{item.titulo}</h2>
                  <p className="mt-2 text-sm text-zinc-300 leading-relaxed">{item.detalhe}</p>
                </Link>
              );
            }
            return (
              <a
                key={item.titulo}
                href={item.href}
                target="_blank"
                rel="noreferrer"
                className="rounded-2xl border border-zinc-800 bg-zinc-900/40 p-5 hover:border-zinc-600 transition"
              >
                <h2 className="text-sm font-semibold tracking-wide text-zinc-100">{item.titulo}</h2>
                <p className="mt-2 text-sm text-zinc-300 leading-relaxed">{item.detalhe}</p>
              </a>
            );
          })}
        </div>
      </section>

      <section className="rounded-2xl border border-zinc-800 bg-zinc-950/60 p-6">
        <div className="text-xs uppercase tracking-[0.3em] text-zinc-500">Pontos auditados</div>
        <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
          {pilares.map((item) => (
            <article key={item.titulo} className="rounded-2xl border border-zinc-800 bg-zinc-900/40 p-5">
              <h2 className="text-sm font-semibold tracking-wide text-zinc-100">{item.titulo}</h2>
              <p className="mt-2 text-sm text-zinc-300 leading-relaxed">{item.texto}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="rounded-2xl border border-zinc-800 bg-zinc-950/60 p-6">
        <div className="text-xs uppercase tracking-[0.3em] text-zinc-500">Limites de uso</div>
        <ul className="mt-3 space-y-2 text-sm text-zinc-300">
          {limites.map((item) => (
            <li key={item}>- {item}</li>
          ))}
        </ul>
      </section>

      <section className="rounded-2xl border border-zinc-800 bg-zinc-950/60 p-6">
        <div className="text-xs uppercase tracking-[0.3em] text-zinc-500">Roadmap até produto pleno</div>
        <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
          {roadmap.map((item) => (
            <article key={item.etapa} className="rounded-2xl border border-zinc-800 bg-zinc-900/40 p-5">
              <h3 className="text-sm font-semibold tracking-wide text-zinc-100">{item.etapa}</h3>
              <p className="mt-1 text-xs uppercase tracking-[0.14em] text-zinc-500">{item.foco}</p>
              <p className="mt-2 text-sm text-zinc-300 leading-relaxed">{item.entrega}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="rounded-2xl border border-zinc-800 bg-zinc-950/60 p-6">
        <div className="text-xs uppercase tracking-[0.3em] text-zinc-500">Fontes de dados usadas</div>
        <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
          {dataSources.map((src) => (
            <article key={src.nome} className="rounded-2xl border border-zinc-800 bg-zinc-900/40 p-5">
              <h3 className="text-sm font-semibold text-zinc-100">{src.nome}</h3>
              <p className="mt-2 text-sm text-zinc-300 leading-relaxed">{src.detalhe}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="rounded-2xl border border-zinc-800 bg-zinc-950/60 p-6">
        <div className="text-xs uppercase tracking-[0.3em] text-zinc-500">Referências metodológicas</div>
        <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-3">
          {methodReferences.map((ref) => {
            const href = String(ref.href || "").trim();
            return href ? (
              <a
                key={ref.id}
                href={href}
                target="_blank"
                rel="noreferrer"
                className="rounded-2xl border border-zinc-800 bg-zinc-900/40 p-4 hover:border-zinc-600 transition"
              >
                <div className="text-[10px] uppercase tracking-[0.16em] text-zinc-500">{ref.id}</div>
                <div className="mt-1 text-sm text-zinc-200">{ref.title}</div>
              </a>
            ) : (
              <div key={ref.id} className="rounded-2xl border border-zinc-800 bg-zinc-900/40 p-4">
                <div className="text-[10px] uppercase tracking-[0.16em] text-zinc-500">{ref.id}</div>
                <div className="mt-1 text-sm text-zinc-200">{ref.title}</div>
                <div className="mt-1 text-xs text-zinc-500">Link indisponível.</div>
              </div>
            );
          })}
        </div>
      </section>
    </div>
  );
}
