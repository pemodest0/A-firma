import Link from "next/link";
import HelpHint from "@/components/ui/HelpHint";
import {
  describeStrategy,
  humanizeGroupName,
  humanizeMethodology,
  humanizeStrategyName,
} from "@/lib/enginePresentation";
import { readSiteFinanceSnapshot } from "@/lib/server/data";

function pct(value: unknown, digits = 1) {
  const n = Number(value);
  return Number.isFinite(n) ? `${(n * 100).toFixed(digits)}%` : "n/d";
}

function countByDomain(rows: Record<string, unknown>[], domain: string) {
  return rows.filter((row) => String(row.domain || "") === domain).length;
}

export default async function EvidenciasLandingPage() {
  const snapshot = (await readSiteFinanceSnapshot()) as Record<string, unknown>;
  const universe = Array.isArray(snapshot.current_universe) ? (snapshot.current_universe as Record<string, unknown>[]) : [];
  const research = ((snapshot.profit_research as Record<string, unknown> | undefined) || {}) as Record<string, unknown>;
  const layered = ((snapshot.layered_engine as Record<string, unknown> | undefined) || {}) as Record<string, unknown>;
  const proof = ((snapshot.proof as Record<string, unknown> | undefined) || {}) as Record<string, unknown>;
  const universeExpansion = ((snapshot.universe_expansion as Record<string, unknown> | undefined) || {}) as Record<string, unknown>;
  const attack = ((layered.best_meta_candidate as Record<string, unknown> | undefined) || {}) as Record<string, unknown>;
  const robust = ((layered.drawdown_best_balanced as Record<string, unknown> | undefined) || {}) as Record<string, unknown>;
  const topCandidate = ((research.top_candidate as Record<string, unknown> | undefined) || {}) as Record<string, unknown>;
  const proofResearch = ((proof.group_suite_best as Record<string, unknown> | undefined) || {}) as Record<string, unknown>;
  const historyWindow = ((proof.history_window as Record<string, unknown> | undefined) || {}) as Record<string, unknown>;
  const universe800 = ((universeExpansion.target_800_cov075 as Record<string, unknown> | undefined) || {}) as Record<string, unknown>;
  const patternHeadlines = Array.isArray(research.pattern_headlines) ? (research.pattern_headlines as string[]) : [];
  const topOos = ((topCandidate.oos as Record<string, unknown> | undefined) || {}) as Record<string, unknown>;

  const cards = [
    { label: "Finanças", value: countByDomain(universe, "finance"), helper: "Ativos no painel financeiro publicado." },
    { label: "Cripto", value: countByDomain(universe, "crypto"), helper: "Moedas líquidas no watchlist publicado." },
    { label: "Pesquisa viva", value: Number(research.event_count || 0), helper: "Padrões recentes registrados pelo copiloto." },
  ];

  return (
    <main className="space-y-10 py-8 md:py-10">
      <section className="rounded-[32px] border border-zinc-800 bg-zinc-950/60 p-8 md:p-10">
        <div className="text-xs uppercase tracking-[0.24em] text-cyan-300/80">Evidências</div>
        <h1 className="mt-4 text-4xl font-semibold tracking-tight text-zinc-100 md:text-5xl">
          O que o motor está provando hoje, sem maquiagem
        </h1>
        <p className="mt-5 max-w-4xl text-base leading-8 text-zinc-300">
          Esta página mostra só o que está vivo e auditável: janela longa, líquido versus bruto, universo ampliado,
          modos do motor e padrões que sobreviveram à pesquisa. Nada aqui deveria parecer bonito só porque ficou bem no
          design.
        </p>
        <div className="mt-6 grid gap-3 md:grid-cols-3">
          {cards.map((card) => (
            <article key={card.label} className="rounded-2xl border border-zinc-800 bg-black/20 p-4">
              <div className="flex items-center gap-2 text-[11px] uppercase tracking-[0.16em] text-zinc-500">
                <span>{card.label}</span>
                <HelpHint text={card.helper} />
              </div>
              <div className="mt-2 text-2xl font-semibold text-zinc-100">{card.value}</div>
              <div className="mt-2 text-xs text-zinc-500">{card.helper}</div>
            </article>
          ))}
        </div>
      </section>

      <section className="grid gap-4 lg:grid-cols-3">
        <article className="rounded-3xl border border-zinc-800 bg-zinc-950/60 p-6">
          <div className="text-xs uppercase tracking-[0.18em] text-zinc-500">Modo ataque</div>
          <h2 className="mt-3 text-2xl font-semibold text-zinc-100">{humanizeStrategyName(attack.candidate_id)}</h2>
          <div className="mt-4 space-y-2 text-sm text-zinc-300">
            <p>Retorno anual: {pct(attack.net_ann_return)}</p>
            <p>Sharpe: {Number(attack.net_sharpe || 0).toFixed(2)}</p>
            <p>Drawdown: {pct(attack.net_max_drawdown)}</p>
            <p className="text-zinc-400">{describeStrategy(attack.candidate_id, attack.notes)}</p>
          </div>
        </article>

        <article className="rounded-3xl border border-zinc-800 bg-zinc-950/60 p-6">
          <div className="text-xs uppercase tracking-[0.18em] text-zinc-500">Modo robusto</div>
          <h2 className="mt-3 text-2xl font-semibold text-zinc-100">{humanizeStrategyName(robust.candidate_id)}</h2>
          <div className="mt-4 space-y-2 text-sm text-zinc-300">
            <p>Retorno anual: {pct(robust.net_ann_return)}</p>
            <p>Sharpe: {Number(robust.net_sharpe || 0).toFixed(2)}</p>
            <p>Drawdown: {pct(robust.net_max_drawdown)}</p>
            <p className="text-zinc-400">{describeStrategy(robust.candidate_id, robust.notes)}</p>
          </div>
        </article>

        <article className="rounded-3xl border border-zinc-800 bg-zinc-950/60 p-6">
          <div className="text-xs uppercase tracking-[0.18em] text-zinc-500">Pesquisa líder</div>
          <h2 className="mt-3 text-2xl font-semibold text-zinc-100">
            {humanizeStrategyName(topCandidate.candidate_id || topCandidate.label)}
          </h2>
          <div className="mt-4 space-y-2 text-sm text-zinc-300">
            <p>Método: {humanizeMethodology(topCandidate.methodology)}</p>
            <p>Retorno líquido anual: {pct(topCandidate.net_ann_return)}</p>
            <p>
              Grupos: {String(topCandidate.groups || "").split(",").filter(Boolean).map((item) => humanizeGroupName(item)).join(" · ")}
            </p>
          </div>
        </article>
      </section>

      <section className="grid gap-4 lg:grid-cols-3">
        <article className="rounded-3xl border border-zinc-800 bg-zinc-950/60 p-6">
          <div className="text-xs uppercase tracking-[0.18em] text-zinc-500">Prova desde 2016</div>
          <div className="mt-3 text-2xl font-semibold text-zinc-100">
            {String(historyWindow.start || "2016-02-18")} → {String(historyWindow.end || snapshot.as_of_date || "n/d")}
          </div>
          <p className="mt-4 text-sm leading-7 text-zinc-300">
            A vitrine usa uma janela longa para separar consistência de surto de curto prazo.
          </p>
        </article>

        <article className="rounded-3xl border border-zinc-800 bg-zinc-950/60 p-6">
          <div className="text-xs uppercase tracking-[0.18em] text-zinc-500">Líquido e bruto</div>
          <div className="mt-3 text-2xl font-semibold text-zinc-100">
            {humanizeStrategyName(proofResearch.candidate_id || topCandidate.candidate_id)}
          </div>
          <div className="mt-4 space-y-2 text-sm text-zinc-300">
            <p>Líquido anual: {pct(proofResearch.net_blended_ann_return || proofResearch.net_ann_return || topCandidate.net_ann_return)}</p>
            <p>Bruto anual: {pct(proofResearch.gross_ann_return || topCandidate.gross_ann_return)}</p>
            <p className="text-zinc-400">Custos e impostos entram como proxy conservadora nos artefatos publicados.</p>
          </div>
        </article>

        <article className="rounded-3xl border border-zinc-800 bg-zinc-950/60 p-6">
          <div className="text-xs uppercase tracking-[0.18em] text-zinc-500">Universo 800 ativos</div>
          <div className="mt-3 text-2xl font-semibold text-zinc-100">
            {String(universe800.assets_ok || "n/d")} ativos · {String(universe800.sector_count || "n/d")} setores
          </div>
          <div className="mt-4 space-y-2 text-sm text-zinc-300">
            <p>Maior setor: {pct(universe800.largest_sector_share)}</p>
            <p>Painel histórico: {Number(universe800.panel_rows || 0).toLocaleString("pt-BR")} linhas</p>
            <p className="text-zinc-400">Mais breadth e menos risco de ficar preso a um setor só.</p>
          </div>
        </article>
      </section>

      <section className="grid gap-4 lg:grid-cols-3">
        <article className="rounded-3xl border border-zinc-800 bg-zinc-950/60 p-6">
          <div className="flex items-center gap-2 text-xs uppercase tracking-[0.18em] text-zinc-500">
            <span>Confiança operacional</span>
            <HelpHint text="Percentual de blocos de teste em que o modo robusto continuou positivo ou útil. Não é garantia; é um resumo de consistência." />
          </div>
          <div className="mt-3 text-2xl font-semibold text-zinc-100">{pct(robust.positive_test_share, 0)}</div>
          <p className="mt-4 text-sm leading-7 text-zinc-300">
            Quanto mais alto esse número, mais vezes o modo robusto sustentou o teste sem depender de um único recorte.
          </p>
        </article>

        <article className="rounded-3xl border border-zinc-800 bg-zinc-950/60 p-6">
          <div className="flex items-center gap-2 text-xs uppercase tracking-[0.18em] text-zinc-500">
            <span>Incerteza da pesquisa</span>
            <HelpHint text="Blocos OOS medem se o candidato continua bom fora do treino. Menos blocos significam mais incerteza." />
          </div>
          <div className="mt-3 text-2xl font-semibold text-zinc-100">{String(topOos.appearances || "n/d")} blocos</div>
          <p className="mt-4 text-sm leading-7 text-zinc-300">
            O candidato líder apareceu de forma consistente em {String(topOos.appearances || "n/d")} blocos fora da amostra, com média de {pct(topOos.mean_test_net_ann_return)} ao ano.
          </p>
        </article>

        <article className="rounded-3xl border border-zinc-800 bg-zinc-950/60 p-6">
          <div className="flex items-center gap-2 text-xs uppercase tracking-[0.18em] text-zinc-500">
            <span>Melhora média vs benchmark</span>
            <HelpHint text="Edge médio nos blocos de teste. Serve para medir se a vantagem aparece repetidamente, não só num período bonito." />
          </div>
          <div className="mt-3 text-2xl font-semibold text-zinc-100">{pct(robust.mean_test_edge)}</div>
          <p className="mt-4 text-sm leading-7 text-zinc-300">
            Esse número resume a folga média do modo robusto nos blocos de teste. É o jeito mais honesto de falar em vantagem sem vender certeza.
          </p>
        </article>
      </section>

      <section className="rounded-3xl border border-zinc-800 bg-zinc-950/60 p-6 md:p-8">
        <div className="flex items-center gap-2 text-xs uppercase tracking-[0.24em] text-zinc-500">
          <span>Padrões recentes do copiloto</span>
          <HelpHint text="São leituras registradas pelo copiloto a partir da pesquisa publicada. Não são promessas de retorno." />
        </div>
        <div className="mt-5 grid gap-3 md:grid-cols-2">
          {patternHeadlines.slice(0, 10).map((item) => (
            <div key={item} className="rounded-2xl border border-zinc-800 bg-black/20 p-4 text-sm leading-6 text-zinc-300">
              {item}
            </div>
          ))}
        </div>
        <div className="mt-6 flex flex-wrap gap-3">
          <Link href="/app/financas" className="rounded-xl bg-cyan-400 px-5 py-3 text-sm font-medium text-zinc-950 hover:bg-cyan-300">
            Abrir finanças no app
          </Link>
          <Link href="/methods" className="rounded-xl border border-zinc-700 px-5 py-3 text-sm text-zinc-200 hover:border-zinc-500">
            Ver metodologia completa
          </Link>
        </div>
      </section>
    </main>
  );
}
