import SectorDashboard from "@/components/SectorDashboard";
import SignalSnapshotSection from "@/components/site/SignalSnapshotSection";
import {
  humanizeEngineState,
  humanizeMethodology,
  humanizeRiskLevel,
  humanizeStatusWord,
  humanizeStrategyName,
} from "@/lib/enginePresentation";
import { readSiteFinanceSnapshot } from "@/lib/server/data";
import Link from "next/link";

function formatPct(value: number | null | undefined) {
  return typeof value === "number" && Number.isFinite(value) ? `${(value * 100).toFixed(0)}%` : "n/d";
}

function describeRiskLevel(level: string) {
  const normalized = String(level || "").toLowerCase();
  if (normalized.includes("high") || normalized.includes("alto")) {
    return "O motor está lendo risco alto. A prioridade vira defesa e redução de agressividade.";
  }
  if (normalized.includes("medium") || normalized.includes("medio") || normalized.includes("moderado")) {
    return "O motor está em zona intermediária. Dá para operar, mas sem all-in e sem pressa.";
  }
  if (normalized.includes("low") || normalized.includes("baixo")) {
    return "O motor está lendo risco mais controlado. Ainda assim, posição grande demais continua sendo erro.";
  }
  return "O motor está em leitura de monitoramento. Use isso como contexto, não como impulso.";
}

function describePublishable(blocked: boolean, publishable: boolean) {
  if (blocked || !publishable) {
    return "Hoje a leitura serve mais para diagnóstico e shadow do que para confiar cegamente na execução.";
  }
  return "Hoje a trilha operacional está íntegra. Ainda não é garantia de lucro, mas o contexto está mais limpo.";
}

function describeExposure(exposure: number | null | undefined) {
  if (typeof exposure !== "number" || !Number.isFinite(exposure)) {
    return "Sem faixa clara de exposição hoje. Melhor operar pequeno ou não operar.";
  }
  if (exposure <= 0.25) {
    return `Exposição alvo perto de ${formatPct(exposure)}. Isso é postura defensiva, quase caixa.`;
  }
  if (exposure <= 0.6) {
    return `Exposição alvo perto de ${formatPct(exposure)}. Isso é mão moderada, sem heroísmo.`;
  }
  return `Exposição alvo perto de ${formatPct(exposure)}. O motor aceita mais risco, mas ainda com disciplina.`;
}

export default async function DashboardPage() {
  const snapshot = (await readSiteFinanceSnapshot()) as Record<string, unknown>;
  const finance = ((snapshot.finance as Record<string, unknown> | undefined) || {}) as Record<string, unknown>;
  const playbook = ((finance.latest_playbook as Record<string, unknown> | undefined) || {}) as Record<string, unknown>;
  const research = ((snapshot.profit_research as Record<string, unknown> | undefined) || {}) as Record<string, unknown>;
  const topCandidate = ((research.top_candidate as Record<string, unknown> | undefined) || {}) as Record<string, unknown>;
  const shadow = ((snapshot.shadow as Record<string, unknown> | undefined) || {}) as Record<string, unknown>;
  const shadowLatest = ((shadow.latest as Record<string, unknown> | undefined) || {}) as Record<string, unknown>;
  const exposure =
    typeof playbook.exposure === "number"
      ? playbook.exposure
      : typeof shadowLatest.target_exposure === "number"
        ? shadowLatest.target_exposure
        : null;
  const grossIdea =
    typeof exposure === "number" && Number.isFinite(exposure) ? Math.max(0, Math.round(10000 * exposure)) : null;
  const cashIdea =
    typeof grossIdea === "number" ? Math.max(0, 10000 - grossIdea) : null;
  const riskLevel = humanizeRiskLevel(finance.risk_level_next_month);
  const regime = humanizeEngineState(playbook.regime || "monitoramento_normal");
  const operationalState = humanizeEngineState(finance.operational_state || playbook.regime || "monitoramento_normal");
  const dataBase = String(snapshot.as_of_date || finance.data_last_date || shadowLatest.price_date || "n/d");
  const publishable = finance.gate_blocked !== true;
  const signalReliability =
    typeof playbook.signal_reliability === "number"
      ? playbook.signal_reliability
      : typeof finance.confidence_score === "number"
        ? finance.confidence_score
        : null;

  return (
    <div className="space-y-5">
      <section className="border-b border-zinc-800/80 px-5 py-5 md:px-6 md:py-6">
        <div className="text-xs uppercase tracking-[0.18em] text-zinc-500">Resumo de hoje</div>
        <h1 className="mt-2 text-2xl font-semibold text-zinc-100 md:text-3xl">O que o Eigen Engine está dizendo agora</h1>
        <p className="mt-3 max-w-3xl text-sm text-zinc-300">
          Aqui a ideia é simples: o motor resume o risco do mercado, sugere uma faixa de exposição e mostra se o
          contexto está limpo ou perigoso. Nada de linguagem de fundo quantitativo para esconder o básico.
        </p>
      </section>

      <section className="grid gap-4 px-5 md:grid-cols-2 md:px-6">
        <article className="rounded-2xl border border-zinc-800 bg-zinc-950/55 p-5">
          <div className="text-xs uppercase tracking-[0.16em] text-zinc-500">Estado</div>
          <div className="mt-2 text-xl font-semibold text-zinc-100">{operationalState}</div>
          <p className="mt-3 text-sm text-zinc-300">{describeRiskLevel(riskLevel)}</p>
          <p className="mt-3 text-sm text-zinc-400">
            Risco do próximo mês: <span className="text-zinc-200">{riskLevel}</span>
            {" · "}
            Regime estrutural: <span className="text-zinc-200">{regime}</span>
          </p>
          <p className="mt-2 text-sm text-zinc-400">
            Confiabilidade publicada: <span className="text-zinc-200">{signalReliability == null ? "sem dado publicado" : `${Math.round(signalReliability * 100)}%`}</span>
          </p>
        </article>

        <article className="rounded-2xl border border-zinc-800 bg-zinc-950/55 p-5">
          <div className="text-xs uppercase tracking-[0.16em] text-zinc-500">Exposição</div>
          <div className="mt-2 text-xl font-semibold text-zinc-100">{formatPct(exposure)}</div>
          <p className="mt-3 text-sm text-zinc-300">{describeExposure(exposure)}</p>
          <p className="mt-3 text-sm text-zinc-400">
            Exemplo didático com R$ 10.000: risco <span className="text-zinc-200">{grossIdea == null ? "n/d" : `R$ ${grossIdea.toLocaleString("pt-BR")}`}</span>
            {" · "}
            caixa/defensivo <span className="text-zinc-200">{cashIdea == null ? "n/d" : `R$ ${cashIdea.toLocaleString("pt-BR")}`}</span>
          </p>
        </article>

        <article className="rounded-2xl border border-zinc-800 bg-zinc-950/55 p-5">
          <div className="text-xs uppercase tracking-[0.16em] text-zinc-500">Integridade</div>
          <div className="mt-2 text-xl font-semibold text-zinc-100">
            {publishable ? "Publicável" : "Modo diagnóstico"}
          </div>
          <p className="mt-3 text-sm text-zinc-300">{describePublishable(finance.gate_blocked === true, publishable)}</p>
          <p className="mt-3 text-sm text-zinc-400">
            Gate bloqueado: <span className="text-zinc-200">{finance.gate_blocked === true ? "sim" : "não"}</span>
            {" · "}
            Data-base: <span className="text-zinc-200">{dataBase}</span>
          </p>
          <p className="mt-2 text-sm text-zinc-400">
            Readiness: <span className="text-zinc-200">{humanizeStatusWord(finance.overall_readiness || "n/d")}</span>
          </p>
        </article>

        <article className="rounded-2xl border border-zinc-800 bg-zinc-950/55 p-5">
          <div className="text-xs uppercase tracking-[0.16em] text-zinc-500">Pesquisa</div>
          <div className="mt-2 text-xl font-semibold text-zinc-100">
            {humanizeStrategyName(topCandidate.candidate_id || topCandidate.label || "sem topo")}
          </div>
          <p className="mt-3 text-sm text-zinc-300">
            O laboratório continua procurando alpha, mas só o que sobrevive em shadow e walk-forward merece tempo.
          </p>
          <p className="mt-3 text-sm text-zinc-400">
            Método: <span className="text-zinc-200">{humanizeMethodology(topCandidate.methodology || "n/d")}</span>
            {" · "}
            Status: <span className="text-zinc-200">{humanizeStatusWord(topCandidate.status || "n/d")}</span>
          </p>
        </article>
      </section>

      <div className="px-5 md:px-6">
        <SignalSnapshotSection snapshot={snapshot as Record<string, unknown>} compact />
      </div>

      <section className="px-5 md:px-6">
        <div className="rounded-2xl border border-cyan-900/40 bg-cyan-950/15 p-5">
          <div className="text-xs uppercase tracking-[0.16em] text-cyan-300/80">Traduzindo para gente normal</div>
          <div className="mt-3 space-y-2 text-sm text-zinc-200">
            <p>- Se o risco sobe, o motor corta agressividade antes de tentar adivinhar o próximo candle.</p>
            <p>- Se o contexto está limpo, ele libera mais exposição, mas nunca transforma isso em certeza.</p>
            <p>- Se o gate trava, a leitura continua útil para estudo, mas não para confiança cega.</p>
          </div>
          <div className="mt-4">
            <Link
              href="/app/aplicacoes"
              className="inline-flex rounded-lg border border-cyan-300/35 bg-white/5 px-4 py-2 text-sm text-zinc-100 hover:border-cyan-200"
            >
              Ver passo a passo de uso
            </Link>
          </div>
        </div>
      </section>

      <SectorDashboard
        title="Eigen Engine | Dashboard operacional"
        showTable
        initialDomain="finance"
        headline="Plataforma operacional"
        description="Visão unificada de finanças, cripto, shadow e estabilidade por ativo."
      />
    </div>
  );
}
