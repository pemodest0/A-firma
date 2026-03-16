import SectorDashboard from "@/components/SectorDashboard";
import SignalSnapshotSection from "@/components/site/SignalSnapshotSection";
import {
  humanizeConfidenceLevel,
  humanizeEngineState,
  humanizeMethodology,
  humanizeModeName,
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

function describeRecommendedMode(mode: string) {
  const normalized = String(mode || "").toLowerCase();
  if (normalized.includes("ataque")) {
    return "Hoje o motor ainda aceita uma postura mais ofensiva, mas sem tratar isso como certeza.";
  }
  if (normalized.includes("prote")) {
    return "Hoje faz mais sentido usar a versão protegida do motor e evitar heroísmo.";
  }
  return "Hoje a melhor postura é seguir o modo que o motor está recomendando e monitorar a vigilância.";
}

function asNumber(value: unknown) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function formatBRL(value: unknown, digits = 0) {
  const numeric = asNumber(value);
  if (numeric == null) return "n/d";
  return numeric.toLocaleString("pt-BR", {
    style: "currency",
    currency: "BRL",
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function formatSignedPct(value: number | null, digits = 1) {
  if (value == null || !Number.isFinite(value)) return "n/d";
  const sign = value > 0 ? "+" : "";
  return `${sign}${(value * 100).toFixed(digits)}%`;
}

function titleCase(raw: string) {
  return String(raw || "")
    .replace(/_/g, " ")
    .split(" ")
    .filter(Boolean)
    .map((chunk) => chunk.charAt(0).toUpperCase() + chunk.slice(1))
    .join(" ");
}

function marketTone(state: string) {
  const normalized = String(state || "").toLowerCase();
  if (normalized.includes("attack")) return "border-emerald-600/60 bg-emerald-500/15 text-emerald-100";
  if (normalized.includes("risk")) return "border-sky-600/60 bg-sky-500/15 text-sky-100";
  if (normalized.includes("transition")) return "border-amber-600/60 bg-amber-500/15 text-amber-100";
  return "border-zinc-700 bg-zinc-900/80 text-zinc-200";
}

function godColor(alias: string) {
  if (alias === "Apollo") return "#f59e0b";
  if (alias === "Zeus") return "#38bdf8";
  if (alias === "Hephaestus") return "#fb923c";
  if (alias === "Hermes") return "#34d399";
  return "#a1a1aa";
}

function DashboardGodArt({ alias }: { alias: string }) {
  const color = godColor(alias);
  if (alias === "Apollo") {
    return (
      <svg viewBox="0 0 120 88" className="h-20 w-24">
        <circle cx="60" cy="44" r="16" fill={color} opacity="0.22" />
        <circle cx="60" cy="44" r="11" fill="none" stroke={color} strokeWidth="3" />
        {[...Array(10)].map((_, i) => {
          const angle = (i / 10) * Math.PI * 2;
          const x1 = 60 + Math.cos(angle) * 18;
          const y1 = 44 + Math.sin(angle) * 18;
          const x2 = 60 + Math.cos(angle) * 30;
          const y2 = 44 + Math.sin(angle) * 30;
          return <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} stroke={color} strokeWidth="3" strokeLinecap="round" />;
        })}
      </svg>
    );
  }
  if (alias === "Zeus") {
    return (
      <svg viewBox="0 0 120 88" className="h-20 w-24">
        <path d="M32 38 C32 24, 44 18, 54 24 C58 14, 75 14, 82 24 C94 20, 104 28, 102 40 C100 50, 90 54, 80 54 H46 C38 54, 30 48, 32 38Z" fill="rgba(255,255,255,0.12)" stroke={color} strokeWidth="2.5" />
        <path d="M62 34 L49 55 H63 L55 74 L77 49 H64 L73 34 Z" fill={color} />
      </svg>
    );
  }
  if (alias === "Hephaestus") {
    return (
      <svg viewBox="0 0 120 88" className="h-20 w-24">
        <path d="M42 66 H82 L76 76 H48 Z" fill="rgba(255,255,255,0.16)" />
        <path d="M52 28 H88 V52 H68 C59 52, 52 45, 52 36 Z" fill="none" stroke={color} strokeWidth="3.5" />
        <path d="M38 66 L53 38" fill="none" stroke="rgba(255,255,255,0.85)" strokeWidth="4" strokeLinecap="round" />
        <path d="M72 14 C68 24, 78 29, 74 38 C84 31, 88 22, 82 14 C78 9, 73 9, 72 14Z" fill={color} />
      </svg>
    );
  }
  return (
    <svg viewBox="0 0 120 88" className="h-20 w-24">
      <path d="M44 38 C50 24, 70 24, 76 38" fill="none" stroke={color} strokeWidth="3.5" strokeLinecap="round" />
      <path d="M60 18 L68 28 L60 38 L52 28 Z" fill="none" stroke="rgba(255,255,255,0.85)" strokeWidth="3" />
      <path d="M60 38 V72" fill="none" stroke="rgba(255,255,255,0.85)" strokeWidth="3" strokeLinecap="round" />
      <path d="M50 48 C56 42, 64 42, 70 48" fill="none" stroke={color} strokeWidth="3" strokeLinecap="round" />
      <path d="M50 60 C56 54, 64 54, 70 60" fill="none" stroke={color} strokeWidth="3" strokeLinecap="round" />
    </svg>
  );
}

export default async function DashboardPage() {
  const snapshot = (await readSiteFinanceSnapshot()) as Record<string, unknown>;
  const finance = ((snapshot.finance as Record<string, unknown> | undefined) || {}) as Record<string, unknown>;
  const playbook = ((finance.latest_playbook as Record<string, unknown> | undefined) || {}) as Record<string, unknown>;
  const research = ((snapshot.profit_research as Record<string, unknown> | undefined) || {}) as Record<string, unknown>;
  const topCandidate = ((research.top_candidate as Record<string, unknown> | undefined) || {}) as Record<string, unknown>;
  const shadow = ((snapshot.shadow as Record<string, unknown> | undefined) || {}) as Record<string, unknown>;
  const shadowLatest = ((shadow.latest as Record<string, unknown> | undefined) || {}) as Record<string, unknown>;
  const confidence = ((snapshot.confidence as Record<string, unknown> | undefined) || {}) as Record<string, unknown>;
  const forecastHorizons = ((snapshot.forecast_horizons as Record<string, unknown> | undefined) || {}) as Record<string, unknown>;
  const dataQuality = ((snapshot.data_quality as Record<string, unknown> | undefined) || {}) as Record<string, unknown>;
  const recommendedLiveMode = ((confidence.recommended_live_mode as Record<string, unknown> | undefined) || {}) as Record<string, unknown>;
  const modeConfidence = ((confidence.mode_confidence as Record<string, unknown> | undefined) || {}) as Record<string, unknown>;
  const forecastDaily = ((forecastHorizons.daily as Record<string, unknown> | undefined) || {}) as Record<string, unknown>;
  const forecastWeekly = ((forecastHorizons.weekly as Record<string, unknown> | undefined) || {}) as Record<string, unknown>;
  const forecastMonthly = ((forecastHorizons.monthly as Record<string, unknown> | undefined) || {}) as Record<string, unknown>;
  const playbookStale = finance.latest_playbook_stale === true;
  const playbookStaleDays =
    typeof finance.latest_playbook_stale_days === "number" ? finance.latest_playbook_stale_days : null;
  const vigilanceAlerts = Array.isArray(confidence.vigilance_alerts)
    ? (confidence.vigilance_alerts as Array<Record<string, unknown>>)
    : [];
  const ingestionStaleDays =
    typeof dataQuality.ingestion_stale_days === "number" ? dataQuality.ingestion_stale_days : null;
  const ingestionFatalReason = String(dataQuality.ingestion_fatal_reason || "").trim();
  const ingestionWarningReasons = Array.isArray(dataQuality.ingestion_warning_reasons)
    ? (dataQuality.ingestion_warning_reasons as unknown[]).map((item) => String(item || "").trim()).filter(Boolean)
    : [];
  const qualityAlerts = Array.isArray(dataQuality.quality_alerts)
    ? (dataQuality.quality_alerts as Array<Record<string, unknown>>)
    : [];
  const exposure =
    !playbookStale && typeof playbook.exposure === "number"
      ? playbook.exposure
      : typeof shadowLatest.target_exposure === "number"
        ? shadowLatest.target_exposure
        : typeof forecastMonthly.exposure_target === "number"
          ? forecastMonthly.exposure_target
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
  const recommendedModeLabel = humanizeModeName(recommendedLiveMode.mode, recommendedLiveMode.label);
  const confidenceLevel = humanizeConfidenceLevel(
    recommendedLiveMode.confidence_level || modeConfidence.confidence_level || "sem leitura",
  );
  const confidenceScore =
    typeof recommendedLiveMode.confidence_score === "number"
      ? recommendedLiveMode.confidence_score
      : typeof modeConfidence.confidence_score === "number"
        ? modeConfidence.confidence_score
        : null;
  const vigilanceStatus = humanizeStatusWord(confidence.vigilance_status || "n/d");
  const scenarioBase = String(modeConfidence.scenario_base || "").trim();
  const confidenceReasons = Array.isArray(modeConfidence.reasons)
    ? (modeConfidence.reasons as unknown[]).map((item) => String(item || "").trim()).filter(Boolean)
    : [];
  const shadowGods = ((snapshot.shadow_gods as Record<string, unknown> | undefined) || {}) as Record<string, unknown>;
  const shadowGodsOverview =
    ((snapshot.shadow_gods_overview as Record<string, unknown> | undefined) || {}) as Record<string, unknown>;
  const shadowGodsList = Array.isArray(shadowGods.gods)
    ? (shadowGods.gods as Array<Record<string, unknown>>)
    : [];

  return (
    <div className="space-y-5">
      <section className="border-b border-zinc-800/80 px-5 py-5 md:px-6 md:py-6">
        <div className="text-xs uppercase tracking-[0.18em] text-zinc-500">Resumo de hoje</div>
        <h1 className="mt-2 text-2xl font-semibold text-zinc-100 md:text-3xl">O que o Eigen Engine está dizendo agora</h1>
        <p className="mt-3 max-w-3xl text-sm text-zinc-300">
          Aqui a ideia é simples: o motor resume o risco do mercado, sugere uma faixa de exposição e mostra se o
          contexto está limpo ou perigoso. Nada de linguagem de fundo quantitativo para esconder o básico.
        </p>
        <p className="mt-3 max-w-3xl rounded-2xl border border-amber-900/40 bg-amber-950/15 px-4 py-3 text-sm text-amber-100/90">
          Leitura para pesquisa, simulação e uso pessoal. Não trate estes sinais como consultoria individualizada nem
          como mandato de execução para terceiros.
        </p>
        <div className="mt-4">
          <div className="flex flex-wrap gap-3">
            <Link
              href="/app/shadow-mode"
              className="inline-flex rounded-xl border border-zinc-700 px-4 py-2 text-sm text-zinc-100 transition hover:border-zinc-500 hover:bg-zinc-800/70"
            >
              Ver todos os modos shadow
            </Link>
            <Link
              href="/app/shadow-mode/historical-simulated"
              className="inline-flex rounded-xl border border-zinc-700 px-4 py-2 text-sm text-zinc-100 transition hover:border-zinc-500 hover:bg-zinc-800/70"
            >
              Ver historico simulado
            </Link>
          </div>
        </div>
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
          {playbookStale ? (
            <p className="mt-2 text-sm text-amber-300">
              A leitura estrutural detalhada ficou {playbookStaleDays == null ? "desatualizada" : `${playbookStaleDays} dias`} atrás.
              A exposição mostrada aqui veio da operação diária mais recente.
            </p>
          ) : null}
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

        <article className="rounded-2xl border border-zinc-800 bg-zinc-950/55 p-5">
          <div className="text-xs uppercase tracking-[0.16em] text-zinc-500">Modo recomendado hoje</div>
          <div className="mt-2 text-xl font-semibold text-zinc-100">{recommendedModeLabel}</div>
          <p className="mt-3 text-sm text-zinc-300">{describeRecommendedMode(recommendedModeLabel)}</p>
          <p className="mt-3 text-sm text-zinc-400">
            Confiança: <span className="text-zinc-200">{confidenceLevel}</span>
            {" · "}
            Score: <span className="text-zinc-200">{confidenceScore == null ? "n/d" : `${Math.round(confidenceScore * 100)}%`}</span>
          </p>
          {scenarioBase ? <p className="mt-2 text-sm text-zinc-400">{scenarioBase}</p> : null}
        </article>

        <article className="rounded-2xl border border-zinc-800 bg-zinc-950/55 p-5">
          <div className="text-xs uppercase tracking-[0.16em] text-zinc-500">Vigilância diária</div>
          <div className="mt-2 text-xl font-semibold text-zinc-100">{vigilanceStatus}</div>
          <p className="mt-3 text-sm text-zinc-300">
            O agente diário vigia se o modo está ficando perigoso, se o dado envelheceu ou se a leitura perdeu força.
          </p>
          {vigilanceAlerts.length ? (
            <div className="mt-3 space-y-2 text-sm text-zinc-400">
              {vigilanceAlerts.slice(0, 3).map((alert, idx) => (
                <p key={`${String(alert.code || idx)}`}>- {String(alert.message || "Alerta sem detalhe.")}</p>
              ))}
            </div>
          ) : confidenceReasons.length ? (
            <div className="mt-3 space-y-2 text-sm text-zinc-400">
              {confidenceReasons.slice(0, 3).map((reason) => (
                <p key={reason}>- {reason}</p>
              ))}
            </div>
          ) : (
            <p className="mt-3 text-sm text-zinc-400">Sem alerta aberto no momento.</p>
          )}
        </article>
      </section>

      <section className="grid gap-4 px-5 md:grid-cols-4 md:px-6">
        <article className="rounded-2xl border border-zinc-800 bg-zinc-950/55 p-5 md:col-span-1">
          <div className="text-xs uppercase tracking-[0.16em] text-zinc-500">Ingestão de dados</div>
          <div className="mt-2 text-xl font-semibold text-zinc-100">
            {humanizeStatusWord(dataQuality.ingestion_status || "n/d")}
          </div>
          <p className="mt-3 text-sm text-zinc-300">
            Último dado conhecido:{" "}
            <span className="text-zinc-100">{String(dataQuality.last_ingestion_data_date || dataBase || "n/d")}</span>
          </p>
          <p className="mt-2 text-sm text-zinc-400">
            Atualizados: <span className="text-zinc-200">{String(dataQuality.ingestion_refreshed_assets ?? dataQuality.ingestion_updated_assets ?? "n/d")}</span>
            {" · "}
            Falhas: <span className="text-zinc-200">{String(dataQuality.ingestion_failed_assets ?? "n/d")}</span>
          </p>
          <p className="mt-2 text-sm text-zinc-400">
            Dias de atraso: <span className="text-zinc-200">{ingestionStaleDays == null ? "n/d" : String(ingestionStaleDays)}</span>
          </p>
          <p className="mt-2 text-sm text-zinc-400">
            Críticos atrasados: <span className="text-zinc-200">{String(dataQuality.quality_critical_stale_assets ?? "n/d")}</span>
            {" · "}
            Núcleo atrasado: <span className="text-zinc-200">{String(dataQuality.quality_core_stale_assets ?? "n/d")}</span>
          </p>
          {ingestionFatalReason ? (
            <p className="mt-2 text-sm text-amber-300">Motivo do alerta: {ingestionFatalReason}</p>
          ) : ingestionWarningReasons.length ? (
            <p className="mt-2 text-sm text-amber-300">Alertas: {ingestionWarningReasons.join(", ")}</p>
          ) : null}
          {qualityAlerts.length ? (
            <p className="mt-2 text-sm text-amber-300">{String(qualityAlerts[0]?.message || "")}</p>
          ) : null}
        </article>

        {[forecastDaily, forecastWeekly, forecastMonthly].map((forecast, idx) => (
          <article key={`forecast-${idx}`} className="rounded-2xl border border-zinc-800 bg-zinc-950/55 p-5">
            <div className="text-xs uppercase tracking-[0.16em] text-zinc-500">
              {String(forecast.label || `Horizonte ${idx + 1}`)}
            </div>
            <div className="mt-2 text-lg font-semibold text-zinc-100">
              {String(forecast.mode || recommendedModeLabel || "Sem leitura")}
            </div>
            <p className="mt-3 text-sm text-zinc-300">
              {String(forecast.summary || "Sem resumo publicado para este horizonte.")}
            </p>
            <p className="mt-3 text-sm text-zinc-400">
              Exposição: <span className="text-zinc-200">{formatPct(typeof forecast.exposure_target === "number" ? forecast.exposure_target : exposure)}</span>
              {" · "}
              Confiança: <span className="text-zinc-200">{humanizeConfidenceLevel(forecast.confidence_level || confidenceLevel)}</span>
            </p>
          </article>
        ))}
      </section>

      <div className="px-5 md:px-6">
        <SignalSnapshotSection snapshot={snapshot as Record<string, unknown>} compact />
      </div>

      {shadowGodsList.length ? (
        <section className="px-5 md:px-6">
          <div className="rounded-2xl border border-zinc-800 bg-zinc-950/55 p-5">
            <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
              <div>
                <div className="text-xs uppercase tracking-[0.16em] text-zinc-500">Shadow gods</div>
                <div className="mt-2 text-xl font-semibold text-zinc-100">
                  Os 4 deuses simulados que rodam todo dia com ordem, fill e historico
                </div>
                <p className="mt-2 max-w-3xl text-sm text-zinc-300">
                  Eles mostram o que cada tese faria com `R$200`, `R$1.000` e `R$10.000`. Se o dia pede defesa, o card mostra
                  caixa e no-trade. Se pede ordem, o pedido aparece.
                </p>
              </div>
              <div className="text-sm text-zinc-400">
                {String(shadowGodsOverview.total_gods || shadowGodsList.length)} deuses ·{" "}
                {String(shadowGodsOverview.total_scenarios || 0)} cenarios ·{" "}
                {String(shadowGodsOverview.order_count_total || 0)} ordens
              </div>
            </div>

            <div className="mt-5 grid gap-4 xl:grid-cols-2">
              {shadowGodsList.map((god) => {
                const alias = String(god.alias || "Shadow");
                const scenarios = Array.isArray(god.scenarios) ? (god.scenarios as Array<Record<string, unknown>>) : [];
                return (
                  <article key={alias} className="rounded-2xl border border-zinc-800 bg-zinc-900/45 p-4">
                    <div className="flex items-start gap-4">
                      <div className="flex h-24 w-24 items-center justify-center rounded-2xl border border-zinc-800 bg-zinc-950/70">
                        <DashboardGodArt alias={alias} />
                      </div>
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-2">
                          <h3 className="text-lg font-semibold text-zinc-100">{alias}</h3>
                          <span className={`rounded-full border px-2 py-1 text-[11px] uppercase tracking-[0.14em] ${marketTone(String(scenarios[0]?.market_state || ""))}`}>
                            {titleCase(String(scenarios[0]?.market_state || "unknown"))}
                          </span>
                        </div>
                        <p className="mt-2 text-sm text-zinc-300">{String(god.thesis || "Sem tese publicada.")}</p>
                        <p className="mt-2 text-xs text-zinc-500">{String(god.candidate_id || "n/d")}</p>
                      </div>
                    </div>

                    <div className="mt-4 grid gap-3 md:grid-cols-3">
                      {scenarios.map((scenario) => {
                        const capital = asNumber(scenario.capital_brl) || 0;
                        const navAfter = asNumber(scenario.nav_after_brl) || capital;
                        const totalReturn = capital > 0 ? navAfter / capital - 1 : null;
                        return (
                          <div key={String(scenario.scenario_id || capital)} className="rounded-xl border border-zinc-800 bg-zinc-950/75 p-3">
                            <div className="text-[11px] uppercase tracking-[0.14em] text-zinc-500">{formatBRL(capital)}</div>
                            <div className="mt-2 text-base font-semibold text-zinc-100">{formatBRL(navAfter, 2)}</div>
                            <div className="mt-1 text-sm text-zinc-300">{formatSignedPct(totalReturn, 1)}</div>
                            <div className="mt-2 text-xs text-zinc-400">
                              ordens {String(scenario.order_count || 0)} · fills {String(scenario.fill_count || 0)}
                            </div>
                          </div>
                        );
                      })}
                    </div>

                    <div className="mt-4">
                      <Link
                        href="/app/shadow-mode"
                        className="inline-flex rounded-xl border border-zinc-700 px-4 py-2 text-sm text-zinc-100 transition hover:border-zinc-500 hover:bg-zinc-800/70"
                      >
                        Abrir painel completo dos deuses
                      </Link>
                    </div>
                  </article>
                );
              })}
            </div>
          </div>
        </section>
      ) : null}

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
