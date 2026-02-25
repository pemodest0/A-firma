import {
  findLatestLabCorrRun,
  readLatestLabCorrActionPlaybook,
  readLatestLabCorrOperationalAlerts,
  readLatestLabCorrTimeseries,
  readLatestLabCorrUiViewModel,
} from "@/lib/server/data";

function toNum(value: unknown): number | null {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function clamp01(value: number | null) {
  if (value == null) return null;
  return Math.max(0, Math.min(1, value));
}

function fmtScore(value: number | null, digits = 2) {
  if (value == null) return "--";
  return value.toFixed(digits);
}

function regimeRiskBase(regime: string) {
  const r = regime.toLowerCase();
  if (r === "stable") return 0.2;
  if (r === "transition") return 0.55;
  if (r === "stress") return 0.85;
  if (r === "dispersion") return 0.7;
  return 0.5;
}

function riskBand(value: number | null) {
  if (value == null) return "indefinido";
  if (value < 0.35) return "baixo";
  if (value < 0.7) return "medio";
  return "alto";
}

function confidenceBand(value: number | null) {
  if (value == null) return "indefinida";
  if (value >= 0.75) return "alta";
  if (value >= 0.5) return "media";
  return "baixa";
}

export default async function DashboardPage() {
  const [labRun, ts, opAlerts, playbook, uiView] = await Promise.all([
    findLatestLabCorrRun(),
    readLatestLabCorrTimeseries(120),
    readLatestLabCorrOperationalAlerts(120),
    readLatestLabCorrActionPlaybook(120),
    readLatestLabCorrUiViewModel(120),
  ]);

  const summary = ((labRun?.summary || {}) as Record<string, unknown>) || {};
  const gate = ((summary.deployment_gate || {}) as Record<string, unknown>) || {};
  const gateBlocked = gate.blocked === true;
  const gateReasons = Array.isArray(gate.reasons) ? gate.reasons.map((v) => String(v)) : [];

  const officialWindow = toNum(summary.official_window) ?? 120;
  const policyPath = String(summary.policy_path || "production_policy_lock.json");
  const nCore = toNum(summary.n_core);

  const latestState = (ts?.latest || ((uiView as Record<string, unknown>)?.latest_state as Record<string, unknown>)) as
    | Record<string, unknown>
    | null;
  const latestNUsed = toNum(latestState?.N_used);
  const structureScore = toNum(latestState?.structure_score);

  const playRows = Array.isArray(playbook) ? playbook : [];
  const latestPlay = (playRows.length
    ? playRows[playRows.length - 1]
    : ((uiView as Record<string, unknown>)?.playbook_latest as Record<string, unknown>)) as Record<
    string,
    unknown
  >;

  const regime = String(
    latestPlay?.regime || ((uiView as Record<string, unknown>)?.latest_regime as Record<string, unknown>)?.regime || "--"
  );
  const signalTier = String(latestPlay?.signal_tier || "--");
  const signalReliability = toNum(latestPlay?.signal_reliability);

  const opObj = (opAlerts || {}) as Record<string, unknown>;
  const latestEvents = Array.isArray(opObj.latest_events) ? opObj.latest_events.map((v) => String(v)) : [];
  const events60d = toNum(opObj.n_events_last_60d) ?? 0;

  const riskRaw = regimeRiskBase(regime) + (gateBlocked ? 0.1 : 0) + (events60d > 30 ? 0.05 : 0);
  const riskScore = clamp01(riskRaw);

  const qMin = toNum((gate.thresholds as Record<string, unknown> | undefined)?.min_joint_majority_60d);
  const qObserved = toNum((summary.scores as Record<string, unknown> | undefined)?.joint_majority_60d);
  const qComponent = qMin != null && qMin > 0 && qObserved != null ? clamp01(qObserved / qMin) : null;
  const coverageComponent = officialWindow > 0 && latestNUsed != null ? clamp01(latestNUsed / officialWindow) : null;
  const qualityComponent = clamp01(structureScore != null ? structureScore : signalReliability);
  const confidenceParts = [coverageComponent, qComponent, qualityComponent].filter(
    (v): v is number => v != null
  );
  const confidenceScore = confidenceParts.length
    ? confidenceParts.reduce((acc, value) => acc + value, 0) / confidenceParts.length
    : null;

  const warmupRemaining =
    latestNUsed != null && officialWindow > 0 ? Math.max(0, Math.trunc(officialWindow - latestNUsed)) : null;
  const warmupActive = warmupRemaining != null ? warmupRemaining > 0 : false;

  return (
    <div className="p-5 md:p-6 lg:p-8 space-y-6">
      <section className="rounded-2xl border border-zinc-800 bg-zinc-950/50 p-5">
        <p className="text-xs tracking-[0.14em] uppercase text-zinc-500">Painel</p>
        <h1 className="mt-2 text-2xl md:text-3xl font-semibold text-zinc-100">Painel operacional do Eigen Engine</h1>
        <p className="mt-3 text-sm text-zinc-300">
          Leitura focada em estado atual, risco, confianca e governanca de publicacao.
        </p>
      </section>

      <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
        <MetricCard
          title="Risco estrutural"
          value={`${fmtScore(riskScore)} (${riskBand(riskScore)})`}
          helper="Escala 0 a 1, separada da confianca."
        />
        <MetricCard
          title="Confianca"
          value={`${fmtScore(confidenceScore)} (${confidenceBand(confidenceScore)})`}
          helper="Baseada em cobertura, robustez e qualidade da janela."
        />
        <MetricCard
          title="Janela e politica"
          value={`T${Math.trunc(officialWindow)} | ${policyPath}`}
          helper="Modo de calculo: causal (walk-forward)."
        />
        <MetricCard
          title="Gate e warmup"
          value={gateBlocked ? "Gate nao verde" : "Gate verde"}
          helper={
            warmupActive
              ? `Warmup estrutural em andamento (${warmupRemaining} dias restantes).`
              : "Eigen Engine totalmente operacional."
          }
        />
      </section>

      <section className="grid grid-cols-1 md:grid-cols-3 gap-3">
        <Card>
          <div className="text-xs uppercase tracking-[0.12em] text-zinc-500">Estado atual do Eigen Engine</div>
          <div className="mt-2 text-zinc-100 font-semibold">Regime: {regime}</div>
          <div className="mt-1 text-sm text-zinc-300">Risco: {riskBand(riskScore)}</div>
          <div className="mt-1 text-sm text-zinc-300">Confianca: {confidenceBand(confidenceScore)}</div>
        </Card>
        <Card>
          <div className="text-xs uppercase tracking-[0.12em] text-zinc-500">Resumo do universo</div>
          <div className="mt-2 text-sm text-zinc-300">Ativos no universo de correlacao: {nCore ?? "--"}</div>
          <div className="mt-1 text-sm text-zinc-300">
            N usado (janela atual): {latestNUsed != null ? Math.trunc(latestNUsed) : "--"}
          </div>
          <div className="mt-1 text-sm text-zinc-300">Eventos nos ultimos 60d: {Math.trunc(events60d)}</div>
        </Card>
        <Card>
          <div className="text-xs uppercase tracking-[0.12em] text-zinc-500">Sinal do dia</div>
          <div className="mt-2 text-zinc-100 font-semibold">Nivel: {signalTier}</div>
          <div className="mt-1 text-sm text-zinc-300">Confianca do sinal: {fmtScore(signalReliability, 3)}</div>
          <div className="mt-1 text-sm text-zinc-300">Run: {labRun?.runId || "unavailable"}</div>
        </Card>
      </section>

      <section className="rounded-2xl border border-zinc-800 bg-zinc-950/45 p-5">
        <h2 className="text-lg font-semibold text-zinc-100">Dicas rapidas</h2>
        <ul className="mt-3 space-y-2 text-sm text-zinc-300">
          <li>- Use o regime para ajustar tamanho de posicao, nao para prever preco.</li>
          <li>- Combine leitura de preco com volatilidade para identificar ruido.</li>
          <li>- Para horizontes curtos (h5), variacao e mais sensivel; use com prudencia.</li>
        </ul>
      </section>

      <details className="rounded-2xl border border-zinc-800 bg-zinc-950/35 p-5">
        <summary className="cursor-pointer text-sm font-semibold text-zinc-200">Detalhes tecnicos</summary>
        <div className="mt-4 space-y-2 text-sm text-zinc-300">
          <div>Janela oficial: {Math.trunc(officialWindow)} dias</div>
          <div>Politica ativa: {policyPath}</div>
          <div>Modo de calculo: causal (walk-forward)</div>
          <div>Gate blocked: {String(gateBlocked)}</div>
          <div>Motivos do gate: {gateReasons.length ? gateReasons.join(", ") : "nenhum"}</div>
          <div>Eventos recentes: {latestEvents.length ? latestEvents.join(", ") : "nenhum"}</div>
          <div>Joint majority 60d: {fmtScore(qObserved, 4)}</div>
          <div>Threshold minimo joint majority 60d: {fmtScore(qMin, 4)}</div>
        </div>
      </details>
    </div>
  );
}

function MetricCard({ title, value, helper }: { title: string; value: string; helper: string }) {
  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-950/55 p-3">
      <div className="text-xs uppercase tracking-[0.12em] text-zinc-500">{title}</div>
      <div className="mt-2 text-lg font-semibold text-zinc-100 break-words">{value}</div>
      <div className="mt-1 text-xs text-zinc-400">{helper}</div>
    </div>
  );
}

function Card({ children }: { children: React.ReactNode }) {
  return <div className="rounded-xl border border-zinc-800 bg-zinc-950/55 p-4">{children}</div>;
}
