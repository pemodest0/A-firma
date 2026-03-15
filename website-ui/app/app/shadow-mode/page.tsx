import Link from "next/link";
import { humanizeConfidenceLevel, humanizeStatusWord } from "@/lib/enginePresentation";
import { readSiteFinanceSnapshot } from "@/lib/server/data";

type AnyRecord = Record<string, unknown>;

const GOD_STYLES: Record<
  string,
  {
    accent: string;
    border: string;
    glow: string;
    badge: string;
    chart: string;
    title: string;
  }
> = {
  Apollo: {
    accent: "from-amber-300/30 via-orange-400/15 to-zinc-950",
    border: "border-amber-700/60",
    glow: "bg-amber-400/10",
    badge: "border-amber-600/60 bg-amber-500/15 text-amber-100",
    chart: "#f59e0b",
    title: "Ancora estrutural",
  },
  Zeus: {
    accent: "from-sky-300/25 via-indigo-400/15 to-zinc-950",
    border: "border-sky-700/60",
    glow: "bg-sky-400/10",
    badge: "border-sky-600/60 bg-sky-500/15 text-sky-100",
    chart: "#38bdf8",
    title: "Seletor de contexto",
  },
  Hephaestus: {
    accent: "from-orange-300/25 via-rose-500/15 to-zinc-950",
    border: "border-orange-700/60",
    glow: "bg-orange-400/10",
    badge: "border-orange-600/60 bg-orange-500/15 text-orange-100",
    chart: "#fb923c",
    title: "Ataque lento calibrado",
  },
  Hermes: {
    accent: "from-emerald-300/25 via-teal-400/15 to-zinc-950",
    border: "border-emerald-700/60",
    glow: "bg-emerald-400/10",
    badge: "border-emerald-600/60 bg-emerald-500/15 text-emerald-100",
    chart: "#34d399",
    title: "Turbo em shadow",
  },
};

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

function formatPctRatio(value: number | null, digits = 1) {
  if (value == null || !Number.isFinite(value)) return "n/d";
  return `${(value * 100).toFixed(digits)}%`;
}

function formatSignedPct(value: number | null, digits = 1) {
  if (value == null || !Number.isFinite(value)) return "n/d";
  const sign = value > 0 ? "+" : "";
  return `${sign}${(value * 100).toFixed(digits)}%`;
}

function formatCount(value: unknown) {
  const numeric = asNumber(value);
  return numeric == null ? "0" : String(Math.round(numeric));
}

function formatDate(value: unknown) {
  const raw = String(value || "").trim();
  return raw ? raw.slice(0, 10) : "n/d";
}

function marketTone(state: string) {
  const normalized = String(state || "").toLowerCase();
  if (normalized.includes("attack")) return "border-emerald-600/60 bg-emerald-500/15 text-emerald-100";
  if (normalized.includes("risk")) return "border-sky-600/60 bg-sky-500/15 text-sky-100";
  if (normalized.includes("transition")) return "border-amber-600/60 bg-amber-500/15 text-amber-100";
  return "border-zinc-700 bg-zinc-900/80 text-zinc-200";
}

function titleCase(raw: string) {
  return raw
    .replace(/_/g, " ")
    .split(" ")
    .filter(Boolean)
    .map((chunk) => chunk.charAt(0).toUpperCase() + chunk.slice(1))
    .join(" ");
}

function getGodStyle(alias: string) {
  return GOD_STYLES[alias] || {
    accent: "from-zinc-400/20 via-zinc-600/10 to-zinc-950",
    border: "border-zinc-700",
    glow: "bg-zinc-700/20",
    badge: "border-zinc-700 bg-zinc-900/80 text-zinc-100",
    chart: "#a1a1aa",
    title: "Shadow",
  };
}

function parseScenarioAssets(scenario: AnyRecord) {
  const holdings = Array.isArray(scenario.holdings) ? (scenario.holdings as AnyRecord[]) : [];
  if (holdings.length) {
    return holdings
      .map((row) => String(row.ticker || "").trim())
      .filter(Boolean);
  }
  const targetWeights = (scenario.target_weights as AnyRecord | undefined) || {};
  return Object.keys(targetWeights).filter((ticker) => ticker !== "CASH-BRL");
}

function buildHistoryPoints(scenario: AnyRecord) {
  const rows = Array.isArray(scenario.history_tail) ? (scenario.history_tail as AnyRecord[]) : [];
  const parsed = rows
    .map((row) => asNumber(row.nav_after_brl))
    .filter((value): value is number => value != null);
  if (parsed.length >= 2) return parsed;
  const navBefore = asNumber(scenario.nav_before_brl);
  const navAfter = asNumber(scenario.nav_after_brl);
  if (navBefore != null && navAfter != null && navBefore !== navAfter) return [navBefore, navAfter];
  if (navAfter != null) return [navAfter, navAfter];
  return [];
}

function sparklinePath(points: number[], width: number, height: number) {
  if (points.length < 2) return "";
  const min = Math.min(...points);
  const max = Math.max(...points);
  const spread = max - min || 1;
  return points
    .map((point, index) => {
      const x = (index / Math.max(points.length - 1, 1)) * width;
      const y = height - ((point - min) / spread) * height;
      return `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(" ");
}

function bucketEntries(source: AnyRecord) {
  return Object.entries(source)
    .map(([ticker, raw]) => ({ ticker, weight: asNumber(raw) || 0 }))
    .filter((row) => row.weight > 0);
}

export default async function ShadowModePage() {
  const snapshot = (await readSiteFinanceSnapshot()) as AnyRecord;
  const shadowGods = ((snapshot.shadow_gods as AnyRecord | undefined) || {}) as AnyRecord;
  const gods = Array.isArray(shadowGods.gods) ? (shadowGods.gods as AnyRecord[]) : [];
  const overview = ((snapshot.shadow_gods_overview as AnyRecord | undefined) ||
    (shadowGods.overview as AnyRecord | undefined) ||
    {}) as AnyRecord;
  const confidence = ((snapshot.confidence as AnyRecord | undefined) || {}) as AnyRecord;
  const recommendedLiveMode =
    ((confidence.recommended_live_mode as AnyRecord | undefined) || {}) as AnyRecord;

  return (
    <div className="space-y-6">
      <section className="border-b border-zinc-800/80 px-5 py-5 md:px-6 md:py-6">
        <div className="text-xs uppercase tracking-[0.18em] text-zinc-500">Shadow gods</div>
        <h1 className="mt-2 text-2xl font-semibold text-zinc-100 md:text-3xl">
          Os 4 deuses que rodam todo dia em shadow, com ordem, fill e historico real
        </h1>
        <p className="mt-3 max-w-4xl text-sm text-zinc-300">
          Esta pagina parou de mostrar o zoo antigo de modos. Agora ela acompanha so os 4 deuses principais do
          laboratorio, cada um com tres blocos de capital, operacao diaria simulada, pedidos emitidos, fills, caixa,
          holdings e historico de navegacao. Se o dia pede defesa, o card mostra isso sem maquiagem. Se pedir compra,
          a ordem aparece aqui.
        </p>
      </section>

      <section className="grid gap-4 px-5 md:grid-cols-4 md:px-6">
        <SummaryCard label="Deuses ativos" value={String(overview.total_gods || gods.length || 0)} />
        <SummaryCard label="Cenarios vivos" value={String(overview.total_scenarios || 0)} />
        <SummaryCard label="Ordens emitidas" value={String(overview.order_count_total || 0)} />
        <SummaryCard
          label="Modo recomendado hoje"
          value={String(recommendedLiveMode.label || recommendedLiveMode.mode || "n/d")}
          caption={`Confianca ${humanizeConfidenceLevel(recommendedLiveMode.confidence_level || "n/d")}`}
        />
      </section>

      {gods.length ? (
        <section className="grid gap-5 px-5 pb-5 md:px-6 md:pb-6">
          {gods.map((god) => (
            <GodCard key={String(god.alias || god.candidate_id || Math.random())} god={god} />
          ))}
        </section>
      ) : (
        <section className="px-5 pb-5 md:px-6 md:pb-6">
          <div className="rounded-3xl border border-zinc-800 bg-zinc-950/60 p-6 text-sm text-zinc-300">
            O snapshot ainda nao publicou `shadow_gods`. Rode o ciclo diario e depois reabra esta pagina.
          </div>
        </section>
      )}

      <section className="px-5 pb-6 md:px-6">
        <div className="rounded-3xl border border-zinc-800 bg-zinc-950/50 p-5">
          <div className="text-sm font-medium text-zinc-100">Como ler os deuses</div>
          <div className="mt-3 grid gap-3 text-sm text-zinc-300 md:grid-cols-3">
            <p>- `Apollo` e `Zeus` medem estrutura e contexto. Eles valem pelo filtro, nao por foguete.</p>
            <p>- `Hephaestus` e `Hermes` testam a parte ofensiva. Um tenta ser mais disciplinado; o outro aceita mais cauda.</p>
            <p>- Se o cenario estiver em `defense`, o card pode deliberadamente nao operar. Isso tambem e decisao.</p>
          </div>
          <div className="mt-4 flex flex-wrap gap-3">
            <Link
              href="/app/dashboard"
              className="rounded-xl border border-zinc-700 px-4 py-2 text-sm text-zinc-100 transition hover:border-zinc-500 hover:bg-zinc-800/70"
            >
              Voltar para o dashboard
            </Link>
            <Link
              href="/app/financas"
              className="rounded-xl border border-zinc-700 px-4 py-2 text-sm text-zinc-100 transition hover:border-zinc-500 hover:bg-zinc-800/70"
            >
              Ver financas
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
}

function GodCard({ god }: { god: AnyRecord }) {
  const alias = String(god.alias || "Shadow");
  const style = getGodStyle(alias);
  const scenarios = Array.isArray(god.scenarios) ? (god.scenarios as AnyRecord[]) : [];
  const latestStates = Array.isArray(god.latest_states) ? (god.latest_states as AnyRecord[]) : [];

  return (
    <article className={`overflow-hidden rounded-[28px] border bg-zinc-950/60 ${style.border}`}>
      <div className={`relative grid gap-5 bg-gradient-to-br ${style.accent} p-5 md:grid-cols-[260px_minmax(0,1fr)] md:p-6`}>
        <div className={`relative overflow-hidden rounded-[24px] border ${style.border} ${style.glow} p-4`}>
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_top,rgba(255,255,255,0.18),transparent_55%)]" />
          <div className="relative">
            <div className={`inline-flex rounded-full border px-3 py-1 text-[11px] uppercase tracking-[0.16em] ${style.badge}`}>
              {style.title}
            </div>
            <div className="mt-4">
              <GodArt alias={alias} color={style.chart} />
            </div>
            <div className="mt-4 space-y-2">
              <div className="text-3xl font-semibold text-zinc-50">{alias}</div>
              <p className="text-sm text-zinc-200/90">{String(god.thesis || "Sem tese publicada.")}</p>
            </div>
          </div>
        </div>

        <div className="space-y-4">
          <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
            <div>
              <div className="text-xs uppercase tracking-[0.16em] text-zinc-400">Identidade tecnica</div>
              <div className="mt-2 text-xl font-semibold text-zinc-100">{titleCase(String(god.role || "shadow"))}</div>
              <p className="mt-2 max-w-3xl text-sm text-zinc-300">{String(god.candidate_id || "n/d")}</p>
            </div>
            <div className="grid min-w-[220px] gap-3 sm:grid-cols-2">
              <MetricCard label="Ordens totais" value={formatCount(god.order_count_total)} />
              <MetricCard label="Fills totais" value={formatCount(god.fill_count_total)} />
            </div>
          </div>

          {latestStates.length ? (
            <div className="grid gap-3 md:grid-cols-3">
              {latestStates.slice(0, 3).map((state) => (
                <div key={String(state.scenario_id || Math.random())} className="rounded-2xl border border-zinc-800/90 bg-zinc-950/55 p-3">
                  <div className="text-[11px] uppercase tracking-[0.14em] text-zinc-500">{String(state.scenario_id || "cenario")}</div>
                  <div className="mt-2 flex items-center justify-between">
                    <span className={`rounded-full border px-2 py-1 text-[11px] uppercase tracking-[0.14em] ${marketTone(String(state.market_state || ""))}`}>
                      {titleCase(String(state.market_state || "unknown"))}
                    </span>
                    <span className="text-xs text-zinc-400">{formatDate(state.as_of_date)}</span>
                  </div>
                  <div className="mt-3 text-sm text-zinc-300">
                    Caixa {formatBRL(state.cash_after_brl)} · NAV {formatBRL(state.nav_after_brl)}
                  </div>
                </div>
              ))}
            </div>
          ) : null}

          <div className="grid gap-4 xl:grid-cols-3">
            {scenarios.map((scenario) => (
              <ScenarioCard key={String(scenario.scenario_id || Math.random())} scenario={scenario} godAlias={alias} chartColor={style.chart} />
            ))}
          </div>
        </div>
      </div>
    </article>
  );
}

function ScenarioCard({
  scenario,
  godAlias,
  chartColor,
}: {
  scenario: AnyRecord;
  godAlias: string;
  chartColor: string;
}) {
  const capital = asNumber(scenario.capital_brl) || 0;
  const navBefore = asNumber(scenario.nav_before_brl) || capital;
  const navAfter = asNumber(scenario.nav_after_brl) || capital;
  const cash = asNumber(scenario.cash_after_brl) || 0;
  const invested = Math.max(0, navAfter - cash);
  const totalReturn = capital > 0 ? navAfter / capital - 1 : null;
  const dailyDelta = navBefore > 0 ? navAfter / navBefore - 1 : null;
  const cashShare = navAfter > 0 ? cash / navAfter : 0;
  const points = buildHistoryPoints(scenario);
  const sparkPath = sparklinePath(points, 220, 58);
  const targetWeights = bucketEntries(((scenario.target_weights as AnyRecord | undefined) || {}) as AnyRecord);
  const holdings = Array.isArray(scenario.holdings) ? (scenario.holdings as AnyRecord[]) : [];
  const orders = Array.isArray(scenario.orders) ? (scenario.orders as AnyRecord[]) : [];
  const fills = Array.isArray(scenario.fills) ? (scenario.fills as AnyRecord[]) : [];
  const blocked = Array.isArray(scenario.blocked) ? (scenario.blocked as AnyRecord[]) : [];
  const marketNotes = Array.isArray(scenario.market_notes) ? (scenario.market_notes as string[]) : [];
  const assets = parseScenarioAssets(scenario);

  return (
    <div className="rounded-[24px] border border-zinc-800/90 bg-zinc-950/70 p-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-[11px] uppercase tracking-[0.16em] text-zinc-500">{godAlias}</div>
          <div className="mt-1 text-xl font-semibold text-zinc-100">{formatBRL(capital)}</div>
        </div>
        <span className={`rounded-full border px-2 py-1 text-[11px] uppercase tracking-[0.14em] ${marketTone(String(scenario.market_state || ""))}`}>
          {titleCase(String(scenario.market_state || "unknown"))}
        </span>
      </div>

      <div className="mt-3 text-sm text-zinc-300">
        {marketNotes[0] || "Sem nota de mercado publicada neste ciclo."}
      </div>

      <div className="mt-4 grid grid-cols-2 gap-3">
        <MetricCard label="NAV atual" value={formatBRL(navAfter, 2)} compact />
        <MetricCard label="Retorno desde o inicio" value={formatSignedPct(totalReturn, 1)} compact />
        <MetricCard label="Mudanca do dia" value={formatSignedPct(dailyDelta, 2)} compact />
        <MetricCard label="Caixa" value={formatPctRatio(cashShare, 0)} compact />
      </div>

      <div className="mt-4 rounded-2xl border border-zinc-800 bg-zinc-900/45 p-3">
        <div className="flex items-center justify-between text-[11px] uppercase tracking-[0.14em] text-zinc-500">
          <span>Grafico do cenario</span>
          <span>{points.length > 1 ? "historia viva" : "sem serie ainda"}</span>
        </div>
        <div className="mt-3">
          {sparkPath ? (
            <svg viewBox="0 0 220 58" className="h-16 w-full overflow-visible">
              <defs>
                <linearGradient id={`spark-${String(scenario.scenario_id)}`} x1="0" x2="1" y1="0" y2="0">
                  <stop offset="0%" stopColor={chartColor} stopOpacity="0.2" />
                  <stop offset="100%" stopColor={chartColor} stopOpacity="1" />
                </linearGradient>
              </defs>
              <path d={sparkPath} fill="none" stroke={`url(#spark-${String(scenario.scenario_id)})`} strokeWidth="3" strokeLinecap="round" />
            </svg>
          ) : (
            <div className="flex h-16 items-center justify-center rounded-xl border border-dashed border-zinc-800 text-xs text-zinc-500">
              Ainda sem pontos suficientes para serie.
            </div>
          )}
        </div>
      </div>

      <div className="mt-4 rounded-2xl border border-zinc-800 bg-zinc-900/45 p-3">
        <div className="text-[11px] uppercase tracking-[0.14em] text-zinc-500">Alocacao alvo e caixa</div>
        <div className="mt-3 overflow-hidden rounded-full bg-zinc-900">
          <div className="flex h-3 w-full">
            <div className="bg-zinc-200/85" style={{ width: `${Math.max(0, Math.min(100, cashShare * 100))}%` }} />
            <div className="bg-zinc-600" style={{ width: `${Math.max(0, Math.min(100, (1 - cashShare) * 100))}%` }} />
          </div>
        </div>
        <div className="mt-2 flex items-center justify-between text-xs text-zinc-400">
          <span>Caixa {formatBRL(cash, 2)}</span>
          <span>Risco {formatBRL(invested, 2)}</span>
        </div>
        <div className="mt-3 flex flex-wrap gap-2">
          {targetWeights.length ? (
            targetWeights.map((bucket) => (
              <span key={`${String(scenario.scenario_id)}-${bucket.ticker}`} className="rounded-full border border-zinc-700 bg-zinc-950 px-2 py-1 text-xs text-zinc-200">
                {bucket.ticker} {formatPctRatio(bucket.weight, 0)}
              </span>
            ))
          ) : (
            <span className="rounded-full border border-zinc-800 bg-zinc-950 px-2 py-1 text-xs text-zinc-500">Sem risco aberto</span>
          )}
        </div>
      </div>

      <div className="mt-4 grid gap-3 md:grid-cols-2">
        <div className="rounded-2xl border border-zinc-800 bg-zinc-900/45 p-3">
          <div className="text-[11px] uppercase tracking-[0.14em] text-zinc-500">Ativos vivos</div>
          <div className="mt-3 flex flex-wrap gap-2">
            {assets.length ? (
              assets.map((asset) => (
                <span key={`${String(scenario.scenario_id)}-asset-${asset}`} className="rounded-full border border-zinc-700 bg-zinc-950 px-2 py-1 text-xs text-zinc-200">
                  {asset}
                </span>
              ))
            ) : (
              <span className="text-sm text-zinc-500">Sem ativo comprado neste bloco.</span>
            )}
          </div>
          {holdings.length ? (
            <div className="mt-3 space-y-2 text-xs text-zinc-300">
              {holdings.slice(0, 3).map((holding) => (
                <div key={`${String(scenario.scenario_id)}-holding-${String(holding.ticker)}`} className="flex items-center justify-between rounded-xl border border-zinc-800 bg-zinc-950/70 px-3 py-2">
                  <span>{String(holding.ticker || "asset")}</span>
                  <span>{formatBRL(holding.market_value_brl, 2)}</span>
                </div>
              ))}
            </div>
          ) : null}
        </div>

        <div className="rounded-2xl border border-zinc-800 bg-zinc-900/45 p-3">
          <div className="text-[11px] uppercase tracking-[0.14em] text-zinc-500">Operacao do dia</div>
          <div className="mt-3 grid grid-cols-2 gap-3">
            <MetricCard label="Pedidos" value={formatCount(scenario.order_count)} compact />
            <MetricCard label="Fills" value={formatCount(scenario.fill_count)} compact />
          </div>
          <div className="mt-3 space-y-2 text-xs text-zinc-300">
            {orders.slice(0, 2).map((order) => (
              <div key={String(order.ticket_id || Math.random())} className="rounded-xl border border-zinc-800 bg-zinc-950/70 px-3 py-2">
                {String(order.side || "").toUpperCase()} {String(order.ticker || "asset")} · {formatBRL(order.notional_brl, 2)}
              </div>
            ))}
            {!orders.length && !fills.length ? (
              <div className="rounded-xl border border-dashed border-zinc-800 px-3 py-2 text-zinc-500">No-trade neste ciclo.</div>
            ) : null}
            {fills.slice(0, 2).map((fill) => (
              <div key={`${String(fill.ticket_id || Math.random())}-fill`} className="rounded-xl border border-zinc-800 bg-zinc-950/70 px-3 py-2">
                Fill {String(fill.ticker || "asset")} · taxa {formatBRL(fill.fee_brl, 2)}
              </div>
            ))}
            {blocked.slice(0, 2).map((item, idx) => (
              <div key={`${String(scenario.scenario_id)}-blocked-${idx}`} className="rounded-xl border border-amber-700/50 bg-amber-500/10 px-3 py-2 text-amber-100/90">
                Bloqueado: {String(item.ticker || "asset")} · {titleCase(String(item.reason || "blocked"))}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function SummaryCard({ label, value, caption }: { label: string; value: string; caption?: string }) {
  return (
    <div className="rounded-2xl border border-zinc-800 bg-zinc-950/55 p-4">
      <div className="text-xs uppercase tracking-[0.15em] text-zinc-500">{label}</div>
      <div className="mt-2 text-2xl font-semibold text-zinc-100">{value}</div>
      {caption ? <div className="mt-2 text-sm text-zinc-400">{caption}</div> : null}
    </div>
  );
}

function MetricCard({
  label,
  value,
  compact = false,
}: {
  label: string;
  value: string;
  compact?: boolean;
}) {
  return (
    <div className={`rounded-2xl border border-zinc-800 bg-zinc-950/70 ${compact ? "p-3" : "p-4"}`}>
      <div className="text-[11px] uppercase tracking-[0.14em] text-zinc-500">{label}</div>
      <div className={`${compact ? "mt-2 text-base" : "mt-2 text-lg"} font-semibold text-zinc-100`}>{value}</div>
    </div>
  );
}

function GodArt({ alias, color }: { alias: string; color: string }) {
  if (alias === "Apollo") {
    return (
      <svg viewBox="0 0 220 180" className="h-44 w-full text-zinc-50">
        <circle cx="110" cy="90" r="38" fill={color} opacity="0.22" />
        <circle cx="110" cy="90" r="27" fill="none" stroke={color} strokeWidth="5" />
        {Array.from({ length: 12 }).map((_, index) => {
          const angle = (index / 12) * Math.PI * 2;
          const x1 = 110 + Math.cos(angle) * 42;
          const y1 = 90 + Math.sin(angle) * 42;
          const x2 = 110 + Math.cos(angle) * 70;
          const y2 = 90 + Math.sin(angle) * 70;
          return <line key={index} x1={x1} y1={y1} x2={x2} y2={y2} stroke={color} strokeWidth="4" strokeLinecap="round" />;
        })}
        <path d="M65 136 C92 122, 128 122, 155 136" fill="none" stroke="rgba(244,244,245,0.85)" strokeWidth="3" strokeLinecap="round" />
      </svg>
    );
  }
  if (alias === "Zeus") {
    return (
      <svg viewBox="0 0 220 180" className="h-44 w-full">
        <path d="M60 80 C60 58, 80 48, 96 58 C102 40, 130 38, 142 56 C160 50, 178 64, 174 84 C170 100, 154 108, 138 106 L82 106 C68 106, 56 96, 60 80Z" fill="rgba(255,255,255,0.12)" stroke={color} strokeWidth="3.5" />
        <path d="M116 76 L92 114 H116 L102 152 L142 104 H118 L136 76 Z" fill={color} opacity="0.88" />
      </svg>
    );
  }
  if (alias === "Hephaestus") {
    return (
      <svg viewBox="0 0 220 180" className="h-44 w-full">
        <path d="M66 128 H156 L145 146 H76 Z" fill="rgba(255,255,255,0.16)" />
        <path d="M88 66 H164 V108 H118 C101 108, 88 95, 88 78 Z" fill="none" stroke={color} strokeWidth="5" />
        <path d="M66 128 L90 82" fill="none" stroke="rgba(255,255,255,0.85)" strokeWidth="6" strokeLinecap="round" />
        <rect x="62" y="62" width="22" height="50" rx="8" fill="rgba(255,255,255,0.18)" />
        <path d="M126 38 C120 54, 138 62, 132 76 C148 64, 154 50, 144 38 C138 30, 128 30, 126 38Z" fill={color} />
      </svg>
    );
  }
  return (
    <svg viewBox="0 0 220 180" className="h-44 w-full">
      <path d="M70 84 C88 44, 138 44, 150 80" fill="none" stroke={color} strokeWidth="5" strokeLinecap="round" />
      <path d="M88 112 C106 96, 126 96, 144 112" fill="none" stroke="rgba(255,255,255,0.85)" strokeWidth="4" strokeLinecap="round" />
      <path d="M94 48 L110 30 L126 48" fill="none" stroke="rgba(255,255,255,0.85)" strokeWidth="4" strokeLinecap="round" />
      <path d="M82 72 C90 58, 102 58, 110 72 C118 58, 130 58, 138 72" fill="none" stroke={color} strokeWidth="4" strokeLinecap="round" />
      <path d="M110 76 V142" fill="none" stroke="rgba(255,255,255,0.8)" strokeWidth="4" strokeLinecap="round" />
      <path d="M96 96 C104 88, 116 88, 124 96" fill="none" stroke={color} strokeWidth="4" strokeLinecap="round" />
      <path d="M96 116 C104 108, 116 108, 124 116" fill="none" stroke={color} strokeWidth="4" strokeLinecap="round" />
    </svg>
  );
}
