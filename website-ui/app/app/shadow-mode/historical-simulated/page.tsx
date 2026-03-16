import Link from "next/link";
import type { ReactNode } from "react";
import { readShadowGodsHistoricalSnapshot } from "@/lib/server/data";

type AnyRecord = Record<string, unknown>;

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

function formatPct(value: unknown, digits = 1) {
  const numeric = asNumber(value);
  if (numeric == null) return "n/d";
  const sign = numeric > 0 ? "+" : "";
  return `${sign}${(numeric * 100).toFixed(digits)}%`;
}

function formatCount(value: unknown) {
  const numeric = asNumber(value);
  return numeric == null ? "0" : String(Math.round(numeric));
}

function titleCase(raw: string) {
  return raw
    .replace(/_/g, " ")
    .split(" ")
    .filter(Boolean)
    .map((chunk) => chunk.charAt(0).toUpperCase() + chunk.slice(1))
    .join(" ");
}

const GOD_COLORS: Record<string, string> = {
  Apollo: "border-amber-700/60 bg-amber-500/10 text-amber-50",
  Zeus: "border-sky-700/60 bg-sky-500/10 text-sky-50",
  Hephaestus: "border-orange-700/60 bg-orange-500/10 text-orange-50",
  Hermes: "border-emerald-700/60 bg-emerald-500/10 text-emerald-50",
};

function godTone(alias: string) {
  return GOD_COLORS[alias] || "border-zinc-700 bg-zinc-900/80 text-zinc-100";
}

export default async function ShadowModeHistoricalSimulatedPage() {
  const payload = (await readShadowGodsHistoricalSnapshot()) as AnyRecord;
  const gods = Array.isArray(payload.gods) ? (payload.gods as AnyRecord[]) : [];
  const overview = ((payload.overview as AnyRecord | undefined) || {}) as AnyRecord;
  const yearsOverview = Array.isArray(payload.years_overview) ? (payload.years_overview as AnyRecord[]) : [];
  const driver = ((payload.driver as AnyRecord | undefined) || {}) as AnyRecord;

  return (
    <div className="space-y-6">
      <section className="border-b border-zinc-800/80 px-5 py-5 md:px-6 md:py-6">
        <div className="text-xs uppercase tracking-[0.18em] text-zinc-500">Historico simulado</div>
        <h1 className="mt-2 text-2xl font-semibold text-zinc-100 md:text-3xl">
          Como cada Deus teria operado em 2023, 2024 e 2025 sem olhar o futuro
        </h1>
        <p className="mt-3 max-w-4xl text-sm text-zinc-300">
          Este replay usa os Deuses congelados de hoje, mas dirige cada dia com um proxy causal do motor oficial,
          deslocado em um dia. O resultado mostra quantas recomendacoes teriam ido para a corretora, quantos fills
          teriam saído, em quantos dias cada Deus teria operado e como o capital teria variado.
        </p>
        <p className="mt-3 max-w-4xl rounded-2xl border border-zinc-800 bg-zinc-950/55 px-4 py-3 text-sm text-zinc-300">
          Driver: observer <span className="text-zinc-100">{String(driver.observer_mode || "n/d")}</span>
          {" · "}
          janela <span className="text-zinc-100">{String(payload.window_start || "n/d")}</span>
          {" -> "}
          <span className="text-zinc-100">{String(payload.window_end || "n/d")}</span>
        </p>
        <div className="mt-4 flex flex-wrap gap-3">
          <Link
            href="/app/shadow-mode"
            className="rounded-xl border border-zinc-700 px-4 py-2 text-sm text-zinc-100 transition hover:border-zinc-500 hover:bg-zinc-800/70"
          >
            Voltar para os Deuses live
          </Link>
          <a
            href="/data/site/latest_shadow_gods_historical.json"
            className="rounded-xl border border-zinc-700 px-4 py-2 text-sm text-zinc-100 transition hover:border-zinc-500 hover:bg-zinc-800/70"
          >
            Baixar JSON publico
          </a>
        </div>
      </section>

      <section className="grid gap-4 px-5 md:grid-cols-4 md:px-6">
        <SummaryCard label="Deuses" value={formatCount(overview.god_count)} />
        <SummaryCard label="Cenarios" value={formatCount(overview.scenario_count)} />
        <SummaryCard label="Ordens totais" value={formatCount(overview.order_count_total)} />
        <SummaryCard label="Fills totais" value={formatCount(overview.fill_count_total)} />
      </section>

      <section className="grid gap-4 px-5 md:grid-cols-3 md:px-6">
        {yearsOverview.map((year) => (
          <div key={String(year.year || Math.random())} className="rounded-2xl border border-zinc-800 bg-zinc-950/60 p-4">
            <div className="text-xs uppercase tracking-[0.14em] text-zinc-500">{String(year.year || "ano")}</div>
            <div className="mt-2 text-2xl font-semibold text-zinc-100">{formatCount(year.trade_days_total)} dias com trade</div>
            <div className="mt-3 text-sm text-zinc-300">
              {formatCount(year.order_count_total)} ordens propostas · {formatCount(year.fill_count_total)} fills
            </div>
          </div>
        ))}
      </section>

      {gods.length ? (
        <section className="grid gap-5 px-5 pb-6 md:px-6">
          {gods.map((god) => (
            <HistoricalGodCard key={String(god.alias || Math.random())} god={god} />
          ))}
        </section>
      ) : (
        <section className="px-5 pb-6 md:px-6">
          <div className="rounded-3xl border border-zinc-800 bg-zinc-950/60 p-6 text-sm text-zinc-300">
            O replay historico ainda nao foi gerado. Rode o agente `daily_shadow_gods_historical` e publique o snapshot.
          </div>
        </section>
      )}
    </div>
  );
}

function HistoricalGodCard({ god }: { god: AnyRecord }) {
  const alias = String(god.alias || "Shadow");
  const scenarios = Array.isArray(god.scenarios) ? (god.scenarios as AnyRecord[]) : [];

  return (
    <article className={`rounded-[28px] border bg-zinc-950/60 p-5 md:p-6 ${godTone(alias)}`}>
      <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
        <div>
          <div className="text-xs uppercase tracking-[0.16em] text-zinc-400">{titleCase(String(god.role || "shadow"))}</div>
          <div className="mt-2 text-3xl font-semibold text-zinc-50">{alias}</div>
          <p className="mt-2 max-w-3xl text-sm text-zinc-200/90">{String(god.thesis || "Sem tese publicada.")}</p>
          <p className="mt-2 text-sm text-zinc-300">{String(god.candidate_id || "n/d")}</p>
        </div>
        <div className="grid gap-3 sm:grid-cols-2">
          {scenarios.map((scenario) => {
            const overall = ((scenario.overall as AnyRecord | undefined) || {}) as AnyRecord;
            return (
              <SummaryCard
                key={String(scenario.scenario_id || Math.random())}
                label={`Bloco ${formatBRL(scenario.capital_brl)}`}
                value={formatPct(overall.total_return)}
                caption={`${formatCount(overall.order_count)} ordens · ${formatCount(overall.trade_days)} dias com trade`}
              />
            );
          })}
        </div>
      </div>

      <div className="mt-5 grid gap-4 xl:grid-cols-3">
        {scenarios.map((scenario) => (
          <HistoricalScenarioCard key={String(scenario.scenario_id || Math.random())} scenario={scenario} />
        ))}
      </div>
    </article>
  );
}

function HistoricalScenarioCard({ scenario }: { scenario: AnyRecord }) {
  const years = Array.isArray(scenario.years) ? (scenario.years as AnyRecord[]) : [];
  const overall = ((scenario.overall as AnyRecord | undefined) || {}) as AnyRecord;

  return (
    <div className="rounded-[24px] border border-zinc-800/90 bg-zinc-950/70 p-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-[11px] uppercase tracking-[0.16em] text-zinc-500">Capital simulado</div>
          <div className="mt-1 text-2xl font-semibold text-zinc-100">{formatBRL(scenario.capital_brl)}</div>
        </div>
        <div className="text-right text-sm text-zinc-300">
          <div>{formatPct(overall.total_return)}</div>
          <div className="text-xs text-zinc-500">{formatCount(overall.order_count)} ordens no total</div>
        </div>
      </div>

      <div className="mt-4 grid grid-cols-2 gap-3">
        <MetricCard label="NAV final" value={formatBRL(overall.end_nav_brl, 2)} />
        <MetricCard label="Drawdown max" value={formatPct(overall.max_drawdown)} />
        <MetricCard label="Dias simulados" value={formatCount(overall.days_total)} />
        <MetricCard label="Dias com trade" value={formatCount(overall.trade_days)} />
      </div>

      <div className="mt-4 space-y-4">
        {years.map((year) => (
          <div key={String(year.year || Math.random())} className="rounded-2xl border border-zinc-800 bg-zinc-900/45 p-3">
            <div className="flex items-center justify-between">
              <div className="text-sm font-medium text-zinc-100">{String(year.year || "ano")}</div>
              <div className="text-sm text-zinc-300">{formatPct(year.total_return)}</div>
            </div>
            <div className="mt-3 grid grid-cols-2 gap-3 md:grid-cols-4">
              <MetricCard label="Ordens" value={formatCount(year.order_count)} compact />
              <MetricCard label="Fills" value={formatCount(year.fill_count)} compact />
              <MetricCard label="Dias com trade" value={formatCount(year.trade_days)} compact />
              <MetricCard label="No-trade" value={formatCount(year.no_trade_days)} compact />
            </div>
            <div className="mt-3 grid gap-3 md:grid-cols-2">
              <div className="rounded-xl border border-zinc-800 bg-zinc-950/70 p-3">
                <div className="text-[11px] uppercase tracking-[0.14em] text-zinc-500">Top ordens para corretora</div>
                <div className="mt-2 space-y-2 text-sm text-zinc-300">
                  {Array.isArray(year.top_requested_tickers) && (year.top_requested_tickers as AnyRecord[]).length ? (
                    (year.top_requested_tickers as AnyRecord[]).slice(0, 4).map((row) => (
                      <div key={`${String(year.year)}-${String(row.ticker)}`} className="flex items-center justify-between">
                        <span>{String(row.ticker || "ticker")}</span>
                        <span>{formatCount(row.count)}</span>
                      </div>
                    ))
                  ) : (
                    <div className="text-zinc-500">Sem ordens emitidas.</div>
                  )}
                </div>
              </div>
              <div className="rounded-xl border border-zinc-800 bg-zinc-950/70 p-3">
                <div className="text-[11px] uppercase tracking-[0.14em] text-zinc-500">Arquivos do ano</div>
                <div className="mt-2 flex flex-wrap gap-2 text-xs">
                  <ArtifactLink href={String(((year.artifacts as AnyRecord | undefined) || {}).public_history_csv || "")}>historico</ArtifactLink>
                  <ArtifactLink href={String(((year.artifacts as AnyRecord | undefined) || {}).public_recommendations_csv || "")}>recomendacoes</ArtifactLink>
                  <ArtifactLink href={String(((year.artifacts as AnyRecord | undefined) || {}).public_requests_csv || "")}>ordens</ArtifactLink>
                  <ArtifactLink href={String(((year.artifacts as AnyRecord | undefined) || {}).public_fills_csv || "")}>fills</ArtifactLink>
                </div>
              </div>
            </div>
            <div className="mt-3 rounded-xl border border-zinc-800 bg-zinc-950/70 p-3">
              <div className="text-[11px] uppercase tracking-[0.14em] text-zinc-500">Ultimas recomendacoes do ano</div>
              <div className="mt-2 space-y-2 text-xs text-zinc-300">
                {Array.isArray(year.recommendation_tail) && (year.recommendation_tail as AnyRecord[]).length ? (
                  (year.recommendation_tail as AnyRecord[]).slice(-4).reverse().map((row, idx) => (
                    <div key={`${String(year.year)}-tail-${idx}`} className="rounded-lg border border-zinc-800 bg-zinc-950/80 px-3 py-2">
                      {String(row.as_of_date || "data")} · {titleCase(String(row.market_state || "unknown"))}
                      {" · "}
                      {String(row.selected_assets || "sem ativo")}
                      {" · "}
                      {formatCount(row.order_count)} ordens
                    </div>
                  ))
                ) : (
                  <div className="text-zinc-500">Sem recomendacoes publicadas.</div>
                )}
              </div>
            </div>
          </div>
        ))}
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

function MetricCard({ label, value, compact = false }: { label: string; value: string; compact?: boolean }) {
  return (
    <div className={`rounded-2xl border border-zinc-800 bg-zinc-950/70 ${compact ? "p-3" : "p-4"}`}>
      <div className="text-[11px] uppercase tracking-[0.14em] text-zinc-500">{label}</div>
      <div className={`${compact ? "mt-2 text-base" : "mt-2 text-lg"} font-semibold text-zinc-100`}>{value}</div>
    </div>
  );
}

function ArtifactLink({ href, children }: { href: string; children: ReactNode }) {
  if (!href) {
    return <span className="rounded-full border border-zinc-800 px-2 py-1 text-zinc-500">{children}</span>;
  }
  return (
    <a
      href={href}
      className="rounded-full border border-zinc-700 px-2 py-1 text-zinc-200 transition hover:border-zinc-500 hover:bg-zinc-800/70"
    >
      {children}
    </a>
  );
}
