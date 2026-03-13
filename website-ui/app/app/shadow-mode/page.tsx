import Link from "next/link";
import {
  humanizeConfidenceLevel,
  humanizeStatusWord,
  humanizeStrategyName,
} from "@/lib/enginePresentation";
import { readSiteFinanceSnapshot } from "@/lib/server/data";

function formatPct(value: unknown, digits = 1) {
  return typeof value === "number" && Number.isFinite(value) ? `${(value * 100).toFixed(digits)}%` : "n/d";
}

function formatDrawdown(value: unknown) {
  return typeof value === "number" && Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "n/d";
}

function formatScore(value: unknown) {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(2) : "n/d";
}

function formatDate(value: unknown) {
  const raw = String(value || "").trim();
  if (!raw) return "n/d";
  if (/^\d{4}-\d{2}$/.test(raw)) return raw;
  return raw.slice(0, 10) || raw;
}

function statusTone(status: string) {
  const normalized = String(status || "").toLowerCase();
  if (normalized === "running") return "border-emerald-700/70 bg-emerald-500/10 text-emerald-200";
  if (normalized === "research") return "border-sky-700/70 bg-sky-500/10 text-sky-200";
  if (normalized === "historical") return "border-amber-700/70 bg-amber-500/10 text-amber-200";
  return "border-zinc-700 bg-zinc-800/60 text-zinc-200";
}

export default async function ShadowModePage() {
  const snapshot = (await readSiteFinanceSnapshot()) as Record<string, unknown>;
  const modes = Array.isArray(snapshot.shadow_modes)
    ? (snapshot.shadow_modes as Array<Record<string, unknown>>)
    : [];
  const overview = ((snapshot.shadow_mode_overview as Record<string, unknown> | undefined) || {}) as Record<
    string,
    unknown
  >;
  const confidence = ((snapshot.confidence as Record<string, unknown> | undefined) || {}) as Record<string, unknown>;
  const recommendedLiveMode =
    ((confidence.recommended_live_mode as Record<string, unknown> | undefined) || {}) as Record<string, unknown>;
  const vigilanceAlerts = Array.isArray(confidence.vigilance_alerts)
    ? (confidence.vigilance_alerts as Array<Record<string, unknown>>)
    : [];

  return (
    <div className="space-y-5">
      <section className="border-b border-zinc-800/80 px-5 py-5 md:px-6 md:py-6">
        <div className="text-xs uppercase tracking-[0.18em] text-zinc-500">Shadow mode</div>
        <h1 className="mt-2 text-2xl font-semibold text-zinc-100 md:text-3xl">
          Tudo o que está rodando, acumulando e sendo comparado
        </h1>
        <p className="mt-3 max-w-3xl text-sm text-zinc-300">
          Esta página junta os modos vivos do laboratório, o que cada um tenta fazer, o que varia entre eles e qual foi
          a última ação conhecida. A ideia aqui é simples: ver o motor como carteira viva de hipóteses, não como caixa
          preta.
        </p>
      </section>

      <section className="grid gap-4 px-5 md:grid-cols-4 md:px-6">
        <SummaryCard label="Modos listados" value={String(overview.total || modes.length || 0)} />
        <SummaryCard label="Rodando agora" value={String(overview.running || 0)} />
        <SummaryCard label="Acumulando" value={String(overview.accumulating || 0)} />
        <SummaryCard
          label="Modo recomendado hoje"
          value={String(recommendedLiveMode.label || recommendedLiveMode.mode || "n/d")}
          caption={`Confiança ${humanizeConfidenceLevel(recommendedLiveMode.confidence_level || "n/d")}`}
        />
      </section>

      {vigilanceAlerts.length ? (
        <section className="px-5 md:px-6">
          <div className="rounded-2xl border border-amber-700/60 bg-amber-500/10 p-4">
            <div className="text-sm font-medium text-amber-200">Alertas da vigilância</div>
            <div className="mt-2 space-y-2 text-sm text-amber-100/90">
              {vigilanceAlerts.slice(0, 4).map((alert, idx) => (
                <p key={`${String(alert.code || idx)}`}>- {String(alert.message || "Alerta sem detalhe.")}</p>
              ))}
            </div>
          </div>
        </section>
      ) : null}

      <section className="grid gap-4 px-5 md:px-6">
        {modes.map((mode) => {
          const label = String(mode.label || mode.slug || "Shadow");
          const weights = Array.isArray(mode.weights) ? (mode.weights as Array<Record<string, unknown>>) : [];
          const sourceLabel = humanizeStatusWord(mode.status || "n/d");
          return (
            <article key={String(mode.slug || label)} className="rounded-2xl border border-zinc-800 bg-zinc-950/55 p-5">
              <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
                <div className="space-y-2">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="text-xl font-semibold text-zinc-100">{label}</span>
                    <span className={`rounded-full border px-2 py-1 text-[11px] uppercase tracking-[0.12em] ${statusTone(String(mode.status || ""))}`}>
                      {sourceLabel}
                    </span>
                  </div>
                  <p className="max-w-3xl text-sm text-zinc-300">{String(mode.what_it_is || "Modo sem descrição publicada.")}</p>
                </div>
                <div className="rounded-xl border border-zinc-800 bg-zinc-900/60 px-3 py-2 text-xs text-zinc-300">
                  Última data conhecida: <span className="text-zinc-100">{formatDate(mode.latest_date)}</span>
                </div>
              </div>

              <div className="mt-4 grid gap-4 md:grid-cols-2">
                <div className="rounded-xl border border-zinc-800 bg-zinc-900/45 p-4">
                  <div className="text-xs uppercase tracking-[0.15em] text-zinc-500">O que varia</div>
                  <p className="mt-2 text-sm text-zinc-300">{String(mode.what_varies || "n/d")}</p>
                </div>
                <div className="rounded-xl border border-zinc-800 bg-zinc-900/45 p-4">
                  <div className="text-xs uppercase tracking-[0.15em] text-zinc-500">Última ação</div>
                  <p className="mt-2 text-sm text-zinc-300">{String(mode.last_action || "n/d")}</p>
                </div>
                <div className="rounded-xl border border-zinc-800 bg-zinc-900/45 p-4">
                  <div className="text-xs uppercase tracking-[0.15em] text-zinc-500">Leitura de hoje</div>
                  <p className="mt-2 text-sm text-zinc-300">{String(mode.forecast || "n/d")}</p>
                </div>
                <div className="rounded-xl border border-zinc-800 bg-zinc-900/45 p-4">
                  <div className="text-xs uppercase tracking-[0.15em] text-zinc-500">Identidade técnica</div>
                  <p className="mt-2 text-sm text-zinc-300">
                    {humanizeStrategyName(mode.candidate_id || "n/d")}
                    {mode.candidate_id ? (
                      <span className="block pt-2 text-xs text-zinc-500">{String(mode.candidate_id)}</span>
                    ) : null}
                  </p>
                </div>
              </div>

              <div className="mt-4 grid gap-3 md:grid-cols-4">
                <MetricCard label="Lucro médio anual" value={formatPct(mode.net_ann_return)} />
                <MetricCard label="Lucro acumulado" value={formatPct(mode.net_total_return)} />
                <MetricCard label="Qualidade" value={formatScore(mode.net_sharpe)} />
                <MetricCard label="Menor preço observado" value={formatDrawdown(mode.net_max_drawdown)} />
              </div>

              {String(mode.weights_preview || "").trim() ? (
                <details className="mt-4 rounded-xl border border-zinc-800 bg-zinc-900/35 p-4">
                  <summary className="cursor-pointer text-sm font-medium text-zinc-200">
                    Exibir mais sobre a carteira
                  </summary>
                  <p className="mt-3 text-sm text-zinc-300">{String(mode.weights_preview)}</p>
                  {weights.length ? (
                    <div className="mt-3 grid gap-2 md:grid-cols-3">
                      {weights.slice(0, 12).map((row) => (
                        <div
                          key={`${String(mode.slug)}-${String(row.asset)}`}
                          className="rounded-lg border border-zinc-800 bg-zinc-950/60 px-3 py-2 text-sm text-zinc-200"
                        >
                          <span className="font-medium">{String(row.asset)}</span>
                          <span className="ml-2 text-zinc-400">{formatPct(row.weight, 0)}</span>
                        </div>
                      ))}
                    </div>
                  ) : null}
                </details>
              ) : null}
            </article>
          );
        })}
      </section>

      <section className="px-5 pb-5 md:px-6 md:pb-6">
        <div className="rounded-2xl border border-zinc-800 bg-zinc-950/45 p-5">
          <div className="text-sm font-medium text-zinc-100">Como usar esta página</div>
          <div className="mt-3 space-y-2 text-sm text-zinc-300">
            <p>- Se quiser acompanhar o laboratório, olhe primeiro os modos marcados como “rodando agora”.</p>
            <p>- Se quiser comparar com o passado, olhe os modos históricos e de pesquisa.</p>
            <p>- Se quiser voltar para a leitura operacional do dia, use o painel principal ou a página de finanças.</p>
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
              Ver finanças
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
}

function SummaryCard({
  label,
  value,
  caption,
}: {
  label: string;
  value: string;
  caption?: string;
}) {
  return (
    <div className="rounded-2xl border border-zinc-800 bg-zinc-950/55 p-4">
      <div className="text-xs uppercase tracking-[0.15em] text-zinc-500">{label}</div>
      <div className="mt-2 text-2xl font-semibold text-zinc-100">{value}</div>
      {caption ? <div className="mt-2 text-sm text-zinc-400">{caption}</div> : null}
    </div>
  );
}

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-3">
      <div className="text-[11px] uppercase tracking-[0.14em] text-zinc-500">{label}</div>
      <div className="mt-2 text-lg font-semibold text-zinc-100">{value}</div>
    </div>
  );
}
