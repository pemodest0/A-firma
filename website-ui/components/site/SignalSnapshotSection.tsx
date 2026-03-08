import Image from "next/image";
import Link from "next/link";
import HelpHint from "@/components/ui/HelpHint";
import {
  describeStrategy,
  humanizeEngineState,
  humanizeGroupName,
  humanizeMethodology,
  humanizeRiskLevel,
  humanizeStrategyName,
} from "@/lib/enginePresentation";

type SnapshotSectionProps = {
  snapshot: Record<string, unknown>;
  compact?: boolean;
};

function toNumber(value: unknown) {
  const num = Number(value);
  return Number.isFinite(num) ? num : null;
}

function pct(value: unknown, digits = 0) {
  const num = toNumber(value);
  if (num == null) return "n/d";
  return `${(num * 100).toFixed(digits)}%`;
}

function shortRisk(level: unknown) {
  const text = humanizeRiskLevel(level);
  if (text.includes("baixo")) return "Mais espaço para risco, ainda com disciplina.";
  if (text.includes("alto")) return "Momento de defesa. O motor não está lendo terreno limpo.";
  if (text.includes("sem leitura")) return "Sem leitura limpa de risco. Melhor operar pequeno e observar.";
  return "Terreno intermediário. Dá para operar, mas sem exagero.";
}

function confidenceLabel(value: unknown) {
  const num = toNumber(value);
  if (num == null) return "sem confiança publicada";
  if (num >= 0.8) return "alta";
  if (num >= 0.6) return "moderada";
  return "baixa";
}

export default function SignalSnapshotSection({ snapshot, compact = false }: SnapshotSectionProps) {
  const finance = (snapshot.finance as Record<string, unknown> | undefined) || {};
  const research = (snapshot.profit_research as Record<string, unknown> | undefined) || {};
  const layered = (snapshot.layered_engine as Record<string, unknown> | undefined) || {};
  const charts = (snapshot.charts as Record<string, unknown> | undefined) || {};
  const proof = (snapshot.proof as Record<string, unknown> | undefined) || {};
  const universeExpansion = (snapshot.universe_expansion as Record<string, unknown> | undefined) || {};

  const currentPlaybook = (finance.latest_playbook as Record<string, unknown> | undefined) || {};
  const topResearch = (research.top_candidate as Record<string, unknown> | undefined) || {};
  const attack = (layered.best_meta_candidate as Record<string, unknown> | undefined) || {};
  const robust = (layered.drawdown_best_balanced as Record<string, unknown> | undefined) || {};
  const historyWindow = (proof.history_window as Record<string, unknown> | undefined) || {};
  const proofResearch = (proof.group_suite_best as Record<string, unknown> | undefined) || {};
  const universe800 = (universeExpansion.target_800_cov075 as Record<string, unknown> | undefined) || {};
  const sectors = Array.isArray(charts.sector_pressure) ? (charts.sector_pressure as Record<string, unknown>[]) : [];
  const cryptos = Array.isArray(charts.crypto_watchlist) ? (charts.crypto_watchlist as Record<string, unknown>[]) : [];
  const exposure = toNumber(currentPlaybook.exposure);
  const cash = exposure == null ? null : Math.max(0, 1 - exposure);
  const signalReliability = toNumber(currentPlaybook.signal_reliability) ?? toNumber(finance.confidence_score);
  const maxImpact = Math.max(0.01, ...sectors.map((row) => toNumber(row.impact_score) || 0.01));
  const exposureWidth = exposure == null ? 0 : Math.max(0, Math.min(100, exposure * 100));
  const cashWidth = cash == null ? 0 : Math.max(0, Math.min(100, cash * 100));

  return (
    <section className="rounded-[26px] border border-zinc-800/80 bg-zinc-950/55 p-6 md:p-7">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="text-xs uppercase tracking-[0.24em] text-zinc-500">Agora no motor</div>
          <h2 className="mt-2 text-2xl font-semibold tracking-tight text-zinc-100 md:text-3xl">
            Leitura atual, modos do motor e pressão do mercado
          </h2>
        </div>
        <div className="rounded-full border border-zinc-800 bg-black/25 px-3 py-1 text-xs text-zinc-300">
          Data-base {String(snapshot.as_of_date || finance.data_last_date || "n/d")}
        </div>
      </div>

      <div className={`mt-6 grid gap-4 ${compact ? "lg:grid-cols-[1.15fr_0.85fr]" : "lg:grid-cols-[1.05fr_0.95fr]"}`}>
        <article className="overflow-hidden rounded-2xl border border-cyan-950/60 bg-[#07101F]">
          <div className="grid gap-4 p-5 md:grid-cols-[1.05fr_0.95fr] md:p-6">
            <div>
              <div className="text-xs uppercase tracking-[0.16em] text-cyan-200/70">Diagnóstico</div>
              <div className="mt-3 text-2xl font-semibold text-zinc-100">
                {humanizeEngineState(finance.operational_state || "monitoramento")}
              </div>
              <p className="mt-3 text-sm text-zinc-300">{shortRisk(finance.risk_level_next_month)}</p>
              <div className="mt-5 grid grid-cols-2 gap-3">
                <div className="rounded-xl border border-zinc-800/80 bg-black/25 p-3">
                  <div className="text-[11px] uppercase tracking-[0.14em] text-zinc-500">Risco alvo</div>
                  <div className="mt-2 text-2xl font-semibold text-zinc-100">{pct(exposure)}</div>
                  <div className="mt-3 h-2 rounded-full bg-zinc-900">
                    <div className="h-2 rounded-full bg-cyan-400" style={{ width: `${exposureWidth}%` }} />
                  </div>
                </div>
                <div className="rounded-xl border border-zinc-800/80 bg-black/25 p-3">
                  <div className="text-[11px] uppercase tracking-[0.14em] text-zinc-500">Caixa alvo</div>
                  <div className="mt-2 text-2xl font-semibold text-zinc-100">{pct(cash)}</div>
                  <div className="mt-3 h-2 rounded-full bg-zinc-900">
                    <div className="h-2 rounded-full bg-zinc-500" style={{ width: `${cashWidth}%` }} />
                  </div>
                </div>
              </div>
              <div className="mt-4 flex flex-wrap items-center gap-3 text-xs text-zinc-400">
                <div className="flex items-center gap-2">
                  <span>Confiabilidade publicada</span>
                  <HelpHint text="Leitura de confiança derivada do regime e dos checks publicados. Não é garantia; é grau de conforto operacional." />
                </div>
                <span className="rounded-full border border-zinc-800 px-2 py-1 text-zinc-200">
                  {signalReliability == null ? "sem dado publicado" : `${Math.round(signalReliability * 100)}% · ${confidenceLabel(signalReliability)}`}
                </span>
              </div>
            </div>
            <div className="relative min-h-[220px] overflow-hidden rounded-2xl border border-zinc-800/70 bg-zinc-950/70">
              <Image
                src="/assets/prints/regime-risk.svg"
                alt="Visual do regime e risco do Eigen Engine"
                fill
                className="object-cover opacity-80"
              />
              <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black via-black/70 to-transparent p-4">
                <div className="text-[11px] uppercase tracking-[0.16em] text-zinc-400">Leitura simples</div>
                <div className="mt-1 text-sm text-zinc-100">
                  {finance.gate_blocked ? "Modo diagnóstico. Serve para estudar, não para confiar cegamente." : "Trilha íntegra. Dá para usar como contexto operacional."}
                </div>
              </div>
            </div>
          </div>
        </article>

        <div className="grid gap-4">
          <article className="overflow-hidden rounded-2xl border border-zinc-800 bg-zinc-950/65">
            <div className="grid grid-cols-[1fr_132px] gap-3 p-4">
              <div>
                <div className="text-xs uppercase tracking-[0.16em] text-zinc-500">Modo ataque</div>
                <div className="mt-2 text-lg font-semibold text-zinc-100">{humanizeStrategyName(attack.candidate_id)}</div>
                <div className="mt-3 grid grid-cols-3 gap-2 text-xs text-zinc-300">
                  <div><span className="block text-zinc-500">Retorno</span>{pct(attack.net_ann_return)}</div>
                  <div><span className="block text-zinc-500">Sharpe</span>{toNumber(attack.net_sharpe)?.toFixed(2) || "sem dado"}</div>
                  <div><span className="block text-zinc-500">Drawdown</span>{pct(attack.net_max_drawdown)}</div>
                </div>
                <div className="mt-3 text-xs leading-relaxed text-zinc-400">{describeStrategy(attack.candidate_id, attack.notes)}</div>
                <div className="mt-3 text-xs text-zinc-500">
                  Benchmark: {String(attack.benchmark_ticker || "não publicado")} · 10x histórico em {toNumber(attack.years_to_10x_full)?.toFixed(1) || "sem dado"} anos.
                </div>
              </div>
              <div className="relative min-h-[110px] overflow-hidden rounded-xl border border-zinc-800">
                <Image src="/assets/prints/walkforward-metrics.svg" alt="Métricas de validação" fill className="object-cover" />
              </div>
            </div>
          </article>

          <article className="overflow-hidden rounded-2xl border border-zinc-800 bg-zinc-950/65">
            <div className="grid grid-cols-[1fr_132px] gap-3 p-4">
              <div>
                <div className="text-xs uppercase tracking-[0.16em] text-zinc-500">Modo robusto</div>
                <div className="mt-2 text-lg font-semibold text-zinc-100">{humanizeStrategyName(robust.candidate_id)}</div>
                <div className="mt-3 grid grid-cols-3 gap-2 text-xs text-zinc-300">
                  <div><span className="block text-zinc-500">Retorno</span>{pct(robust.net_ann_return)}</div>
                  <div><span className="block text-zinc-500">Sharpe</span>{toNumber(robust.net_sharpe)?.toFixed(2) || "sem dado"}</div>
                  <div><span className="block text-zinc-500">Drawdown</span>{pct(robust.net_max_drawdown)}</div>
                </div>
                <div className="mt-3 text-xs leading-relaxed text-zinc-400">{describeStrategy(robust.candidate_id, robust.notes)}</div>
                <div className="mt-3 text-xs text-zinc-500">
                  Testes favoráveis: {pct(robust.positive_test_share)} · melhora média de edge: {pct(robust.mean_test_edge)}.
                </div>
              </div>
              <div className="relative min-h-[110px] overflow-hidden rounded-xl border border-zinc-800">
                <Image src="/assets/prints/dashboard-main.svg" alt="Painel principal do Eigen Engine" fill className="object-cover" />
              </div>
            </div>
          </article>

          <article className="rounded-2xl border border-zinc-800 bg-zinc-950/65 p-4">
            <div className="text-xs uppercase tracking-[0.16em] text-zinc-500">Pesquisa viva</div>
            <div className="mt-2 text-lg font-semibold text-zinc-100">{humanizeStrategyName(topResearch.candidate_id || topResearch.label)}</div>
            <div className="mt-3 flex flex-wrap gap-2 text-xs text-zinc-300">
              <span className="rounded-full border border-zinc-800 px-2 py-1">{humanizeMethodology(topResearch.methodology)}</span>
              {String(topResearch.groups || "")
                .split(",")
                .map((item) => item.trim())
                .filter(Boolean)
                .slice(0, compact ? 2 : 3)
                .map((item) => (
                  <span key={item} className="rounded-full border border-zinc-800 px-2 py-1">
                    {humanizeGroupName(item)}
                  </span>
                ))}
            </div>
            <div className="mt-3 text-xs leading-relaxed text-zinc-400">{describeStrategy(topResearch.candidate_id || topResearch.label, topResearch.notes)}</div>
            <div className="mt-3 text-sm text-zinc-400">
              O copiloto usa esse banco de pesquisa para separar o que vale continuar testando do que já morreu.
            </div>
            <div className="mt-3 text-xs text-zinc-500">
              OOS consistente: {String((topResearch.oos as Record<string, unknown> | undefined)?.appearances || "sem dado")} blocos ·
              média de teste {pct((topResearch.oos as Record<string, unknown> | undefined)?.mean_test_net_ann_return)}.
            </div>
          </article>
        </div>
      </div>

      <div className={`mt-4 grid gap-4 ${compact ? "lg:grid-cols-[1.1fr_0.9fr]" : "lg:grid-cols-2"}`}>
        <article className="rounded-2xl border border-zinc-800 bg-zinc-950/65 p-5">
          <div className="text-xs uppercase tracking-[0.16em] text-zinc-500">Pressão por setor</div>
          <div className="mt-4 space-y-3">
            {sectors.slice(0, compact ? 4 : 6).map((row) => {
              const impact = toNumber(row.impact_score) || 0;
              return (
                <div key={String(row.sector || impact)}>
                  <div className="flex items-center justify-between text-sm text-zinc-200">
                    <span>{humanizeGroupName(row.sector)}</span>
                    <span className="text-zinc-400">{(impact * 100).toFixed(0)}</span>
                  </div>
                  <div className="mt-1 h-2 rounded-full bg-zinc-900">
                    <div
                      className="h-2 rounded-full bg-gradient-to-r from-cyan-500 via-sky-400 to-emerald-400"
                      style={{ width: `${Math.max(8, (impact / maxImpact) * 100)}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </article>

        <article className="rounded-2xl border border-zinc-800 bg-zinc-950/65 p-5">
          <div className="flex items-center justify-between gap-3">
            <div className="text-xs uppercase tracking-[0.16em] text-zinc-500">Watchlist cripto</div>
            <Link href="/app/cripto" className="text-xs text-cyan-300 hover:text-cyan-200">
              abrir cripto
            </Link>
          </div>
          <div className="mt-4 flex flex-wrap gap-2">
            {cryptos.slice(0, compact ? 8 : 12).map((row) => (
              <div key={String(row.asset || "")} className="rounded-xl border border-zinc-800 bg-black/25 px-3 py-2">
                <div className="text-sm font-semibold text-zinc-100">{String(row.asset || "")}</div>
                <div className="text-[11px] uppercase tracking-[0.12em] text-zinc-500">
                  confiança {pct(row.confidence)}
                </div>
              </div>
            ))}
            {!cryptos.length ? (
              <div className="text-sm text-zinc-400">Sem watchlist cripto limpa publicada no snapshot atual.</div>
            ) : null}
          </div>
        </article>
      </div>

      <div className={`mt-4 grid gap-3 ${compact ? "md:grid-cols-3" : "lg:grid-cols-3"}`}>
        <div className="rounded-2xl border border-zinc-800 bg-zinc-950/65 p-4">
          <div className="text-[11px] uppercase tracking-[0.14em] text-zinc-500">Prova longa</div>
          <div className="mt-2 text-lg font-semibold text-zinc-100">
            {String(historyWindow.start || "2016-02-18")} → {String(historyWindow.end || snapshot.as_of_date || "n/d")}
          </div>
          <div className="mt-2 text-xs text-zinc-400">Janela longa para não vender sorte curta como se fosse método.</div>
        </div>
        <div className="rounded-2xl border border-zinc-800 bg-zinc-950/65 p-4">
          <div className="text-[11px] uppercase tracking-[0.14em] text-zinc-500">Líquido e bruto</div>
          <div className="mt-2 text-lg font-semibold text-zinc-100">{pct(proofResearch.net_blended_ann_return || proofResearch.net_ann_return || topResearch.net_ann_return)}</div>
          <div className="mt-2 text-xs text-zinc-400">
            Líquido anual. Bruto: {pct(proofResearch.gross_ann_return || topResearch.gross_ann_return)}.
          </div>
        </div>
        <div className="rounded-2xl border border-zinc-800 bg-zinc-950/65 p-4">
          <div className="text-[11px] uppercase tracking-[0.14em] text-zinc-500">Universo 800</div>
          <div className="mt-2 text-lg font-semibold text-zinc-100">{String(universe800.assets_ok || "n/d")} ativos</div>
          <div className="mt-2 text-xs text-zinc-400">
            {String(universe800.sector_count || "n/d")} setores e maior concentração em {pct(universe800.largest_sector_share)}.
          </div>
        </div>
      </div>
    </section>
  );
}
