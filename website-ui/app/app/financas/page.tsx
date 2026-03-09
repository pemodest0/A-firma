import { humanizeConfidenceLevel, humanizeGroupName, humanizeModeName, humanizeStrategyName } from "@/lib/enginePresentation";
import { readSiteFinanceSnapshot } from "@/lib/server/data";
import SectorDashboard from "@/components/SectorDashboard";

function pct(value: unknown, digits = 1) {
  const n = Number(value);
  return Number.isFinite(n) ? `${(n * 100).toFixed(digits)}%` : "n/d";
}

export default async function FinancasPage() {
  const snapshot = (await readSiteFinanceSnapshot()) as Record<string, unknown>;
  const proof = ((snapshot.proof as Record<string, unknown> | undefined) || {}) as Record<string, unknown>;
  const researchBest = ((proof.group_suite_best as Record<string, unknown> | undefined) || {}) as Record<string, unknown>;
  const universe800 = ((snapshot.universe_expansion as Record<string, unknown> | undefined)?.target_800_cov075 || {}) as Record<string, unknown>;
  const confidence = ((snapshot.confidence as Record<string, unknown> | undefined) || {}) as Record<string, unknown>;
  const recommendedLiveMode = ((confidence.recommended_live_mode as Record<string, unknown> | undefined) || {}) as Record<string, unknown>;
  const modeConfidence = ((confidence.mode_confidence as Record<string, unknown> | undefined) || {}) as Record<string, unknown>;
  const vigilanceAlerts = Array.isArray(confidence.vigilance_alerts)
    ? (confidence.vigilance_alerts as Array<Record<string, unknown>>)
    : [];
  const recommendedLabel = humanizeModeName(recommendedLiveMode.mode, recommendedLiveMode.label);
  const confidenceLevel = humanizeConfidenceLevel(
    recommendedLiveMode.confidence_level || modeConfidence.confidence_level || "sem leitura",
  );
  const confidenceScore =
    typeof recommendedLiveMode.confidence_score === "number"
      ? recommendedLiveMode.confidence_score
      : typeof modeConfidence.confidence_score === "number"
        ? modeConfidence.confidence_score
        : null;

  return (
    <div className="space-y-5">
      <section className="grid gap-4 px-4 pt-4 md:grid-cols-3 md:px-5">
        <article className="rounded-2xl border border-zinc-800 bg-zinc-950/55 p-5">
          <div className="text-xs uppercase tracking-[0.16em] text-zinc-500">Prova desde 2016</div>
          <div className="mt-2 text-xl font-semibold text-zinc-100">
            {String((proof.history_window as Record<string, unknown> | undefined)?.start || "2016-02-18")} {"→"}{" "}
            {String((proof.history_window as Record<string, unknown> | undefined)?.end || "n/d")}
          </div>
          <p className="mt-3 text-sm text-zinc-300">
            Janela longa usada para mostrar que o motor não está vendendo um truque de seis meses.
          </p>
        </article>

        <article className="rounded-2xl border border-zinc-800 bg-zinc-950/55 p-5">
          <div className="text-xs uppercase tracking-[0.16em] text-zinc-500">Pesquisa líquida e bruta</div>
          <div className="mt-2 text-xl font-semibold text-zinc-100">{humanizeStrategyName(researchBest.candidate_id || "n/d")}</div>
          <p className="mt-3 text-sm text-zinc-300">
            Líquido: <span className="text-zinc-100">{pct(researchBest.net_blended_ann_return || researchBest.net_ann_return)}</span>
            {" · "}
            Bruto: <span className="text-zinc-100">{pct(researchBest.gross_ann_return)}</span>
          </p>
          <p className="mt-2 text-sm text-zinc-400">
            Grupos: {String(researchBest.groups || "").split(",").filter(Boolean).map((item) => humanizeGroupName(item)).join(" · ") || "n/d"}
          </p>
        </article>

        <article className="rounded-2xl border border-zinc-800 bg-zinc-950/55 p-5">
          <div className="text-xs uppercase tracking-[0.16em] text-zinc-500">Universo 800 ativos</div>
          <div className="mt-2 text-xl font-semibold text-zinc-100">{String(universe800.assets_ok || "n/d")} ativos</div>
          <p className="mt-3 text-sm text-zinc-300">
            {String(universe800.sector_count || "n/d")} setores e concentração do maior setor em {pct(universe800.largest_sector_share)}.
          </p>
          <p className="mt-2 text-sm text-zinc-400">
            Histórico preservado: {String(universe800.period_start || "n/d")} {"→"} {String(universe800.period_end || "n/d")}
          </p>
        </article>
      </section>

      <section className="grid gap-4 px-4 md:grid-cols-2 md:px-5">
        <article className="rounded-2xl border border-zinc-800 bg-zinc-950/55 p-5">
          <div className="text-xs uppercase tracking-[0.16em] text-zinc-500">Leitura operacional de hoje</div>
          <div className="mt-2 text-xl font-semibold text-zinc-100">{recommendedLabel}</div>
          <p className="mt-3 text-sm text-zinc-300">
            Confiança {confidenceLevel}
            {confidenceScore == null ? "" : ` (${Math.round(confidenceScore * 100)}%)`}. O motor já não fala só “risco alto ou baixo”; ele também indica qual modo parece mais adequado para o dia.
          </p>
          <p className="mt-3 text-sm text-zinc-400">
            Cenário central: {String(modeConfidence.scenario_base || "seguir o modo recomendado e monitorar a vigilância.")}
          </p>
        </article>

        <article className="rounded-2xl border border-zinc-800 bg-zinc-950/55 p-5">
          <div className="text-xs uppercase tracking-[0.16em] text-zinc-500">Alertas que pedem respeito</div>
          {vigilanceAlerts.length ? (
            <div className="mt-3 space-y-2 text-sm text-zinc-300">
              {vigilanceAlerts.slice(0, 4).map((alert, idx) => (
                <p key={`${String(alert.code || idx)}`}>- {String(alert.message || "Alerta sem detalhe.")}</p>
              ))}
            </div>
          ) : (
            <p className="mt-3 text-sm text-zinc-300">
              Sem alerta crítico aberto. Isso não é sinal para relaxar; é só sinal de que o contexto está mais limpo agora.
            </p>
          )}
          <p className="mt-3 text-sm text-zinc-400">
            Se a vigilância apertar, o uso certo é reduzir agressividade antes do estrago, não depois.
          </p>
        </article>
      </section>

      <SectorDashboard
        title="Eigen Engine | Finanças"
        showTable
        initialDomain="finance"
        headline="Painel financeiro por ativo"
        description="Leitura de preço, risco, estabilidade e contexto estrutural para finanças globais."
      />
    </div>
  );
}
