import { humanizeGroupName, humanizeStrategyName } from "@/lib/enginePresentation";
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
