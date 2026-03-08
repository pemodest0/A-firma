import Link from "next/link";
import EngineStoryDeck from "@/components/visuals/EngineStoryDeck";
import HelpHint from "@/components/ui/HelpHint";
import { humanizeStrategyName } from "@/lib/enginePresentation";
import { readSiteFinanceSnapshot } from "@/lib/server/data";

function pct(value: unknown, digits = 1) {
  const n = Number(value);
  return Number.isFinite(n) ? `${(n * 100).toFixed(digits)}%` : "n/d";
}

export default async function FinancasLandingPage() {
  const snapshot = (await readSiteFinanceSnapshot()) as Record<string, unknown>;
  const proof = ((snapshot.proof as Record<string, unknown> | undefined) || {}) as Record<string, unknown>;
  const researchBest = ((proof.group_suite_best as Record<string, unknown> | undefined) || {}) as Record<string, unknown>;
  const universe800 = ((snapshot.universe_expansion as Record<string, unknown> | undefined)?.target_800_cov075 || {}) as Record<string, unknown>;
  const layered = ((snapshot.layered_engine as Record<string, unknown> | undefined) || {}) as Record<string, unknown>;
  const attack = ((layered.best_meta_candidate as Record<string, unknown> | undefined) || {}) as Record<string, unknown>;

  return (
    <main className="space-y-10 py-8 md:py-10">
      <section className="grid gap-6 lg:grid-cols-[0.95fr_1.05fr]">
        <div className="rounded-[32px] border border-zinc-800 bg-zinc-950/60 p-8 md:p-10">
          <div className="text-xs uppercase tracking-[0.24em] text-cyan-300/80">Finanças</div>
          <h1 className="mt-4 text-4xl font-semibold tracking-tight text-zinc-100 md:text-6xl">
            Diagnóstico estrutural diário para regime, risco e alocação
          </h1>
          <p className="mt-5 max-w-2xl text-base leading-8 text-zinc-300">
            O Eigen Engine observa o mercado como um sistema coletivo. Em vez de prometer a próxima alta, ele mede
            quando a estrutura fica mais frágil, mais concentrada ou mais saudável, e transforma isso em orçamento de
            risco, modos do motor e leitura auditável.
          </p>
          <div className="mt-6 grid gap-3 md:grid-cols-2">
            <div className="rounded-2xl border border-zinc-800 bg-black/20 p-4">
              <div className="flex items-center gap-2 text-[11px] uppercase tracking-[0.16em] text-zinc-500">
                <span>Prova longa</span>
                <HelpHint text="A vitrine usa uma janela longa para não vender sorte curta como se fosse método." />
              </div>
              <div className="mt-2 text-lg font-semibold text-zinc-100">
                {String((proof.history_window as Record<string, unknown> | undefined)?.start || "2016-02-18")} →{" "}
                {String((proof.history_window as Record<string, unknown> | undefined)?.end || snapshot.as_of_date || "n/d")}
              </div>
            </div>
            <div className="rounded-2xl border border-zinc-800 bg-black/20 p-4">
              <div className="flex items-center gap-2 text-[11px] uppercase tracking-[0.16em] text-zinc-500">
                <span>Pesquisa líquida</span>
                <HelpHint text="Líquido já desconta o modelo conservador de custos e impostos da pesquisa publicada." />
              </div>
              <div className="mt-2 text-lg font-semibold text-zinc-100">
                {pct(researchBest.net_blended_ann_return || researchBest.net_ann_return)}
              </div>
              <div className="mt-1 text-xs text-zinc-500">
                bruto {pct(researchBest.gross_ann_return)} · {humanizeStrategyName(researchBest.candidate_id)}
              </div>
            </div>
          </div>

          <div className="mt-7 flex flex-wrap gap-3">
            <Link href="/app/financas" className="rounded-xl bg-cyan-400 px-5 py-3 text-sm font-medium text-zinc-950 hover:bg-cyan-300">
              Abrir finanças no app
            </Link>
            <Link href="/evidencias" className="rounded-xl border border-zinc-700 px-5 py-3 text-sm text-zinc-200 hover:border-zinc-500">
              Ver evidências publicadas
            </Link>
            <Link href="/methods" className="rounded-xl border border-zinc-700 px-5 py-3 text-sm text-zinc-200 hover:border-zinc-500">
              Entender o Eigen Engine
            </Link>
          </div>
        </div>

        <EngineStoryDeck />
      </section>

      <section className="grid gap-4 lg:grid-cols-3">
        <article className="rounded-3xl border border-zinc-800 bg-zinc-950/60 p-6">
          <div className="text-xs uppercase tracking-[0.2em] text-zinc-500">Modo principal</div>
          <div className="mt-3 text-2xl font-semibold text-zinc-100">{humanizeStrategyName(attack.candidate_id)}</div>
          <div className="mt-4 space-y-2 text-sm text-zinc-300">
            <p>Retorno anual: {pct(attack.net_ann_return)}</p>
            <p>Sharpe: {Number(attack.net_sharpe || 0).toFixed(2)}</p>
            <p>Drawdown: {pct(attack.net_max_drawdown)}</p>
          </div>
        </article>

        <article className="rounded-3xl border border-zinc-800 bg-zinc-950/60 p-6">
          <div className="text-xs uppercase tracking-[0.2em] text-zinc-500">Universo ampliado</div>
          <div className="mt-3 text-2xl font-semibold text-zinc-100">
            {String(universe800.assets_ok || "n/d")} ativos
          </div>
          <div className="mt-4 space-y-2 text-sm text-zinc-300">
            <p>{String(universe800.sector_count || "n/d")} setores cobertos</p>
            <p>Maior setor: {pct(universe800.largest_sector_share)}</p>
            <p>Painel histórico: {Number(universe800.panel_rows || 0).toLocaleString("pt-BR")} linhas</p>
          </div>
        </article>

        <article className="rounded-3xl border border-zinc-800 bg-zinc-950/60 p-6">
          <div className="text-xs uppercase tracking-[0.2em] text-zinc-500">O que está à venda aqui</div>
          <div className="mt-3 text-2xl font-semibold text-zinc-100">Contexto melhor, risco mais controlado</div>
          <p className="mt-4 text-sm leading-7 text-zinc-300">
            O valor do produto não está em “adivinhar o próximo candle”. Está em ler melhor o terreno, dosar exposição
            e mostrar quando o contexto piora antes de o investidor agir no impulso.
          </p>
        </article>
      </section>

      <section className="rounded-3xl border border-zinc-800 bg-zinc-950/60 p-6 md:p-8">
        <div className="text-xs uppercase tracking-[0.24em] text-zinc-500">Rigor matemático aplicado</div>
        <div className="mt-5 grid gap-5 lg:grid-cols-2">
          <article className="rounded-2xl border border-zinc-800 bg-black/20 p-5">
            <h2 className="text-xl font-semibold text-zinc-100">Como aplicamos em finanças</h2>
            <div className="mt-4 space-y-3 text-sm leading-7 text-zinc-300">
              <p>1. Transformamos preços em retornos comparáveis e verificamos a cobertura temporal.</p>
              <p>2. Construímos a matriz de correlação e limpamos o que parece ruído estatístico.</p>
              <p>3. O espectro mostra quando poucos fatores passam a dominar tudo e o mercado perde diversidade.</p>
              <p>4. Essa leitura estrutural vira regime, faixa de exposição e priorização de sleeves.</p>
            </div>
          </article>
          <article className="rounded-2xl border border-zinc-800 bg-black/20 p-5">
            <h2 className="text-xl font-semibold text-zinc-100">Como aplicamos em cripto</h2>
            <div className="mt-4 space-y-3 text-sm leading-7 text-zinc-300">
              <p>1. O motor reaproveita o mesmo núcleo estrutural, mas com sleeves mais agressivos e benchmark em BTC.</p>
              <p>2. O ranking mede tração e breadth real, em vez de confiar só em um nome explodindo isolado.</p>
              <p>3. O meta-switch escolhe entre cripto, ações e caixa conforme o contexto e a integridade do regime.</p>
              <p>4. O resultado final é um ataque mais disciplinado, não uma aposta cega em volatilidade.</p>
            </div>
          </article>
        </div>
      </section>
    </main>
  );
}
