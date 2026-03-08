import Link from "next/link";
import HelpHint from "@/components/ui/HelpHint";
import {
  humanizeEngineState,
  humanizeStatusWord,
  humanizeStrategyName,
} from "@/lib/enginePresentation";
import { readSiteFinanceSnapshot } from "@/lib/server/data";

function pct(value: unknown, digits = 1) {
  const n = Number(value);
  return Number.isFinite(n) ? `${(n * 100).toFixed(digits)}%` : "n/d";
}

const checklist = [
  "Leitura diária de risco e exposição",
  "Finanças e cripto na mesma trilha",
  "Copiloto com artefatos reais",
  "Shadow e comparação contra benchmark",
  "Checklist operacional com gate",
  "Trilha auditável para pesquisa e execução",
];

export default async function ProductPage() {
  let snapshot: Record<string, unknown> = {};
  try {
    snapshot = (await readSiteFinanceSnapshot()) as Record<string, unknown>;
  } catch {
    snapshot = {};
  }
  const finance = ((snapshot.finance as Record<string, unknown> | undefined) || {}) as Record<string, unknown>;
  const profit = ((snapshot.profit_research as Record<string, unknown> | undefined) || {}) as Record<string, unknown>;
  const shadow = ((snapshot.shadow as Record<string, unknown> | undefined) || {}) as Record<string, unknown>;
  const topCandidate = ((profit.top_candidate as Record<string, unknown> | undefined) || {}) as Record<string, unknown>;
  const shadowLatest = ((shadow.latest as Record<string, unknown> | undefined) || {}) as Record<string, unknown>;
  const shadowReplay = ((shadow.historical_proxy_replay as Record<string, unknown> | undefined) || {}) as Record<string, unknown>;

  const states = [
    {
      title: "Leitura operacional",
      badge: humanizeStatusWord(finance.overall_readiness || "missing"),
      detail: `${humanizeEngineState(finance.operational_state || "monitoramento_normal")} · risco ${String(finance.risk_level_next_month || "n/d")}`,
    },
    {
      title: "Pesquisa de alpha",
      badge: humanizeStatusWord(topCandidate.status || "missing"),
      detail: `${humanizeStrategyName(topCandidate.candidate_id || "n/d")} · líquido ${pct(topCandidate.net_ann_return)}`,
    },
    {
      title: "Paper trading",
      badge: shadow.run_id ? "publicado" : "não publicado",
      detail: `${String(shadowLatest.regime || "n/d")} · replay ${pct(shadowReplay.ann_return)}`,
    },
  ];

  return (
    <main className="space-y-10 py-8 md:py-10">
      <section className="grid gap-6 lg:grid-cols-[0.95fr_1.05fr]">
        <div className="rounded-[32px] border border-zinc-800 bg-zinc-950/60 p-8 md:p-10">
          <div className="text-xs uppercase tracking-[0.24em] text-emerald-300/80">Produto</div>
          <h1 className="mt-4 text-4xl font-semibold tracking-tight text-zinc-100 md:text-5xl">
            Plataforma pessoal para usar o motor com clareza, não no susto
          </h1>
          <p className="mt-5 max-w-2xl text-base leading-8 text-zinc-300">
            O produto junta diagnóstico estrutural, orçamento de risco, sleeves, pesquisa e shadow numa interface que
            mostra o contexto de forma humana. A promessa aqui não é milagre. É disciplina, trilha auditável e menos
            chance de operar na base do impulso.
          </p>
          <div className="mt-6 flex flex-wrap gap-3">
            <Link href="/app/dashboard" className="rounded-xl bg-emerald-400 px-5 py-3 text-sm font-medium text-zinc-950 hover:bg-emerald-300">
              Abrir plataforma
            </Link>
            <Link href="/app/copiloto" className="rounded-xl border border-zinc-700 px-5 py-3 text-sm text-zinc-200 hover:border-zinc-500">
              Conversar com o copiloto
            </Link>
          </div>
        </div>

        <div className="rounded-[32px] border border-zinc-800 bg-zinc-950/60 p-6">
          <div className="flex items-center gap-2 text-xs uppercase tracking-[0.24em] text-zinc-500">
            <span>Status real dos artefatos</span>
            <HelpHint text="Lido do snapshot canônico do site, sem inventar disponibilidade nem esconder artefato ruim." />
          </div>
          <div className="mt-5 grid gap-3">
            {states.map((state) => (
              <article key={state.title} className="rounded-2xl border border-zinc-800 bg-black/20 p-4">
                <div className="flex items-center justify-between gap-3">
                  <div className="text-sm font-semibold text-zinc-100">{state.title}</div>
                  <div className="rounded-full border border-zinc-700 px-2.5 py-1 text-[11px] text-zinc-200">{state.badge}</div>
                </div>
                <div className="mt-3 text-sm leading-6 text-zinc-300">{state.detail}</div>
              </article>
            ))}
          </div>
        </div>
      </section>

      <section className="grid gap-4 lg:grid-cols-[1.05fr_0.95fr]">
        <div className="rounded-3xl border border-zinc-800 bg-zinc-950/60 p-6">
          <div className="text-xs uppercase tracking-[0.24em] text-zinc-500">Checklist elegante</div>
          <div className="mt-5 grid gap-3 md:grid-cols-2">
            {checklist.map((item) => (
              <div key={item} className="rounded-2xl border border-emerald-900/40 bg-emerald-950/10 px-4 py-3 text-sm text-zinc-200">
                <span className="mr-2 text-emerald-300">✓</span>
                {item}
              </div>
            ))}
          </div>
        </div>

        <div className="rounded-3xl border border-zinc-800 bg-zinc-950/60 p-6">
          <div className="text-xs uppercase tracking-[0.24em] text-zinc-500">Como isso funciona na prática</div>
          <div className="mt-5 space-y-3 text-sm leading-7 text-zinc-300">
            <p>
              <span className="text-emerald-300">1.</span> O Eigen Engine lê a estrutura do mercado e decide se o
              contexto está limpo, intermediário ou hostil.
            </p>
            <p>
              <span className="text-emerald-300">2.</span> O produto traduz isso em faixa de exposição, modos do motor
              e contexto por ativo.
            </p>
            <p>
              <span className="text-emerald-300">3.</span> O shadow compara o que parece bonito com o que sobrevive ao
              tempo e ao benchmark.
            </p>
            <p>
              <span className="text-emerald-300">4.</span> O copiloto ajuda a ler o app em português claro, sem te
              jogar um monte de sigla sem contexto.
            </p>
          </div>
        </div>
      </section>

      <section className="rounded-3xl border border-zinc-800 bg-zinc-950/60 p-6 md:p-8">
        <div className="text-xs uppercase tracking-[0.24em] text-zinc-500">Rigor matemático aplicado</div>
        <div className="mt-5 grid gap-5 lg:grid-cols-2">
          <article className="rounded-2xl border border-zinc-800 bg-black/20 p-5">
            <h2 className="text-xl font-semibold text-zinc-100">Em finanças</h2>
            <div className="mt-4 space-y-3 text-sm leading-7 text-zinc-300">
              <p>O produto usa a camada estrutural para medir correlação, concentração de risco e mudança de regime.</p>
              <p>Isso serve para ajustar exposição e entender quando o mercado fica mais perigoso ou mais respirável.</p>
              <p>O valor comercial aqui é menos ruído operacional e mais contexto antes da decisão.</p>
            </div>
          </article>
          <article className="rounded-2xl border border-zinc-800 bg-black/20 p-5">
            <h2 className="text-xl font-semibold text-zinc-100">Em cripto</h2>
            <div className="mt-4 space-y-3 text-sm leading-7 text-zinc-300">
              <p>O mesmo motor organiza sleeves agressivos, ranking, meta-switch e proteção de drawdown.</p>
              <p>Cripto entra como aceleração tática, não como licença para abandonar a leitura de risco.</p>
              <p>O produto mostra quando faz sentido atacar e quando a resposta adulta é reduzir o tamanho.</p>
            </div>
          </article>
        </div>
      </section>
    </main>
  );
}
