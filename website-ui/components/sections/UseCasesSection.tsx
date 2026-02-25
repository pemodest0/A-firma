"use client";

import { KeyboardEvent, useRef, useState } from "react";
import { FinanceVisual } from "@/components/visuals/UseCasePanels";

function GovernanceVisual() {
  return (
    <div className="relative rounded-3xl border border-zinc-800 bg-black/70 p-6 overflow-hidden">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_top,_rgba(16,185,129,0.2),_transparent_60%)]" />
      <div className="relative z-10 space-y-4">
        <div className="text-xs uppercase tracking-[0.2em] text-zinc-400">Governanca - Trilha de decisao</div>
        <div className="rounded-xl border border-zinc-800 bg-black/35 p-4 space-y-2 text-sm">
          <div className="flex items-center justify-between text-zinc-300">
            <span>Run</span>
            <span className="text-zinc-100">20260210_contractfix</span>
          </div>
          <div className="flex items-center justify-between text-zinc-300">
            <span>Status</span>
            <span className="text-amber-300">watch</span>
          </div>
          <div className="flex items-center justify-between text-zinc-300">
            <span>Gate</span>
            <span className="text-zinc-100">publish green</span>
          </div>
          <div className="rounded-lg border border-zinc-800 bg-zinc-950/60 p-2 text-xs text-zinc-400">
            Registro de decisao, justificativa de exposicao e rastreabilidade para comites e auditoria.
          </div>
        </div>
      </div>
    </div>
  );
}

const tabs = [
  {
    id: "financas",
    label: "Financas",
    title: "Financas",
    description:
      "Utilize o regime como um barometro estrutural. Em mercado estavel, mantenha posicionamentos. Em transicao, reduza exposicao e monitore setores de risco. Em estresse, corte posicoes e proteja capital.",
    note: "Leitura estrutural nao e recomendacao de compra ou venda. A decisao final e do usuario.",
    Visual: FinanceVisual,
  },
  {
    id: "governanca",
    label: "Governanca",
    title: "Governanca",
    description:
      "Mantenha auditoria continua. Registre decisoes, justifique mudancas de exposicao e demonstre transparencia para investidores e reguladores. O Eigen Engine preserva historico e criterios objetivos por regime.",
    note: "Governanca forte reduz risco reputacional e melhora consistencia de comites de investimento.",
    Visual: GovernanceVisual,
  },
];

export default function UseCasesSection() {
  const [activeIndex, setActiveIndex] = useState(0);
  const tabRefs = useRef<Array<HTMLButtonElement | null>>([]);
  const current = tabs[activeIndex] || tabs[0];
  const Visual = current.Visual;

  function handleTabKeyDown(event: KeyboardEvent<HTMLButtonElement>, currentIndex: number) {
    const { key } = event;
    if (!["ArrowRight", "ArrowLeft", "Home", "End"].includes(key)) return;
    event.preventDefault();
    let nextIndex = currentIndex;
    if (key === "ArrowRight") nextIndex = (currentIndex + 1) % tabs.length;
    if (key === "ArrowLeft") nextIndex = (currentIndex - 1 + tabs.length) % tabs.length;
    if (key === "Home") nextIndex = 0;
    if (key === "End") nextIndex = tabs.length - 1;
    setActiveIndex(nextIndex);
    tabRefs.current[nextIndex]?.focus();
  }

  return (
    <section className="space-y-6 py-10 md:py-12 lg:py-14 xl:py-16">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="text-xs uppercase tracking-[0.3em] text-zinc-400">Casos de uso</div>
          <h2 className="mt-2 text-3xl md:text-4xl font-semibold tracking-tight">{current.title}</h2>
          <p className="mt-2 text-zinc-300 max-w-3xl text-base lg:text-lg">{current.description}</p>
          <p className="mt-2 text-xs text-zinc-500">{current.note}</p>
        </div>
        <div role="tablist" aria-label="Casos de uso" className="flex gap-2">
          {tabs.map((tab, index) => (
            <button
              key={tab.id}
              id={`use-case-tab-${tab.id}`}
              role="tab"
              aria-selected={index === activeIndex}
              aria-controls={`use-case-panel-${tab.id}`}
              tabIndex={index === activeIndex ? 0 : -1}
              ref={(el) => {
                tabRefs.current[index] = el;
              }}
              className={`rounded-full border px-4 py-2 text-sm transition ${
                index === activeIndex
                  ? "border-cyan-400/70 bg-cyan-400/10 text-cyan-200"
                  : "border-zinc-800 text-zinc-400 hover:text-white"
              }`}
              onClick={() => setActiveIndex(index)}
              onKeyDown={(event) => handleTabKeyDown(event, index)}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>
      <div
        key={current.id}
        id={`use-case-panel-${current.id}`}
        role="tabpanel"
        aria-labelledby={`use-case-tab-${current.id}`}
        className="outline-none"
      >
        <div className="grid grid-cols-1 lg:grid-cols-[1.05fr_0.95fr] gap-6">
          <Visual />
          <div className="rounded-3xl border border-zinc-800 bg-zinc-950/60 p-5 space-y-3">
            <div className="text-xs uppercase tracking-[0.2em] text-zinc-400">Uso pratico</div>
            <ul className="space-y-2 text-sm text-zinc-300">
              <li>- Ajuste exposicao com base no regime, nao em previsao de preco.</li>
              <li>- Use o gate para documentar quando agir e quando esperar.</li>
              <li>- Mantenha registro historico para auditoria e revisao de estrategia.</li>
            </ul>
          </div>
        </div>
      </div>
    </section>
  );
}
