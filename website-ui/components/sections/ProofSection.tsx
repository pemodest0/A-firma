"use client";

import { motion } from "framer-motion";
import ProofMatrix from "@/components/visuals/ProofMatrix";

const limits = [
  {
    title: "Sem look-ahead",
    text: "Os regimes sao calculados de forma causal: em cada instante t usamos apenas informacoes observadas ate t-1.",
  },
  {
    title: "Historico e indicativo",
    text: "Metricas historicas (F1, MCC e correlatas) servem para avaliar algoritmo e robustez, nao para prometer retorno.",
  },
  {
    title: "Nao e recomendacao",
    text: "Nao dizemos compre ou venda. O objetivo e reduzir erro operacional e melhorar governanca de risco.",
  },
];

export default function ProofSection() {
  return (
    <section className="space-y-6 py-10 md:py-12 lg:py-14 xl:py-16">
      <motion.div
        className="space-y-2"
        initial={{ opacity: 0, y: 8 }}
        whileInView={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <div className="text-xs uppercase tracking-[0.3em] text-zinc-400">Confiabilidade e limites</div>
        <h2 className="text-3xl md:text-4xl font-semibold tracking-tight">Transparencia metodologica para uso institucional</h2>
      </motion.div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {limits.map((item) => (
          <article key={item.title} className="rounded-2xl border border-zinc-800 bg-zinc-950/60 p-5">
            <h3 className="text-lg font-semibold text-zinc-100">{item.title}</h3>
            <p className="mt-2 text-sm text-zinc-300">{item.text}</p>
          </article>
        ))}
      </div>

      <ProofMatrix />
      <div className="text-xs text-zinc-500">Uso para diagnostico estrutural de risco, sem promessa de retorno financeiro.</div>
    </section>
  );
}
