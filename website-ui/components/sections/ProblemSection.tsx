import Link from "next/link";

export default function ProblemSection() {
  return (
    <section className="grid grid-cols-1 lg:grid-cols-[1.02fr_0.98fr] gap-8 lg:gap-10 items-stretch py-10 md:py-12 lg:py-14 xl:py-16">
      <div className="rounded-[26px] border border-zinc-800/80 bg-zinc-950/55 p-8 md:p-9 ax-glow">
        <div className="text-xs uppercase tracking-[0.25em] text-zinc-400">Sobre</div>
        <h2 className="mt-3 text-3xl md:text-4xl font-semibold tracking-tight text-zinc-100">
          Somos físicos e engenheiros focados na análise estrutural de sistemas complexos.
        </h2>
        <p className="mt-4 text-zinc-300 text-base lg:text-lg">
          Aplicamos métodos de análise espectral, estatística robusta e teoria de sistemas dinâmicos para compreender
          como estruturas se organizam, se concentram e mudam ao longo do tempo.
        </p>
        <p className="mt-4 text-zinc-300 text-base lg:text-lg">
          A Assyntrax não prevê preço. Não gera sinais de compra ou venda. Ela mede estrutura.
        </p>
      </div>
      <div className="rounded-[26px] border border-cyan-300/20 bg-zinc-950/50 p-8 md:p-9">
        <div className="text-xs uppercase tracking-[0.25em] text-zinc-400">Posicionamento</div>
        <div className="mt-4 space-y-3 text-zinc-200">
          <p>- Diagnóstico estrutural em vez de narrativa de mercado.</p>
          <p>- Leitura causal em vez de ajuste retrospectivo.</p>
          <p>- Evidência quantitativa em vez de opinião.</p>
        </div>
        <Link className="mt-6 inline-flex rounded-lg border border-cyan-300/35 bg-white/5 px-4 py-2 text-sm text-zinc-100 hover:border-cyan-200" href="/methods">
          Entenda como medimos
        </Link>
      </div>
    </section>
  );
}
