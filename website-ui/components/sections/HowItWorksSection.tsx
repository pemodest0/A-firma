import Link from "next/link";

export default function HowItWorksSection() {
  return (
    <section className="rounded-[26px] border border-zinc-800/80 bg-zinc-950/55 p-8 md:p-9 space-y-5 py-10 md:py-12 lg:py-14 xl:py-16">
      <div className="text-xs uppercase tracking-[0.25em] text-zinc-400">Como funciona</div>
      <h2 className="text-3xl md:text-4xl font-semibold tracking-tight text-zinc-100">
        Quatro camadas, uma decisão operacional.
      </h2>
      <div className="space-y-3 text-zinc-300 text-base md:text-lg max-w-4xl">
        <p>1. Correlação e espectro definem o regime e o orçamento de risco do portfólio.</p>
        <p>2. Ranking e força relativa escolhem os ativos e sleeves que entram no jogo.</p>
        <p>3. Regras de execução controlam caps, turnover, stops e fallback para caixa.</p>
        <p>4. O copiloto resume artefatos, shadow e limites antes de qualquer ação.</p>
      </div>
      <Link className="inline-flex rounded-lg border border-cyan-300/35 bg-white/5 px-4 py-2 text-sm text-zinc-100 hover:border-cyan-200" href="/app/copiloto">
        Abrir copiloto
      </Link>
    </section>
  );
}
