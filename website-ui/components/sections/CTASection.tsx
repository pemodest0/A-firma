import Link from "next/link";

export default function CTASection() {
  return (
    <section className="rounded-[28px] border border-zinc-800 bg-zinc-950/70 p-8 lg:p-10 flex flex-col md:flex-row items-center justify-between gap-5 ax-glow py-10 md:py-12 lg:py-14 xl:py-16">
      <div>
        <div className="text-xs uppercase tracking-[0.3em] text-zinc-400">Proximo passo</div>
        <h3 className="mt-3 text-3xl md:text-4xl font-semibold tracking-tight">
          Solicite uma demonstracao guiada
        </h3>
        <p className="mt-2 text-zinc-300 max-w-2xl text-base lg:text-lg">
          Veja o diagnostico estrutural no seu fluxo de decisao, com governanca e rastreabilidade.
        </p>
      </div>
      <div className="flex gap-3">
        <Link
          className="rounded-xl bg-zinc-100 text-black px-6 py-3 font-medium hover:bg-white transition"
          href="/contact"
        >
          Solicitar demonstracao
        </Link>
        <Link
          className="rounded-xl border border-zinc-800 px-6 py-3 text-zinc-200 hover:border-zinc-600 transition"
          href="/app/dashboard"
        >
          Abrir painel
        </Link>
      </div>
    </section>
  );
}
