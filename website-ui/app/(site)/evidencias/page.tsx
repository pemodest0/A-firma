import Link from "next/link";

export default function EvidenciasLandingPage() {
  return (
    <main className="mx-auto max-w-6xl px-6 py-12">
      <section className="rounded-2xl border border-zinc-800 bg-zinc-900/45 p-8 md:p-10">
        <div className="text-xs uppercase tracking-[0.22em] text-cyan-300">Evidências</div>
        <h1 className="mt-3 text-3xl md:text-5xl font-semibold tracking-tight text-zinc-100">
          Contexto histórico e simulações de uso do Eigen Engine
        </h1>
        <p className="mt-4 max-w-4xl text-zinc-300">
          Casos reais de regime em finanças, energia e agro, com leitura de como o motor teria sinalizado o contexto
          operacional em cada janela.
        </p>
        <div className="mt-7 flex flex-wrap gap-3">
          <Link href="/app/evidencias" className="rounded-lg bg-cyan-500 px-5 py-2.5 text-sm font-medium text-zinc-950 hover:bg-cyan-400 transition">
            Abrir evidências no app
          </Link>
          <Link href="/app/copiloto" className="rounded-lg border border-zinc-600 px-5 py-2.5 text-sm font-medium text-zinc-200 hover:border-zinc-400 transition">
            Conversar com o copiloto
          </Link>
        </div>
      </section>
    </main>
  );
}
