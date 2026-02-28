import Link from "next/link";

export default function AgroLandingPage() {
  return (
    <main className="mx-auto max-w-6xl px-6 py-12">
      <section className="rounded-2xl border border-zinc-800 bg-zinc-900/40 p-8 md:p-12">
        <div className="text-xs uppercase tracking-[0.22em] text-cyan-300">Vertical Agro Brasil</div>
        <h1 className="mt-3 text-3xl md:text-5xl font-semibold tracking-tight text-zinc-100">
          Diagnóstico estrutural mensal para cadeia agro
        </h1>
        <p className="mt-4 max-w-3xl text-zinc-300">
          O Eigen Engine reaproveita o núcleo estrutural em frequência mensal para monitorar macro, fluxo externo e
          sinais de safra/estoque com governança causal e artefatos auditáveis.
        </p>
        <div className="mt-8 flex flex-wrap gap-3">
          <Link
            href="/app/agro"
            className="rounded-lg bg-cyan-500 px-5 py-2.5 text-sm font-medium text-zinc-950 hover:bg-cyan-400 transition"
          >
            Abrir app Agro
          </Link>
          <Link
            href="/methods"
            className="rounded-lg border border-zinc-600 px-5 py-2.5 text-sm font-medium text-zinc-200 hover:border-zinc-400 transition"
          >
            Ver metodologia
          </Link>
        </div>
      </section>
    </main>
  );
}

