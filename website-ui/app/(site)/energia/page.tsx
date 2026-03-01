import Image from "next/image";
import Link from "next/link";

export default function EnergiaLandingPage() {
  return (
    <main className="mx-auto max-w-6xl px-6 py-12">
      <section className="grid gap-6 lg:grid-cols-[1.05fr_0.95fr] rounded-2xl border border-zinc-800 bg-zinc-900/45 p-8 md:p-10">
        <div>
          <div className="text-xs uppercase tracking-[0.22em] text-cyan-300">Setor Energia Brasil</div>
          <h1 className="mt-3 text-3xl md:text-5xl font-semibold tracking-tight text-zinc-100">
            Sinal estrutural para transições operacionais de energia
          </h1>
          <p className="mt-4 text-zinc-300">
            O Eigen Engine avalia acoplamento sistêmico, mudança de estrutura e antecipação de eventos em base diária,
            com validação temporal em blocos.
          </p>
          <div className="mt-7 flex flex-wrap gap-3">
            <Link href="/app/energia" className="rounded-lg bg-cyan-500 px-5 py-2.5 text-sm font-medium text-zinc-950 hover:bg-cyan-400 transition">
              Abrir energia no app
            </Link>
            <Link href="/methods" className="rounded-lg border border-zinc-600 px-5 py-2.5 text-sm font-medium text-zinc-200 hover:border-zinc-400 transition">
              Métodos por setor
            </Link>
          </div>
        </div>
        <div className="overflow-hidden rounded-xl border border-zinc-700/60 bg-zinc-950/60">
          <Image
            src="/visuals/hero-flow.svg"
            alt="Fluxo estrutural para monitoramento setorial de energia"
            width={900}
            height={560}
            className="h-full w-full object-cover"
          />
        </div>
      </section>
    </main>
  );
}
