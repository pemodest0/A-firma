import Link from "next/link";
import HeroStructureCard from "@/components/visuals/HeroStructureCard";

export default function HeroSection() {
  return (
    <section className="relative overflow-hidden rounded-[30px] border border-cyan-300/20 bg-[#050C1D]/72 p-8 md:p-10 lg:p-12 min-h-[72vh] ax-glow py-10 md:py-12 lg:py-14 xl:py-16">
      <div aria-hidden className="ax-hero-wave absolute inset-0" />
      <div aria-hidden className="ax-hero-stars absolute inset-0" />
      <div className="relative z-10 grid grid-cols-1 xl:grid-cols-[1.08fr_0.92fr] items-center gap-8 lg:gap-10">
        <div className="max-w-3xl">
          <div className="text-xs uppercase tracking-[0.3em] text-cyan-100/70">Assyntrax</div>
          <h1 className="mt-4 text-4xl md:text-5xl lg:text-6xl font-semibold tracking-tight text-zinc-100">
            Diagnóstico Estrutural para Sistemas Complexos
          </h1>
          <p className="mt-5 text-zinc-200/90 text-base md:text-lg max-w-2xl">
            Plataforma de diagnóstico estrutural para sistemas complexos de alta dimensionalidade.
          </p>
          <p className="mt-3 text-zinc-200/85 text-base md:text-lg max-w-2xl">
            Identifique mudanças estruturais antes que elas se tornem instabilidade operacional.
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            <Link
              className="rounded-xl bg-[#2D7DFF] text-white px-5 py-3 font-medium hover:bg-[#3A89FF] transition"
              href="/contact"
            >
              Solicitar demonstração
            </Link>
            <Link
              className="rounded-xl border border-zinc-300/35 bg-white/5 px-5 py-3 text-zinc-100 hover:border-zinc-100/65 transition"
              href="/app/dashboard"
            >
              Abrir app
            </Link>
          </div>
          <div className="mt-8 text-sm text-zinc-300/90 tracking-wide">
            Regimes • Acoplamento • Ranking de impacto • Governança de publicação
          </div>
        </div>
        <HeroStructureCard />
      </div>
    </section>
  );
}
