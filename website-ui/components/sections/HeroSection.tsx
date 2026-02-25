import Link from "next/link";
import SectorRegimeShowcase from "@/components/visuals/SectorRegimeShowcase";

export default function HeroSection() {
  return (
    <section className="grid grid-cols-1 lg:grid-cols-[1.08fr_0.92fr] gap-8 lg:gap-10 items-center min-h-[70vh] lg:min-h-[74vh] py-10 md:py-12 lg:py-14 xl:py-16">
      <div className="rounded-[28px] border border-zinc-800 bg-zinc-950/70 p-8 lg:p-10 backdrop-blur ax-glow">
        <div className="text-xs uppercase tracking-[0.3em] text-zinc-400">Diagnóstico estrutural</div>
        <h1 className="mt-5 text-4xl md:text-5xl font-semibold tracking-tight">
          Diagnóstico estrutural baseado em matemática e física para mercados complexos
        </h1>
        <p className="mt-4 text-zinc-300 max-w-2xl text-base lg:text-lg">
          Identifique mudanças de regime antes que o mercado reaja. Classificação causal, sem uso de dados futuros,
          para tomar decisões com clareza e controle.
        </p>
        <p className="mt-4 text-zinc-400 max-w-2xl text-sm lg:text-base">
          Somos fisicos e engenheiros apaixonados pela estrutura dos mercados. Aplicamos teoria de sistemas dinamicos,
          analise espectral e estatistica robusta para identificar mudancas estruturais no mercado brasileiro. O
          Eigen Engine e causal: em cada momento t usa apenas informacoes observadas ate t-1, evitando vies de look-ahead.
        </p>
        <ul className="mt-5 space-y-2 text-sm text-zinc-300">
          <li>Sem look-ahead: cálculo causal em modo walk-forward.</li>
          <li>Saída com regime, risco estrutural e confiança do sinal.</li>
          <li>Mesma leitura no painel, API e artefatos de auditoria.</li>
        </ul>
        <div className="mt-6 flex flex-wrap gap-3">
          <Link
            className="rounded-xl bg-zinc-100 text-black px-5 py-3 font-medium hover:bg-white transition"
            href="/contact"
          >
            Solicitar demonstração
          </Link>
          <Link
            className="rounded-xl border border-zinc-800 px-5 py-3 text-zinc-200 hover:border-zinc-600 transition"
            href="/app/dashboard"
          >
            Abrir painel
          </Link>
        </div>
      </div>
      <div className="animate-float-slow">
        <SectorRegimeShowcase />
      </div>
    </section>
  );
}
