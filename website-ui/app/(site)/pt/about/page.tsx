import type { Metadata } from "next";
import MethodLattice from "@/components/visuals/MethodLattice";
import { buildPageMetadata } from "@/lib/site/metadata";

export const metadata: Metadata = buildPageMetadata({
  title: "Sobre o Eigen Engine",
  description: "Versão espelhada em /pt. Conteúdo principal publicado em /about.",
  path: "/pt/about",
  locale: "pt-BR",
  noIndex: true,
  canonicalPath: "/about",
});

export default function AboutPage() {
  return (
    <div className="space-y-12">
      <section className="relative isolate grid grid-cols-1 lg:grid-cols-[1.1fr_0.9fr] gap-10 items-center">
        <div className="space-y-4">
          <div className="text-xs uppercase tracking-[0.3em] text-zinc-400">Sobre</div>
          <h1 className="text-4xl md:text-5xl font-semibold tracking-tight">
            Por que o Eigen Engine existe
          </h1>
          <p className="text-zinc-300 max-w-3xl text-lg">
            Mercados mudam de regime. Modelos lineares continuam extrapolando como se nada tivesse
            mudado. O Eigen Engine existe para diagnosticar estrutura, risco e confiabilidade com trilha auditavel.
          </p>
        </div>
        <div className="overflow-hidden rounded-[28px]">
          <MethodLattice />
        </div>
      </section>

      <section className="relative isolate rounded-[32px] border border-zinc-800 bg-zinc-950/60 p-10">
        <div className="text-sm uppercase tracking-[0.3em] text-zinc-400">Autor e criador</div>
        <div className="mt-4 grid grid-cols-1 md:grid-cols-[1fr_auto] gap-4 items-center">
          <div>
            <div className="text-2xl font-semibold tracking-tight">Pedro Henrique Modesto</div>
            <p className="mt-2 text-zinc-300">
              Pesquisa aplicada em dinâmica de regimes, risco estrutural e arquitetura de diagnóstico para decisão.
            </p>
          </div>
          <div className="flex flex-wrap gap-3 text-sm">
            <a className="rounded-lg border border-zinc-700 px-3 py-2 text-zinc-300 hover:text-white hover:border-zinc-500" href="https://github.com/pemodest0/Assyntrax">GitHub</a>
            <a className="rounded-lg border border-zinc-700 px-3 py-2 text-zinc-300 hover:text-white hover:border-zinc-500" href="https://www.linkedin.com/in/pedro-henrique-gesualdo-modesto-39a135272/">LinkedIn</a>
            <a className="rounded-lg border border-zinc-700 px-3 py-2 text-zinc-300 hover:text-white hover:border-zinc-500" href="https://pemodest0.github.io/">Portfólio</a>
          </div>
        </div>
      </section>

      <section className="relative isolate grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="rounded-[28px] border border-zinc-800 bg-zinc-950/60 p-8">
          <div className="text-sm uppercase tracking-[0.3em] text-zinc-400">História</div>
          <p className="mt-3 text-zinc-300">
            O projeto começou com uma pergunta simples: por que modelos de previsão falham exatamente
            quando o risco mais importa? A resposta levou ao foco em estrutura dinâmica e controle de decisão.
          </p>
          <p className="mt-3 text-zinc-300">
            Hoje, o núcleo combina análise espectral de correlações, validação causal walk-forward,
            winsorização de outliers e gate de publicação com trilha auditável por execução.
          </p>
        </div>
        <div className="rounded-[28px] border border-zinc-800 bg-zinc-950/60 p-8">
          <div className="text-sm uppercase tracking-[0.3em] text-zinc-400">Direção</div>
          <ul className="mt-3 space-y-3 text-zinc-300">
            <li>- Consolidar cobertura por domínio com dados rastreáveis e contrato único de saída.</li>
            <li>- Fortalecer validação diária com detecção de drift e bloqueio automático de sinais fracos.</li>
            <li>- Entregar API e dashboard com utilidade operacional clara para cada setor.</li>
          </ul>
        </div>
      </section>
    </div>
  );
}
