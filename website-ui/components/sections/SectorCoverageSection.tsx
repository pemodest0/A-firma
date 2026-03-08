import Image from "next/image";
import Link from "next/link";

const cards = [
  {
    id: "financas",
    title: "Finanças",
    description:
      "Leitura diária de regime, risco estrutural, ranking por ativo e painel operacional com foco em alocação.",
    image: "/assets/prints/dashboard-main.svg",
    href: "/financas",
    cta: "Abrir finanças",
  },
  {
    id: "cripto",
    title: "Cripto",
    description:
      "Sleeve cripto líquido sob o mesmo controle estrutural, com foco em majors, meta-switch e disciplina de risco.",
    image: "/visuals/hero-embedding.svg",
    href: "/cripto",
    cta: "Abrir cripto",
  },
  {
    id: "copiloto",
    title: "Copiloto e Shadow",
    description:
      "Pesquisa de alpha, paper trading e trilha de evidências para testar hipóteses antes de arriscar capital.",
    image: "/assets/prints/walkforward-metrics.svg",
    href: "/app/copiloto",
    cta: "Abrir copiloto",
  },
];

export default function SectorCoverageSection() {
  return (
    <section className="rounded-[26px] border border-zinc-800/80 bg-zinc-950/55 p-8 md:p-9">
      <div className="text-xs uppercase tracking-[0.25em] text-zinc-400">Módulos ativos</div>
      <h2 className="mt-3 text-3xl md:text-4xl font-semibold tracking-tight text-zinc-100">
        O que já está ativo na plataforma
      </h2>
      <div className="mt-6 grid gap-4 md:grid-cols-3">
        {cards.map((card) => (
          <article key={card.id} className="rounded-2xl border border-zinc-800 bg-zinc-900/45 overflow-hidden">
            <div className="border-b border-zinc-800 bg-zinc-950/70">
              <Image
                src={card.image}
                alt={`Visual do setor ${card.title}`}
                width={560}
                height={320}
                className="h-36 w-full object-cover"
              />
            </div>
            <div className="p-4">
              <h3 className="text-lg font-semibold text-zinc-100">{card.title}</h3>
              <p className="mt-2 text-sm text-zinc-300">{card.description}</p>
              <Link
                href={card.href}
                className="mt-4 inline-flex rounded-lg border border-cyan-300/35 bg-white/5 px-3 py-1.5 text-sm text-zinc-100 hover:border-cyan-200"
              >
                {card.cta}
              </Link>
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}
