import Image from "next/image";
import Link from "next/link";

const cards = [
  {
    id: "financas",
    title: "Finanças",
    description:
      "Motor diário com detecção de regimes, ranking de impacto ativo/setor e validação temporal por blocos.",
    image: "/assets/prints/walkforward-metrics.svg",
    href: "/financas",
  },
  {
    id: "energia",
    title: "Energia Brasil",
    description:
      "Leitura estrutural diária para acoplamento e transição operacional com evidência de pré-sinal em eventos reais.",
    image: "/visuals/hero-flow.svg",
    href: "/energia",
  },
  {
    id: "agro",
    title: "Agro Brasil",
    description:
      "Pipeline mensal com macro, safra e comércio externo, usando o mesmo núcleo matemático do Eigen Engine.",
    image: "/visuals/hero-embedding.svg",
    href: "/agro",
  },
];

export default function SectorCoverageSection() {
  return (
    <section className="rounded-[26px] border border-zinc-800/80 bg-zinc-950/55 p-8 md:p-9">
      <div className="text-xs uppercase tracking-[0.25em] text-zinc-400">Domínios ativos</div>
      <h2 className="mt-3 text-3xl md:text-4xl font-semibold tracking-tight text-zinc-100">
        O que já colocamos em produção por setor
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
                Abrir setor
              </Link>
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}
