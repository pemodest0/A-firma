"use client";

const blocks = [
  {
    icon: "1",
    title: "Ingestao e Tratamento",
    text: "Coletamos retornos por ativo e aplicamos winsorizacao para remover outliers de forma leve.",
  },
  {
    icon: "2",
    title: "Analise Espectral e Estrutura",
    text: "Utilizamos matematica avancada (EWMA, decomposicao espectral e teoria de matrizes aleatorias) para separar sinal e ruido e medir a dispersao estrutural do mercado.",
  },
  {
    icon: "3",
    title: "Classificacao Causal e Publicacao",
    text: "Classificamos o regime em estavel, transicao, estresse ou dispersao de forma causal, sem olhar o futuro, e so publicamos quando o publish gate esta verde.",
  },
];

export default function HowItWorksSection() {
  return (
    <section className="space-y-6 py-10 md:py-12 lg:py-14 xl:py-16">
      <div className="space-y-2">
        <div className="text-xs uppercase tracking-[0.3em] text-zinc-400">Como funciona</div>
        <h2 className="text-3xl md:text-4xl font-semibold tracking-tight">
          Tres etapas para transformar dados em decisao clara
        </h2>
        <p className="text-zinc-300 max-w-3xl text-base lg:text-lg">
          Fluxo simplificado para operacao: tratamento de dados, leitura estrutural e publicacao com governanca.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {blocks.map((block, index) => (
          <div
            key={block.title}
            className="rounded-2xl border border-zinc-800 bg-zinc-950/60 p-5"
            style={{ animationDelay: `${index * 80}ms` }}
          >
            <div className="inline-flex h-8 w-8 items-center justify-center rounded-full border border-cyan-600/60 text-xs text-cyan-300">
              {block.icon}
            </div>
            <h3 className="mt-3 text-lg font-semibold text-zinc-100">{block.title}</h3>
            <p className="mt-2 text-sm text-zinc-300">{block.text}</p>
          </div>
        ))}
      </div>
    </section>
  );
}
