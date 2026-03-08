import HelpHint from "@/components/ui/HelpHint";

const sections = [
  {
    title: "Espaço probabilístico e causalidade",
    summary:
      "O motor é formulado como filtragem temporal: em cada instante, usa apenas o que já estava observável até t-1. Isso impede o uso de informação futura no cálculo do regime.",
    formula: "F_t = σ(X_1, ..., X_t) e a decisão_t depende de F_{t-1}.",
    purpose:
      "Na prática, isso significa que o produto não se aproveita do futuro para parecer brilhante no passado.",
  },
  {
    title: "Retornos, normalização e espectro",
    summary:
      "Preços são transformados em retornos para tornar ativos comparáveis. Depois, a leitura espectral mede concentração, breadth e organização do sistema.",
    formula: "r_t = log(P_t / P_{t-1})",
    purpose:
      "O espectro ajuda a separar a parte coletiva do mercado da oscilação que parece padrão só porque o olho humano quer ver desenho em ruído.",
  },
  {
    title: "Teoria de matrizes aleatórias",
    summary:
      "Os limites de Marchenko-Pastur ajudam a distinguir autovalores compatíveis com ruído de autovalores que carregam estrutura de verdade.",
    formula: "λ fora da banda de Marchenko-Pastur → sinal estrutural.",
    purpose:
      "Isso reduz a chance de tratar coincidência estatística como se fosse mudança real de regime.",
  },
  {
    title: "Classificação com histerese",
    summary:
      "A leitura estrutural não troca de estado a cada espirro do mercado. Histerese e persistência mínima existem para reduzir falso alarme.",
    formula: "estado_t = H(estado_bruto_t, persistência, limiares)",
    purpose:
      "O objetivo é preferir uma leitura estável e útil a um motor nervoso que muda de opinião a cada janela.",
  },
  {
    title: "Validação e incerteza",
    summary:
      "O motor é comparado em blocos walk-forward, com bootstrap e subamostragem, para medir se o efeito parece método ou só um intervalo favorável.",
    formula: "score = função(retorno, drawdown, OOS, robustez)",
    purpose:
      "Quando a incerteza sobe, o produto deveria ficar mais humilde. Esse é o próximo passo da camada de confiança.",
  },
  {
    title: "Do modelo à decisão",
    summary:
      "A matemática não termina no gráfico bonito. Ela vira faixa de exposição, seleção de sleeve, proteção de drawdown e contexto para o usuário operar melhor.",
    formula: "regime + sleeves + execução + gate → decisão",
    purpose:
      "O valor do sistema está em transformar teoria auditável em decisão prática e controlada.",
  },
];

export default function TeoriaPage() {
  return (
    <div className="p-5 md:p-6 lg:p-8 space-y-6">
      <section className="rounded-2xl border border-zinc-800 bg-zinc-950/50 p-5">
        <p className="text-xs tracking-[0.14em] uppercase text-zinc-500">Teoria</p>
        <h1 className="mt-2 text-2xl md:text-3xl font-semibold text-zinc-100">Base matemática, linguagem acessível</h1>
        <p className="mt-3 max-w-4xl text-sm leading-7 text-zinc-300">
          Esta página funciona como um artigo resumido: explica o que o Eigen Engine mede, por que isso faz sentido em
          mercados complexos e como a matemática vira uma decisão de risco mais disciplinada.
        </p>
      </section>

      <section className="grid grid-cols-1 gap-4 md:grid-cols-2">
        {sections.map((item) => (
          <article key={item.title} className="rounded-2xl border border-zinc-800 bg-zinc-950/45 p-5 space-y-3">
            <div className="flex items-center gap-2">
              <h2 className="text-lg font-semibold text-zinc-100">{item.title}</h2>
              <HelpHint text={item.purpose} />
            </div>
            <p className="text-sm leading-7 text-zinc-300">{item.summary}</p>
            <pre className="overflow-x-auto rounded-xl border border-zinc-800 bg-black/60 p-3 text-xs text-zinc-200">
              {item.formula}
            </pre>
            <p className="text-xs leading-6 text-zinc-400">{item.purpose}</p>
          </article>
        ))}
      </section>
    </div>
  );
}
