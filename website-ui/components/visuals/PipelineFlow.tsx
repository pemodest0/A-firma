import HelpHint from "@/components/ui/HelpHint";

const steps = [
  {
    step: "01",
    title: "Dados",
    helper: "Cobertura temporal, consistência e janela ativa por ativo.",
    text: "Retornos por ativo com checagem de cobertura, consistência temporal e histórico mínimo.",
    detail: "Sem dado limpo, o motor não entra na camada estrutural.",
  },
  {
    step: "02",
    title: "Winsorização",
    helper: "Corte de extremos em janela de 252 dias entre 0,5% e 99,5%.",
    text: "Tratamento de outliers para reduzir distorção de choques extremos na matriz.",
    detail: "A ideia é cortar distorção, não apagar risco real.",
  },
  {
    step: "03",
    title: "Espectro",
    helper: "Autovalores, autovetores, dimensão efetiva e concentração de risco.",
    text: "Análise espectral da matriz de correlação para medir estrutura do sistema.",
    detail: "É aqui que o ruído tenta se passar por sinal, e o motor tenta separar os dois.",
  },
  {
    step: "04",
    title: "Regime",
    helper: "Classificação causal walk-forward com histerese operacional.",
    text: "Leitura em estável, transição, estresse ou dispersão para ajustar o orçamento de risco.",
    detail: "O motor prefere errar devagar a ficar flipando estado por barulho curto.",
  },
  {
    step: "05",
    title: "Gate",
    helper: "Bloqueio automático se cobertura, universo ou QA falham.",
    text: "Publicação automática só quando os checks mínimos são aprovados.",
    detail: "Se a integridade cai, a UI deve mostrar diagnóstico, não fantasia de produção.",
  },
];

export default function PipelineFlow() {
  return (
    <section className="rounded-3xl border border-zinc-800 bg-zinc-950/60 p-6">
      <div className="text-xs uppercase tracking-[0.3em] text-zinc-500">Pipeline do motor</div>
      <div className="mt-5 grid gap-4 lg:grid-cols-5">
        {steps.map((item) => (
          <article
            key={item.step}
            className="rounded-2xl border border-zinc-800 bg-black/20 p-4 transition hover:-translate-y-1 hover:border-zinc-600"
          >
            <div className="flex items-center gap-2 text-[11px] uppercase tracking-[0.18em] text-zinc-500">
              <span>{item.step}</span>
              <HelpHint text={item.helper} />
            </div>
            <h2 className="mt-3 text-lg font-semibold text-zinc-100">{item.title}</h2>
            <p className="mt-2 text-sm leading-6 text-zinc-300">{item.text}</p>
            <div className="mt-4 rounded-2xl border border-zinc-800 bg-zinc-950/60 px-3 py-3 text-xs leading-5 text-zinc-400">
              {item.detail}
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}
