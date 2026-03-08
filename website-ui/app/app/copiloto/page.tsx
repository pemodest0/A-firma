import EigenEngineAssistantFloatingChat from "@/components/EigenEngineAssistantFloatingChat";

export default function CopilotoPage() {
  const prompts = [
    "Me explica o risco de hoje sem economês.",
    "Se eu tivesse R$ 10 mil, quanto iria para risco e quanto ficaria defensivo?",
    "Hoje faz mais sentido olhar finanças ou cripto?",
  ];

  return (
    <section className="p-5 md:p-6 lg:p-8 space-y-5">
      <header className="rounded-2xl border border-zinc-800 bg-zinc-950/50 p-5">
        <p className="text-xs tracking-[0.14em] uppercase text-zinc-500">Copiloto</p>
        <h1 className="mt-2 text-2xl md:text-3xl font-semibold text-zinc-100">Converse com o copiloto do motor</h1>
        <p className="mt-3 text-sm text-zinc-300">
          Pense nele como alguém tentando te ajudar a usar o app sem pânico e sem teatro quantitativo. Ele lê os
          artefatos de finanças, cripto, shadow e pesquisa, e responde em linguagem simples.
        </p>
      </header>

      <section className="rounded-2xl border border-zinc-800 bg-zinc-950/55 p-5">
        <div className="text-xs uppercase tracking-[0.16em] text-zinc-500">Comece por aqui</div>
        <div className="mt-3 rounded-xl border border-zinc-800 bg-black/20 p-4 text-sm leading-7 text-zinc-300">
          “Eu olho o estado do motor, a faixa de exposição, o que está mais forte no momento e o que ainda é só
          pesquisa. Se algo estiver fraco ou bloqueado, eu vou te dizer isso sem enfeitar.”
        </div>
        <div className="mt-3 grid gap-3 md:grid-cols-3">
          {prompts.map((prompt) => (
            <div key={prompt} className="rounded-xl border border-zinc-800 bg-black/20 p-4 text-sm text-zinc-300">
              {prompt}
            </div>
          ))}
        </div>
      </section>

      <EigenEngineAssistantFloatingChat mode="embedded" className="w-full" />
    </section>
  );
}
