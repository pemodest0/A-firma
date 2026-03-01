import EigenEngineAssistantFloatingChat from "@/components/EigenEngineAssistantFloatingChat";

export default function CopilotoPage() {
  return (
    <section className="p-5 md:p-6 lg:p-8 space-y-5">
      <header className="rounded-2xl border border-zinc-800 bg-zinc-950/50 p-5">
        <p className="text-xs tracking-[0.14em] uppercase text-zinc-500">Copiloto</p>
        <h1 className="mt-2 text-2xl md:text-3xl font-semibold text-zinc-100">Eigen Engine Assistant</h1>
        <p className="mt-3 text-sm text-zinc-300">
          Chat guiado por artefatos do motor (finanças, energia e agro). Sem recomendação de compra ou venda.
        </p>
      </header>

      <EigenEngineAssistantFloatingChat mode="embedded" className="w-full" />
    </section>
  );
}
