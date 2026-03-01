import Link from "next/link";

export default function HowItWorksSection() {
  return (
    <section className="rounded-[26px] border border-zinc-800/80 bg-zinc-950/55 p-8 md:p-9 space-y-5 py-10 md:py-12 lg:py-14 xl:py-16">
      <div className="text-xs uppercase tracking-[0.25em] text-zinc-400">O problema</div>
      <h2 className="text-3xl md:text-4xl font-semibold tracking-tight text-zinc-100">
        Sistemas complexos mudam antes que os efeitos sejam visíveis.
      </h2>
      <div className="space-y-3 text-zinc-300 text-base md:text-lg max-w-4xl">
        <p>Instabilidades não surgem do nada. Elas aparecem quando a estrutura interna se reorganiza.</p>
        <p>Quando a concentração aumenta e a diversidade diminui, o sistema entra em nova fase.</p>
        <p>A maioria das ferramentas mede superfície. A Assyntrax mede profundidade estrutural.</p>
      </div>
      <Link className="inline-flex rounded-lg border border-cyan-300/35 bg-white/5 px-4 py-2 text-sm text-zinc-100 hover:border-cyan-200" href="/methods">
        Entenda como medimos
      </Link>
    </section>
  );
}
