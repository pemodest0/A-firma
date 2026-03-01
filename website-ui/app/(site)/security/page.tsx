export default function SecurityPage() {
  return (
    <main className="mx-auto max-w-5xl px-6 py-12">
      <section className="rounded-2xl border border-zinc-800 bg-zinc-900/45 p-8 md:p-10">
        <div className="text-xs uppercase tracking-[0.22em] text-cyan-300">Segurança do site</div>
        <h1 className="mt-3 text-3xl md:text-5xl font-semibold tracking-tight text-zinc-100">
          Controles técnicos ativos na camada web
        </h1>
        <ul className="mt-6 space-y-2 text-zinc-300 text-sm">
          <li>- Content-Security-Policy aplicada para reduzir risco de injeção e recursos indevidos.</li>
          <li>- X-Frame-Options = DENY para bloquear clickjacking.</li>
          <li>- X-Content-Type-Options = nosniff para endurecer parsing de MIME.</li>
          <li>- Referrer-Policy restritiva e Permissions-Policy sem câmera/microfone/geolocalização.</li>
          <li>- Chat e endpoints de dados com `Cache-Control: no-store` para contexto operacional atualizado.</li>
        </ul>
      </section>
    </main>
  );
}
