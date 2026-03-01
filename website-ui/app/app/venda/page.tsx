import Link from "next/link";

type PackageKey = "basico" | "completo" | "sobmedida";

const features: Array<{ label: string; basico: boolean; completo: boolean; sobmedida: boolean }> = [
  { label: "Dashboard Eigen Engine com leitura diária", basico: true, completo: true, sobmedida: true },
  { label: "Gráfico histórico por ativo", basico: true, completo: true, sobmedida: true },
  { label: "Resumo por ativo com recomendações operacionais", basico: true, completo: true, sobmedida: true },
  { label: "Histórico completo de runs e auditoria ampliada", basico: false, completo: true, sobmedida: true },
  { label: "Suporte técnico de implantação", basico: false, completo: true, sobmedida: true },
  { label: "Integração API externa", basico: false, completo: false, sobmedida: true },
  { label: "Política/gate customizado por cliente", basico: false, completo: false, sobmedida: true },
  { label: "Acompanhamento dedicado com ajustes de operação", basico: false, completo: false, sobmedida: true },
];

const packageInfo: Array<{ key: PackageKey; title: string; note: string }> = [
  {
    key: "basico",
    title: "Básico",
    note: "Entrada rápida para mesa/comitê: leitura diária + histórico operacional direto no app.",
  },
  {
    key: "completo",
    title: "Completo",
    note: "Para operação institucional com trilha de auditoria mais forte e apoio técnico.",
  },
  {
    key: "sobmedida",
    title: "Sob medida",
    note: "Para times que precisam API, integração e política de risco customizada.",
  },
];

function hasFeature(feature: (typeof features)[number], pkg: PackageKey) {
  return pkg === "basico" ? feature.basico : pkg === "completo" ? feature.completo : feature.sobmedida;
}

export default function VendaPage() {
  return (
    <div className="p-5 md:p-6 lg:p-8 space-y-6">
      <section className="rounded-2xl border border-zinc-800 bg-zinc-950/50 p-5">
        <p className="text-xs tracking-[0.14em] uppercase text-zinc-500">Venda</p>
        <h1 className="mt-2 text-2xl md:text-3xl font-semibold text-zinc-100">Pacotes comerciais do Eigen Engine</h1>
        <p className="mt-3 text-sm text-zinc-300">
          Assyntrax entrega diagnóstico estrutural com operação diária. Escolha o pacote conforme nível de cobertura e integração desejado.
        </p>
        <div className="mt-4 flex flex-wrap gap-2">
          <Link
            href="/contact"
            className="rounded-lg bg-zinc-100 px-4 py-2 text-sm font-medium text-black hover:bg-white"
          >
            Falar com comercial
          </Link>
          <Link
            href="/app/dashboard"
            className="rounded-lg border border-zinc-700 px-4 py-2 text-sm text-zinc-200 hover:border-zinc-500"
          >
            Abrir app
          </Link>
        </div>
      </section>

      <section className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {packageInfo.map((pkg) => (
          <article key={pkg.key} className="rounded-2xl border border-zinc-800 bg-zinc-950/55 p-5">
            <h2 className="text-lg font-semibold text-zinc-100">{pkg.title}</h2>
            <p className="mt-2 text-sm text-zinc-300">{pkg.note}</p>
          </article>
        ))}
      </section>

      <section className="rounded-2xl border border-zinc-800 bg-zinc-950/55 p-5">
        <h2 className="text-lg font-semibold text-zinc-100">Checklist de features por pacote</h2>
        <div className="mt-4 overflow-x-auto">
          <table className="w-full min-w-[860px] text-sm">
            <thead className="text-zinc-400">
              <tr className="border-b border-zinc-800">
                <th className="py-2 text-left">Feature</th>
                <th className="py-2 text-center">Básico</th>
                <th className="py-2 text-center">Completo</th>
                <th className="py-2 text-center">Sob medida</th>
              </tr>
            </thead>
            <tbody>
              {features.map((feature) => (
                <tr key={feature.label} className="border-b border-zinc-900 text-zinc-300">
                  <td className="py-2">{feature.label}</td>
                  <td className="py-2 text-center">{hasFeature(feature, "basico") ? "✓" : "—"}</td>
                  <td className="py-2 text-center">{hasFeature(feature, "completo") ? "✓" : "—"}</td>
                  <td className="py-2 text-center">{hasFeature(feature, "sobmedida") ? "✓" : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="rounded-2xl border border-amber-800/40 bg-amber-950/15 p-5">
        <h2 className="text-lg font-semibold text-zinc-100">Limites declarados</h2>
        <ul className="mt-3 space-y-2 text-sm text-zinc-300">
          <li>- Não prevê data de crash e não substitui decisão humana.</li>
          <li>- Não é recomendação de compra/venda e não promete retorno.</li>
          <li>- Uso focado em governança de risco e execução operacional.</li>
        </ul>
      </section>
    </div>
  );
}
