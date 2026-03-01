import Link from "next/link";
import { existsSync } from "node:fs";
import path from "path";
import { promises as fs } from "fs";

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

type DomainReadiness = {
  domain: "finance" | "energy" | "agro";
  status: string;
  dataLastDate: string;
  detail: string;
};

function resolveResultsRoot() {
  const candidates = [path.join(process.cwd(), "results"), path.join(process.cwd(), "..", "results")];
  for (const candidate of candidates) {
    if (existsSync(candidate)) return candidate;
  }
  return candidates[1];
}

async function readJsonSafe(filePath: string) {
  try {
    const raw = await fs.readFile(filePath, "utf-8");
    return JSON.parse(raw) as Record<string, unknown>;
  } catch {
    return {};
  }
}

async function readFinanceReadiness(resultsRoot: string): Promise<DomainReadiness> {
  const latestPath = path.join(resultsRoot, "ops", "finance_product_ready", "latest_finance_product_ready.json");
  const latest = await readJsonSafe(latestPath);
  const reportPathRaw = String(latest.finance_product_ready_json || "").trim();
  const reportPath = reportPathRaw
    ? path.isAbsolute(reportPathRaw)
      ? reportPathRaw
      : path.join(resultsRoot, reportPathRaw)
    : "";
  const report = reportPath ? await readJsonSafe(reportPath) : {};
  const status = String(report.overall_readiness || latest.overall_readiness || "missing");
  const dataLastDate = String(report.data_last_date || latest.data_last_date || "");
  const opState = String(report.operational_state || "");
  const risk = String(report.risk_level_next_month || "");
  return {
    domain: "finance",
    status,
    dataLastDate,
    detail: [opState ? `estado=${opState}` : "", risk ? `risco=${risk}` : ""].filter(Boolean).join(" | "),
  };
}

async function readReleaseDomain(resultsRoot: string, domain: "energy" | "agro"): Promise<DomainReadiness> {
  const releasePath = path.join(resultsRoot, `${domain}_br`, `latest_release_${domain}_br.json`);
  const release = await readJsonSafe(releasePath);
  const status = String(release.status || "missing");
  const latestDir = String(release.latest_dir || "").trim();
  const evidencePath = latestDir
    ? path.join(latestDir, `historical_structure_summary_${domain}_br.json`)
    : path.join(resultsRoot, `${domain}_br`, "latest", `historical_structure_summary_${domain}_br.json`);
  const evidence = await readJsonSafe(evidencePath);
  const dataLastDate = String(evidence.last_date || evidence.data_last_date || "");
  const schemaChecks = (release.schema_checks as Record<string, unknown>) || {};
  const schemaAllOk = Boolean(schemaChecks.all_ok);
  const mode = String(((release.ranking_builder as Record<string, unknown>) || {}).mode || "");
  return {
    domain,
    status,
    dataLastDate,
    detail: `schema_all_ok=${schemaAllOk ? "sim" : "não"}${mode ? ` | ranking=${mode}` : ""}`,
  };
}

function badgeTone(status: string) {
  const normalized = status.toLowerCase();
  if (normalized === "ok" || normalized === "pass") return "border-emerald-700/60 bg-emerald-950/20 text-emerald-200";
  if (normalized === "warn" || normalized === "warning") return "border-amber-700/60 bg-amber-950/20 text-amber-200";
  return "border-rose-700/60 bg-rose-950/20 text-rose-200";
}

export default async function VendaPage() {
  const resultsRoot = resolveResultsRoot();
  const [financeState, energyState, agroState] = await Promise.all([
    readFinanceReadiness(resultsRoot),
    readReleaseDomain(resultsRoot, "energy"),
    readReleaseDomain(resultsRoot, "agro"),
  ]);
  const domainStates = [financeState, energyState, agroState];

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

      <section className="rounded-2xl border border-zinc-800 bg-zinc-950/55 p-5">
        <h2 className="text-lg font-semibold text-zinc-100">Status real dos artefatos por domínio</h2>
        <p className="mt-2 text-sm text-zinc-400">
          Esta seção lê os artefatos publicados em `results/ops/finance_product_ready`, `results/energy_br` e `results/agro_br`.
        </p>
        <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-3">
          {domainStates.map((state) => (
            <article key={state.domain} className="rounded-xl border border-zinc-800 bg-black/20 p-3 space-y-2">
              <div className="flex items-center justify-between gap-2">
                <div className="text-sm font-semibold text-zinc-100 capitalize">{state.domain}</div>
                <span className={`rounded-md border px-2 py-1 text-[11px] ${badgeTone(state.status)}`}>{state.status || "missing"}</span>
              </div>
              <div className="text-xs text-zinc-400">Última data efetiva: {state.dataLastDate || "n/d"}</div>
              <div className="text-xs text-zinc-500">{state.detail || "Sem detalhe adicional no artefato atual."}</div>
            </article>
          ))}
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
