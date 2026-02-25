import Link from "next/link";
import {
  findLatestLabCorrRun,
  readLatestLabCorrActionPlaybook,
  readLatestLabCorrTimeseries,
  readLatestValidationSummary,
} from "@/lib/server/data";

function toNum(value: unknown): number | null {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function fmt(value: number | null, digits = 3) {
  if (value == null) return "--";
  return value.toFixed(digits);
}

function fmtPct(value: number | null) {
  if (value == null) return "--";
  return `${(value * 100).toFixed(1)}%`;
}

export default async function DashboardPage() {
  const [labRun, ts, playbook, latestValidation] = await Promise.all([
    findLatestLabCorrRun(),
    readLatestLabCorrTimeseries(120),
    readLatestLabCorrActionPlaybook(120),
    readLatestValidationSummary(),
  ]);

  const playRows = Array.isArray(playbook) ? playbook : [];
  const latestPlay = playRows.length ? (playRows[playRows.length - 1] as Record<string, unknown>) : {};
  const validationObj = (latestValidation || {}) as Record<string, unknown>;
  const evidence = ((validationObj.evidence || {}) as Record<string, unknown>) || {};

  const regime = String(latestPlay.regime || "--");
  const signalTier = String(latestPlay.signal_tier || "--");
  const nUsed = toNum((ts?.latest as Record<string, unknown> | undefined)?.N_used);
  const structureScore = toNum((ts?.latest as Record<string, unknown> | undefined)?.structure_score);
  const asOfDate = String(validationObj.as_of_date || "--");
  const eventRate = toNum(evidence.event_rate);
  const alertRate = toNum(evidence.alert_rate);
  const lift = toNum(evidence.lift);

  return (
    <div className="p-5 md:p-6 lg:p-8 space-y-4">
      <section className="grid grid-cols-1 md:grid-cols-3 gap-3">
        <Card title="Regime">{regime}</Card>
        <Card title="Nível do sinal">{signalTier}</Card>
        <Card title="Run">{labRun?.runId || "--"}</Card>
      </section>

      <section className="grid grid-cols-1 md:grid-cols-4 gap-3">
        <Card title="As of">{asOfDate}</Card>
        <Card title="Event rate">{fmtPct(eventRate)}</Card>
        <Card title="Alert rate">{fmtPct(alertRate)}</Card>
        <Card title="Lift">{fmt(lift, 2)}</Card>
      </section>

      <section className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <Card title="N usado">{nUsed == null ? "--" : String(Math.trunc(nUsed))}</Card>
        <Card title="Structure score">{fmt(structureScore)}</Card>
      </section>

      <section className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <ActionCard
          href="/app/universo-observavel"
          title="Universo observável"
          text="Resumo de ativos, gráfico de espalhamento risco x confiança e gráfico de regimes."
        />
        <ActionCard
          href="/app/copiloto"
          title="Leonardo"
          text="Guia operacional para contexto de empresa, motor e evidências."
        />
      </section>
    </div>
  );
}

function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-950/55 p-4">
      <div className="text-xs uppercase tracking-[0.12em] text-zinc-500">{title}</div>
      <div className="mt-2 text-lg font-semibold text-zinc-100">{children}</div>
    </div>
  );
}

function ActionCard({ href, title, text }: { href: string; title: string; text: string }) {
  return (
    <Link href={href} className="rounded-xl border border-zinc-800 bg-zinc-950/55 p-4 hover:border-zinc-600 transition">
      <div className="text-xs uppercase tracking-[0.12em] text-zinc-500">{title}</div>
      <div className="mt-2 text-sm text-zinc-300">{text}</div>
    </Link>
  );
}
