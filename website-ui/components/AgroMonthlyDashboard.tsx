"use client";

import { useEffect, useMemo, useState } from "react";

type JsonLike = Record<string, unknown>;

function fmtNum(v: unknown, digits = 3) {
  const n = Number(v);
  if (!Number.isFinite(n)) return "n/d";
  return n.toFixed(digits);
}

function arr(x: unknown): JsonLike[] {
  return Array.isArray(x) ? (x.filter((it) => it && typeof it === "object") as JsonLike[]) : [];
}

function txt(x: unknown, fallback = "n/d") {
  const s = String(x ?? "").trim();
  return s ? s : fallback;
}

export default function AgroMonthlyDashboard() {
  const [state, setState] = useState<JsonLike | null>(null);
  const [rankings, setRankings] = useState<JsonLike | null>(null);
  const [evidence, setEvidence] = useState<JsonLike | null>(null);
  const [error, setError] = useState<string>("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let alive = true;
    const load = async () => {
      setLoading(true);
      setError("");
      try {
        const [s, r, e] = await Promise.all([
          fetch("/api/agro/state", { cache: "no-store" }),
          fetch("/api/agro/rankings", { cache: "no-store" }),
          fetch("/api/agro/evidence", { cache: "no-store" }),
        ]);
        if (!s.ok || !r.ok || !e.ok) {
          const bad = [s, r, e].find((x) => !x.ok);
          throw new Error(`artefatos_agro_indisponiveis_status_${bad?.status ?? 503}`);
        }
        const [sj, rj, ej] = await Promise.all([s.json(), r.json(), e.json()]);
        if (!alive) return;
        setState(sj);
        setRankings(rj);
        setEvidence(ej);
      } catch (err) {
        if (!alive) return;
        setError(err instanceof Error ? err.message : "erro_desconhecido");
      } finally {
        if (alive) setLoading(false);
      }
    };
    load();
    return () => {
      alive = false;
    };
  }, []);

  const topAssets = useMemo(() => arr(rankings?.top_assets_global_mode).slice(0, 10), [rankings]);
  const topSectors = useMemo(() => arr(rankings?.top_sectors_global_mode).slice(0, 8), [rankings]);
  const coupling = useMemo(() => arr(state?.top_sectors_by_overlap).slice(0, 8), [state]);
  const evidenceObj = (evidence?.evidence ?? {}) as JsonLike;
  const globalState = (rankings?.global_state ?? {}) as JsonLike;

  return (
    <section className="p-6 md:p-8">
      <header className="mb-6">
        <h1 className="text-2xl font-semibold">Agro Brasil (mensal)</h1>
        <p className="text-sm text-zinc-400 mt-1">
          Leitura estrutural por frequência mensal com hierarquia setor-global. Sem fallback automático.
        </p>
      </header>

      {loading ? (
        <div className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-5 text-zinc-300">Carregando artefatos Agro...</div>
      ) : null}

      {!loading && error ? (
        <div className="rounded-xl border border-red-900/60 bg-red-950/30 p-5 text-red-200">
          Dados indisponíveis no momento ({error}). Execute o pipeline mensal Agro para publicar os artefatos.
        </div>
      ) : null}

      {!loading && !error ? (
        <div className="space-y-5">
          <section className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-5">
            <h2 className="text-sm uppercase tracking-widest text-zinc-400">1) Estado atual</h2>
            <div className="mt-3 grid gap-3 md:grid-cols-5 text-sm">
              <K label="Data base" value={txt(rankings?.date || state?.date)} />
              <K label="Score global" value={fmtNum(state?.global_score, 3)} />
              <K label="Q" value={fmtNum(globalState.q, 3)} />
              <K label="N usado" value={fmtNum(globalState.n_used, 0)} />
              <K label="Janela T" value={fmtNum(globalState.t_window, 0)} />
            </div>
          </section>

          <section className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-5">
            <h2 className="text-sm uppercase tracking-widest text-zinc-400">2) Evidência</h2>
            <div className="mt-3 grid gap-3 md:grid-cols-4 text-sm">
              <K label="Eventos válidos" value={fmtNum(evidenceObj.events_valid, 0)} />
              <K label="Pré-sinal" value={fmtNum(evidenceObj.pre_signal_count, 0)} />
              <K label="Taxa pré-sinal" value={fmtNum(evidenceObj.pre_signal_rate, 3)} />
              <K label="Interpretação" value={txt(evidenceObj.interpretation)} />
            </div>
          </section>

          <section className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-5">
            <h2 className="text-sm uppercase tracking-widest text-zinc-400">3) Impacto (ranking)</h2>
            <div className="mt-3 grid gap-4 md:grid-cols-2 text-sm">
              <div>
                <div className="text-zinc-300 mb-2">Top setores</div>
                <ul className="space-y-1">
                  {topSectors.map((row, idx) => (
                    <li key={`sector-${idx}`} className="flex items-center justify-between rounded bg-zinc-900/70 px-3 py-2">
                      <span>{txt(row.sector)}</span>
                      <span className="text-zinc-400">{fmtNum(row.impact, 4)}</span>
                    </li>
                  ))}
                </ul>
              </div>
              <div>
                <div className="text-zinc-300 mb-2">Top ativos</div>
                <ul className="space-y-1">
                  {topAssets.map((row, idx) => (
                    <li key={`asset-${idx}`} className="flex items-center justify-between rounded bg-zinc-900/70 px-3 py-2">
                      <span>{txt(row.ticker || row.asset_id)}</span>
                      <span className="text-zinc-400">{fmtNum(row.impact, 4)}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </section>

          <section className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-5">
            <h2 className="text-sm uppercase tracking-widest text-zinc-400">4) Coupling/hierarquia</h2>
            <ul className="mt-3 space-y-1 text-sm">
              {coupling.map((row, idx) => (
                <li key={`coupling-${idx}`} className="flex items-center justify-between rounded bg-zinc-900/70 px-3 py-2">
                  <span>{txt(row.sector)}</span>
                  <span className="text-zinc-400">{fmtNum(row.overlap_sector_global, 4)}</span>
                </li>
              ))}
            </ul>
          </section>
        </div>
      ) : null}
    </section>
  );
}

function K({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/60 px-3 py-2">
      <div className="text-xs text-zinc-500 uppercase tracking-widest">{label}</div>
      <div className="text-zinc-100 mt-1">{value}</div>
    </div>
  );
}

