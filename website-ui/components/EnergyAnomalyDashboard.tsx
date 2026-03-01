"use client";

import { useMemo, useState } from "react";
import EnergyBrazilRiskMap from "@/components/EnergyBrazilRiskMap";

type EnergyStateRisk = {
  uf: string;
  name: string;
  submarket_id: string;
  submarket_label: string;
  probability: number;
  risk_level: string;
};

type EnergySubmarket = {
  id: string;
  label: string;
  probability: number;
  risk_level: string;
  drivers: {
    load_z: number;
    cmo_z: number;
  };
};

type EnergyEdge = {
  from: string;
  to: string;
  correlation: number;
  abs_correlation: number;
  critical: boolean;
};

type Props = {
  payload: {
    as_of_date: string;
    national: {
      bottleneck_probability: number;
      risk_level: string;
      model_score_latest: number | null;
      model_score_date: string | null;
    };
    states: EnergyStateRisk[];
    submarkets: EnergySubmarket[];
    network: {
      edges: EnergyEdge[];
    };
    evidence: {
      pre_signal_rate: number;
      events_valid: number;
      interpretation: string;
    };
  };
};

function pct(v: number) {
  if (!Number.isFinite(v)) return "--";
  return `${(v * 100).toFixed(1)}%`;
}

function fmt(v: number) {
  if (!Number.isFinite(v)) return "--";
  return v.toFixed(3);
}

function riskLabel(level: string) {
  const norm = String(level || "").toLowerCase();
  if (norm === "critico") return "crítico";
  if (norm === "alto") return "alto";
  if (norm === "moderado") return "moderado";
  return "baixo";
}

export default function EnergyAnomalyDashboard({ payload }: Props) {
  const topStates = useMemo(() => payload.states.slice(0, 12), [payload.states]);
  const [selectedUF, setSelectedUF] = useState<string>(topStates[0]?.uf || "SP");

  const selected = useMemo(
    () => payload.states.find((state) => state.uf === selectedUF) || topStates[0] || null,
    [payload.states, selectedUF, topStates]
  );
  const selectedSubmarket = useMemo(
    () => payload.submarkets.find((row) => row.id === selected?.submarket_id) || null,
    [payload.submarkets, selected]
  );

  const topEdges = useMemo(() => {
    const critical = payload.network.edges.filter((edge) => edge.critical).slice(0, 8);
    return critical.length ? critical : payload.network.edges.slice(0, 8);
  }, [payload.network.edges]);

  return (
    <section className="space-y-6 p-5 md:p-6 lg:p-8">
      <header className="rounded-2xl border border-zinc-800 bg-zinc-950/70 p-5">
        <p className="text-xs uppercase tracking-[0.14em] text-zinc-500">Eigen Engine | Energia BR</p>
        <h1 className="mt-2 text-2xl font-semibold text-zinc-100 md:text-3xl">Detector de anomalias da rede elétrica</h1>
        <p className="mt-3 text-sm text-zinc-300">
          Mapa estrutural por estado com probabilidade de gargalo por submercado (N, NE, SE/CO, S).
        </p>
        <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <div className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-3">
            <p className="text-[11px] uppercase tracking-[0.12em] text-zinc-500">Data-base</p>
            <p className="mt-1 text-lg font-semibold text-zinc-100">{payload.as_of_date || "--"}</p>
          </div>
          <div className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-3">
            <p className="text-[11px] uppercase tracking-[0.12em] text-zinc-500">Risco nacional</p>
            <p className="mt-1 text-lg font-semibold text-zinc-100">{pct(payload.national.bottleneck_probability)}</p>
          </div>
          <div className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-3">
            <p className="text-[11px] uppercase tracking-[0.12em] text-zinc-500">Nível nacional</p>
            <p className="mt-1 text-lg font-semibold text-zinc-100">{riskLabel(payload.national.risk_level)}</p>
          </div>
          <div className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-3">
            <p className="text-[11px] uppercase tracking-[0.12em] text-zinc-500">Pré-sinal validado</p>
            <p className="mt-1 text-lg font-semibold text-zinc-100">{pct(payload.evidence.pre_signal_rate)}</p>
          </div>
        </div>
      </header>

      <div className="grid gap-6 xl:grid-cols-[1.2fr_0.8fr]">
        <div className="rounded-2xl border border-zinc-800 bg-zinc-950/60 p-4">
          <p className="mb-3 text-sm font-medium text-zinc-300">Mapa de gargalo por estado</p>
          <EnergyBrazilRiskMap selectedUF={selectedUF} onSelectUF={setSelectedUF} stateRisk={payload.states} />
        </div>

        <div className="space-y-4">
          <div className="rounded-2xl border border-zinc-800 bg-zinc-950/60 p-4">
            <p className="text-sm font-medium text-zinc-300">Estado selecionado</p>
            {selected ? (
              <div className="mt-3 space-y-1 text-sm text-zinc-200">
                <div>
                  <span className="text-zinc-400">UF:</span> {selected.name} ({selected.uf})
                </div>
                <div>
                  <span className="text-zinc-400">Probabilidade de gargalo:</span> {pct(selected.probability)}
                </div>
                <div>
                  <span className="text-zinc-400">Nível:</span> {riskLabel(selected.risk_level)}
                </div>
                <div>
                  <span className="text-zinc-400">Submercado:</span> {selected.submarket_label}
                </div>
                {selectedSubmarket ? (
                  <div className="pt-2 text-xs text-zinc-400">
                    drivers: carga z={fmt(selectedSubmarket.drivers.load_z)} | preço z={fmt(selectedSubmarket.drivers.cmo_z)}
                  </div>
                ) : null}
              </div>
            ) : (
              <p className="mt-2 text-sm text-zinc-500">Sem estado selecionado.</p>
            )}
          </div>

          <div className="rounded-2xl border border-zinc-800 bg-zinc-950/60 p-4">
            <p className="text-sm font-medium text-zinc-300">Estados com maior risco</p>
            <div className="mt-3 space-y-2">
              {topStates.map((row) => (
                <button
                  key={row.uf}
                  type="button"
                  onClick={() => setSelectedUF(row.uf)}
                  className={`flex w-full items-center justify-between rounded-lg border px-3 py-2 text-left text-sm transition ${
                    row.uf === selectedUF
                      ? "border-cyan-500/70 bg-cyan-950/40 text-zinc-100"
                      : "border-zinc-800 bg-zinc-900/60 text-zinc-200 hover:border-zinc-700"
                  }`}
                >
                  <span>
                    {row.name} ({row.uf})
                  </span>
                  <span>{pct(row.probability)}</span>
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <div className="rounded-2xl border border-zinc-800 bg-zinc-950/60 p-4">
          <p className="text-sm font-medium text-zinc-300">Submercados (probabilidade de gargalo)</p>
          <div className="mt-3 overflow-x-auto">
            <table className="min-w-full text-sm text-zinc-200">
              <thead className="text-xs uppercase tracking-[0.12em] text-zinc-500">
                <tr>
                  <th className="px-2 py-2 text-left">Submercado</th>
                  <th className="px-2 py-2 text-left">Prob.</th>
                  <th className="px-2 py-2 text-left">Nível</th>
                  <th className="px-2 py-2 text-left">Carga z</th>
                  <th className="px-2 py-2 text-left">Preço z</th>
                </tr>
              </thead>
              <tbody>
                {payload.submarkets.map((row) => (
                  <tr key={row.id} className="border-t border-zinc-800">
                    <td className="px-2 py-2">{row.label}</td>
                    <td className="px-2 py-2">{pct(row.probability)}</td>
                    <td className="px-2 py-2">{riskLabel(row.risk_level)}</td>
                    <td className="px-2 py-2">{fmt(row.drivers.load_z)}</td>
                    <td className="px-2 py-2">{fmt(row.drivers.cmo_z)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="rounded-2xl border border-zinc-800 bg-zinc-950/60 p-4">
          <p className="text-sm font-medium text-zinc-300">Acoplamentos críticos entre submercados</p>
          <div className="mt-3 space-y-2">
            {topEdges.map((edge, idx) => (
              <div key={`${edge.from}-${edge.to}-${idx}`} className="rounded-lg border border-zinc-800 bg-zinc-900/60 p-3 text-sm text-zinc-200">
                <div className="flex items-center justify-between">
                  <span>
                    {edge.from} ↔ {edge.to}
                  </span>
                  <span>{fmt(edge.abs_correlation)}</span>
                </div>
                <div className="mt-1 text-xs text-zinc-400">
                  correlação={fmt(edge.correlation)} | {edge.critical ? "conexão crítica" : "conexão monitorada"}
                </div>
              </div>
            ))}
          </div>
          <p className="mt-4 text-xs text-zinc-500">
            Evidência histórica: {payload.evidence.events_valid} eventos | interpretação: {payload.evidence.interpretation}.
          </p>
        </div>
      </div>
    </section>
  );
}
