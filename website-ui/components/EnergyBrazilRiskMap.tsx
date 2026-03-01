"use client";

import { useEffect, useMemo, useState } from "react";
import { ComposableMap, Geographies, Geography } from "react-simple-maps";

type GeoFeatureProps = {
  name?: string;
  sigla?: string;
  UF?: string;
  uf?: string;
  NAME_1?: string;
};

type StateRisk = {
  uf: string;
  name: string;
  probability: number;
  risk_level: string;
};

type Props = {
  selectedUF: string;
  onSelectUF: (uf: string) => void;
  stateRisk: StateRisk[];
};

const NAME_TO_UF: Record<string, string> = {
  acre: "AC",
  alagoas: "AL",
  amapa: "AP",
  amazonas: "AM",
  bahia: "BA",
  ceara: "CE",
  "distrito federal": "DF",
  "espirito santo": "ES",
  goias: "GO",
  maranhao: "MA",
  "mato grosso": "MT",
  "mato grosso do sul": "MS",
  "minas gerais": "MG",
  para: "PA",
  paraiba: "PB",
  parana: "PR",
  pernambuco: "PE",
  piaui: "PI",
  "rio de janeiro": "RJ",
  "rio grande do norte": "RN",
  "rio grande do sul": "RS",
  rondonia: "RO",
  roraima: "RR",
  "santa catarina": "SC",
  "sao paulo": "SP",
  sergipe: "SE",
  tocantins: "TO",
};

function normalizeText(value: string) {
  return value
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .trim();
}

function resolveUF(props: GeoFeatureProps): string {
  if (props.sigla) return props.sigla.toUpperCase();
  if (props.UF) return props.UF.toUpperCase();
  if (props.uf) return props.uf.toUpperCase();
  const name = props.name || props.NAME_1 || "";
  return NAME_TO_UF[normalizeText(name)] || "";
}

function colorByProbability(p: number) {
  if (!Number.isFinite(p)) return "rgba(255,255,255,0.08)";
  if (p >= 0.78) return "rgba(239, 68, 68, 0.70)";
  if (p >= 0.62) return "rgba(251, 146, 60, 0.64)";
  if (p >= 0.42) return "rgba(250, 204, 21, 0.56)";
  if (p >= 0.24) return "rgba(34, 211, 238, 0.48)";
  return "rgba(59, 130, 246, 0.42)";
}

export default function EnergyBrazilRiskMap({ selectedUF, onSelectUF, stateRisk }: Props) {
  const [geoData, setGeoData] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [hover, setHover] = useState<{ uf: string; name: string } | null>(null);
  const [reloadKey, setReloadKey] = useState(0);

  useEffect(() => {
    let mounted = true;
    fetch(`/geo/br-states.geojson?v=${reloadKey}`)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((json) => {
        if (!mounted) return;
        setError(null);
        setGeoData(json);
      })
      .catch(() => {
        if (!mounted) return;
        setGeoData(null);
        setError("Falha ao carregar mapa do Brasil.");
      });
    return () => {
      mounted = false;
    };
  }, [reloadKey]);

  const riskByUf = useMemo(() => {
    const m = new Map<string, StateRisk>();
    for (const row of stateRisk) {
      m.set(String(row.uf || "").toUpperCase(), row);
    }
    return m;
  }, [stateRisk]);

  if (!geoData) {
    return (
      <div className="flex h-[520px] w-full items-center justify-center rounded-xl border border-zinc-800 bg-zinc-950/70 text-sm text-zinc-400">
        {error || "Carregando mapa de risco da rede..."}
        {error ? (
          <button
            className="ml-4 rounded-md border border-zinc-700 px-3 py-1 text-xs text-zinc-200 hover:border-zinc-500"
            onClick={() => setReloadKey((k) => k + 1)}
          >
            Recarregar
          </button>
        ) : null}
      </div>
    );
  }

  return (
    <div className="relative h-[520px] w-full rounded-xl border border-zinc-800 bg-zinc-950/70 p-2">
      <ComposableMap
        projection="geoMercator"
        projectionConfig={{ scale: 680, center: [-52, -15] }}
        style={{ width: "100%", height: "100%" }}
      >
        <Geographies geography={geoData}>
          {({ geographies }: { geographies: Array<{ rsmKey: string; properties: unknown }> }) =>
            geographies.map((geo) => {
              const props = (geo.properties ?? {}) as GeoFeatureProps;
              const uf = resolveUF(props);
              const risk = riskByUf.get(uf);
              const isSelected = uf === selectedUF;
              const name = String(props.name || props.NAME_1 || uf || "Estado");
              return (
                <Geography
                  key={geo.rsmKey}
                  geography={geo}
                  onMouseEnter={() => setHover({ uf: uf || "--", name })}
                  onMouseLeave={() => setHover(null)}
                  onClick={() => {
                    if (uf) onSelectUF(uf);
                  }}
                  style={{
                    default: {
                      fill: colorByProbability(risk?.probability ?? NaN),
                      stroke: isSelected ? "rgba(255,255,255,0.95)" : "rgba(255,255,255,0.24)",
                      strokeWidth: isSelected ? 1.3 : 0.8,
                      outline: "none",
                    },
                    hover: {
                      fill: colorByProbability((risk?.probability ?? 0) + 0.12),
                      stroke: "rgba(255,255,255,0.84)",
                      strokeWidth: 1.1,
                      outline: "none",
                      cursor: "pointer",
                    },
                    pressed: {
                      fill: colorByProbability(risk?.probability ?? 0),
                      stroke: "rgba(255,255,255,1)",
                      strokeWidth: 1.2,
                      outline: "none",
                    },
                  }}
                />
              );
            })
          }
        </Geographies>
      </ComposableMap>

      {hover ? (
        <div className="pointer-events-none absolute right-3 top-3 rounded-md border border-zinc-700 bg-black/80 px-2 py-1 text-xs text-zinc-200">
          {hover.name} ({hover.uf})
          {riskByUf.get(hover.uf) ? (
            <div className="text-[11px] text-zinc-300">
              gargalo: {(100 * (riskByUf.get(hover.uf)?.probability || 0)).toFixed(1)}%
            </div>
          ) : null}
        </div>
      ) : null}

      <div className="absolute bottom-3 left-3 rounded-md border border-zinc-700 bg-black/70 px-2 py-1 text-[11px] text-zinc-300">
        baixo → moderado → alto → crítico
      </div>
    </div>
  );
}
