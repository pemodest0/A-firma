"use client";

import { useMemo, useState } from "react";

type Point = { date: string; price: number | null; confidence: number | null; regime: string };
type LegacyPoint = { t: number; confidence: number | null; regime: string };

type SeriesMap = Record<string, Point[]>;

type Props = {
  data: SeriesMap | Point[] | LegacyPoint[];
  selected?: string[];
  normalize?: boolean;
  showRegimeBands?: boolean;
  smoothing?: "none" | "ema_short" | "ema_long";
  rangePreset?: string;
  tooltipMode?: "full" | "price_only";
  chartTitle?: string;
  yUnitLabel?: string;
};

const regimeColors: Record<string, string> = {
  STABLE: "rgba(52,211,153,0.14)",
  TRANSITION: "rgba(251,191,36,0.14)",
  UNSTABLE: "rgba(251,113,133,0.14)",
  UNKNOWN: "rgba(113,113,122,0.10)",
};

const lineColors = ["#38bdf8", "#22c55e", "#f97316", "#a855f7", "#facc15", "#14b8a6", "#f472b6", "#60a5fa"];

function formatDate(d: string) {
  if (!d) return "";
  const t = new Date(`${d}T00:00:00Z`);
  const mm = String(t.getUTCMonth() + 1).padStart(2, "0");
  const yy = String(t.getUTCFullYear()).slice(-2);
  return `${mm}/${yy}`;
}

function formatValue(v: number) {
  const a = Math.abs(v);
  if (a >= 1_000_000) return `${(v / 1_000_000).toFixed(1)}M`;
  if (a >= 1_000) return `${(v / 1_000).toFixed(1)}K`;
  return v.toFixed(2);
}

function ema(values: number[], period: number) {
  if (!values.length) return values;
  const alpha = 2 / (period + 1);
  const out = [values[0]];
  for (let i = 1; i < values.length; i += 1) {
    out.push(alpha * values[i] + (1 - alpha) * out[i - 1]);
  }
  return out;
}

function downsample<T>(arr: T[], max = 2000) {
  if (arr.length <= max) return arr;
  const step = Math.ceil(arr.length / max);
  return arr.filter((_, idx) => idx % step === 0);
}

function rangeCount(preset: string) {
  if (preset === "30d") return 30;
  if (preset === "90d") return 90;
  if (preset === "180d") return 180;
  if (preset === "1y") return 252;
  return 0;
}

export default function RegimeChart(props: Props) {
  const {
    data,
    selected = [],
    normalize = false,
    showRegimeBands = true,
    smoothing = "none",
    rangePreset = "all",
    tooltipMode = "full",
    chartTitle = "Grafico principal",
    yUnitLabel,
  } = props;
  const [hidden, setHidden] = useState<Record<string, boolean>>({});
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);

  const prepared = useMemo(() => {
    const normalizeLegacy = (arr: Array<Point | LegacyPoint>): Point[] =>
      arr.map((p, i) => {
        if ("date" in p) return p;
        return {
          date: String(p.t ?? i),
          price: Number.isFinite(p.confidence) ? p.confidence : null,
          confidence: Number.isFinite(p.confidence) ? p.confidence : null,
          regime: p.regime ?? "UNKNOWN",
        };
      });

    const dataMap: SeriesMap = Array.isArray(data) ? { SERIES: normalizeLegacy(data) } : data;
    const baseSelection = selected.length ? selected : Object.keys(dataMap);
    const active = baseSelection.filter((a) => !hidden[a] && (dataMap[a] || []).length);
    if (!active.length) return null;

    let commonLen = Math.min(...active.map((a) => dataMap[a].length));
    const rc = rangeCount(rangePreset);
    if (rc > 0) commonLen = Math.min(commonLen, rc);

    const aligned = active.map((a) => ({
      asset: a,
      points: dataMap[a].slice(-commonLen),
    }));

    const sampled = aligned.map((s) => ({
      asset: s.asset,
      points: downsample(s.points, 1600),
    }));

    const series = sampled.map((s) => {
      const prices = s.points.map((p) => (p.price == null ? NaN : p.price));
      const base = prices.find((v) => Number.isFinite(v)) || 1;
      const normalized = prices.map((v) => (Number.isFinite(v) ? (v / base) * 100 : NaN));
      const source = normalize ? normalized : prices;
      const smooth =
        smoothing === "ema_short" ? ema(source as number[], 8) : smoothing === "ema_long" ? ema(source as number[], 20) : source;
      return {
        asset: s.asset,
        points: s.points,
        values: smooth,
      };
    });

    const allValues = series.flatMap((s) => s.values).filter((v) => Number.isFinite(v)) as number[];
    if (!allValues.length) return null;

    const ymin = Math.min(...allValues);
    const ymax = Math.max(...allValues);

    return {
      series,
      ymin,
      ymax,
      count: series[0].values.length,
      focus: series[0].points,
    };
  }, [data, selected, normalize, smoothing, hidden, rangePreset]);

  const width = 1200;
  const height = 480;
  const pad = 56;

  const scaleX = (i: number, total: number) => pad + (i / Math.max(1, total - 1)) * (width - pad * 2);
  const scaleY = (v: number, ymin: number, ymax: number, h: number) => h - pad - ((v - ymin) / Math.max(1e-9, ymax - ymin)) * (h - pad * 2);

  if (!prepared) return <div className="text-sm text-zinc-500">Selecione ativos para visualizar o grafico.</div>;

  const h = height;
  const hover = hoverIndex != null ? Math.min(hoverIndex, prepared.count - 1) : null;

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between text-xs text-zinc-400">
        <div>{chartTitle}</div>
        <button className="rounded border border-zinc-700 px-2 py-1 hover:border-zinc-500" onClick={() => setHidden({})}>
          Reset series
        </button>
      </div>
      <div className="text-xs text-zinc-500">
        Eixo X: tempo do periodo selecionado. Eixo Y: {yUnitLabel || (normalize ? "indice base 100" : "preco do ativo")}.
      </div>

      <div className="rounded-xl border border-zinc-800 p-4 md:p-5 bg-transparent">
        <svg
          viewBox={`0 0 ${width} ${h}`}
          className="w-full h-[360px] lg:h-[500px]"
          onMouseMove={(e) => {
            const rect = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
            const localX = e.clientX - rect.left;
            const idx = Math.round(((localX - pad) / Math.max(1, rect.width - 2 * pad)) * (prepared.count - 1));
            setHoverIndex(Math.max(0, Math.min(prepared.count - 1, idx)));
          }}
          onMouseLeave={() => setHoverIndex(null)}
        >
          {showRegimeBands
            ? prepared.focus.map((p, i, arr) => {
                if (i === 0) return null;
                const x0 = scaleX(i - 1, arr.length);
                const x1 = scaleX(i, arr.length);
                const regime = p.regime in regimeColors ? p.regime : "UNKNOWN";
                return <rect key={`band-${i}`} x={x0} y={pad} width={Math.max(1, x1 - x0)} height={h - 2 * pad} fill={regimeColors[regime]} />;
              })
            : null}

          {Array.from({ length: 5 }).map((_, i) => {
            const y = pad + ((h - 2 * pad) * i) / 4;
            const v = prepared.ymax - ((prepared.ymax - prepared.ymin) * i) / 4;
            return (
              <g key={`y-${i}`}>
                <line x1={pad} y1={y} x2={width - pad} y2={y} stroke="rgba(148,163,184,0.14)" />
                <text x={pad - 10} y={y + 4} textAnchor="end" fill="#94a3b8" fontSize="11">
                  {formatValue(v)}
                </text>
              </g>
            );
          })}

          {Array.from({ length: 6 }).map((_, i) => {
            const idx = Math.round((i / 5) * (prepared.count - 1));
            const x = scaleX(idx, prepared.count);
            const dt = prepared.series[0].points[idx]?.date;
            return (
              <g key={`x-${i}`}>
                <line x1={x} y1={pad} x2={x} y2={h - pad} stroke="rgba(148,163,184,0.08)" />
                <text x={x} y={h - pad + 18} textAnchor="middle" fill="#94a3b8" fontSize="11">
                  {formatDate(dt)}
                </text>
              </g>
            );
          })}
          <text x={width / 2} y={h - 8} textAnchor="middle" fill="#94a3b8" fontSize="11">
            Tempo
          </text>
          <text x={14} y={h / 2} textAnchor="middle" fill="#94a3b8" fontSize="11" transform={`rotate(-90 14 ${h / 2})`}>
            {normalize ? "Indice (base 100)" : "Preco"}
          </text>

          {prepared.series.map((s, idx) => {
            const d = s.values
              .map((v, i) => (Number.isFinite(v) ? `${i === 0 ? "M" : "L"} ${scaleX(i, s.values.length)} ${scaleY(v, prepared.ymin, prepared.ymax, h)}` : ""))
              .join(" ");
            return (
              <g key={s.asset}>
                <path d={d} fill="none" stroke={lineColors[idx % lineColors.length]} strokeWidth="2.2" />
                {s.values.map((v, i) => {
                  if (!Number.isFinite(v) || i % 18 !== 0) return null;
                  return (
                    <circle key={`${s.asset}-${i}`} cx={scaleX(i, s.values.length)} cy={scaleY(v, prepared.ymin, prepared.ymax, h)} r={1.8} fill={lineColors[idx % lineColors.length]} />
                  );
                })}
              </g>
            );
          })}

          {hover != null ? <line x1={scaleX(hover, prepared.count)} y1={pad} x2={scaleX(hover, prepared.count)} y2={h - pad} stroke="rgba(226,232,240,0.5)" strokeDasharray="4 4" /> : null}
        </svg>
      </div>

      <div className="flex flex-wrap gap-2 text-xs">
        {prepared.series.map((s, idx) => (
          <button
            key={s.asset}
            className={`rounded-full border px-2 py-1 ${hidden[s.asset] ? "border-zinc-700 text-zinc-500" : "border-zinc-600 text-zinc-200"}`}
            onClick={() => setHidden((prev) => ({ ...prev, [s.asset]: !prev[s.asset] }))}
          >
            <span className="inline-block h-2 w-2 rounded-full mr-2" style={{ background: lineColors[idx % lineColors.length] }} />
            {s.asset}
          </button>
        ))}
      </div>

      {hover != null ? (
        <div className="rounded-lg border border-zinc-800 bg-zinc-950 px-3 py-2 text-xs text-zinc-300">
          <div>Data: {prepared.series[0].points[hover]?.date || "n/d"}</div>
          {tooltipMode === "price_only" ? (
            prepared.series.map((series) => (
              <div key={`${series.asset}-hover`}>
                {series.asset}: {Number.isFinite(series.values[hover]) ? formatValue(series.values[hover]) : "n/d"}
              </div>
            ))
          ) : (
            <>
              <div>Regime: {prepared.series[0].points[hover]?.regime || "n/d"}</div>
              <div>
                Confiança:{" "}
                {(() => {
                  const confidence = prepared.series[0].points[hover]?.confidence;
                  if (!Number.isFinite(confidence as number)) return "n/d";
                  return `${((confidence as number) * 100).toFixed(1)}%`;
                })()}
              </div>
            </>
          )}
        </div>
      ) : null}
    </div>
  );
}
