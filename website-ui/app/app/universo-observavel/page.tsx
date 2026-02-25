import {
  readLatestLabCorrAssetDiagnostics,
  readLatestLabCorrRegimeSeries,
  readLatestLabCorrSectorDiagnostics,
  readLatestLabCorrTimeseries,
} from "@/lib/server/data";

type AssetPoint = {
  ticker: string;
  sector: string;
  risk: number;
  confidence: number;
  regime: string;
};

type RegimePoint = {
  date: string;
  transition: number;
  exposure: number;
};

function toNum(value: unknown): number | null {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function clamp01(value: number) {
  return Math.max(0, Math.min(1, value));
}

function fmt(value: number | null, digits = 3) {
  if (value == null) return "--";
  return value.toFixed(digits);
}

function buildPath(points: Array<{ x: number; y: number }>) {
  if (!points.length) return "";
  const [first, ...rest] = points;
  return `M ${first.x.toFixed(2)} ${first.y.toFixed(2)} ` + rest.map((p) => `L ${p.x.toFixed(2)} ${p.y.toFixed(2)}`).join(" ");
}

export default async function UniversoObservavelPage() {
  const [assetRaw, sectorRaw, regimeRaw, ts] = await Promise.all([
    readLatestLabCorrAssetDiagnostics(1200),
    readLatestLabCorrSectorDiagnostics(),
    readLatestLabCorrRegimeSeries(120, 220),
    readLatestLabCorrTimeseries(120),
  ]);

  const assetPoints: AssetPoint[] = (Array.isArray(assetRaw) ? assetRaw : [])
    .map((row) => {
      const risk = toNum((row as Record<string, unknown>).risk_score);
      const confidence = toNum((row as Record<string, unknown>).confidence_score);
      if (risk == null || confidence == null) return null;
      return {
        ticker: String((row as Record<string, unknown>).ticker || "").trim(),
        sector: String((row as Record<string, unknown>).sector || "sem setor").trim(),
        risk: clamp01(risk),
        confidence: clamp01(confidence),
        regime: String((row as Record<string, unknown>).regime_asset || "").trim(),
      };
    })
    .filter((row): row is AssetPoint => row !== null)
    .slice(0, 500);

  const sectors = new Set(assetPoints.map((x) => x.sector).filter((x) => x.length > 0));
  const avgRisk = assetPoints.length ? assetPoints.reduce((acc, x) => acc + x.risk, 0) / assetPoints.length : null;
  const avgConfidence = assetPoints.length ? assetPoints.reduce((acc, x) => acc + x.confidence, 0) / assetPoints.length : null;

  const scatterWidth = 860;
  const scatterHeight = 360;
  const padL = 52;
  const padR = 20;
  const padT = 16;
  const padB = 42;
  const scatterInnerW = scatterWidth - padL - padR;
  const scatterInnerH = scatterHeight - padT - padB;

  const scatterPlot = assetPoints.map((p) => ({
    ...p,
    x: padL + p.risk * scatterInnerW,
    y: padT + (1 - p.confidence) * scatterInnerH,
  }));

  const regimePoints: RegimePoint[] = (Array.isArray(regimeRaw) ? regimeRaw : [])
    .map((row) => {
      const transition = toNum((row as Record<string, unknown>).transition_score);
      const exposure = toNum((row as Record<string, unknown>).exposure);
      if (transition == null || exposure == null) return null;
      return {
        date: String((row as Record<string, unknown>).date || ""),
        transition: clamp01(transition),
        exposure: clamp01(exposure),
      };
    })
    .filter((row): row is RegimePoint => row !== null)
    .slice(-160);

  const regimeWidth = 860;
  const regimeHeight = 320;
  const rPadL = 52;
  const rPadR = 20;
  const rPadT = 18;
  const rPadB = 38;
  const rInnerW = regimeWidth - rPadL - rPadR;
  const rInnerH = regimeHeight - rPadT - rPadB;

  const transitionPath = buildPath(
    regimePoints.map((p, idx) => ({
      x: rPadL + (idx / Math.max(1, regimePoints.length - 1)) * rInnerW,
      y: rPadT + (1 - p.transition) * rInnerH,
    }))
  );
  const exposurePath = buildPath(
    regimePoints.map((p, idx) => ({
      x: rPadL + (idx / Math.max(1, regimePoints.length - 1)) * rInnerW,
      y: rPadT + (1 - p.exposure) * rInnerH,
    }))
  );

  const sectorRows = (Array.isArray(sectorRaw) ? sectorRaw : []).slice(0, 18);

  return (
    <div className="p-5 md:p-6 lg:p-8 space-y-5">
      <section className="rounded-2xl border border-zinc-800 bg-zinc-950/55 p-5">
        <h1 className="text-2xl md:text-3xl font-semibold text-zinc-100">Universo observável</h1>
        <p className="mt-2 text-sm text-zinc-300">
          Leitura dos ativos e setores realmente recebidos pela API, com gráficos de espalhamento e de regimes.
        </p>
        <div className="mt-3 grid grid-cols-1 md:grid-cols-4 gap-3">
          <K title="Ativos recebidos" value={String(assetPoints.length)} />
          <K title="Setores recebidos" value={String(sectors.size)} />
          <K title="Risco médio" value={fmt(avgRisk)} />
          <K title="Confiança média" value={fmt(avgConfidence)} />
        </div>
        <div className="mt-3 text-xs text-zinc-500">
          Janela base: T120 | período da série macro: {ts?.start || "--"} até {ts?.end || "--"}.
        </div>
      </section>

      <section className="rounded-2xl border border-zinc-800 bg-zinc-950/55 p-5">
        <div className="text-sm font-semibold text-zinc-100">Mapa de espalhamento: risco x confiança (ativos)</div>
        <div className="mt-2 text-xs text-zinc-400">Eixo X: risco (`risk_score`) | Eixo Y: confiança (`confidence_score`).</div>
        <div className="mt-3 overflow-x-auto">
          <svg viewBox={`0 0 ${scatterWidth} ${scatterHeight}`} className="min-w-[700px] w-full h-[320px]">
            <rect x="0" y="0" width={scatterWidth} height={scatterHeight} fill="transparent" />
            <line x1={padL} y1={padT + scatterInnerH} x2={padL + scatterInnerW} y2={padT + scatterInnerH} stroke="#334155" />
            <line x1={padL} y1={padT} x2={padL} y2={padT + scatterInnerH} stroke="#334155" />
            <line x1={padL} y1={padT + scatterInnerH * 0.5} x2={padL + scatterInnerW} y2={padT + scatterInnerH * 0.5} stroke="#1f2937" />
            <line x1={padL + scatterInnerW * 0.5} y1={padT} x2={padL + scatterInnerW * 0.5} y2={padT + scatterInnerH} stroke="#1f2937" />
            <text x={padL + scatterInnerW - 6} y={scatterHeight - 10} textAnchor="end" fill="#94a3b8" fontSize="11">
              risco
            </text>
            <text x={12} y={padT + 8} fill="#94a3b8" fontSize="11">
              confiança
            </text>
            {scatterPlot.map((p) => (
              <circle key={`${p.ticker}-${p.sector}`} cx={p.x} cy={p.y} r="2.8" fill="#38bdf8" opacity="0.8">
                <title>{`${p.ticker} | ${p.sector} | risco=${fmt(p.risk)} | confiança=${fmt(p.confidence)} | estado=${p.regime || "--"}`}</title>
              </circle>
            ))}
          </svg>
        </div>
      </section>

      <section className="rounded-2xl border border-zinc-800 bg-zinc-950/55 p-5">
        <div className="text-sm font-semibold text-zinc-100">Gráfico de regimes (transição e exposição)</div>
        <div className="mt-2 text-xs text-zinc-400">
          Série causal vinda de `regime_series_T120.csv`: linha ciano=`transition_score`, linha laranja=`exposure`.
        </div>
        <div className="mt-3 overflow-x-auto">
          <svg viewBox={`0 0 ${regimeWidth} ${regimeHeight}`} className="min-w-[700px] w-full h-[280px]">
            <rect x="0" y="0" width={regimeWidth} height={regimeHeight} fill="transparent" />
            <line x1={rPadL} y1={rPadT + rInnerH} x2={rPadL + rInnerW} y2={rPadT + rInnerH} stroke="#334155" />
            <line x1={rPadL} y1={rPadT} x2={rPadL} y2={rPadT + rInnerH} stroke="#334155" />
            <line x1={rPadL} y1={rPadT + rInnerH * 0.5} x2={rPadL + rInnerW} y2={rPadT + rInnerH * 0.5} stroke="#1f2937" />
            <path d={transitionPath} fill="none" stroke="#22d3ee" strokeWidth="2.3" />
            <path d={exposurePath} fill="none" stroke="#f97316" strokeWidth="2.1" />
          </svg>
        </div>
        <div className="mt-2 text-xs text-zinc-500">Pontos de regime carregados: {regimePoints.length}</div>
      </section>

      <section className="rounded-2xl border border-zinc-800 bg-zinc-950/55 p-5">
        <div className="text-sm font-semibold text-zinc-100">Resumo setorial (dados recebidos)</div>
        <div className="mt-3 overflow-x-auto">
          <table className="w-full min-w-[720px] text-sm text-zinc-300">
            <thead className="text-zinc-400">
              <tr className="border-b border-zinc-800">
                <th className="py-2 text-left">Setor</th>
                <th className="py-2 text-left">n_assets</th>
                <th className="py-2 text-left">risk_mean</th>
                <th className="py-2 text-left">confidence_mean</th>
                <th className="py-2 text-left">pct_instavel</th>
              </tr>
            </thead>
            <tbody>
              {sectorRows.map((row, idx) => {
                const x = row as Record<string, unknown>;
                return (
                  <tr key={`${String(x.sector || "setor")}-${idx}`} className="border-b border-zinc-900">
                    <td className="py-2">{String(x.sector || "--")}</td>
                    <td className="py-2">{fmt(toNum(x.n_assets), 0)}</td>
                    <td className="py-2">{fmt(toNum(x.risk_mean))}</td>
                    <td className="py-2">{fmt(toNum(x.confidence_mean))}</td>
                    <td className="py-2">{fmt(toNum(x.pct_instavel))}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      <section className="rounded-2xl border border-zinc-800 bg-zinc-950/45 p-5">
        <div className="text-sm font-semibold text-zinc-100">Verificação de labels e origem dos dados</div>
        <div className="mt-3 overflow-x-auto">
          <table className="w-full min-w-[680px] text-sm text-zinc-300">
            <thead className="text-zinc-400">
              <tr className="border-b border-zinc-800">
                <th className="py-2 text-left">Label na UI</th>
                <th className="py-2 text-left">Campo lido</th>
                <th className="py-2 text-left">Origem</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-b border-zinc-900">
                <td className="py-2">Risco</td>
                <td className="py-2">`risk_score`</td>
                <td className="py-2">`asset_regime_diagnostics.csv`</td>
              </tr>
              <tr className="border-b border-zinc-900">
                <td className="py-2">Confiança</td>
                <td className="py-2">`confidence_score`</td>
                <td className="py-2">`asset_regime_diagnostics.csv`</td>
              </tr>
              <tr className="border-b border-zinc-900">
                <td className="py-2">Estado (regime)</td>
                <td className="py-2">`regime_asset`</td>
                <td className="py-2">`asset_regime_diagnostics.csv`</td>
              </tr>
              <tr className="border-b border-zinc-900">
                <td className="py-2">Força de transição</td>
                <td className="py-2">`transition_score`</td>
                <td className="py-2">`regime_series_T120.csv`</td>
              </tr>
              <tr>
                <td className="py-2">Exposição estrutural</td>
                <td className="py-2">`exposure`</td>
                <td className="py-2">`regime_series_T120.csv`</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}

function K({ title, value }: { title: string; value: string }) {
  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-950/60 p-3">
      <div className="text-xs uppercase tracking-[0.12em] text-zinc-500">{title}</div>
      <div className="mt-2 text-lg font-semibold text-zinc-100">{value}</div>
    </div>
  );
}
