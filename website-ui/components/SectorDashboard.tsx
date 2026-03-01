"use client";

import { useEffect, useMemo, useState } from "react";
import DashboardFilters from "@/components/DashboardFilters";
import RegimeChart from "@/components/RegimeChart";

type Domain = "finance" | "energy" | "realestate";

type SeriesPoint = {
  date: string;
  price: number | null;
  confidence: number;
  regime: string;
  volume?: number | null;
};

type UniverseAsset = {
  asset: string;
  group?: string;
};

type AssetRow = {
  asset: string;
  group?: string;
  startDate?: string;
  endDate?: string;
  period: string;
  priceToday: number | null;
  pricePrev: number | null;
  changeAbs: number | null;
  changePct: number | null;
  ret5d: number | null;
  vol20d: number | null;
  volume: number | null;
  retH1: number | null;
  retH5: number | null;
  retH10: number | null;
};

type AssetRecommendation = {
  level: "ESTÁVEL" | "MONITORAR" | "ATENÇÃO" | "DEFENSIVO";
  action: string;
  rationale: string;
};

const MISSING = "n/d";

const groupLabels: Record<string, string> = {
  crypto: "Cripto",
  volatility: "Volatilidade",
  commodities_broad: "Commodities",
  energy: "Commodities cíclicas",
  metals: "Metais",
  bonds_rates: "Juros e Bonds",
  fx: "Moedas",
  equities_us_broad: "Ações EUA - índice amplo",
  equities_us_sectors: "Ações EUA - setores",
  equities_international: "Ações internacionais",
  realestate: "Imobiliário",
};

const financeGroupFilter: Array<{ value: string; label: string }> = [
  { value: "all", label: "Todos os grupos" },
  { value: "equities_us_broad", label: "Ações EUA - índice amplo" },
  { value: "equities_us_sectors", label: "Ações EUA - setores" },
  { value: "equities_international", label: "Ações internacionais" },
  { value: "commodities_broad", label: "Commodities" },
  { value: "metals", label: "Metais" },
  { value: "bonds_rates", label: "Juros e Bonds" },
  { value: "fx", label: "Câmbio" },
  { value: "crypto", label: "Cripto" },
  { value: "volatility", label: "Volatilidade" },
];

const PREFERRED_BY_DOMAIN: Record<Domain, string[]> = {
  finance: ["SPY", "QQQ", "IWM", "TLT", "GLD", "BTC-USD"],
  energy: ["XLE", "USO", "XOP"],
  realestate: ["FipeZap_Índice_FipeZAP_Total", "FipeZap_São_Paulo_Total", "FipeZap_Rio_de_Janeiro_Total"],
};

const DEFAULT_SAMPLE_SIZE = 20;
const EXPANDED_SAMPLE_SIZE = 40;

function mean(values: number[]) {
  if (!values.length) return null;
  return values.reduce((acc, value) => acc + value, 0) / values.length;
}

function std(values: number[]) {
  if (values.length < 2) return 0;
  const avg = mean(values);
  if (avg == null) return 0;
  const variance = values.reduce((acc, value) => acc + (value - avg) ** 2, 0) / values.length;
  return Math.sqrt(variance);
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function formatNumber(value: number | null | undefined, digits = 2) {
  if (!isFiniteNumber(value)) return MISSING;
  return value.toFixed(digits);
}

function formatPrice(value: number | null | undefined, digits = 2) {
  if (!isFiniteNumber(value)) return MISSING;
  return `${value.toFixed(digits)}`;
}

function formatPercent(value: number | null | undefined, digits = 2) {
  if (!isFiniteNumber(value)) return MISSING;
  const sign = value > 0 ? "+" : "";
  return `${sign}${(value * 100).toFixed(digits)}%`;
}

function computeReturn(points: Array<SeriesPoint & { price: number }>, horizonBars: number) {
  if (points.length <= horizonBars) return null;
  const last = points[points.length - 1];
  const ref = points[points.length - 1 - horizonBars];
  if (!isFiniteNumber(last?.price) || !isFiniteNumber(ref?.price) || ref.price === 0) return null;
  return (last.price - ref.price) / ref.price;
}

function toneFromPct(value: number | null | undefined) {
  if (!isFiniteNumber(value)) return "text-zinc-300";
  if (value > 0) return "text-emerald-300";
  if (value < 0) return "text-rose-300";
  return "text-zinc-300";
}

function recommendationTone(level: AssetRecommendation["level"]) {
  if (level === "ESTÁVEL") return "border-emerald-500/40 bg-emerald-500/10 text-emerald-200";
  if (level === "MONITORAR") return "border-cyan-500/40 bg-cyan-500/10 text-cyan-200";
  if (level === "ATENÇÃO") return "border-amber-500/40 bg-amber-500/10 text-amber-200";
  return "border-rose-500/40 bg-rose-500/10 text-rose-200";
}

function buildAssetRecommendation(row: AssetRow, horizon: 1 | 5 | 10): AssetRecommendation {
  const hRet = horizon === 1 ? row.retH1 : horizon === 5 ? row.retH5 : row.retH10;
  const changePct = isFiniteNumber(row.changePct) ? row.changePct : null;
  const vol20d = isFiniteNumber(row.vol20d) ? row.vol20d : null;

  if (changePct != null && vol20d != null) {
    if ((changePct <= -0.02 && vol20d >= 0.025) || (hRet != null && hRet <= -0.05)) {
      return {
        level: "DEFENSIVO",
        action: "Reduzir exposição e priorizar proteção",
        rationale: "Queda relevante combinada com volatilidade elevada ou perda acumulada no horizonte.",
      };
    }
    if (Math.abs(changePct) >= 0.015 && vol20d >= 0.02) {
      return {
        level: "ATENÇÃO",
        action: "Manter monitoramento intradiário e revisar gatilhos",
        rationale: "Movimento curto intenso com regime de volatilidade acima da média.",
      };
    }
    if (Math.abs(changePct) <= 0.006 && vol20d <= 0.015 && (hRet == null || Math.abs(hRet) <= 0.03)) {
      return {
        level: "ESTÁVEL",
        action: "Manter leitura com ajustes graduais",
        rationale: "Oscilação diária baixa e risco curto controlado no ativo.",
      };
    }
  }

  return {
    level: "MONITORAR",
    action: "Acompanhar continuidade do movimento",
    rationale: "Sem sinal extremo; validar direção com mais barras no horizonte ativo.",
  };
}

function buildAssetNarrative(row: AssetRow | null, horizon: 1 | 5 | 10) {
  if (!row) return "Selecione um ativo para ver a leitura do comportamento recente.";
  const hKey = horizon === 1 ? row.retH1 : horizon === 5 ? row.retH5 : row.retH10;
  const daily = isFiniteNumber(row.changePct) ? row.changePct : null;
  const tone = daily == null ? "com variação diária indisponível" : daily > 0 ? "de força" : daily < 0 ? "de pressão" : "lateral";
  const vol = isFiniteNumber(row.vol20d) ? row.vol20d : null;
  const volText = vol == null ? "sem vol 20d suficiente" : vol > 0.02 ? "volatilidade alta" : "volatilidade controlada";
  const hText = isFiniteNumber(hKey) ? formatPercent(hKey) : "sem leitura";

  return `Hoje o ativo está em movimento ${tone}, com ${volText}. No horizonte h${horizon}, a variação é ${hText}. Use como diagnóstico de risco e contexto de exposição, não como ordem automática.`;
}

function buildAssetTips(row: AssetRow | null, horizon: 1 | 5 | 10) {
  if (!row) {
    return ["Selecione um ativo para liberar as dicas de leitura contextual."];
  }
  const hRet = horizon === 1 ? row.retH1 : horizon === 5 ? row.retH5 : row.retH10;
  const changePct = isFiniteNumber(row.changePct) ? row.changePct : null;
  const vol20d = isFiniteNumber(row.vol20d) ? row.vol20d : null;

  const tips: string[] = [];

  if (changePct != null && vol20d != null && Math.abs(changePct) >= 0.02 && vol20d >= 0.025) {
    tips.push("Movimento forte com volatilidade alta: trate como fase de risco elevado e revise tamanho de exposição.");
  } else if (changePct != null && vol20d != null && Math.abs(changePct) <= 0.005 && vol20d <= 0.015) {
    tips.push("Movimento curto e vol baixa: cenário mais estável para comparar com outros ativos do grupo.");
  } else {
    tips.push("Use a leitura de preço junto da volatilidade para separar ruído de mudança estrutural.");
  }

  if (hRet != null) {
    if (hRet <= -0.05) {
      tips.push(`No h${horizon}, queda forte (${formatPercent(hRet)}): observar persistência antes de mudar posição.`);
    } else if (hRet >= 0.05) {
      tips.push(`No h${horizon}, alta forte (${formatPercent(hRet)}): monitorar se a vol acompanha ou se o movimento perde fôlego.`);
    } else {
      tips.push(`No h${horizon}, variação moderada (${formatPercent(hRet)}): útil para manter leitura de tendência sem exagero.`);
    }
  } else {
    tips.push(`Sem histórico suficiente para h${horizon}: interpretar com cautela até completar a amostra.`);
  }

  if (vol20d != null) {
    tips.push(vol20d >= 0.03 ? "Vol 20d elevada: aumente atenção ao risco de reversão curta." : "Vol 20d controlada: bom cenário para comparar estabilidade relativa.");
  } else {
    tips.push("Vol 20d indisponível: a série ainda não tem pontos suficientes para medir risco de curto prazo.");
  }

  return tips.slice(0, 3);
}

export default function SectorDashboard({
  title,
  showTable = true,
  initialDomain = "finance",
}: {
  title: string;
  showTable?: boolean;
  initialDomain?: Domain;
}) {
  const [timeframe, setTimeframe] = useState("daily");
  const [groupFilter, setGroupFilter] = useState("all");
  const [rangePreset, setRangePreset] = useState("180d");
  const [normalize, setNormalize] = useState(false);
  const [showRegimeBands, setShowRegimeBands] = useState(true);
  const [smoothing, setSmoothing] = useState<"none" | "ema_short" | "ema_long">("none");
  const [summaryHorizon, setSummaryHorizon] = useState<1 | 5 | 10>(5);
  const [focusAsset, setFocusAsset] = useState<string>("");
  const [showAllRecommendationCards, setShowAllRecommendationCards] = useState(false);

  const [universe, setUniverse] = useState<UniverseAsset[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [seriesByAsset, setSeriesByAsset] = useState<Record<string, SeriesPoint[]>>({});
  const [loading, setLoading] = useState(false);
  const [universeLoaded, setUniverseLoaded] = useState(true);

  const pickUniverseSample = (limit: number) => {
    const safeLimit = Math.max(1, Math.min(EXPANDED_SAMPLE_SIZE, limit));
    const preferred = PREFERRED_BY_DOMAIN[initialDomain] || [];
    const available = new Set(universe.map((x) => x.asset));
    const picked = preferred.filter((asset) => available.has(asset));
    const selectedSet = new Set<string>(picked.slice(0, safeLimit));
    for (const item of universe) {
      if (selectedSet.size >= safeLimit) break;
      selectedSet.add(item.asset);
    }
    setSelected(Array.from(selectedSet));
  };

  const loadExamples = () => {
    pickUniverseSample(DEFAULT_SAMPLE_SIZE);
  };

  const loadExpanded = () => {
    pickUniverseSample(EXPANDED_SAMPLE_SIZE);
  };

  useEffect(() => {
    const loadUniverse = async () => {
      try {
        let data: UniverseAsset[] = [];
        const assetsQueries = [
          `/api/assets?domain=${encodeURIComponent(initialDomain)}&status=${encodeURIComponent("validated,watch")}&include_inconclusive=1`,
          `/api/assets?domain=${encodeURIComponent(initialDomain)}&status=${encodeURIComponent("validated,watch")}&include_inconclusive=0`,
        ];

        for (const query of assetsQueries) {
          const res = await fetch(query);
          if (!res.ok) continue;
          const assetsJson = await res.json();
          const parsed = Array.isArray(assetsJson?.records) ? (assetsJson.records as UniverseAsset[]) : [];
          if (parsed.length) {
            data = parsed;
            break;
          }
        }

        if (!data.length) {
          setUniverse([]);
          setSelected([]);
          setUniverseLoaded(false);
          return;
        }

        const normalized = data.map((r) => ({
          asset: String(r.asset || ""),
          group: String(r.group || ""),
        }));
        const byGroup = groupFilter === "all" ? normalized : normalized.filter((r) => (r.group || "") === groupFilter);

        setUniverseLoaded(true);
        setUniverse(byGroup);
        setSelected((prev) => {
          const scoped = byGroup.map((u) => u.asset);
          const keep = prev.filter((asset) => scoped.includes(asset));
          if (keep.length) return keep;
          return scoped.slice(0, DEFAULT_SAMPLE_SIZE);
        });
      } catch {
        setUniverse([]);
        setSelected([]);
        setUniverseLoaded(false);
      }
    };

    loadUniverse();
  }, [initialDomain, groupFilter]);

  useEffect(() => {
    const loadSeries = async () => {
      if (!selected.length) {
        setSeriesByAsset({});
        return;
      }

      setLoading(true);
      try {
        const seriesRes = await fetch(`/api/graph/series-batch?assets=${selected.join(",")}&tf=${timeframe}&limit=2000`);
        if (!seriesRes.ok) {
          setSeriesByAsset({});
          return;
        }
        const seriesJson = await seriesRes.json();
        setSeriesByAsset(seriesJson || {});
      } finally {
        setLoading(false);
      }
    };

    loadSeries();
  }, [selected, timeframe]);

  useEffect(() => {
    if (!selected.length) {
      setFocusAsset("");
      return;
    }
    if (!focusAsset || !selected.includes(focusAsset)) {
      setFocusAsset(selected[0]);
    }
  }, [selected, focusAsset]);

  const tableRows = useMemo<AssetRow[]>(() => {
    const rows = selected.map((asset) => {
      const series = seriesByAsset[asset] || [];
      const rowMeta = universe.find((u) => u.asset === asset);
      const pricedPoints = series.filter((p): p is SeriesPoint & { price: number } => isFiniteNumber(p.price));

      const first = pricedPoints[0];
      const last = pricedPoints[pricedPoints.length - 1];
      const prev = pricedPoints.length >= 2 ? pricedPoints[pricedPoints.length - 2] : undefined;
      const point5 = pricedPoints.length >= 6 ? pricedPoints[pricedPoints.length - 6] : undefined;

      const priceToday = last?.price ?? null;
      const pricePrev = prev?.price ?? null;
      const changeAbs = isFiniteNumber(priceToday) && isFiniteNumber(pricePrev) ? priceToday - pricePrev : null;
      const changePct = isFiniteNumber(changeAbs) && isFiniteNumber(pricePrev) && pricePrev !== 0 ? changeAbs / pricePrev : null;
      const ret5d =
        isFiniteNumber(priceToday) && isFiniteNumber(point5?.price) && point5.price !== 0 ? (priceToday - point5.price) / point5.price : null;

      const returns = [] as number[];
      for (let i = 1; i < pricedPoints.length; i += 1) {
        const p0 = pricedPoints[i - 1]?.price;
        const p1 = pricedPoints[i]?.price;
        if (!isFiniteNumber(p0) || !isFiniteNumber(p1) || p0 === 0) continue;
        returns.push((p1 - p0) / p0);
      }
      const vol20d = returns.length >= 20 ? std(returns.slice(-20)) : null;
      const volume = isFiniteNumber(last?.volume) ? last.volume : null;

      return {
        asset,
        group: groupLabels[String(rowMeta?.group || "")] || String(rowMeta?.group || "Sem classificação"),
        startDate: first?.date,
        endDate: last?.date,
        period: first && last ? `${first.date} até ${last.date}` : MISSING,
        priceToday,
        pricePrev,
        changeAbs,
        changePct,
        ret5d,
        vol20d,
        volume,
        retH1: computeReturn(pricedPoints, 1),
        retH5: computeReturn(pricedPoints, 5),
        retH10: computeReturn(pricedPoints, 10),
      };
    });

    return rows.sort((a, b) => {
      const aScore = isFiniteNumber(a.changePct) ? Math.abs(a.changePct) : -1;
      const bScore = isFiniteNumber(b.changePct) ? Math.abs(b.changePct) : -1;
      return bScore - aScore;
    });
  }, [selected, seriesByAsset, universe]);

  const metrics = useMemo(() => {
    const allSeriesPoints = selected.flatMap((asset) => seriesByAsset[asset] || []);
    const absChanges = tableRows.map((row) => row.changePct).filter(isFiniteNumber).map((value) => Math.abs(value));
    const vols = tableRows.map((row) => row.vol20d).filter(isFiniteNumber);
    const lastPrices = tableRows.map((row) => row.priceToday).filter(isFiniteNumber);

    return {
      sampleSize: allSeriesPoints.length,
      avgAbsChange: mean(absChanges),
      avgVol20d: mean(vols),
      avgPrice: mean(lastPrices),
    };
  }, [selected, seriesByAsset, tableRows]);

  const summary = useMemo(() => {
    const withDaily = tableRows.filter((row) => isFiniteNumber(row.changePct));
    const topGain = [...withDaily].sort((a, b) => (b.changePct as number) - (a.changePct as number))[0];
    const topDrop = [...withDaily].sort((a, b) => (a.changePct as number) - (b.changePct as number))[0];

    const horizonKey = summaryHorizon === 1 ? "retH1" : summaryHorizon === 5 ? "retH5" : "retH10";
    const withHorizon = tableRows.filter((row) => isFiniteNumber(row[horizonKey] as number | null));
    const topGainH = [...withHorizon].sort((a, b) => (b[horizonKey] as number) - (a[horizonKey] as number))[0];
    const topDropH = [...withHorizon].sort((a, b) => (a[horizonKey] as number) - (b[horizonKey] as number))[0];

    const starts = tableRows.map((row) => row.startDate).filter((value): value is string => Boolean(value));
    const ends = tableRows.map((row) => row.endDate).filter((value): value is string => Boolean(value));
    const periodStart = starts.length ? starts.sort()[0] : null;
    const periodEnd = ends.length ? ends.sort()[ends.length - 1] : null;

    const avgVol20d = mean(tableRows.map((row) => row.vol20d).filter(isFiniteNumber));

    return {
      period: periodStart && periodEnd ? `${periodStart} até ${periodEnd}` : MISSING,
      topGain,
      topDrop,
      topGainH,
      topDropH,
      avgVol20d: avgVol20d != null && avgVol20d > 0 ? avgVol20d : null,
    };
  }, [tableRows, summaryHorizon]);

  const focusRow = useMemo(() => tableRows.find((row) => row.asset === focusAsset) || null, [tableRows, focusAsset]);
  const recommendationRows = useMemo(() => {
    const sorted = [...tableRows].sort((a, b) => {
      const av = isFiniteNumber(a.vol20d) ? a.vol20d : -1;
      const bv = isFiniteNumber(b.vol20d) ? b.vol20d : -1;
      return bv - av;
    });
    return showAllRecommendationCards ? sorted : sorted.slice(0, 12);
  }, [tableRows, showAllRecommendationCards]);

  const sectors =
    initialDomain === "finance"
      ? financeGroupFilter
      : initialDomain === "energy"
      ? [
          { value: "all", label: "Todos os grupos" },
          { value: "energy", label: "Energia" },
        ]
      : [{ value: "all", label: "Todos os grupos imobiliários" }];

  return (
    <div className="p-4 md:p-5 space-y-4 md:space-y-5">
      <div className="space-y-2">
        <div className="text-xs uppercase tracking-[0.2em] text-zinc-400">{title}</div>
        <h1 className="text-2xl font-semibold">Painel financeiro por ativo</h1>
        <p className="text-sm text-zinc-400">Leitura de comportamento de preço, risco e estabilidade por ativo selecionado.</p>
      </div>

      <div className="rounded-2xl border border-zinc-800 bg-zinc-950/60 p-4 md:p-5 space-y-4">
        <DashboardFilters
          assets={universe}
          selected={selected}
          onSelectedChange={setSelected}
          sector={groupFilter}
          onSectorChange={setGroupFilter}
          sectors={sectors}
          timeframe={timeframe}
          onTimeframeChange={setTimeframe}
          rangePreset={rangePreset}
          onRangePresetChange={setRangePreset}
          normalize={normalize}
          onNormalizeChange={setNormalize}
          showRegimeBands={showRegimeBands}
          onShowRegimeBandsChange={setShowRegimeBands}
          regimeBandsLabel="Destaques"
          regimeBandsTitle="Exibir ou ocultar destaques de fundo no gráfico"
          smoothing={smoothing}
          onSmoothingChange={setSmoothing}
        />

        <div className="flex flex-wrap items-center gap-2">
          <button
            onClick={loadExamples}
            className="rounded-md border border-zinc-700 px-2 py-1 text-xs text-zinc-200 hover:border-zinc-500"
            aria-label="Carregar amostra completa base"
          >
            Amostra completa (20)
          </button>
          <button
            onClick={loadExpanded}
            className="rounded-md border border-zinc-700 px-2 py-1 text-xs text-zinc-200 hover:border-zinc-500"
            aria-label="Expandir para amostra maior"
          >
            Expandir 20+
          </button>
          <span className="text-xs text-zinc-500">Amostra ativa: {selected.length} ativos (máximo 40).</span>
        </div>
        {!universeLoaded ? (
          <div className="rounded-lg border border-rose-800/60 bg-rose-950/20 p-3 text-xs text-rose-200">
            Não foi possível carregar o universo de ativos no momento. Verifique os artefatos publicados para o app.
          </div>
        ) : null}

        <div className="rounded-xl border border-zinc-800 bg-black/20 px-3 py-2">
          <div className="text-sm text-zinc-200">Gráfico estrutural dos ativos selecionados</div>
          <div className="text-xs text-zinc-500">
            Eixo X: tempo da janela ativa. Eixo Y: {normalize ? "índice base 100" : "preço na unidade original da série (USD/pts/índice)"}.
          </div>
        </div>

        {loading ? <div className="text-sm text-zinc-500">Carregando séries...</div> : null}
        {!selected.length ? (
          <div className="rounded-lg border border-zinc-800 bg-black/30 p-3 text-xs text-zinc-400">
            Sem ativos selecionados. Clique em <strong>Carregar exemplos</strong> para iniciar.
          </div>
        ) : null}

        <RegimeChart
          data={seriesByAsset}
          selected={selected}
          normalize={normalize}
          showRegimeBands={showRegimeBands}
          smoothing={smoothing}
          rangePreset={rangePreset}
          tooltipMode="price_only"
          chartTitle="Evolução temporal por ativo"
          yUnitLabel={normalize ? "Índice base 100" : "Preço (unidade original da série)"}
        />

        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
          <Card label="Ativos" value={String(selected.length)} helper="Quantidade de ativos selecionados na leitura atual." compact />
          <Card label="|Delta %| médio" value={formatPercent(metrics.avgAbsChange)} helper="Média da variação percentual absoluta diária." compact />
          <Card label="Vol média 20d" value={formatPercent(metrics.avgVol20d)} helper="Média da volatilidade de 20 períodos." compact />
          <Card label="Período" value={summary.period} helper="Janela temporal usada na leitura atual." compact />
        </div>

        <details className="rounded-xl border border-zinc-800 bg-black/20 p-3">
          <summary className="cursor-pointer text-sm text-zinc-200">Saiba mais: métricas adicionais do painel</summary>
            <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-3">
            <Card
              label="Preço médio"
              value={`${formatPrice(metrics.avgPrice)}`}
              helper="Média simples do preço mais recente. A unidade depende de cada ativo (ex.: USD, pontos ou índice)."
            />
            <Card
              label="Ativos com volume"
              value={String(tableRows.filter((r) => isFiniteNumber(r.volume)).length)}
              helper="Quantidade de ativos com volume disponível na fonte atual."
            />
          </div>
        </details>
      </div>

      <div className="rounded-2xl border border-zinc-800 bg-zinc-950/60 p-4 md:p-5 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <div className="text-sm uppercase tracking-widest text-zinc-400">Holders por ativo e recomendação</div>
            <div className="text-xs text-zinc-500">
              Cards calculados sobre os ativos selecionados no filtro atual. Horizonte ativo: h{summaryHorizon}.
            </div>
          </div>
          {tableRows.length > 12 ? (
            <button
              onClick={() => setShowAllRecommendationCards((v) => !v)}
              className="rounded-md border border-zinc-700 px-2 py-1 text-xs text-zinc-200 hover:border-zinc-500"
            >
              {showAllRecommendationCards ? "Mostrar menos" : `Ver todos (${tableRows.length})`}
            </button>
          ) : null}
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
          {recommendationRows.map((row) => {
            const rec = buildAssetRecommendation(row, summaryHorizon);
            return (
              <article key={`rec-${row.asset}`} className="rounded-xl border border-zinc-800 bg-black/20 p-3 space-y-2">
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <div className="text-sm font-semibold text-zinc-100">{row.asset}</div>
                    <div className="text-xs text-zinc-500">{row.group || MISSING}</div>
                  </div>
                  <span className={`rounded-md border px-2 py-1 text-[10px] font-medium ${recommendationTone(rec.level)}`}>{rec.level}</span>
                </div>

                <div className="grid grid-cols-2 gap-2 text-xs">
                  <MetricMini label="Preço hoje" value={formatPrice(row.priceToday)} />
                  <MetricMini label="Preço ontem" value={formatPrice(row.pricePrev)} />
                  <MetricMini label="Delta %" value={formatPercent(row.changePct)} tone={toneFromPct(row.changePct)} />
                  <MetricMini label="Vol 20D" value={formatPercent(row.vol20d)} />
                </div>

                <div className="rounded-lg border border-zinc-800 bg-zinc-950/60 p-2">
                  <div className="text-[11px] uppercase tracking-[0.12em] text-zinc-500">Recomendação operacional</div>
                  <div className="mt-1 text-sm text-zinc-200">{rec.action}</div>
                  <div className="mt-1 text-xs text-zinc-400">{rec.rationale}</div>
                </div>
              </article>
            );
          })}
        </div>
      </div>

      {showTable ? (
        <div className="grid grid-cols-1 lg:grid-cols-[1fr_400px] gap-4 md:gap-5">
          <div className="rounded-2xl border border-zinc-800 bg-zinc-950/60 p-4 md:p-5">
            <div className="flex items-center justify-between gap-2">
              <div className="text-sm uppercase tracking-widest text-zinc-400">Amostra completa por ativo</div>
              <div
                className="text-xs text-zinc-500"
                title="Unidade base do preço: padrão USD. Exceções principais: VIX em pontos e séries FIPEZAP em índice."
              >
                Unidade do preço: USD (VIX=pts, FIPEZAP=índice)
              </div>
            </div>
            <div className="mt-3 overflow-auto max-h-[560px]">
              <table className="w-full text-xs">
                <thead className="text-zinc-500 uppercase sticky top-0 bg-zinc-950">
                  <tr>
                    <th className="text-left py-2" title="Ticker do ativo.">Ativo</th>
                    <th className="text-left py-2" title="Grupo/setor de classificação.">Setor</th>
                    <th className="text-left py-2" title="Intervalo de datas disponível no ativo.">Período</th>
                    <th className="text-left py-2" title="Último preço disponível no período (na unidade da série).">Preço hoje</th>
                    <th className="text-left py-2" title="Preço do ponto imediatamente anterior (na unidade da série).">Preço ontem</th>
                    <th className="text-left py-2" title="Diferença absoluta entre preço hoje e ontem (na unidade da série).">Delta abs</th>
                    <th className="text-left py-2" title="Variação percentual diária.">Delta %</th>
                    <th className="text-left py-2" title="Retorno acumulado dos últimos 5 pontos.">Ret 5D</th>
                    <th className="text-left py-2" title="Volatilidade dos retornos em 20 pontos.">Vol 20D</th>
                    <th className="text-left py-2" title="Volume do último ponto. Se ausente na fonte, aparece n/d.">Volume</th>
                  </tr>
                </thead>
                <tbody>
                  {tableRows.map((row) => {
                    return (
                      <tr key={row.asset} className="border-t border-zinc-800/70 text-zinc-300">
                        <td className="py-2 font-medium">{row.asset}</td>
                        <td className="py-2 text-zinc-400">{row.group || MISSING}</td>
                        <td className="py-2 text-zinc-400">{row.period}</td>
                        <td className="py-2">{formatPrice(row.priceToday)}</td>
                        <td className="py-2">{formatPrice(row.pricePrev)}</td>
                        <td className="py-2">{formatNumber(row.changeAbs)}</td>
                        <td className={`py-2 ${toneFromPct(row.changePct)}`}>{formatPercent(row.changePct)}</td>
                        <td className="py-2">{formatPercent(row.ret5d)}</td>
                        <td className="py-2">{formatPercent(row.vol20d)}</td>
                        <td className="py-2">{isFiniteNumber(row.volume) ? row.volume.toLocaleString("en-US") : "n/d"}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          <div className="rounded-2xl border border-zinc-800 bg-zinc-950/60 p-4 md:p-5 space-y-4">
            <div className="text-sm uppercase tracking-widest text-zinc-400">Resumo por ativo</div>

            <div>
              <div className="mb-1 flex items-center gap-1 text-[11px] uppercase tracking-[0.12em] text-zinc-500">
                <span>Ativo selecionado</span>
                <Help text="Mostra o resumo detalhado de um ativo por vez. Use este seletor para trocar o foco." />
              </div>
              <select
                value={focusAsset}
                onChange={(e) => setFocusAsset(e.target.value)}
                className="w-full rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm"
                aria-label="Selecionar ativo para resumo"
              >
                {selected.map((asset) => (
                  <option key={asset} value={asset}>
                    {asset}
                  </option>
                ))}
              </select>
            </div>

            <div className="rounded-xl border border-zinc-800 bg-black/20 p-3 text-sm text-zinc-200 leading-relaxed">
              {buildAssetNarrative(focusRow, summaryHorizon)}
            </div>
            <div className="rounded-xl border border-zinc-800 bg-zinc-950/70 p-3">
              <div className="text-[11px] uppercase tracking-[0.12em] text-zinc-500">Dicas rápidas de uso</div>
              <div className="mt-2 space-y-1.5 text-xs text-zinc-300">
                {buildAssetTips(focusRow, summaryHorizon).map((tip, idx) => (
                  <div key={`tip-${idx}`}>{idx + 1}. {tip}</div>
                ))}
              </div>
            </div>

            <div className="flex items-center gap-2 text-xs">
              <span className="text-zinc-400">Horizonte</span>
              <Help text="h1 = retorno de 1 barra, h5 = 5 barras, h10 = 10 barras (na frequência atual: diário ou semanal)." />
              {[1, 5, 10].map((h) => (
                <button
                  key={h}
                  className={`rounded-md border px-2 py-1 ${
                    summaryHorizon === h ? "border-cyan-400 text-cyan-300" : "border-zinc-700 text-zinc-300"
                  }`}
                  onClick={() => setSummaryHorizon(h as 1 | 5 | 10)}
                  title={`Retorno visual em ${h} barra(s)`}
                >
                  h{h}
                </button>
              ))}
            </div>

            <div className="space-y-1 text-xs text-zinc-300">
              <div>
                Nome do ativo: <strong>{focusRow?.asset || MISSING}</strong>
              </div>
              <div>
                Período exibido: <strong>{summary.period}</strong>
              </div>
            </div>

            <details className="rounded-xl border border-zinc-800 bg-black/20 p-3">
              <summary className="cursor-pointer text-sm text-zinc-200">Ver mais métricas associadas</summary>
              <div className="mt-3 space-y-1 text-xs text-zinc-300">
                <div title="Maior variação percentual positiva no dia entre os ativos selecionados.">
                  Maior alta do dia: {summary.topGain ? `${summary.topGain.asset} (${formatPercent(summary.topGain.changePct)})` : MISSING}
                </div>
                <div title="Maior variação percentual negativa no dia entre os ativos selecionados.">
                  Maior queda do dia: {summary.topDrop ? `${summary.topDrop.asset} (${formatPercent(summary.topDrop.changePct)})` : MISSING}
                </div>
                <div title="Maior variação positiva no horizonte H selecionado.">
                  Maior alta H{summaryHorizon}:{" "}
                  {summary.topGainH
                    ? `${summary.topGainH.asset} (${formatPercent(
                        summary.topGainH[
                          summaryHorizon === 1 ? "retH1" : summaryHorizon === 5 ? "retH5" : "retH10"
                        ] as number
                      )})`
                    : MISSING}
                </div>
                <div title="Maior variação negativa no horizonte H selecionado.">
                  Maior queda H{summaryHorizon}:{" "}
                  {summary.topDropH
                    ? `${summary.topDropH.asset} (${formatPercent(
                        summary.topDropH[
                          summaryHorizon === 1 ? "retH1" : summaryHorizon === 5 ? "retH5" : "retH10"
                        ] as number
                      )})`
                    : MISSING}
                </div>
                <div title="Média da volatilidade de 20 dias dos ativos atualmente selecionados.">
                  Volatilidade média (20d): {summary.avgVol20d != null ? formatPercent(summary.avgVol20d) : MISSING}
                </div>
              </div>
            </details>
          </div>
        </div>
      ) : null}
    </div>
  );
}

function Help({ text }: { text: string }) {
  return (
    <span
      className="inline-flex h-4 w-4 items-center justify-center rounded-full border border-zinc-700 text-[10px] text-zinc-400"
      title={text}
      aria-label={text}
    >
      ?
    </span>
  );
}

function Card({
  label,
  value,
  helper,
  compact = false,
}: {
  label: string;
  value: string;
  helper: string;
  compact?: boolean;
}) {
  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-950/70 p-3 md:p-4" title={helper}>
      <div className="flex items-center gap-1 text-[11px] uppercase tracking-[0.14em] text-zinc-500">
        <span>{label}</span>
        <Help text={helper} />
      </div>
      <div className={`mt-1 font-semibold text-zinc-100 ${compact ? "text-lg" : "text-xl"}`}>{value}</div>
      {!compact ? <div className="mt-1 text-[11px] text-zinc-500">{helper}</div> : null}
    </div>
  );
}

function MetricMini({ label, value, tone = "text-zinc-200" }: { label: string; value: string; tone?: string }) {
  return (
    <div className="rounded-md border border-zinc-800 bg-zinc-950/60 px-2 py-1.5">
      <div className="text-[10px] uppercase tracking-[0.12em] text-zinc-500">{label}</div>
      <div className={`text-sm font-medium ${tone}`}>{value}</div>
    </div>
  );
}
