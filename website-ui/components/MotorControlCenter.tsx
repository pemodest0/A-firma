"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";

type RegimeHistoryPoint = {
  date: string;
  regime: string;
  exposure?: number | null;
  transition_score?: number | null;
};

type AssetPricePoint = {
  price: number | null;
};

type LabResponse = {
  run?: { id?: string };
  summary?: {
    deployment_gate?: {
      blocked?: boolean;
      reasons?: string[];
      checks?: Record<string, number | boolean | string | null | undefined>;
      thresholds?: Record<string, number | boolean | string | null | undefined>;
    };
  };
  metrics?: {
    latest_state?: { date?: string; N_used?: number; p1?: number; deff?: number };
    delta_20d?: { p1?: number; deff?: number };
    n_used_stats?: { min?: number; mean?: number; max?: number };
  };
  playbook?: { latest?: { regime?: string; action_code?: string; signal_tier?: string; tradeoff_label?: string } };
  view_model?: {
    latest_state?: { date?: string; N_used?: number; p1?: number; deff?: number };
    latest_regime?: { regime?: string; action_code?: string; interpretation?: string; exposure?: number };
    playbook_latest?: { regime?: string; action_code?: string; signal_tier?: string; tradeoff_label?: string };
  };
  alerts?: {
    operational?: {
      latest_events?: string[];
      latest_event_rows?: Array<{ message?: string }>;
      n_events_last_60d?: number;
      n_events_total?: number;
      event_counts?: Record<string, number>;
    };
  };
  regime_history?: RegimeHistoryPoint[];
  alert_levels?: Array<{
    date?: string;
    alert_level?: string;
    alert_level_raw?: string;
    risk_score?: number | null;
    signal_confidence?: number | null;
    transition_score?: number | null;
  }>;
  significance?: Array<{
    window?: number | null;
    metric?: string;
    significant_share_p_lt_0_05?: number | null;
    latest_pvalue?: number | null;
  }>;
  qa_checks?: {
    ok?: boolean;
    failed_checks?: string[];
    checks?: Array<Record<string, unknown>>;
  };
  asset_sector_summary?: {
    n_assets?: number;
    n_sectors?: number;
    top_risk_asset?: string;
    top_risk_sector?: string;
    regime_mix_assets?: Record<string, number>;
  };
  asset_diagnostics?: {
    count?: number;
    items?: Array<{
      ticker: string;
      sector: string;
      risk_score?: number | null;
      confidence_score?: number | null;
      regime_asset?: string;
      switches_30d?: number | null;
      switches_90d?: number | null;
    }>;
  };
  sector_diagnostics?: {
    items?: Array<{
      sector: string;
      n_assets?: number | null;
      risk_mean?: number | null;
      confidence_mean?: number | null;
      pct_instavel?: number | null;
      alerta_setor?: string;
      plano_acao?: string;
    }>;
  };
};

function num(v: number | null | undefined, d = 3) {
  if (v == null || !Number.isFinite(v)) return "n/d";
  return v.toFixed(d);
}

function pct(v: number | null | undefined, d = 1) {
  if (v == null || !Number.isFinite(v)) return "n/d";
  return `${(v * 100).toFixed(d)}%`;
}

function fmtPrice(v: number | null | undefined) {
  if (v == null || !Number.isFinite(v)) return "n/d";
  return v.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function levelBadge(level: string) {
  const k = String(level || "").toLowerCase();
  if (k.includes("vermelho")) return "text-rose-200 border-rose-700/50 bg-rose-950/30";
  if (k.includes("amarelo")) return "text-amber-200 border-amber-700/50 bg-amber-950/30";
  return "text-emerald-200 border-emerald-700/50 bg-emerald-950/30";
}

function alertLevelBadge(level: string) {
  const k = String(level || "").toLowerCase();
  if (k.includes("red") || k.includes("vermelh")) return "text-rose-200 border-rose-700/50 bg-rose-950/30";
  if (k.includes("yellow") || k.includes("amarel")) return "text-amber-200 border-amber-700/50 bg-amber-950/30";
  return "text-emerald-200 border-emerald-700/50 bg-emerald-950/30";
}

function significanceTone(share: number | null | undefined, pvalue: number | null | undefined) {
  const s = safeNum(share);
  const p = safeNum(pvalue);
  if (s != null && p != null && s >= 0.8 && p <= 0.05) return "forte";
  if ((s != null && s >= 0.5) || (p != null && p <= 0.10)) return "moderada";
  return "fraca";
}

function explainRegime(regime: string) {
  const key = String(regime || "").toLowerCase();
  if (key.includes("stress") || key.includes("instavel")) return "Instável: risco alto e correlação mais concentrada.";
  if (key.includes("transition") || key.includes("trans")) return "Transição: estrutura mudando, pede cautela.";
  if (key.includes("dispersion")) return "Dispersão: comportamento mais heterogêneo entre ativos.";
  if (key.includes("stable") || key.includes("estavel")) return "Estável: estrutura mais previsível no curto prazo.";
  return "Sem classificação de regime disponível.";
}

function shortRegime(regime: string) {
  const key = String(regime || "").toLowerCase();
  if (key.includes("stress") || key.includes("instavel")) return "Estresse";
  if (key.includes("transition") || key.includes("trans")) return "Transição";
  if (key.includes("dispersion")) return "Dispersão";
  if (key.includes("stable") || key.includes("estavel")) return "Estável";
  return "Sem regime publicado";
}

function dotTone(regime: string) {
  const key = String(regime || "").toLowerCase();
  if (key.includes("stress") || key.includes("instavel")) return "bg-rose-400 ring-rose-500/40";
  if (key.includes("transition") || key.includes("trans")) return "bg-amber-400 ring-amber-500/40";
  if (key.includes("dispersion")) return "bg-violet-400 ring-violet-500/40";
  if (key.includes("stable") || key.includes("estavel")) return "bg-emerald-400 ring-emerald-500/40";
  return "bg-zinc-500 ring-zinc-500/40";
}

function dotHex(regime: string) {
  const key = String(regime || "").toLowerCase();
  if (key.includes("stress") || key.includes("instavel")) return "#fb7185";
  if (key.includes("transition") || key.includes("trans")) return "#fbbf24";
  if (key.includes("dispersion")) return "#a78bfa";
  if (key.includes("stable") || key.includes("estavel")) return "#34d399";
  return "#71717a";
}

function normalizeText(value: string) {
  return String(value || "")
    .trim()
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "");
}

function cleanSectorName(value: string) {
  const raw = String(value || "").trim();
  if (!raw) return "Sem setor";
  const norm = normalizeText(raw);
  if (norm === "unknown" || norm === "n/a" || norm === "na" || norm === "--") return "Sem setor";
  return raw.replace(/_/g, " ");
}

function readableInterpretation(actionCode: string, regime: string) {
  const key = normalizeText(actionCode);
  if (!key) return explainRegime(regime);
  if (key.includes("reduce") || key.includes("de-risk") || key.includes("derisk")) {
    return "Estrutura frágil: priorizar postura defensiva e reduzir risco operacional.";
  }
  if (key.includes("wait") || key.includes("hold")) {
    return "Estrutura em ajuste: manter cautela e aguardar confirmação de direção.";
  }
  if (key.includes("dispersion")) {
    return "Dispersão ativa: comparar ativos/setores com mais seletividade.";
  }
  if (key.includes("stable") || key.includes("normal")) {
    return "Estrutura estável: risco controlado no conjunto agregado.";
  }
  return explainRegime(regime);
}

function bandByScore(score: number) {
  if (score >= 0.7) return "Crítico";
  if (score >= 0.45) return "Atenção";
  return "Controlado";
}

function safeNum(v: number | null | undefined) {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function clamp01(v: number | null | undefined) {
  const n = safeNum(v);
  if (n == null) return null;
  return Math.max(0, Math.min(1, n));
}

function textOrNd(value: unknown) {
  if (value == null) return "n/d";
  const text = String(value).trim();
  return text.length ? text : "n/d";
}

export default function MotorControlCenter() {
  const [windowSize, setWindowSize] = useState(120);
  const [periodDays, setPeriodDays] = useState(180);
  const [assetSearch, setAssetSearch] = useState("");
  const [sectorSearch, setSectorSearch] = useState("");
  const [assetPage, setAssetPage] = useState(1);
  const [assetPageSize, setAssetPageSize] = useState(100);
  const [data, setData] = useState<LabResponse | null>(null);
  const [hoveredDot, setHoveredDot] = useState<RegimeHistoryPoint | null>(null);
  const [assetPrices, setAssetPrices] = useState<Record<string, { today: number | null; prev: number | null }>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [emptyNotice, setEmptyNotice] = useState<string | null>(null);

  useEffect(() => {
    const ac = new AbortController();
    const run = async () => {
      setLoading(true);
      setEmptyNotice(null);
      try {
        const params = new URLSearchParams({
          window: String(windowSize),
          include_rows: "1",
          period_days: String(periodDays),
          asset: assetSearch,
          sector: sectorSearch,
        });
        const res = await fetch(`/api/lab/corr/latest?${params.toString()}`, { cache: "no-store", signal: ac.signal });
        if (!res.ok) {
          const body = (await res.json().catch(() => ({}))) as { error?: string; message?: string };
          if (res.status === 503 && body.error === "no_valid_lab_run") {
            setData(null);
            setError(null);
            setEmptyNotice("Painel do motor aguardando a primeira publicação de run válido.");
            return;
          }
          throw new Error(
            `Erro ao consultar painel (${res.status}). ${body.message || body.error || "Snapshot do motor indisponível."}`
          );
        }
        const payload = (await res.json()) as LabResponse;
        setData(payload);
        setError(null);
      } catch (err) {
        if (ac.signal.aborted) return;
        setError(err instanceof Error ? err.message : "erro");
      } finally {
        if (!ac.signal.aborted) setLoading(false);
      }
    };
    void run();
    return () => ac.abort();
  }, [windowSize, periodDays, assetSearch, sectorSearch]);

  const latest = data?.metrics?.latest_state || null;
  const gateBlocked = Boolean(data?.summary?.deployment_gate?.blocked);
  const gateChecks = (data?.summary?.deployment_gate?.checks || {}) as Record<string, number | boolean | string | null | undefined>;
  const gateThresholds = (data?.summary?.deployment_gate?.thresholds || {}) as Record<string, number | boolean | string | null | undefined>;
  const gateReasons = Array.isArray(data?.summary?.deployment_gate?.reasons) ? data?.summary?.deployment_gate?.reasons : [];
  const play = data?.playbook?.latest || {};
  const qaOk = data?.qa_checks?.ok === true;
  const qaFailed = Array.isArray(data?.qa_checks?.failed_checks) ? data?.qa_checks?.failed_checks : [];
  const nStats = data?.metrics?.n_used_stats || {};
  const delta20 = data?.metrics?.delta_20d || {};
  const op = data?.alerts?.operational || {};
  const latestAlert = useMemo(() => {
    const arr = Array.isArray(data?.alert_levels) ? data?.alert_levels : [];
    return arr.length ? arr[arr.length - 1] : null;
  }, [data?.alert_levels]);
  const sigRows = useMemo(() => {
    const rows = Array.isArray(data?.significance) ? data.significance : [];
    const p1Rows = rows.filter((x) => String(x.metric || "").includes("delta_p1"));
    const deffRows = rows.filter((x) => String(x.metric || "").includes("delta_deff"));
    const byWindow = [60, 120, 252].map((window) => {
      const p1 = p1Rows.find((x) => Number(x.window) === window) || null;
      const deff = deffRows.find((x) => Number(x.window) === window) || null;
      return { window, p1, deff };
    });
    return byWindow;
  }, [data?.significance]);
  const regimeHistory = useMemo(
    () => (Array.isArray(data?.regime_history) ? data.regime_history : []),
    [data?.regime_history]
  );
  const historyChanges = useMemo(() => {
    const out: Array<{ date: string; from: string; to: string }> = [];
    for (let i = 1; i < regimeHistory.length; i += 1) {
      const prev = String(regimeHistory[i - 1]?.regime || "");
      const now = String(regimeHistory[i]?.regime || "");
      if (prev && now && prev !== now) {
        out.push({ date: String(regimeHistory[i].date), from: prev, to: now });
      }
    }
    return out.slice(-12).reverse();
  }, [regimeHistory]);

  const sectorRows = useMemo(
    () => (Array.isArray(data?.sector_diagnostics?.items) ? data.sector_diagnostics?.items : []),
    [data?.sector_diagnostics?.items]
  );
  const sectorMathRows = useMemo(() => {
    return sectorRows
      .map((row) => {
        const risk = clamp01(row.risk_mean ?? null);
        const instavel = clamp01(row.pct_instavel ?? null);
        const conf = clamp01(row.confidence_mean ?? null);
        if (risk == null || instavel == null || conf == null) return null;
        const score = 0.5 * risk + 0.35 * instavel + 0.15 * (1 - conf);
        return {
          sector: String(row.sector || "sem setor"),
          alerta: String(row.alerta_setor || ""),
          score,
          risk,
          instavel,
          confidence: conf,
          nAssets: safeNum(row.n_assets ?? null),
        };
      })
      .filter(
        (
          row
        ): row is {
          sector: string;
          alerta: string;
          score: number;
          risk: number;
          instavel: number;
          confidence: number;
          nAssets: number | null;
        } => row !== null
      )
      .sort((a, b) => b.score - a.score)
      .slice(0, 10);
  }, [sectorRows]);
  const incompleteSectorRows = Math.max(0, sectorRows.length - sectorMathRows.length);
  const maxSectorScore = useMemo(
    () => sectorMathRows.reduce((acc, row) => Math.max(acc, row.score), 0.01),
    [sectorMathRows]
  );
  const universeRisk = useMemo(() => {
    if (!sectorMathRows.length) return null;
    const withWeight = sectorMathRows.filter((x) => x.nAssets != null && x.nAssets > 0);
    if (withWeight.length) {
      const weightedSum = withWeight.reduce((acc, row) => acc + row.score * (row.nAssets as number), 0);
      const weightSum = withWeight.reduce((acc, row) => acc + (row.nAssets as number), 0);
      if (weightSum > 0) return weightedSum / weightSum;
    }
    return sectorMathRows.reduce((acc, row) => acc + row.score, 0) / sectorMathRows.length;
  }, [sectorMathRows]);
  const universeConfidence = useMemo(() => {
    if (!sectorMathRows.length) return null;
    const withWeight = sectorMathRows.filter((x) => x.nAssets != null && x.nAssets > 0);
    if (withWeight.length) {
      const weightedSum = withWeight.reduce((acc, row) => acc + row.confidence * (row.nAssets as number), 0);
      const weightSum = withWeight.reduce((acc, row) => acc + (row.nAssets as number), 0);
      if (weightSum > 0) return weightedSum / weightSum;
    }
    return sectorMathRows.reduce((acc, row) => acc + row.confidence, 0) / sectorMathRows.length;
  }, [sectorMathRows]);
  const assetRows = useMemo(
    () => (Array.isArray(data?.asset_diagnostics?.items) ? data.asset_diagnostics?.items : []),
    [data?.asset_diagnostics?.items]
  );
  const filteredAssetsAll = useMemo(() => {
    const a = assetSearch.trim().toUpperCase();
    const s = normalizeText(sectorSearch);
    return assetRows
      .filter((x) => (!a ? true : String(x.ticker || "").toUpperCase().includes(a)))
      .filter((x) => (!s ? true : normalizeText(cleanSectorName(String(x.sector || ""))).includes(s)))
      .sort((x, y) => {
        const xr = safeNum(x.risk_score ?? null);
        const yr = safeNum(y.risk_score ?? null);
        if (xr == null && yr == null) return String(x.ticker || "").localeCompare(String(y.ticker || ""));
        if (xr == null) return 1;
        if (yr == null) return -1;
        return yr - xr;
      });
  }, [assetRows, assetSearch, sectorSearch]);
  const topAssetsForPrice = useMemo(() => filteredAssetsAll.slice(0, 40), [filteredAssetsAll]);

  const sectorAlertMap = useMemo(() => {
    const map = new Map<string, string>();
    for (const row of sectorRows) {
      const key = normalizeText(cleanSectorName(String(row.sector || "")));
      if (!key) continue;
      const alerta = String(row.alerta_setor || "").trim();
      if (alerta) map.set(key, alerta);
    }
    return map;
  }, [sectorRows]);

  const sectorBreakdownRows = useMemo(() => {
    const grouped = new Map<
      string,
      {
        sector: string;
        count: number;
        riskSum: number;
        riskCount: number;
        confSum: number;
        confCount: number;
        stable: number;
        trans: number;
        stress: number;
        dispersion: number;
      }
    >();
    for (const row of filteredAssetsAll) {
      const sector = cleanSectorName(String(row.sector || "Sem setor"));
      const key = normalizeText(sector);
      if (!grouped.has(key)) {
        grouped.set(key, {
          sector,
          count: 0,
          riskSum: 0,
          riskCount: 0,
          confSum: 0,
          confCount: 0,
          stable: 0,
          trans: 0,
          stress: 0,
          dispersion: 0,
        });
      }
      const bucket = grouped.get(key)!;
      bucket.count += 1;
      const risk = safeNum(row.risk_score ?? null);
      const conf = safeNum(row.confidence_score ?? null);
      if (risk != null) {
        bucket.riskSum += Math.max(0, Math.min(1, risk));
        bucket.riskCount += 1;
      }
      if (conf != null) {
        bucket.confSum += Math.max(0, Math.min(1, conf));
        bucket.confCount += 1;
      }
      const regime = normalizeText(String(row.regime_asset || ""));
      if (regime.includes("stress") || regime.includes("instavel")) bucket.stress += 1;
      else if (regime.includes("transition") || regime.includes("trans")) bucket.trans += 1;
      else if (regime.includes("dispersion")) bucket.dispersion += 1;
      else bucket.stable += 1;
    }
    return Array.from(grouped.values())
      .map((row) => {
        const riskMean = row.riskCount ? row.riskSum / row.riskCount : null;
        const confMean = row.confCount ? row.confSum / row.confCount : null;
        const pctStress = row.count ? row.stress / row.count : 0;
        const pctTrans = row.count ? row.trans / row.count : 0;
        const score = riskMean != null && confMean != null ? 0.55 * riskMean + 0.3 * pctStress + 0.15 * (1 - confMean) : null;
        const sectorDataAlert = sectorAlertMap.get(normalizeText(row.sector)) || "";
        const alerta =
          sectorDataAlert ||
          (score == null ? "sem classificação" : score >= 0.7 ? "vermelho" : score >= 0.45 ? "amarelo" : "verde");
        return {
          ...row,
          riskMean,
          confMean,
          pctStress,
          pctTrans,
          score,
          alerta,
        };
      })
      .sort((a, b) => {
        if (a.score == null && b.score == null) return a.sector.localeCompare(b.sector);
        if (a.score == null) return 1;
        if (b.score == null) return -1;
        return b.score - a.score;
      });
  }, [filteredAssetsAll, sectorAlertMap]);

  const universeByAsset = useMemo(() => {
    if (!filteredAssetsAll.length) return null;
    let riskSum = 0;
    let riskCount = 0;
    let confSum = 0;
    let confCount = 0;
    let stress = 0;
    let trans = 0;
    let stable = 0;
    let dispersion = 0;
    for (const row of filteredAssetsAll) {
      const risk = safeNum(row.risk_score ?? null);
      const conf = safeNum(row.confidence_score ?? null);
      if (risk != null) {
        riskSum += Math.max(0, Math.min(1, risk));
        riskCount += 1;
      }
      if (conf != null) {
        confSum += Math.max(0, Math.min(1, conf));
        confCount += 1;
      }
      const regime = normalizeText(String(row.regime_asset || ""));
      if (regime.includes("stress") || regime.includes("instavel")) stress += 1;
      else if (regime.includes("transition") || regime.includes("trans")) trans += 1;
      else if (regime.includes("dispersion")) dispersion += 1;
      else stable += 1;
    }
    const total = filteredAssetsAll.length;
    return {
      total,
      sectors: sectorBreakdownRows.length,
      riskMean: riskCount ? riskSum / riskCount : null,
      confMean: confCount ? confSum / confCount : null,
      stressPct: total ? stress / total : 0,
      transPct: total ? trans / total : 0,
      stablePct: total ? stable / total : 0,
      dispersionPct: total ? dispersion / total : 0,
    };
  }, [filteredAssetsAll, sectorBreakdownRows.length]);

  const scatterPoints = useMemo(() => {
    const points: Array<{ ticker: string; sector: string; regime: string; risk: number; confidence: number }> = [];
    for (const row of filteredAssetsAll) {
      const risk = safeNum(row.risk_score ?? null);
      const confidence = safeNum(row.confidence_score ?? null);
      if (risk == null || confidence == null) continue;
      const ticker = String(row.ticker || "").trim();
      if (!ticker) continue;
      points.push({
        ticker,
        sector: cleanSectorName(String(row.sector || "Sem setor")),
        regime: String(row.regime_asset || ""),
        risk: Math.max(0, Math.min(1, risk)),
        confidence: Math.max(0, Math.min(1, confidence)),
      });
    }
    return points;
  }, [filteredAssetsAll]);

  const totalPages = Math.max(1, Math.ceil(filteredAssetsAll.length / assetPageSize));
  const currentPage = Math.min(assetPage, totalPages);
  const pagedAssets = useMemo(() => {
    const start = (currentPage - 1) * assetPageSize;
    return filteredAssetsAll.slice(start, start + assetPageSize);
  }, [filteredAssetsAll, currentPage, assetPageSize]);

  useEffect(() => {
    const tickers = topAssetsForPrice.map((x) => String(x.ticker || "").trim()).filter(Boolean).slice(0, 40);
    if (!tickers.length) {
      setAssetPrices({});
      return;
    }
    const ac = new AbortController();
    const run = async () => {
      try {
        const res = await fetch(`/api/graph/series-batch?assets=${encodeURIComponent(tickers.join(","))}&tf=daily&limit=2`, {
          cache: "no-store",
          signal: ac.signal,
        });
        if (!res.ok) {
          setAssetPrices({});
          return;
        }
        const payload = (await res.json()) as Record<string, AssetPricePoint[]>;
        const snap: Record<string, { today: number | null; prev: number | null }> = {};
        for (const t of tickers) {
          const arr = Array.isArray(payload?.[t]) ? payload[t] : [];
          const last = arr.length ? arr[arr.length - 1] : null;
          const prev = arr.length >= 2 ? arr[arr.length - 2] : null;
          snap[t] = {
            today: typeof last?.price === "number" && Number.isFinite(last.price) ? last.price : null,
            prev: typeof prev?.price === "number" && Number.isFinite(prev.price) ? prev.price : null,
          };
        }
        setAssetPrices(snap);
      } catch {
        if (!ac.signal.aborted) setAssetPrices({});
      }
    };
    void run();
    return () => ac.abort();
  }, [topAssetsForPrice]);

  const regimeDots = useMemo(() => {
    const target = Math.max(120, Math.min(540, Number.isFinite(periodDays) ? periodDays * 2 : 240));
    const sliced = regimeHistory.slice(-target);
    if (sliced.length > 1) return sliced;
    if (regimeHistory.length > 1) return regimeHistory.slice(-Math.min(540, regimeHistory.length));
    return sliced;
  }, [regimeHistory, periodDays]);
  const activeDot = hoveredDot || (regimeDots.length ? regimeDots[regimeDots.length - 1] : null);
  const displayRegimeRaw =
    String(play.regime || "").trim() || String(data?.view_model?.latest_regime?.regime || "").trim();
  const displayRegime = displayRegimeRaw || "Sem regime publicado";
  const displayInterpretation = readableInterpretation(String(play.action_code || ""), displayRegime);
  const latestDate = textOrNd(latest?.date);
  const latestNUsed = latest?.N_used != null ? num(latest.N_used, 0) : "n/d";
  const latestP1 = latest?.p1 ?? null;
  const latestDeff = latest?.deff ?? null;
  const latestAlertLevel = String(latestAlert?.alert_level || "").trim();
  const hasLatestAlert = latestAlertLevel.length > 0;
  const topRiskAsset = String(data?.asset_sector_summary?.top_risk_asset || "").trim();
  const topRiskSectorRaw = String(data?.asset_sector_summary?.top_risk_sector || "").trim();
  const topRiskSector = topRiskSectorRaw ? cleanSectorName(topRiskSectorRaw) : "";
  const topRiskSummary = topRiskAsset ? `${topRiskAsset}${topRiskSector ? ` | ${topRiskSector}` : ""}` : "n/d";
  const universeAssetsCount = safeNum(data?.asset_sector_summary?.n_assets ?? latest?.N_used ?? null);
  const sectorsCount = safeNum(data?.asset_sector_summary?.n_sectors ?? null);

  useEffect(() => {
    setHoveredDot(null);
  }, [windowSize, periodDays, assetSearch, sectorSearch]);

  useEffect(() => {
    setAssetPage(1);
  }, [assetSearch, sectorSearch, assetPageSize]);

  useEffect(() => {
    if (assetPage > totalPages) setAssetPage(totalPages);
  }, [assetPage, totalPages]);

  if (loading) return <div className="p-6 text-sm text-zinc-400">Carregando painel do motor...</div>;
  if (emptyNotice) {
    return (
      <div className="p-6 space-y-2">
        <div className="text-sm text-zinc-200">{emptyNotice}</div>
        <div className="text-xs text-zinc-400">
          Próximo passo: execute a rotina diária e publique artefatos em `public/data/lab_corr_macro/latest`.
        </div>
      </div>
    );
  }
  if (error || !data) {
    return (
      <div className="p-6 space-y-2">
        <div className="text-sm text-rose-300">Erro no painel do motor: {error || "não foi possível consultar os dados agora."}</div>
        <div className="text-xs text-zinc-400">
          Ação sugerida: valide `public/data/lab_corr_macro/latest` e rode a rotina diária para atualizar os artefatos.
        </div>
      </div>
    );
  }

  return (
    <div className="p-4 md:p-6 space-y-5">
      <section className="rounded-2xl border border-zinc-800 bg-zinc-950/60 p-4 md:p-5">
        <div className="text-xs uppercase tracking-[0.16em] text-zinc-500">Motor</div>
        <h1 className="mt-2 text-2xl font-semibold text-zinc-100">Estado atual do motor</h1>
        <div className="mt-2 text-sm text-zinc-300">
          {gateBlocked
            ? "Publicação travada por regra mínima de segurança."
            : "Leitura ativa. Ferramenta de suporte quantitativo para contexto de risco."}
        </div>
        <div className="mt-2 text-xs text-zinc-400">{explainRegime(displayRegime)}</div>
        <div className="mt-3 grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
          <K label="Gate" value={gateBlocked ? "bloqueado" : "ok"} hint="Se bloqueado, os critérios mínimos de publicação não foram atendidos." />
          <K label="Regime" value={displayRegime} hint="Estado estrutural agregado do mercado na janela selecionada." />
          <K label="Leitura" value={displayInterpretation} hint="Interpretação estatística em linguagem humana. Não é recomendação de compra ou venda." />
          <K
            label="Confiança média"
            value={num(universeConfidence, 3)}
            hint="Confiança agregada do diagnóstico, ponderada pelo número de ativos por setor."
          />
        </div>
        <details className="mt-3 rounded-lg border border-zinc-800 bg-black/20 p-2 text-xs text-zinc-400">
          <summary className="cursor-pointer text-zinc-300">Detalhes técnicos</summary>
          <div className="mt-2">
            data={latestDate} | universo analisado={latestNUsed} | p1={num(latestP1, 4)} | deff={num(latestDeff, 2)} | id_execucao={textOrNd(data?.run?.id)}
          </div>
        </details>
        <div className="mt-3 flex flex-wrap items-center gap-2 text-[11px] text-zinc-400">
          <span className="inline-flex items-center gap-1 rounded-md border border-zinc-700 bg-black/30 px-2 py-1" title="Estado com menor fragilidade estrutural no momento.">
            <span className="h-2.5 w-2.5 rounded-full bg-emerald-400" />
            Estável
          </span>
          <span className="inline-flex items-center gap-1 rounded-md border border-zinc-700 bg-black/30 px-2 py-1" title="Mudança de estrutura em andamento. Maior risco de erro operacional.">
            <span className="h-2.5 w-2.5 rounded-full bg-amber-400" />
            Transição
          </span>
          <span className="inline-flex items-center gap-1 rounded-md border border-zinc-700 bg-black/30 px-2 py-1" title="Fragilidade alta e sincronização elevada.">
            <span className="h-2.5 w-2.5 rounded-full bg-rose-400" />
            Estresse
          </span>
          <span className="inline-flex items-center gap-1 rounded-md border border-zinc-700 bg-black/30 px-2 py-1" title="Comportamento mais heterogêneo entre ativos.">
            <span className="h-2.5 w-2.5 rounded-full bg-violet-400" />
            Dispersão
          </span>
        </div>
        {gateReasons.length ? <div className="mt-2 text-xs text-rose-300">motivo gate: {gateReasons.join(" | ")}</div> : null}
      </section>

      <section className="rounded-2xl border border-zinc-800 bg-zinc-950/60 p-4 md:p-5">
        <div className="text-xs uppercase tracking-[0.16em] text-zinc-500">Saúde e confiança</div>
        <h2 className="mt-1 text-lg font-semibold text-zinc-100">Resumo simples do que sustenta o sinal</h2>
        <div className="mt-3 grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-3 text-sm">
          <K
            label="Qualidade dos dados"
            value={qaOk ? "ok" : "atenção"}
            hint="Se esta parte falha, o motor perde confiabilidade e a publicação deve ser bloqueada."
          />
          <K
            label="Falhas de QA"
            value={String(qaFailed.length)}
            hint="Quantidade de checagens técnicas que falharam no run atual."
          />
          <K
            label="Universo analisado"
            value={universeAssetsCount == null ? "n/d" : `${num(universeAssetsCount, 0)} ativos`}
            hint="Total de ativos que entraram na leitura estrutural."
          />
          <K
            label="Setores analisados"
            value={sectorsCount == null ? "n/d" : num(sectorsCount, 0)}
            hint="Total de setores representados no universo atual."
          />
        </div>
        <div className="mt-3 grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-3 text-sm">
          <K
            label="Cobertura do universo"
            value={`min ${num(nStats.min ?? null, 0)} | médio ${num(nStats.mean ?? null, 1)} | max ${num(nStats.max ?? null, 0)}`}
            hint="Quanto do universo está de fato entrando nas janelas do motor."
          />
          <K
            label="Mudança em 20 dias"
            value={`p1 ${num(delta20.p1 ?? null, 4)} | deff ${num(delta20.deff ?? null, 2)}`}
            hint="Mostra se a estrutura agregada acelerou ou desacelerou no curto prazo."
          />
          <K
            label="Consistência 60d"
            value={num(gateChecks.joint_majority_60d as number, 3)}
            hint="Quanto as janelas concordam entre si na direção do sinal."
          />
          <K
            label="Alertas de cluster"
            value={textOrNd(gateChecks.active_cluster_alerts)}
            hint="Contagem de alertas de estrutura de cluster ativos no run."
          />
        </div>
        <details className="mt-3 rounded-lg border border-zinc-800 bg-black/20 p-2 text-xs text-zinc-400">
          <summary className="cursor-pointer text-zinc-300">Ver regras do gate</summary>
          <div className="mt-2 break-all">
            min_majority_60d={textOrNd(gateThresholds.min_joint_majority_60d)} | max_delta_p1={textOrNd(gateThresholds.max_abs_delta_p1)} | max_delta_deff={textOrNd(gateThresholds.max_abs_delta_deff)} | max_cluster_alerts={textOrNd(gateThresholds.max_active_cluster_alerts)}
          </div>
        </details>
      </section>

      <section className="rounded-2xl border border-zinc-800 bg-zinc-950/60 p-4 md:p-5">
        <div className="text-xs uppercase tracking-[0.16em] text-zinc-500">Alertas</div>
        <h2 className="mt-1 text-lg font-semibold text-zinc-100">Sinal do dia e histórico recente</h2>
        <div className="mt-3 grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-3 text-sm">
          <div className="rounded-xl border border-zinc-800 bg-black/30 p-3">
            <div className="flex items-center gap-1.5 text-[11px] uppercase tracking-[0.12em] text-zinc-500">
              <span>Nível do dia</span>
              <Hint text="Leitura operacional diária do motor: verde, amarelo ou vermelho." />
            </div>
            <div className="mt-2">
              {hasLatestAlert ? (
                <span className={`rounded-md border px-2 py-1 text-xs ${alertLevelBadge(latestAlertLevel)}`}>{latestAlertLevel}</span>
              ) : (
                <span className="rounded-md border border-zinc-700 px-2 py-1 text-xs text-zinc-300">sem leitura publicada</span>
              )}
            </div>
            <div className="mt-2 text-xs text-zinc-400">
              {hasLatestAlert
                ? `risco=${num(latestAlert?.risk_score ?? null, 3)} | confiança=${num(latestAlert?.signal_confidence ?? null, 3)}`
                : "Sem dados de alerta para a janela atual."}
            </div>
          </div>
          <K
            label="Eventos (60 dias)"
            value={textOrNd(op.n_events_last_60d)}
            hint="Quantidade de eventos estruturais recentes no histórico curto."
          />
          <K
            label="Eventos (total)"
            value={textOrNd(op.n_events_total)}
            hint="Quantidade total de eventos acumulados no run."
          />
          <K
            label="Topo de risco"
            value={topRiskSummary}
            hint="Ativo e setor mais sensíveis no diagnóstico atual."
          />
        </div>
        <div className="mt-3 rounded-xl border border-zinc-800 bg-black/30 p-3">
          <div className="flex items-center gap-2 text-xs uppercase tracking-[0.12em] text-zinc-500">
            <span>Força estatística por janela</span>
            <Hint text="Mede se o sinal está mais próximo de padrão forte ou de ruído." />
          </div>
          <div className="mt-3 overflow-auto">
            <table className="w-full text-xs">
              <thead className="text-zinc-500 uppercase">
                <tr>
                  <th className="text-left py-2">Janela</th>
                  <th className="text-left py-2">P1 (força)</th>
                  <th className="text-left py-2">DEFF (força)</th>
                  <th className="text-left py-2">Leitura</th>
                </tr>
              </thead>
              <tbody>
                {sigRows.map((row) => {
                  const p1Share = row.p1?.significant_share_p_lt_0_05 ?? null;
                  const p1p = row.p1?.latest_pvalue ?? null;
                  const dShare = row.deff?.significant_share_p_lt_0_05 ?? null;
                  const dp = row.deff?.latest_pvalue ?? null;
                  const toneP1 = significanceTone(p1Share, p1p);
                  const toneD = significanceTone(dShare, dp);
                  const read =
                    toneP1 === "forte" || toneD === "forte"
                      ? "forte"
                      : toneP1 === "moderada" || toneD === "moderada"
                        ? "moderada"
                        : "fraca";
                  return (
                    <tr key={`sig-${row.window}`} className="border-t border-zinc-800/70 text-zinc-300">
                      <td className="py-2">T{row.window}</td>
                      <td className="py-2">{toneP1} (p={num(p1p, 4)})</td>
                      <td className="py-2">{toneD} (p={num(dp, 4)})</td>
                      <td className="py-2">{read}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <div className="mt-2 text-xs text-zinc-500">
            Eventos recentes: {Object.entries(op.event_counts || {}).slice(0, 4).map(([k, v]) => `${k}=${v}`).join(" | ") || "sem eventos listados"}
          </div>
        </div>
      </section>

      <section className="rounded-2xl border border-zinc-800 bg-zinc-950/60 p-4 md:p-5">
        <div className="mb-3 flex flex-wrap gap-2 items-center">
          <div className="text-xs text-zinc-400">Filtros:</div>
          <select
            value={windowSize}
            onChange={(e) => setWindowSize(Number(e.target.value) || 120)}
            aria-label="Selecionar janela do motor"
            className="rounded-md border border-zinc-700 bg-black/30 px-2 py-1 text-xs text-zinc-200"
            title="Janela oficial de produção: T120."
          >
            <option value={120}>janela 120</option>
          </select>
          <select
            value={periodDays}
            onChange={(e) => setPeriodDays(Number(e.target.value) || 180)}
            aria-label="Selecionar período da análise"
            className="rounded-md border border-zinc-700 bg-black/30 px-2 py-1 text-xs text-zinc-200"
          >
            <option value={30}>30 dias</option>
            <option value={90}>90 dias</option>
            <option value={180}>180 dias</option>
            <option value={365}>365 dias</option>
          </select>
          <input
            value={sectorSearch}
            onChange={(e) => setSectorSearch(e.target.value)}
            placeholder="setor"
            aria-label="Filtrar setor"
            className="rounded-md border border-zinc-700 bg-black/30 px-2 py-1 text-xs text-zinc-200"
          />
          <input
            value={assetSearch}
            onChange={(e) => setAssetSearch(e.target.value)}
            placeholder="ativo"
            aria-label="Filtrar ativo"
            className="rounded-md border border-zinc-700 bg-black/30 px-2 py-1 text-xs text-zinc-200"
          />
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <div className="rounded-xl border border-zinc-800 bg-black/30 p-3">
            <div className="flex items-center gap-2 text-xs uppercase tracking-[0.12em] text-zinc-500">
              <span>Resumo por setor</span>
              <Hint text="Score matemático setorial: 0.50*risco + 0.35*instável + 0.15*(1-confiança)." />
            </div>
            <div className="mt-2 rounded-md border border-zinc-800 bg-zinc-950/40 p-2 text-[11px] text-zinc-400">
              score_setor = 0.50*risk_mean + 0.35*pct_instavel + 0.15*(1-confidence_mean)
            </div>
            <div className="mt-2 text-xs text-zinc-400">
              Risco agregado atual:{" "}
              <span className="text-zinc-200">{universeRisk == null ? "n/d" : `${num(universeRisk, 3)} (${bandByScore(universeRisk)})`}</span>
            </div>
            {incompleteSectorRows > 0 ? (
              <div className="mt-2 text-[11px] text-zinc-500">
                {incompleteSectorRows} setor(es) não entraram no score por dados incompletos nesta execução.
              </div>
            ) : null}
            <div className="mt-2 space-y-2 text-xs">
              {sectorMathRows.map((r, idx) => {
                const width = Math.max(4, Math.round((r.score / maxSectorScore) * 100));
                return (
                  <div key={r.sector} className="border border-zinc-800 rounded-md p-2">
                    <div className="flex items-center justify-between gap-2">
                      <div className="text-zinc-200">{`${idx + 1}. ${cleanSectorName(r.sector)}`}</div>
                      <span className={`rounded-md border px-2 py-0.5 ${levelBadge(r.alerta)}`}>{r.alerta}</span>
                    </div>
                    <div className="mt-2 h-2 rounded-full bg-zinc-900">
                      <div
                        className="h-2 rounded-full bg-gradient-to-r from-cyan-500 via-amber-500 to-rose-500"
                        style={{ width: `${width}%` }}
                      />
                    </div>
                    <div className="mt-1 text-zinc-400">
                      score={num(r.score, 3)} ({bandByScore(r.score)}) | risco={num(r.risk, 3)} | instável={pct(r.instavel, 1)} | confiança={num(r.confidence, 3)}
                      {r.nAssets != null ? ` | ativos=${num(r.nAssets, 0)}` : ""}
                    </div>
                  </div>
                );
              })}
              {!sectorMathRows.length ? <div className="text-zinc-500">Sem dados setoriais disponíveis nesta atualização.</div> : null}
            </div>
          </div>
          <div className="rounded-xl border border-zinc-800 bg-black/30 p-3">
            <div className="flex items-center gap-2 text-xs uppercase tracking-[0.12em] text-zinc-500">
              <span>Linha diária de regime</span>
              <Hint text="Cada bolinha representa 1 dia. Passe o mouse para ver data, regime e score de transição." />
            </div>
            <div className="mt-2 overflow-x-auto pb-1">
              <div className="flex min-w-max items-center gap-1">
                {regimeDots.map((point, idx) => (
                  <button
                    key={`${point.date}-${idx}`}
                    type="button"
                    className={`h-2.5 w-2.5 rounded-full ring-1 transition hover:scale-125 focus:scale-125 ${dotTone(point.regime)}`}
                    title={`Data: ${point.date}\nRegime: ${shortRegime(point.regime)}\nForça de transição: ${num(point.transition_score, 3)}\nExposição estrutural: ${num(point.exposure, 3)}`}
                    onMouseEnter={() => setHoveredDot(point)}
                    onFocus={() => setHoveredDot(point)}
                    onMouseLeave={() => setHoveredDot(null)}
                    onBlur={() => setHoveredDot(null)}
                  />
                ))}
                {!regimeDots.length ? <div className="text-zinc-500 text-xs">Sem histórico diário no período.</div> : null}
              </div>
            </div>
            <div className="mt-2 text-xs text-zinc-300">
              {activeDot ? (
                <>
                  {activeDot.date} | {shortRegime(activeDot.regime)} | força de transição={num(activeDot.transition_score, 3)} | exposição={num(activeDot.exposure, 3)}
                </>
              ) : (
                "Passe o mouse nas bolinhas para detalhes."
              )}
            </div>
            <div className="mt-1 text-[11px] text-zinc-500">Dias exibidos: {regimeDots.length}</div>
            <div className="mt-3 space-y-1 text-xs text-zinc-300">
              {historyChanges.slice(0, 6).map((x, idx) => (
                <div key={`${x.date}-${idx}`}>
                  {x.date}: {shortRegime(x.from)} para {shortRegime(x.to)}
                </div>
              ))}
              {!historyChanges.length ? <div className="text-zinc-500">Sem mudanças no período filtrado.</div> : null}
            </div>
          </div>
        </div>
      </section>

      <section className="rounded-2xl border border-zinc-800 bg-zinc-950/60 p-4 md:p-5">
        <div className="flex items-center gap-2 text-sm uppercase tracking-widest text-zinc-400">
          <span>Amostra rápida de preços (top 40 por risco)</span>
          <Hint text="Tabela de leitura rápida com preços apenas para os 40 ativos mais arriscados do filtro atual." />
        </div>
        <div className="mt-3 overflow-auto">
          <table className="w-full text-xs">
            <thead className="text-zinc-500 uppercase">
              <tr>
                <th className="text-left py-2" title="Ticker do ativo.">Ativo</th>
                <th className="text-left py-2" title="Setor ao qual o ativo pertence no mapeamento do motor.">Setor</th>
                <th className="text-left py-2" title="Preço mais recente disponível para o ativo (unidade da fonte, geralmente USD ou pontos).">Preço hoje</th>
                <th className="text-left py-2" title="Preço imediatamente anterior ao preço de hoje (mesma unidade da fonte).">Preço ontem</th>
                <th className="text-left py-2" title="Score de risco estrutural (quanto maior, mais frágil).">Risco</th>
                <th className="text-left py-2" title="Confiança do diagnóstico para o ativo.">Confiança</th>
                <th className="text-left py-2" title="Regime atual do ativo.">Estado</th>
                <th className="text-left py-2" title="Quantidade de trocas de regime nos últimos 30 dias.">Trocas 30d</th>
                <th className="text-left py-2" title="Quantidade de trocas de regime nos últimos 90 dias.">Trocas 90d</th>
              </tr>
            </thead>
            <tbody>
              {topAssetsForPrice.map((r) => (
                <tr key={r.ticker} className="border-t border-zinc-800/70 text-zinc-300">
                  <td className="py-2">{r.ticker}</td>
                  <td className="py-2">{cleanSectorName(r.sector)}</td>
                  <td className="py-2">{fmtPrice(assetPrices[String(r.ticker || "")]?.today)}</td>
                  <td className="py-2">{fmtPrice(assetPrices[String(r.ticker || "")]?.prev)}</td>
                  <td className="py-2">{num(r.risk_score, 3)}</td>
                  <td className="py-2">{num(r.confidence_score, 3)}</td>
                  <td className="py-2">{shortRegime(String(r.regime_asset || ""))}</td>
                  <td className="py-2">{num(r.switches_30d, 0)}</td>
                  <td className="py-2">{num(r.switches_90d, 0)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {!topAssetsForPrice.length ? (
            <div className="mt-2 text-xs text-zinc-500">
              {assetSearch || sectorSearch
                ? "Nenhum ativo para esse filtro. Tente reduzir o texto do filtro."
                : "Selecione um setor ou ativo para começar."}
            </div>
          ) : null}
        </div>
      </section>

      <section className="rounded-2xl border border-zinc-800 bg-zinc-950/60 p-4 md:p-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <div className="text-sm uppercase tracking-widest text-zinc-400">Universo completo por ativo e setor</div>
            <div className="mt-1 text-xs text-zinc-500">
              Sem cortar conteúdo: visão consolidada dos ativos do filtro atual com estado, confiança e risco.
            </div>
          </div>
          <div className="flex items-center gap-2 text-xs">
            <label htmlFor="asset-page-size" className="text-zinc-500">
              ativos por página
            </label>
            <select
              id="asset-page-size"
              value={assetPageSize}
              onChange={(e) => setAssetPageSize(Number(e.target.value) || 100)}
              className="rounded-md border border-zinc-700 bg-black/30 px-2 py-1 text-zinc-200"
            >
              <option value={50}>50</option>
              <option value={100}>100</option>
              <option value={200}>200</option>
              <option value={500}>todos (até 500)</option>
            </select>
          </div>
        </div>

        <div className="mt-3 grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
          <K label="Ativos no filtro" value={universeByAsset ? String(universeByAsset.total) : "n/d"} hint="Total de ativos carregados para o filtro atual." />
          <K label="Setores no filtro" value={universeByAsset ? String(universeByAsset.sectors) : "n/d"} hint="Total de setores presentes nos ativos filtrados." />
          <K
            label="Risco médio (ativos)"
            value={num(universeByAsset?.riskMean ?? null, 3)}
            hint="Média de risco estrutural por ativo."
          />
          <K
            label="Confiança média (ativos)"
            value={num(universeByAsset?.confMean ?? null, 3)}
            hint="Média de confiança do diagnóstico por ativo."
          />
        </div>
        <div className="mt-2 text-xs text-zinc-400">
          Distribuição de estado:
          {" "}
          <span className="text-emerald-300">estável {pct(universeByAsset?.stablePct, 1)}</span>
          {" | "}
          <span className="text-amber-300">transição {pct(universeByAsset?.transPct, 1)}</span>
          {" | "}
          <span className="text-rose-300">estresse {pct(universeByAsset?.stressPct, 1)}</span>
          {" | "}
          <span className="text-violet-300">dispersão {pct(universeByAsset?.dispersionPct, 1)}</span>
        </div>

        <div className="mt-4 grid grid-cols-1 xl:grid-cols-2 gap-4">
          <div className="rounded-xl border border-zinc-800 bg-black/30 p-3">
            <div className="flex items-center gap-2 text-xs uppercase tracking-[0.12em] text-zinc-500">
              <span>Mapa risco x confiança (ativos)</span>
              <Hint text="Cada ponto é um ativo: eixo X=confiança, eixo Y=risco. Cores representam o estado do ativo." />
            </div>
            <div className="mt-3 overflow-auto">
              <svg viewBox="0 0 760 330" className="min-w-[640px] w-full h-[300px]">
                <rect x="0" y="0" width="760" height="330" fill="rgba(24,24,27,0.35)" rx="12" />
                <line x1="48" y1="280" x2="724" y2="280" stroke="rgba(113,113,122,0.8)" strokeWidth="1" />
                <line x1="48" y1="40" x2="48" y2="280" stroke="rgba(113,113,122,0.8)" strokeWidth="1" />
                {[0, 0.25, 0.5, 0.75, 1].map((tick) => {
                  const x = 48 + tick * 676;
                  const y = 280 - tick * 240;
                  return (
                    <g key={`tick-${tick}`}>
                      <line x1={x} y1="280" x2={x} y2="284" stroke="rgba(113,113,122,0.8)" strokeWidth="1" />
                      <line x1="44" y1={y} x2="48" y2={y} stroke="rgba(113,113,122,0.8)" strokeWidth="1" />
                      <text x={x} y="300" fill="rgba(161,161,170,0.9)" textAnchor="middle" fontSize="10">
                        {tick.toFixed(2)}
                      </text>
                      <text x="34" y={y + 3} fill="rgba(161,161,170,0.9)" textAnchor="end" fontSize="10">
                        {tick.toFixed(2)}
                      </text>
                    </g>
                  );
                })}
                {scatterPoints.map((point) => {
                  const x = 48 + point.confidence * 676;
                  const y = 280 - point.risk * 240;
                  return (
                    <circle key={`${point.ticker}-${point.sector}`} cx={x} cy={y} r="3" fill={dotHex(point.regime)} opacity="0.82">
                      <title>
                        {`${point.ticker} | ${point.sector} | risco=${num(point.risk, 3)} | confiança=${num(point.confidence, 3)} | estado=${shortRegime(point.regime)}`}
                      </title>
                    </circle>
                  );
                })}
                <text x="388" y="320" fill="rgba(161,161,170,0.95)" textAnchor="middle" fontSize="11">
                  confiança
                </text>
                <text x="12" y="160" fill="rgba(161,161,170,0.95)" fontSize="11" transform="rotate(-90 12 160)">
                  risco
                </text>
              </svg>
            </div>
            <div className="mt-2 text-[11px] text-zinc-500">Pontos plotados: {scatterPoints.length}</div>
          </div>

          <div className="rounded-xl border border-zinc-800 bg-black/30 p-3">
            <div className="flex items-center gap-2 text-xs uppercase tracking-[0.12em] text-zinc-500">
              <span>Setores com divisão de regimes</span>
              <Hint text="Cada linha mostra o setor, métricas médias e a proporção de ativos em cada estado." />
            </div>
            <div className="mt-3 space-y-2 max-h-[355px] overflow-auto pr-1">
              {sectorBreakdownRows.map((row) => {
                const stablePct = row.count ? (row.stable / row.count) * 100 : 0;
                const transPct = row.count ? (row.trans / row.count) * 100 : 0;
                const stressPct = row.count ? (row.stress / row.count) * 100 : 0;
                const dispersionPct = row.count ? (row.dispersion / row.count) * 100 : 0;
                return (
                  <div key={row.sector} className="rounded-md border border-zinc-800 p-2">
                    <div className="flex items-center justify-between gap-2">
                      <div className="text-xs text-zinc-200">{row.sector}</div>
                      <span className={`rounded-md border px-2 py-0.5 text-[11px] ${levelBadge(row.alerta)}`}>{row.alerta}</span>
                    </div>
                    <div className="mt-1 text-[11px] text-zinc-400">
                      ativos={row.count} | risco={num(row.riskMean, 3)} | confiança={num(row.confMean, 3)} | score={num(row.score, 3)}
                    </div>
                    <div className="mt-2 h-2 overflow-hidden rounded-full bg-zinc-900">
                      <div className="flex h-2 w-full">
                        <div className="h-2 bg-emerald-500/80" style={{ width: `${stablePct}%` }} />
                        <div className="h-2 bg-amber-500/80" style={{ width: `${transPct}%` }} />
                        <div className="h-2 bg-rose-500/80" style={{ width: `${stressPct}%` }} />
                        <div className="h-2 bg-violet-500/80" style={{ width: `${dispersionPct}%` }} />
                      </div>
                    </div>
                    <div className="mt-1 text-[10px] text-zinc-500">
                      estável {stablePct.toFixed(1)}% | transição {transPct.toFixed(1)}% | estresse {stressPct.toFixed(1)}% | dispersão {dispersionPct.toFixed(1)}%
                    </div>
                  </div>
                );
              })}
              {!sectorBreakdownRows.length ? <div className="text-xs text-zinc-500">Sem dados setoriais para este filtro.</div> : null}
            </div>
          </div>
        </div>

        <div className="mt-4 overflow-auto">
          <table className="w-full text-xs">
            <thead className="text-zinc-500 uppercase">
              <tr>
                <th className="text-left py-2" title="Ticker do ativo no universo.">Ativo</th>
                <th className="text-left py-2" title="Setor mapeado para o ativo.">Setor</th>
                <th className="text-left py-2" title="Estado estrutural atual do ativo.">Estado</th>
                <th className="text-left py-2" title="Score de risco estrutural do ativo (0-1).">Risco</th>
                <th className="text-left py-2" title="Confiança do diagnóstico do ativo (0-1).">Confiança</th>
                <th className="text-left py-2" title="Trocas de estado em 30 dias.">Trocas 30d</th>
                <th className="text-left py-2" title="Trocas de estado em 90 dias.">Trocas 90d</th>
              </tr>
            </thead>
            <tbody>
              {pagedAssets.map((row) => (
                <tr key={`${row.ticker}-${row.sector}`} className="border-t border-zinc-800/70 text-zinc-300">
                  <td className="py-2">{row.ticker}</td>
                  <td className="py-2">{cleanSectorName(row.sector)}</td>
                  <td className="py-2">
                    <span className="inline-flex items-center gap-1">
                      <span className={`h-2 w-2 rounded-full ring-1 ${dotTone(String(row.regime_asset || ""))}`} />
                      {shortRegime(String(row.regime_asset || ""))}
                    </span>
                  </td>
                  <td className="py-2">{num(row.risk_score, 3)}</td>
                  <td className="py-2">{num(row.confidence_score, 3)}</td>
                  <td className="py-2">{num(row.switches_30d, 0)}</td>
                  <td className="py-2">{num(row.switches_90d, 0)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {!pagedAssets.length ? <div className="mt-2 text-xs text-zinc-500">Sem ativos para mostrar no filtro atual.</div> : null}
        </div>
        <div className="mt-3 flex flex-wrap items-center justify-between gap-2 text-xs text-zinc-400">
          <div>
            página {currentPage} de {totalPages} | exibindo {pagedAssets.length} de {filteredAssetsAll.length} ativos
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setAssetPage((p) => Math.max(1, p - 1))}
              disabled={currentPage <= 1}
              className="rounded-md border border-zinc-700 px-2 py-1 text-zinc-200 disabled:opacity-40"
            >
              anterior
            </button>
            <button
              type="button"
              onClick={() => setAssetPage((p) => Math.min(totalPages, p + 1))}
              disabled={currentPage >= totalPages}
              className="rounded-md border border-zinc-700 px-2 py-1 text-zinc-200 disabled:opacity-40"
            >
              próxima
            </button>
          </div>
        </div>
      </section>

      <section className="rounded-2xl border border-zinc-800 bg-zinc-950/60 p-4 md:p-5">
        <div className="text-sm uppercase tracking-widest text-zinc-400">Produto e venda</div>
        <div className="mt-2 text-sm text-zinc-300">
          Pacotes prontos: básico, completo e sob medida, com relatório executivo e estudo de caso.
        </div>
        <div className="mt-3 flex flex-wrap gap-2 text-xs">
          <Link className="rounded-md border border-zinc-700 px-2 py-1 text-zinc-200 hover:border-zinc-500" href="/methods">
            Eigen Engine
          </Link>
          <Link className="rounded-md border border-zinc-700 px-2 py-1 text-zinc-200 hover:border-zinc-500" href="/app/casos">
            Estudo de caso
          </Link>
        </div>
      </section>
    </div>
  );
}

function Hint({ text }: { text: string }) {
  return (
    <span
      title={text}
      className="inline-flex h-4 w-4 items-center justify-center rounded-full border border-zinc-700 text-[10px] text-zinc-400"
      aria-label={text}
    >
      ?
    </span>
  );
}

function K({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="rounded-xl border border-zinc-800 bg-black/30 p-3">
      <div className="flex items-center gap-1.5 text-[11px] uppercase tracking-[0.12em] text-zinc-500">
        <span>{label}</span>
        {hint ? <Hint text={hint} /> : null}
      </div>
      <div className="mt-1 text-sm leading-snug text-zinc-100">{value}</div>
    </div>
  );
}
