"use client";

import { useEffect, useMemo, useState } from "react";
import DashboardFilters from "@/components/DashboardFilters";
import RegimeChart from "@/components/RegimeChart";
import {
  humanizeEngineState,
  humanizeGroupName,
} from "@/lib/enginePresentation";

type Domain = "finance" | "crypto";

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
  sector?: string;
  regime?: string;
  confidence?: number | null;
  signal_status?: string;
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
  vol60d: number | null;
  volume: number | null;
  retH1: number | null;
  retH5: number | null;
  retH10: number | null;
  retH20: number | null;
  retH60: number | null;
  retH120: number | null;
  distMa20: number | null;
  distMa60: number | null;
  rangePos60: number | null;
  rangePos120: number | null;
  drawdown60: number | null;
  drawdown120: number | null;
  upShare20: number | null;
  upShare60: number | null;
  streak: number | null;
  continuationAfterUp: number | null;
  reboundAfterDown: number | null;
  avgUpMove20: number | null;
  avgDownMove20: number | null;
  confidence: number | null;
  regime: string;
  signalStatus: string;
};

type AssetRecommendation = {
  level: "ESTÁVEL" | "MONITORAR" | "ATENÇÃO" | "DEFENSIVO";
  action: string;
  rationale: string;
};

type PlatformSectorImpact = {
  sector?: string;
  sector_kind?: string;
  impact?: number | null;
};

type PlatformLatestPayload = {
  rankings?: {
    date?: string;
    top_sectors_global_mode?: PlatformSectorImpact[];
  };
};

type InvestAdvisoryAsset = {
  asset_id?: string;
  ticker?: string;
  sector_gics?: string;
  weight?: number | null;
  amount_1000?: number | null;
  amount_10000?: number | null;
  amount_100000?: number | null;
};

type InvestAdvisoryPayload = {
  ok?: boolean;
  status?: string;
  missing?: string[];
  strategy_state?: string;
  guidance?: string[];
  guardrails?: {
    publishable?: boolean;
    advisory_ready?: boolean;
    step_status?: Record<string, boolean>;
  };
  simulation?: {
    run_id?: string;
    test_start?: string;
    test_end?: string;
    latest_rebalance?: {
      date?: string;
      regime?: string;
      risk_bucket?: string;
      cash_weight?: number | null;
    };
    top_assets?: InvestAdvisoryAsset[];
    performance?: {
      ann_strategy?: number | null;
      ann_eqw?: number | null;
      ann_edge?: number | null;
      max_drawdown_strategy?: number | null;
      max_drawdown_eqw?: number | null;
      drawdown_edge?: number | null;
      signal_reliability?: number | null;
    };
  };
  systematic?: {
    run_id?: string;
    years_tested?: number[];
    worth_it_rate_vs_eqw?: number | null;
    monthly_alpha_prob_positive_vs_eqw?: number | null;
  };
};

type InvestmentShadowPayload = {
  ok?: boolean;
  status?: string;
  run_id?: string;
  generated_at_utc?: string;
  proxies?: {
    risk_proxy?: string;
    defensive_proxy?: string;
  };
  latest?: {
    price_date?: string;
    signal_date?: string;
    effective_date?: string;
    regime?: string;
    target_exposure?: number | null;
    gate_blocked?: boolean;
    freshness_days?: number | null;
  };
  live?: {
    status?: string;
    capital_start?: number | null;
    capital_end?: number | null;
    n_days?: number | null;
    latest_target_exposure?: number | null;
    latest_executed_exposure?: number | null;
    latest_regime?: string;
    edge_vs_benchmark_total_return?: number | null;
    portfolio?: {
      total_return?: number | null;
      ann_return?: number | null;
      ann_vol?: number | null;
      sharpe?: number | null;
      max_drawdown?: number | null;
    };
  };
  historical_proxy_replay?: {
    status?: string;
    edge_vs_benchmark_total_return?: number | null;
    portfolio?: {
      total_return?: number | null;
      ann_return?: number | null;
      ann_vol?: number | null;
      sharpe?: number | null;
      max_drawdown?: number | null;
    };
  };
  refresh_prices?: {
    ok?: number | null;
    failed?: number | null;
  };
};

const MISSING = "n/d";

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
  crypto: ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "LINK-USD"],
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

function formatCurrency(value: number | null | undefined, digits = 2) {
  if (!isFiniteNumber(value)) return MISSING;
  return value.toLocaleString("pt-BR", {
    style: "currency",
    currency: "BRL",
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
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

function computeAverage(points: Array<SeriesPoint & { price: number }>, windowBars: number) {
  const tail = points.slice(-windowBars);
  if (!tail.length) return null;
  const values = tail.map((point) => point.price).filter(isFiniteNumber);
  return mean(values);
}

function computeGapToAverage(points: Array<SeriesPoint & { price: number }>, windowBars: number) {
  const avg = computeAverage(points, windowBars);
  const last = points[points.length - 1];
  if (!isFiniteNumber(avg) || !isFiniteNumber(last?.price) || avg === 0) return null;
  return (last.price - avg) / avg;
}

function computeRangePosition(points: Array<SeriesPoint & { price: number }>, windowBars: number) {
  const tail = points.slice(-windowBars);
  if (tail.length < 2) return null;
  const prices = tail.map((point) => point.price).filter(isFiniteNumber);
  if (prices.length < 2) return null;
  const low = Math.min(...prices);
  const high = Math.max(...prices);
  const last = prices[prices.length - 1];
  if (high === low) return 0.5;
  return (last - low) / (high - low);
}

function computeTailDrawdown(points: Array<SeriesPoint & { price: number }>, windowBars: number) {
  const tail = points.slice(-windowBars);
  if (tail.length < 2) return null;
  let peak = tail[0].price;
  let worst = 0;
  for (const point of tail) {
    peak = Math.max(peak, point.price);
    if (peak > 0) worst = Math.min(worst, (point.price - peak) / peak);
  }
  return worst;
}

function computeUpShare(returns: number[], windowBars: number) {
  const tail = returns.slice(-windowBars);
  if (!tail.length) return null;
  const positive = tail.filter((value) => value > 0).length;
  return positive / tail.length;
}

function computeStreak(returns: number[]) {
  if (!returns.length) return null;
  const latest = returns[returns.length - 1];
  if (!isFiniteNumber(latest) || latest === 0) return 0;
  const sign = latest > 0 ? 1 : -1;
  let streak = 0;
  for (let i = returns.length - 1; i >= 0; i -= 1) {
    const value = returns[i];
    if (!isFiniteNumber(value) || value === 0) break;
    if ((value > 0 ? 1 : -1) !== sign) break;
    streak += 1;
  }
  return sign * streak;
}

function computeContinuationAfterUp(returns: number[]) {
  let total = 0;
  let continued = 0;
  for (let i = 0; i < returns.length - 1; i += 1) {
    if (returns[i] <= 0) continue;
    total += 1;
    if (returns[i + 1] > 0) continued += 1;
  }
  return total >= 5 ? continued / total : null;
}

function computeReboundAfterDown(returns: number[]) {
  let total = 0;
  let rebound = 0;
  for (let i = 0; i < returns.length - 1; i += 1) {
    if (returns[i] >= 0) continue;
    total += 1;
    if (returns[i + 1] > 0) rebound += 1;
  }
  return total >= 5 ? rebound / total : null;
}

function computeAverageMoveBySign(returns: number[], sign: "up" | "down", windowBars: number) {
  const tail = returns.slice(-windowBars);
  const filtered = tail.filter((value) => (sign === "up" ? value > 0 : value < 0));
  return filtered.length ? mean(filtered) : null;
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

function describeVolBucket(value: number | null) {
  if (!isFiniteNumber(value)) return "sem medicao limpa de volatilidade";
  if (value >= 0.04) return "volatilidade muito alta";
  if (value >= 0.025) return "volatilidade alta";
  if (value >= 0.015) return "volatilidade moderada";
  return "volatilidade controlada";
}

function buildAssetInsightDeck(row: AssetRow | null, horizon: 1 | 5 | 10) {
  if (!row) return ["Selecione um ativo para abrir o caderno de leitura contextual."];

  const insights: string[] = [];
  const push = (text: string | null | undefined) => {
    const clean = String(text || "").trim();
    if (!clean) return;
    if (!insights.includes(clean)) insights.push(clean);
  };

  const horizonMap = [
    { label: "1 barra", value: row.retH1 },
    { label: "5 barras", value: row.retH5 },
    { label: "10 barras", value: row.retH10 },
    { label: "20 barras", value: row.retH20 },
    { label: "60 barras", value: row.retH60 },
    { label: "120 barras", value: row.retH120 },
  ];

  push(`${row.asset} pertence hoje ao grupo ${humanizeGroupName(row.group)}.`);
  push(`O status do motor para ${row.asset} é ${humanizeEngineState(row.signalStatus || "monitoramento_normal")}.`);
  push(`O regime associado ao ativo está rotulado como ${humanizeEngineState(row.regime)}.`);
  if (isFiniteNumber(row.confidence)) push(`A confiança estrutural do ativo está em ${(row.confidence * 100).toFixed(0)}%.`);
  if (isFiniteNumber(row.confidence)) {
    push(row.confidence >= 0.75 ? "O motor está relativamente confortável com a leitura estrutural deste ativo." : row.confidence <= 0.45 ? "A leitura estrutural deste ativo ainda é fraca; ele pede mais confirmação." : "A leitura estrutural existe, mas ainda não é daquelas para confiar relaxado.");
  }
  if (isFiniteNumber(row.priceToday)) push(`O último preço observado de ${row.asset} foi ${formatPrice(row.priceToday)}.`);
  if (isFiniteNumber(row.pricePrev)) push(`O preço anterior foi ${formatPrice(row.pricePrev)}.`);
  if (isFiniteNumber(row.changeAbs)) push(`A variação absoluta mais recente foi ${formatNumber(row.changeAbs)} na unidade da série.`);
  if (isFiniteNumber(row.changePct)) {
    push(`No último fechamento útil, ${row.asset} mexeu ${formatPercent(row.changePct)}.`);
    push(row.changePct > 0 ? "O movimento diário mais recente foi de força compradora." : row.changePct < 0 ? "O movimento diário mais recente foi de pressão vendedora." : "O último movimento diário foi neutro.");
    push(Math.abs(row.changePct) >= 0.03 ? "Essa oscilação diária foi grande para uma leitura de rotina." : "A oscilação diária ficou dentro de uma faixa mais normal.");
  }

  horizonMap.forEach(({ label, value }) => {
    if (!isFiniteNumber(value)) {
      push(`Ainda não há histórico suficiente para tirar conclusão limpa em ${label}.`);
      return;
    }
    push(`Em ${label}, ${row.asset} acumulou ${formatPercent(value)}.`);
    push(value >= 0.1 ? `O retorno em ${label} está forte e acima do que se espera de um ativo parado.` : value <= -0.1 ? `O retorno em ${label} está fraco e mostra perda importante de tração.` : `Em ${label}, o comportamento ainda parece administrável.`);
    push(value > 0 ? `A leitura de ${label} favorece manutenção ou aumento gradual, nunca all-in.` : value < 0 ? `A leitura de ${label} pede humildade: tamanho menor e mais validação.` : `Em ${label}, o ativo ainda não escolheu uma direção clara.`);
    push(value > 0 ? `Se você já carrega ${row.asset}, ${label} ajuda mais a gerenciar lucro do que a adivinhar topo.` : `Se você já carrega ${row.asset}, ${label} pede regra de defesa antes de coragem.`);
    push(Math.abs(value) >= 0.05 ? `O deslocamento em ${label} é grande o bastante para afetar a percepção de risco do ativo.` : `O deslocamento em ${label} ainda está em faixa que não precisa dramatização.`);
  });

  push(`A volatilidade de 20 barras está em ${describeVolBucket(row.vol20d)}.`);
  if (isFiniteNumber(row.vol20d)) push(`Volatilidade curta medida em 20 barras: ${formatPercent(row.vol20d)}.`);
  if (isFiniteNumber(row.vol60d)) {
    push(`Volatilidade mais longa, em 60 barras, está em ${formatPercent(row.vol60d)}.`);
    push(row.vol60d > (row.vol20d ?? 0) ? "O risco mais longo ainda está mais carregado que o curto." : "O risco de longo prazo está mais domado do que o curto.");
    push(row.vol20d != null && row.vol60d != null && row.vol20d > row.vol60d ? "O curto prazo está mais nervoso do que a memória mais longa do ativo." : "O curto prazo ainda não superou o estresse observado no quadro mais longo.");
  }

  if (isFiniteNumber(row.distMa20)) {
    push(`O preço está ${formatPercent(row.distMa20)} distante da média curta de 20 barras.`);
    push(row.distMa20 > 0 ? "O ativo está acima da média curta, o que costuma sustentar leitura de tendência." : "O ativo está abaixo da média curta, o que pede mais cautela.");
  }
  if (isFiniteNumber(row.distMa60)) {
    push(`Contra a média de 60 barras, a distância atual é ${formatPercent(row.distMa60)}.`);
    push(row.distMa60 > 0 ? "No quadro mais amplo, o ativo continua acima da média longa." : "No quadro mais amplo, o ativo ainda não recuperou a média longa.");
  }

  if (isFiniteNumber(row.rangePos60)) {
    push(`${row.asset} está ${(row.rangePos60 * 100).toFixed(0)}% acima do fundo e dentro da faixa dos últimos 60 pontos.`);
    push(row.rangePos60 >= 0.8 ? "O ativo está perto do topo recente; perseguir preço aqui pede sangue frio." : row.rangePos60 <= 0.2 ? "O ativo está perto do fundo recente; faça mais diagnóstico antes de chamar de oportunidade." : "O ativo está no miolo da faixa recente.");
    push(row.rangePos60 >= 0.8 ? "Quando o ativo passa muito tempo perto do topo, qualquer falha de continuidade fica mais sensível." : "Enquanto o ativo não encosta no topo da faixa, ainda há espaço para consolidação sem drama.");
  }
  if (isFiniteNumber(row.rangePos120)) {
    push(`Na faixa de 120 pontos, a posição relativa atual está em ${(row.rangePos120 * 100).toFixed(0)}%.`);
    push(row.rangePos120 <= 0.3 ? "Na fotografia mais longa, o ativo ainda opera mais perto do piso do que da euforia." : "Na fotografia mais longa, o ativo já saiu da zona de pessimismo extremo.");
  }

  if (isFiniteNumber(row.drawdown60)) {
    push(`O drawdown local de 60 pontos está em ${formatPercent(row.drawdown60)}.`);
    push(row.drawdown60 <= -0.2 ? "A queda desde o pico recente ainda é relevante e pede respeito." : "A queda desde o pico recente está relativamente controlada.");
  }
  if (isFiniteNumber(row.drawdown120)) {
    push(`O drawdown de 120 pontos está em ${formatPercent(row.drawdown120)}.`);
    push(row.drawdown120 <= -0.35 ? "No horizonte mais longo, o ativo ainda carrega uma cicatriz importante de perda." : "No horizonte mais longo, o ativo não está numa zona de estrago extremo.");
  }

  if (isFiniteNumber(row.upShare20)) {
    push(`Nos últimos 20 pontos, ${(row.upShare20 * 100).toFixed(0)}% das barras fecharam em alta.`);
    push(row.upShare20 >= 0.6 ? "A taxa curta de barras positivas sugere mercado aceitando o ativo." : "A taxa curta de barras positivas ainda não mostra domínio comprador.");
  }
  if (isFiniteNumber(row.upShare60)) {
    push(`Nos últimos 60 pontos, ${(row.upShare60 * 100).toFixed(0)}% das barras fecharam em alta.`);
    push(row.upShare60 >= 0.55 ? "No quadro de 60 pontos, o histórico ainda pesa mais a favor do comprador." : "No quadro de 60 pontos, o histórico ainda não está claramente amigável.");
  }

  if (isFiniteNumber(row.streak)) {
    if (row.streak > 0) push(`${row.asset} vem de ${row.streak} fechamentos positivos seguidos.`);
    if (row.streak < 0) push(`${row.asset} vem de ${Math.abs(row.streak)} fechamentos negativos seguidos.`);
    if (row.streak === 0) push("Não há sequência recente clara de altas ou quedas.");
  }

  if (isFiniteNumber(row.continuationAfterUp)) {
    push(`Historicamente, depois de um fechamento positivo, ${row.asset} continua subindo no dia seguinte em ${(row.continuationAfterUp * 100).toFixed(0)}% dos casos medidos.`);
    push(row.continuationAfterUp >= 0.58 ? "O ativo tem um habito razoavel de continuar o impulso quando entra em ritmo." : "Quando sobe, este ativo nem sempre carrega o embalo para o dia seguinte.");
  }
  if (isFiniteNumber(row.reboundAfterDown)) {
    push(`Depois de um fechamento negativo, ${row.asset} reage para cima no dia seguinte em ${(row.reboundAfterDown * 100).toFixed(0)}% dos casos medidos.`);
    push(row.reboundAfterDown >= 0.55 ? "Historicamente, esse ativo costuma brigar contra quedas curtas com alguma frequência." : "Historicamente, esse ativo não mostra grande reflexo de repique depois de cair.");
  }
  if (isFiniteNumber(row.avgUpMove20)) {
    push(`Nas últimas 20 barras, a alta média nos dias positivos foi ${formatPercent(row.avgUpMove20)}.`);
  }
  if (isFiniteNumber(row.avgDownMove20)) {
    push(`Nas últimas 20 barras, a queda média nos dias negativos foi ${formatPercent(row.avgDownMove20)}.`);
    if (isFiniteNumber(row.avgUpMove20) && isFiniteNumber(row.avgDownMove20)) {
      push(Math.abs(row.avgDownMove20) > Math.abs(row.avgUpMove20) ? "Quando erra a mão, este ativo costuma cair mais forte do que sobe nos dias bons." : "Nos últimos 20 pontos, os dias bons ainda compensaram bem os dias ruins.");
    }
  }

  const groupText = String(row.group || "").toLowerCase();
  if (groupText.includes("cripto")) {
    push("Cripto pede mais humildade do que convicção: um mesmo sinal pode parecer perfeito e virar violência em poucas barras.");
  } else if (groupText.includes("tecnologia")) {
    push("Tecnologia costuma premiar continuidade, mas pune caro quando a narrativa perde tração.");
  } else if (groupText.includes("finance")) {
    push("Financeiro reage muito a juros, liquidez e medo sistêmico; a leitura estrutural costuma importar bastante.");
  } else if (groupText.includes("materiais")) {
    push("Materiais tendem a responder a ciclo global e expectativa de atividade; o motor costuma captar isso via regime e momentum.");
  } else if (groupText.includes("industr")) {
    push("Indústria costuma ser bom termômetro de ciclo. Quando a tração some aqui, vale respeitar o sinal.");
  } else if (groupText.includes("consumo")) {
    push("Consumo discricionário costuma acelerar cedo em fases otimistas e sofrer forte quando a tolerância a risco seca.");
  }

  const rec = buildAssetRecommendation(row, horizon);
  push(`Leitura operacional atual: ${rec.level.toLowerCase()}.`);
  push(`Ação simples sugerida hoje: ${rec.action}.`);
  push(`Motivo principal: ${rec.rationale}.`);

  if (rec.level === "ESTÁVEL") {
    push("Para quem já está posicionado, o ativo permite ajustes pequenos em vez de mudanças bruscas.");
    push("Para quem está de fora, faz mais sentido entrar em parcelas do que numa tacada só.");
  } else if (rec.level === "MONITORAR") {
    push("O melhor uso agora é acompanhar continuidade e não tentar adivinhar o fundo ou topo.");
    push("Se você operar esse ativo, prefira tamanho médio e revisão frequente.");
  } else if (rec.level === "ATENÇÃO") {
    push("A combinação de movimento forte com risco curto pede mão mais leve.");
    push("Se o ativo continuar acelerando, o risco de reversão também cresce.");
  } else {
    push("Esse não é um momento amigável para insistir em tamanho cheio.");
    push("O ativo entrou em zona onde proteger capital vale mais do que buscar heroísmo.");
  }

  for (let i = 0; i < horizonMap.length; i += 1) {
    const item = horizonMap[i];
    const value = item.value;
    if (!isFiniteNumber(value)) continue;
    push(value > 0 ? `No recorte de ${item.label}, o histórico recente ainda favorece continuidade, não pressa.` : `No recorte de ${item.label}, o histórico recente recomenda seletividade e paciência.`);
    push(Math.abs(value) >= 0.08 ? `O módulo do movimento em ${item.label} já é grande o bastante para exigir regra de saída.` : `O movimento em ${item.label} ainda cabe numa leitura normal de mercado.`);
    push(value > 0 ? `Quem entrar agora em ${row.asset} no recorte de ${item.label} precisa aceitar comprar depois da alta, não antes dela.` : `Quem entrar agora em ${row.asset} no recorte de ${item.label} precisa aceitar que o ativo ainda não mostrou recuperação suficiente.`);
  }

  return insights;
}

export default function SectorDashboard({
  title,
  showTable = true,
  initialDomain = "finance",
  initialGroupFilter = "all",
  headline = "Painel financeiro por ativo",
  description = "Leitura de comportamento de preço, risco e estabilidade por ativo selecionado.",
}: {
  title: string;
  showTable?: boolean;
  initialDomain?: Domain;
  initialGroupFilter?: string;
  headline?: string;
  description?: string;
}) {
  const [timeframe, setTimeframe] = useState("daily");
  const [groupFilter, setGroupFilter] = useState(initialGroupFilter);
  const [rangePreset, setRangePreset] = useState("180d");
  const [normalize, setNormalize] = useState(false);
  const [showRegimeBands, setShowRegimeBands] = useState(true);
  const [smoothing, setSmoothing] = useState<"none" | "ema_short" | "ema_long">("none");
  const [summaryHorizon, setSummaryHorizon] = useState<1 | 5 | 10>(5);
  const [focusAsset, setFocusAsset] = useState<string>("");
  const [showAllRecommendationCards, setShowAllRecommendationCards] = useState(false);
  const [showAllFocusInsights, setShowAllFocusInsights] = useState(false);

  const [universe, setUniverse] = useState<UniverseAsset[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [seriesByAsset, setSeriesByAsset] = useState<Record<string, SeriesPoint[]>>({});
  const [loading, setLoading] = useState(false);
  const [universeLoaded, setUniverseLoaded] = useState(true);
  const [platformLatest, setPlatformLatest] = useState<PlatformLatestPayload | null>(null);
  const [investAdvisory, setInvestAdvisory] = useState<InvestAdvisoryPayload | null>(null);
  const [investAdvisoryError, setInvestAdvisoryError] = useState("");
  const [investmentShadow, setInvestmentShadow] = useState<InvestmentShadowPayload | null>(null);
  const [investmentShadowError, setInvestmentShadowError] = useState("");

  useEffect(() => {
    setGroupFilter(initialGroupFilter);
  }, [initialGroupFilter]);

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
          sector: String(r.sector || r.group || ""),
          regime: String(r.regime || ""),
          confidence: isFiniteNumber(r.confidence) ? r.confidence : null,
          signal_status: String(r.signal_status || ""),
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
    const loadPlatformLatest = async () => {
      try {
        const res = await fetch("/api/platform/latest", { cache: "no-store" });
        if (!res.ok) return;
        const payload = (await res.json()) as PlatformLatestPayload;
        setPlatformLatest(payload);
      } catch {
        setPlatformLatest(null);
      }
    };
    loadPlatformLatest();
  }, []);

  useEffect(() => {
    const loadInvestmentAdvisory = async () => {
      try {
        const res = await fetch("/api/invest/advisory", { cache: "no-store" });
        const payload = (await res.json()) as InvestAdvisoryPayload;
        if (!res.ok) {
          const missing = Array.isArray(payload?.missing) ? payload.missing.join(", ") : "artefatos indisponíveis";
          setInvestAdvisoryError(`Guia indisponível: ${missing}`);
          setInvestAdvisory(null);
          return;
        }
        setInvestAdvisoryError("");
        setInvestAdvisory(payload);
      } catch {
        setInvestAdvisoryError("Guia indisponível: falha ao carregar endpoint /api/invest/advisory.");
        setInvestAdvisory(null);
      }
    };
    loadInvestmentAdvisory();
  }, []);

  useEffect(() => {
    const loadInvestmentShadow = async () => {
      try {
        const res = await fetch("/api/invest/shadow", { cache: "no-store" });
        const payload = (await res.json()) as InvestmentShadowPayload;
        if (!res.ok) {
          setInvestmentShadowError("Teste sombra indisponível: nenhum snapshot local publicado ainda.");
          setInvestmentShadow(null);
          return;
        }
        setInvestmentShadowError("");
        setInvestmentShadow(payload);
      } catch {
        setInvestmentShadowError("Teste sombra indisponível: falha ao carregar endpoint /api/invest/shadow.");
        setInvestmentShadow(null);
      }
    };
    loadInvestmentShadow();
  }, []);

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
      const vol60d = returns.length >= 60 ? std(returns.slice(-60)) : null;
      const volume = isFiniteNumber(last?.volume) ? last.volume : null;

      return {
        asset,
        group: humanizeGroupName(rowMeta?.group || rowMeta?.sector || "Sem classificação"),
        startDate: first?.date,
        endDate: last?.date,
        period: first && last ? `${first.date} até ${last.date}` : MISSING,
        priceToday,
        pricePrev,
        changeAbs,
        changePct,
        ret5d,
        vol20d,
        vol60d,
        volume,
        retH1: computeReturn(pricedPoints, 1),
        retH5: computeReturn(pricedPoints, 5),
        retH10: computeReturn(pricedPoints, 10),
        retH20: computeReturn(pricedPoints, 20),
        retH60: computeReturn(pricedPoints, 60),
        retH120: computeReturn(pricedPoints, 120),
        distMa20: computeGapToAverage(pricedPoints, 20),
        distMa60: computeGapToAverage(pricedPoints, 60),
        rangePos60: computeRangePosition(pricedPoints, 60),
        rangePos120: computeRangePosition(pricedPoints, 120),
        drawdown60: computeTailDrawdown(pricedPoints, 60),
        drawdown120: computeTailDrawdown(pricedPoints, 120),
        upShare20: computeUpShare(returns, 20),
        upShare60: computeUpShare(returns, 60),
        streak: computeStreak(returns),
        continuationAfterUp: computeContinuationAfterUp(returns),
        reboundAfterDown: computeReboundAfterDown(returns),
        avgUpMove20: computeAverageMoveBySign(returns, "up", 20),
        avgDownMove20: computeAverageMoveBySign(returns, "down", 20),
        confidence: rowMeta?.confidence ?? null,
        regime: String(rowMeta?.regime || ""),
        signalStatus: String(rowMeta?.signal_status || ""),
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
  const focusInsights = useMemo(() => buildAssetInsightDeck(focusRow, summaryHorizon), [focusRow, summaryHorizon]);
  const recommendationRows = useMemo(() => {
    const sorted = [...tableRows].sort((a, b) => {
      const av = isFiniteNumber(a.vol20d) ? a.vol20d : -1;
      const bv = isFiniteNumber(b.vol20d) ? b.vol20d : -1;
      return bv - av;
    });
    return showAllRecommendationCards ? sorted : sorted.slice(0, 12);
  }, [tableRows, showAllRecommendationCards]);

  const topSectorImpactRows = useMemo(() => {
    const rows = Array.isArray(platformLatest?.rankings?.top_sectors_global_mode)
      ? platformLatest?.rankings?.top_sectors_global_mode
      : [];
    return rows
      .filter((row) => String(row?.sector_kind || "").toLowerCase() === "gics")
      .slice(0, 6)
      .map((row) => ({
        sector: String(row?.sector || MISSING),
        impact: isFiniteNumber(row?.impact) ? row.impact : null,
      }));
  }, [platformLatest]);

  const investStepRows = useMemo(() => {
    const stepStatus = investAdvisory?.guardrails?.step_status || {};
    return Object.entries(stepStatus);
  }, [investAdvisory]);

  const shadowProfit = useMemo(() => {
    const capitalStart = investmentShadow?.live?.capital_start ?? null;
    const capitalEnd = investmentShadow?.live?.capital_end ?? null;
    if (!isFiniteNumber(capitalStart) || !isFiniteNumber(capitalEnd)) return null;
    return capitalEnd - capitalStart;
  }, [investmentShadow]);
  const advisoryMetricLabel =
    isFiniteNumber(investAdvisory?.simulation?.performance?.ann_edge ?? null) ? "Diferença vs referência" : "Confiabilidade publicada";
  const advisoryMetricValue = isFiniteNumber(investAdvisory?.simulation?.performance?.ann_edge ?? null)
    ? formatPercent(investAdvisory?.simulation?.performance?.ann_edge ?? null)
    : formatPercent(
        investAdvisory?.simulation?.performance?.signal_reliability ??
          investAdvisory?.simulation?.latest_rebalance?.signal_reliability ??
          null,
        0
      );
  const advisoryMetricTone = isFiniteNumber(investAdvisory?.simulation?.performance?.ann_edge ?? null)
    ? toneFromPct(investAdvisory?.simulation?.performance?.ann_edge ?? null)
    : "text-cyan-200";
  const shadowSharpeValue = isFiniteNumber(investmentShadow?.live?.portfolio?.sharpe ?? null)
    ? formatNumber(investmentShadow?.live?.portfolio?.sharpe ?? null, 2)
    : formatNumber(investmentShadow?.historical_proxy_replay?.portfolio?.sharpe ?? null, 2);
  const shadowSharpeLabel = isFiniteNumber(investmentShadow?.live?.portfolio?.sharpe ?? null) ? "Sharpe live" : "Sharpe replay";

  const sectors = financeGroupFilter;

  return (
    <div className="p-4 md:p-5 space-y-4 md:space-y-5">
      <div className="space-y-2">
        <div className="text-xs uppercase tracking-[0.2em] text-zinc-400">{title}</div>
        <h1 className="text-2xl font-semibold">{headline}</h1>
        <p className="text-sm text-zinc-400">{description}</p>
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

        <div className="rounded-xl border border-zinc-800 bg-black/20 p-3">
          <div className="text-sm text-zinc-200">Ranking setor → global (impacto estrutural)</div>
          <div className="text-xs text-zinc-500">
            Fonte: snapshot financeiro atual do Eigen Engine ({platformLatest?.rankings?.date || MISSING}).
          </div>
          <div className="mt-2 grid grid-cols-2 md:grid-cols-3 gap-2">
            {topSectorImpactRows.length ? (
              topSectorImpactRows.map((row) => (
                <div key={`sector-impact-${row.sector}`} className="rounded-md border border-zinc-800 bg-zinc-950/60 px-2 py-1.5">
                  <div className="text-[11px] uppercase tracking-[0.12em] text-zinc-500">{humanizeGroupName(row.sector)}</div>
                  <div className="text-sm font-semibold text-zinc-100">{isFiniteNumber(row.impact) ? row.impact.toFixed(4) : MISSING}</div>
                </div>
              ))
            ) : (
              <div className="col-span-full text-xs text-zinc-500">Sem ranking setorial publicado no artefato atual.</div>
            )}
          </div>
        </div>

        <div className="rounded-xl border border-zinc-800 bg-black/20 p-3 space-y-3">
          <div>
            <div className="text-sm text-zinc-200">Guia de decisão estatística (advisory)</div>
            <div className="text-xs text-zinc-500">
              Integração direta de guardrails + simulação de alocação. Sem execução automática e sem promessa de retorno.
            </div>
          </div>
          {investAdvisoryError ? (
            <div className="rounded-lg border border-rose-800/60 bg-rose-950/20 px-3 py-2 text-xs text-rose-200">{investAdvisoryError}</div>
          ) : null}
          {!investAdvisoryError && !investAdvisory ? <div className="text-xs text-zinc-500">Carregando guia...</div> : null}
          {investAdvisory ? (
            <div className="space-y-3">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                <MetricMini label="Estado do guia" value={humanizeEngineState(investAdvisory.strategy_state || MISSING)} />
                <MetricMini
                  label="Janela de teste"
                  value={`${String(investAdvisory.simulation?.test_start || MISSING)} → ${String(
                    investAdvisory.simulation?.test_end || MISSING
                  )}`}
                />
                <MetricMini
                  label="Retorno anual (estratégia)"
                  value={formatPercent(investAdvisory.simulation?.performance?.ann_strategy ?? null)}
                  tone={toneFromPct(investAdvisory.simulation?.performance?.ann_edge ?? null)}
                />
                <MetricMini
                  label={advisoryMetricLabel}
                  value={advisoryMetricValue}
                  tone={advisoryMetricTone}
                />
              </div>

              {Array.isArray(investAdvisory.guidance) && investAdvisory.guidance.length ? (
                <div className="rounded-lg border border-zinc-800 bg-zinc-950/60 p-2">
                  <div className="text-[11px] uppercase tracking-[0.12em] text-zinc-500">Diretrizes do motor</div>
                  <div className="mt-1 space-y-1 text-xs text-zinc-300">
                    {investAdvisory.guidance.map((tip, idx) => (
                      <div key={`advisory-tip-${idx}`}>{idx + 1}. {tip}</div>
                    ))}
                  </div>
                </div>
              ) : null}

              {investStepRows.length ? (
                <div className="flex flex-wrap gap-1.5">
                  {investStepRows.map(([step, ok]) => (
                    <span
                      key={`invest-step-${step}`}
                      className={`rounded-md border px-2 py-1 text-[10px] ${
                        ok ? "border-emerald-600/40 bg-emerald-600/10 text-emerald-200" : "border-amber-600/40 bg-amber-600/10 text-amber-200"
                      }`}
                    >
                      {step.replace(/^step\d+_/, "")}: {ok ? "ok" : "falha"}
                    </span>
                  ))}
                </div>
              ) : null}

              {Array.isArray(investAdvisory.simulation?.top_assets) && investAdvisory.simulation?.top_assets.length ? (
                <div className="overflow-auto">
                  <table className="w-full text-xs">
                    <thead className="text-zinc-500 uppercase">
                      <tr>
                        <th className="text-left py-1.5">Ativo</th>
                        <th className="text-left py-1.5">Setor</th>
                        <th className="text-left py-1.5">Peso</th>
                        <th className="text-left py-1.5">R$ 1.000</th>
                        <th className="text-left py-1.5">R$ 10.000</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(investAdvisory.simulation?.top_assets || []).slice(0, 10).map((row) => (
                        <tr key={`inv-top-${row.asset_id || row.ticker || ""}`} className="border-t border-zinc-800/70 text-zinc-300">
                          <td className="py-1.5">{row.ticker || row.asset_id || MISSING}</td>
                          <td className="py-1.5 text-zinc-400">{humanizeGroupName(row.sector_gics || MISSING)}</td>
                          <td className="py-1.5">{formatPercent(row.weight ?? null)}</td>
                          <td className="py-1.5">{formatNumber(row.amount_1000 ?? null, 2)}</td>
                          <td className="py-1.5">{formatNumber(row.amount_10000 ?? null, 2)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : null}
            </div>
          ) : null}
        </div>

        <div className="rounded-xl border border-zinc-800 bg-black/20 p-3 space-y-3">
          <div>
            <div className="text-sm text-zinc-200">Teste sombra local</div>
            <div className="text-xs text-zinc-500">
              Paper trading diário com proxies investíveis. O motor roda localmente, registra sinal, valor da carteira e comparação contra benchmark.
            </div>
          </div>
          {investmentShadowError ? (
            <div className="rounded-lg border border-amber-800/60 bg-amber-950/20 px-3 py-2 text-xs text-amber-200">{investmentShadowError}</div>
          ) : null}
          {!investmentShadowError && !investmentShadow ? <div className="text-xs text-zinc-500">Carregando teste sombra...</div> : null}
          {investmentShadow ? (
            <div className="space-y-3">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                <MetricMini
                  label="Capital atual"
                  value={formatCurrency(investmentShadow.live?.capital_end ?? null)}
                  tone={toneFromPct(investmentShadow.live?.portfolio?.total_return ?? null)}
                />
                <MetricMini label="Lucro acumulado" value={formatCurrency(shadowProfit)} tone={toneFromPct(shadowProfit ?? null)} />
                <MetricMini
                  label="Regime / exposição"
                  value={`${humanizeEngineState(investmentShadow.latest?.regime || MISSING)} / ${formatPercent(investmentShadow.latest?.target_exposure ?? null, 0)}`}
                />
                <MetricMini
                  label="Replay anualizado"
                  value={formatPercent(investmentShadow.historical_proxy_replay?.portfolio?.ann_return ?? null)}
                  tone={toneFromPct(investmentShadow.historical_proxy_replay?.edge_vs_benchmark_total_return ?? null)}
                />
              </div>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                <MetricMini
                  label={shadowSharpeLabel}
                  value={shadowSharpeValue}
                  tone={toneFromPct(investmentShadow.live?.edge_vs_benchmark_total_return ?? null)}
                />
                <MetricMini
                  label="MDD replay"
                  value={formatPercent(investmentShadow.historical_proxy_replay?.portfolio?.max_drawdown ?? null)}
                  tone={toneFromPct(-(investmentShadow.historical_proxy_replay?.portfolio?.max_drawdown ?? 0))}
                />
                <MetricMini
                  label="Atualização"
                  value={`${String(investmentShadow.latest?.price_date || MISSING)}${isFiniteNumber(investmentShadow.latest?.freshness_days) ? ` (${investmentShadow.latest?.freshness_days}d)` : ""}`}
                />
                <MetricMini
                  label="Proxies"
                  value={`${String(investmentShadow.proxies?.risk_proxy || MISSING)} / ${String(
                    investmentShadow.proxies?.defensive_proxy || MISSING
                  )}`}
                />
              </div>

              <div className="rounded-lg border border-zinc-800 bg-zinc-950/60 p-2 text-xs text-zinc-300">
                <div>
                  Run: <span className="text-zinc-100">{String(investmentShadow.run_id || MISSING)}</span>
                </div>
                <div>
                  Sinal: <span className="text-zinc-100">{String(investmentShadow.latest?.signal_date || MISSING)}</span> | efetivação:{" "}
                  <span className="text-zinc-100">{String(investmentShadow.latest?.effective_date || MISSING)}</span>
                </div>
                <div>
                  Gate:{" "}
                  <span className={investmentShadow.latest?.gate_blocked ? "text-amber-300" : "text-emerald-300"}>
                    {investmentShadow.latest?.gate_blocked ? "bloqueado" : "livre"}
                  </span>{" "}
                  | refresh preços ok/fail:{" "}
                  <span className="text-zinc-100">
                    {formatNumber(investmentShadow.refresh_prices?.ok ?? null, 0)}/{formatNumber(investmentShadow.refresh_prices?.failed ?? null, 0)}
                  </span>
                </div>
              </div>
            </div>
          ) : null}
        </div>

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
              label="Leituras com confiança"
              value={String(tableRows.filter((r) => isFiniteNumber(r.confidence)).length)}
              helper="Quantidade de ativos com nota de confiança estrutural publicada no snapshot atual."
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

                <div className="rounded-lg border border-zinc-800 bg-black/25 p-2">
                  <div className="text-[11px] uppercase tracking-[0.12em] text-zinc-500">Leituras rapidas do ativo</div>
                  <div className="mt-2 space-y-1 text-xs text-zinc-300">
                    {buildAssetInsightDeck(row, summaryHorizon)
                      .slice(0, 5)
                      .map((tip, idx) => (
                        <div key={`card-tip-${row.asset}-${idx}`}>{idx + 1}. {tip}</div>
                      ))}
                  </div>
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
                    <th className="text-left py-2" title="Nota de confiança estrutural publicada para o ativo.">Confiança</th>
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
                        <td className="py-2">{isFiniteNumber(row.confidence) ? `${(row.confidence * 100).toFixed(0)}%` : "n/d"}</td>
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

            <div className="rounded-xl border border-zinc-800 bg-zinc-950/70 p-3">
              <div className="flex items-center justify-between gap-2">
                <div className="text-[11px] uppercase tracking-[0.12em] text-zinc-500">Caderno do ativo</div>
                <button
                  onClick={() => setShowAllFocusInsights((value) => !value)}
                  className="rounded-md border border-zinc-700 px-2 py-1 text-[10px] text-zinc-200 hover:border-zinc-500"
                >
                  {showAllFocusInsights ? "Mostrar menos" : `Abrir tudo (${focusInsights.length})`}
                </button>
              </div>
              <div className="mt-2 text-xs text-zinc-500">
                Comentários simples montados a partir do histórico, da volatilidade, do regime atual e da forma como o ativo costuma reagir no curto e médio prazo.
              </div>
              <div className="mt-3 grid gap-2 text-xs text-zinc-300">
                {(showAllFocusInsights ? focusInsights : focusInsights.slice(0, 28)).map((tip, idx) => (
                  <div key={`focus-insight-${idx}`} className="rounded-md border border-zinc-800 bg-black/20 px-2 py-1.5">
                    {idx + 1}. {tip}
                  </div>
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
