const CANDIDATE_LABELS: Array<[RegExp, string]> = [
  [/^meta_v1__btc63_vs_equity$/i, "Ataque tático entre cripto e ações"],
  [/^alpha_attack_major8_equity25$/i, "Ataque por confiança com liquidação cripto"],
  [/^alpha_attack_major8_equity25_mc_guard$/i, "Ataque por confiança com guarda Monte Carlo"],
  [/^entry_fast14_exit63_m2_h0__wrapped$/i, "Ataque tático com entrada cripto mais rápida"],
  [/^meta_dd_conviction__25_100$/i, "Proteção por convicção"],
  [/^meta_dd_guard__12_06_reduce35$/i, "Freio global de drawdown"],
  [/^meta_dd_regime_guard__global45_crypto10$/i, "Proteção adaptativa por regime"],
  [/^meta_dd_crypto_guard__/i, "Proteção focada em cripto"],
  [/^static_equities_us_other__materials__technology$/i, "Cesta global de materiais, tecnologia e oportunidades especiais"],
  [/^meta_v1__crypto_vs_equity_meta_search__trail_switch__a2__r1$/i, "Ataque tático com ações balanceadas"],
  [/^static_materials__technology$/i, "Cesta de materiais e tecnologia"],
  [/^equity_meta_search__trail_switch__a2__r1$/i, "Ações balanceadas com troca por tração"],
  [/^equity_meta_search__trail_switch__a2__r3$/i, "Ações balanceadas com foco em retorno"],
  [/^equity_v2__/i, "Sleeve de ações com rotação lenta"],
  [/^equity_v3__/i, "Sleeve de ações balanceado"],
  [/^dynamic_lb12_k2$/i, "Rotação dinâmica por grupos"],
  [/^group_static_combo$/i, "Cesta estática por grupos"],
  [/^group_dynamic_combo$/i, "Rotação dinâmica por grupos"],
];

const METHOD_LABELS: Record<string, string> = {
  group_static_combo: "Cesta estática por grupos",
  group_dynamic_combo: "Rotação dinâmica por grupos",
  meta_switch: "Troca de sleeves por regime",
  drawdown_control: "Controle de drawdown",
  conviction: "Escala por convicção",
};

const GROUP_LABELS: Record<string, string> = {
  equities_us_other: "ações americanas especiais",
  materials: "materiais",
  technology: "tecnologia",
  consumer_discretionary: "consumo discricionário",
  industrials: "indústria",
  financials: "financeiro",
  health_care: "saúde",
  energy: "energia",
  crypto: "cripto",
  bonds_rates: "juros e bonds",
  fx: "câmbio",
  metals: "metais",
  commodities_broad: "commodities",
  equities_us_broad: "ações americanas amplas",
  equities_us_sectors: "setores americanos",
  equities_international: "ações internacionais",
};

const STATE_LABELS: Record<string, string> = {
  monitoramento_normal: "Monitoramento normal",
  monitoring_normal: "Monitoramento normal",
  advisory_controlado: "Diagnóstico controlado",
  operacional: "Operacional",
  restrito: "Apenas pesquisa",
  validated: "validado",
  watch: "observação",
  inconclusive: "inconclusivo",
  stress: "Stress",
  transition: "Transição",
  stable: "Estável",
  dispersion: "Dispersão",
};

const RISK_LABELS: Record<string, string> = {
  high: "alto",
  medium: "moderado",
  moderate: "moderado",
  low: "baixo",
  unknown: "sem leitura limpa",
};

const MODE_LABELS: Record<string, string> = {
  ataque: "Modo ataque",
  attack: "Modo ataque",
  protection: "Modo principal com proteção",
  protecao: "Modo principal com proteção",
  proteção: "Modo principal com proteção",
  principal: "Modo principal",
  balanced: "Modo principal com proteção",
};

function prettifyWords(text: string) {
  return text
    .replace(/__/g, " ")
    .replace(/_/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function titleCase(text: string) {
  return text
    .split(" ")
    .filter(Boolean)
    .map((word) => {
      const lower = word.toLowerCase();
      if (["de", "da", "do", "das", "dos", "e", "por", "em"].includes(lower)) return lower;
      return lower.charAt(0).toUpperCase() + lower.slice(1);
    })
    .join(" ");
}

export function humanizeGroupName(value: unknown) {
  const raw = String(value || "").trim();
  if (!raw || raw === "n/d" || raw === "--") return "n/d";
  const mapped = GROUP_LABELS[raw.toLowerCase()];
  return mapped ? titleCase(mapped) : titleCase(prettifyWords(raw));
}

export function humanizeMethodology(value: unknown) {
  const raw = String(value || "").trim();
  if (!raw || raw === "n/d" || raw === "--") return "n/d";
  return METHOD_LABELS[raw] || titleCase(prettifyWords(raw));
}

export function humanizeStrategyName(value: unknown) {
  const raw = String(value || "").trim();
  if (!raw || raw === "n/d" || raw === "--") return "n/d";
  for (const [pattern, label] of CANDIDATE_LABELS) {
    if (pattern.test(raw)) return label;
  }
  return titleCase(prettifyWords(raw));
}

export function humanizeEngineState(value: unknown) {
  const raw = String(value || "").trim().toLowerCase();
  if (!raw || raw === "n/d" || raw === "--") return "Sem leitura limpa";
  return STATE_LABELS[raw] || titleCase(prettifyWords(raw));
}

export function humanizeRiskLevel(value: unknown) {
  const raw = String(value || "").trim().toLowerCase();
  if (!raw || raw === "n/d" || raw === "--") return "sem leitura limpa";
  if (RISK_LABELS[raw]) return RISK_LABELS[raw];
  return prettifyWords(raw).toLowerCase();
}

export function humanizeModeName(value: unknown, fallbackLabel?: unknown) {
  const raw = String(value || "").trim().toLowerCase();
  const fallback = String(fallbackLabel || "").trim();
  if (raw && MODE_LABELS[raw]) return MODE_LABELS[raw];
  if (fallback) return fallback;
  if (!raw || raw === "n/d" || raw === "--") return "Sem modo definido";
  return titleCase(prettifyWords(raw));
}

export function humanizeConfidenceLevel(value: unknown) {
  const raw = String(value || "").trim().toLowerCase();
  if (!raw || raw === "n/d" || raw === "--") return "sem leitura";
  if (raw === "alta" || raw === "high") return "alta";
  if (raw === "média" || raw === "media" || raw === "medium") return "média";
  if (raw === "baixa" || raw === "low") return "baixa";
  return prettifyWords(raw).toLowerCase();
}

export function describeStrategy(value: unknown, fallbackNotes?: unknown) {
  const raw = String(value || "").trim();
  if (/^meta_v1__btc63_vs_equity$/i.test(raw)) {
    return "Liga cripto quando o BTC mostra força e o sleeve cripto supera ações. Se o terreno piora, volta para ações ou caixa.";
  }
  if (/^alpha_attack_major8_equity25$/i.test(raw)) {
    return "Usa confiança relativa ao histórico recente para dosar o ataque e corta um pouco a mão quando o ecossistema cripto parece perto de liquidação forçada.";
  }
  if (/^alpha_attack_major8_equity25_mc_guard$/i.test(raw)) {
    return "Parte do mesmo ataque por confiança, mas adiciona uma trava extra quando a simulação por regime mostra risco de cauda acima do normal.";
  }
  if (/^entry_fast14_exit63_m2_h0__wrapped$/i.test(raw)) {
    return "Mantém a leitura por confiança, mas libera o ataque cripto mais cedo quando a aceleração aparece de forma limpa.";
  }
  if (/^meta_dd_conviction__25_100$/i.test(raw)) {
    return "Usa o mesmo motor agressivo, mas reduz o tamanho quando a convicção cai. Mantém um piso para não desligar tarde demais.";
  }
  if (/^static_equities_us_other__materials__technology$/i.test(raw)) {
    return "Cesta estática que combina nomes americanos fora do índice amplo com materiais e tecnologia para buscar retorno sem depender de um único tema.";
  }
  if (/^equity_meta_search__trail_switch__a2__r1$/i.test(raw)) {
    return "Mistura uma perna mais forte com outra mais robusta e troca entre elas quando a tração muda.";
  }
  const notes = String(fallbackNotes || "").trim();
  if (notes) {
    return notes
      .replace("crypto if BTC risk-on and trailing 63d beats equities; else equities; cash if both BTC and SPY below MM200.", "Ataca cripto quando o BTC está em regime favorável; caso contrário fica em ações e vai para caixa quando o contexto dos dois piora.")
      .replace("scale por conviccao; piso ativo=25%", "Escala a exposição pela convicção do motor e mantém um piso de 25% para evitar zigue-zague excessivo.")
      .replace("modo ataque promovido com entrada cripto mais rapida, sizing por confianca relativa ao historico recente e overlay de liquidacao cripto", "Ataca mais cedo quando o cripto acelera, ajusta o tamanho pelo grau de confiança e reduz a mão quando o ambiente parece próximo de liquidação.")
      .replace("modo ataque promovido com entrada cripto mais rapida e sizing por confianca relativa ao historico recente", "Ataca cedo quando o cripto acelera e varia o tamanho da posição conforme a confiança recente.");
  }
  return "Leitura quantitativa traduzida para uso humano: mais risco quando o contexto está limpo, menos risco quando a estrutura enfraquece.";
}

export function humanizeStatusWord(value: unknown) {
  const raw = String(value || "").trim().toLowerCase();
  if (!raw || raw === "n/d" || raw === "--") return "n/d";
  if (raw === "keep") return "seguir";
  if (raw === "watch") return "observação";
  if (raw === "validated") return "validado";
  if (raw === "ok") return "publicado";
  if (raw === "fail") return "em ajuste";
  if (raw === "missing") return "não publicado";
  return titleCase(prettifyWords(raw));
}
