import { promises as fs } from "fs";
import path from "path";
import {
  dataDirs,
  findLatestLabCorrRun,
  findLatestValidRun,
  readLatestLabCorrActionPlaybook,
  readLatestLabCorrAssetDiagnostics,
  readLatestLabCorrOperationalAlerts,
  readLatestLabCorrTimeseries,
  readLatestSnapshot,
  readPlatformDbSnapshot,
  readProfitMethodAuditLatest,
  readProfitResearchLatest,
  readRiskTruthPanel,
  readSiteFinanceSnapshot,
} from "@/lib/server/data";
import {
  humanizeEngineState,
  humanizeMethodology,
  humanizeRiskLevel,
  humanizeStatusWord,
  humanizeStrategyName,
} from "@/lib/enginePresentation";

type GenericRow = Record<string, unknown>;

function asObj(value: unknown): GenericRow {
  return value && typeof value === "object" ? (value as GenericRow) : {};
}

function toNum(value: unknown, fallback: number | null = null): number | null {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function toText(value: unknown, fallback = ""): string {
  return typeof value === "string" ? value : fallback;
}

function lc(value: unknown): string {
  return toText(value).toLowerCase();
}

function yesNo(value: boolean): string {
  return value ? "sim" : "não";
}

function pctText(value: number | null | undefined, digits = 0): string {
  return typeof value === "number" && Number.isFinite(value) ? `${(value * 100).toFixed(digits)}%` : "--";
}

async function readJsonFile<T>(target: string, fallback: T): Promise<T> {
  try {
    const raw = await fs.readFile(target, "utf-8");
    try {
      return JSON.parse(raw) as T;
    } catch {
      return JSON.parse(
        raw
          .replace(/\bNaN\b/g, "null")
          .replace(/\bInfinity\b/g, "null")
          .replace(/\b-Infinity\b/g, "null")
      ) as T;
    }
  } catch {
    return fallback;
  }
}

function statusCountsFromRecords(rows: GenericRow[]) {
  const out = { assets: rows.length, validated: 0, watch: 0, inconclusive: 0 };
  for (const row of rows) {
    const status = lc(row.signal_status || row.status);
    if (status === "validated") out.validated += 1;
    else if (status === "watch") out.watch += 1;
    else out.inconclusive += 1;
  }
  return out;
}

function sampleAssets(rows: GenericRow[], desiredStatus: "watch" | "inconclusive", limit = 6) {
  return rows
    .filter((row) => lc(row.signal_status || row.status) === desiredStatus)
    .map((row) => ({
      asset: toText(row.asset, "--"),
      confidence: toNum(row.confidence, 0) ?? 0,
      quality: toNum(row.quality, 0) ?? 0,
    }))
    .sort((a, b) => a.confidence - b.confidence)
    .slice(0, limit);
}

function runIdFromRunDir(runDir: string): string {
  const clean = String(runDir || "").trim();
  if (!clean) return "";
  const normalized = clean.replace(/\\/g, "/").replace(/\/+$/, "");
  const parts = normalized.split("/");
  const last = parts[parts.length - 1] || "";
  return /^20\d{6}T\d{6}Z$/.test(last) ? last : "";
}

function statusCountsFromLabAssetDiagnostics(rows: GenericRow[]) {
  const out = { assets: rows.length, validated: 0, watch: 0, inconclusive: 0 };
  for (const row of rows) {
    const regime = lc(row.regime_asset);
    if (regime === "estavel" || regime === "stable") out.validated += 1;
    else if (regime === "transicao" || regime === "transition") out.watch += 1;
    else out.inconclusive += 1;
  }
  return out;
}

async function readCopilotShadow(runId: string | null) {
  const { results } = dataDirs();
  const root = path.join(results, "ops", "copilot");
  if (runId) {
    const byRun = path.join(root, runId, "shadow_summary.json");
    const payload = await readJsonFile<GenericRow | null>(byRun, null);
    if (payload) return payload;
  }

  const latest = await readJsonFile<GenericRow | null>(path.join(root, "latest_release.json"), null);
  const latestPath = toText(latest?.shadow_summary, "");
  if (!latestPath) return null;
  return readJsonFile<GenericRow | null>(latestPath, null);
}

async function readInstructionCoreVersion() {
  const repoRoot = path.resolve(process.cwd(), "..");
  const cfg = await readJsonFile<GenericRow>(
    path.join(repoRoot, "config", "copilot_instruction_core.v1.json"),
    {}
  );
  return {
    version: toText(cfg.version, "unknown"),
    name: toText(asObj(cfg.identity).name, "Eigen Engine Assistant"),
    statement: toText(asObj(cfg.identity).statement, ""),
  };
}

async function readOperationalBrief() {
  const { results } = dataDirs();
  const root = path.join(results, "ops", "ai_knowledge");
  const latest = await readJsonFile<GenericRow | null>(path.join(root, "latest_operational_brief.json"), null);

  const fromPointer = toText(latest?.operational_brief_path, "");
  if (fromPointer) {
    const payload = await readJsonFile<GenericRow | null>(fromPointer, null);
    if (payload) return { latest, payload, path: fromPointer };
  }

  try {
    const files = (await fs.readdir(root))
      .filter((name) => name.startsWith("operational_brief_") && name.endsWith(".json"))
      .sort()
      .reverse();
    if (!files.length) return null;
    const candidate = path.join(root, files[0]);
    const payload = await readJsonFile<GenericRow | null>(candidate, null);
    if (!payload) return null;
    return { latest, payload, path: candidate };
  } catch {
    return null;
  }
}

type FinanceProductReady = {
  available: boolean;
  overall_readiness: string;
  data_last_date: string;
  risk_level_next_month: string;
  operational_state: string;
  confidence_score: number | null;
  warnings: string[];
  report_path: string;
};

async function readFinanceProductReady(): Promise<FinanceProductReady> {
  const { results } = dataDirs();
  const latestPath = path.join(results, "ops", "finance_product_ready", "latest_finance_product_ready.json");
  const latest = await readJsonFile<GenericRow | null>(latestPath, null);
  const reportPath = toText(latest?.finance_product_ready_json, "");
  if (!reportPath) {
    return {
      available: false,
      overall_readiness: "missing",
      data_last_date: "",
      risk_level_next_month: "",
      operational_state: "",
      confidence_score: null,
      warnings: [],
      report_path: "",
    };
  }
  const report = await readJsonFile<GenericRow | null>(reportPath, null);
  if (!report) {
    return {
      available: false,
      overall_readiness: "missing",
      data_last_date: "",
      risk_level_next_month: "",
      operational_state: "",
      confidence_score: null,
      warnings: [],
      report_path: reportPath,
    };
  }
  const warnings = Array.isArray(report.warnings) ? report.warnings.map((v) => String(v)) : [];
  return {
    available: true,
    overall_readiness: toText(report.overall_readiness, "unknown"),
    data_last_date: toText(report.data_last_date, ""),
    risk_level_next_month: toText(report.risk_level_next_month, ""),
    operational_state: toText(report.operational_state, ""),
    confidence_score: toNum(report.confidence_score),
    warnings,
    report_path: reportPath,
  };
}

type CopilotContext = {
  generated_at_utc: string;
  assistant: {
    name: string;
    role: string;
  };
  run: {
    id: string;
    status: string;
    gate_blocked: boolean;
    gate_reasons: string[];
    policy: string;
    window_days: number | null;
  };
  universe: {
    assets: number;
    validated: number;
    watch: number;
    inconclusive: number;
  };
  lab: {
    run_id: string;
    regime: string;
    signal_tier: string;
    signal_reliability: number | null;
    structure_score: number | null;
    n_used: number | null;
    n_events_60d: number;
  };
  model_b: {
    status: string;
    detail: string;
    regime: string;
    risk_score: number | null;
    confidence: number | null;
    mode: string;
  };
  model_c: {
    status: string;
    detail: string;
    regime: string;
    risk_score: number | null;
    confidence: number | null;
    mode: string;
    publish_ready: boolean;
    reasons: string[];
  };
  governance: {
    publishable: boolean;
    risk_structural: number | null;
    confidence: number | null;
    risk_level: string;
    publish_blockers: string[];
  };
  instruction_core: {
    version: string;
    name: string;
    statement: string;
  };
  platform_db: {
    status: string;
    run_id: string;
    rows_for_run: number;
    runs_total: number;
    db_path: string;
    copilot_row_exists: boolean;
  };
  operational_brief: {
    brief_available: boolean;
    brief_path: string;
    data_last_date: string;
    freshness_status: string;
    freshness_days_lag: number | null;
    risk_level_next_month: string;
    operational_state: string;
    action_hint: string;
    confidence_score: number | null;
    allocation_mode: string;
    target_exposure: number | null;
    target_exposure_min: number | null;
    target_exposure_max: number | null;
    profit_reinforcement_enabled: boolean;
    top_sector_global: string;
    top_asset_global: string;
    insight_headlines: string[];
  };
  domain_scenarios: {
    finance: FinanceProductReady;
  };
  profit_research: {
    available: boolean;
    top_candidate: string;
    top_methodology: string;
    top_net_ann_return: number | null;
    oos_candidate: string;
    oos_mean_test_net_ann_return: number | null;
    promotable_now: boolean;
    audit_findings: string[];
    keep_count: number;
    watch_count: number;
    kill_count: number;
    registry_path: string;
    insight_headlines: string[];
    pattern_headlines: string[];
    event_count: number;
  };
  improvement_backlog: string[];
  watch_assets: Array<{ asset: string; confidence: number; quality: number }>;
  inconclusive_assets: Array<{ asset: string; confidence: number; quality: number }>;
  sources: string[];
};

export async function buildCopilotContext(): Promise<CopilotContext> {
  const [
    run,
    snap,
    panel,
    labRun,
    labTs,
    playbook,
    alerts,
    instruction,
    platformSnapshot,
    opBrief,
    financeReady,
    labAssetDiagnostics,
    profitResearch,
    profitMethodAudit,
    siteSnapshot,
  ] =
    await Promise.all([
      findLatestValidRun(),
      readLatestSnapshot(),
      readRiskTruthPanel(),
      findLatestLabCorrRun(),
      readLatestLabCorrTimeseries(120),
      readLatestLabCorrActionPlaybook(120),
      readLatestLabCorrOperationalAlerts(120),
      readInstructionCoreVersion(),
      readPlatformDbSnapshot(),
      readOperationalBrief(),
      readFinanceProductReady(),
      readLatestLabCorrAssetDiagnostics(2500),
      readProfitResearchLatest(),
      readProfitMethodAuditLatest(),
      readSiteFinanceSnapshot(),
    ]);

  const shadow = await readCopilotShadow(run?.runId || null);
  const runSummary = asObj(run?.summary);
  const runGate = asObj(runSummary.deployment_gate);
  const runReasons = Array.isArray(runGate.reasons) ? runGate.reasons.map((v) => String(v)) : [];
  const runBlocked = runGate.blocked === true;

  const rows = Array.isArray(snap?.records) ? (snap.records as GenericRow[]) : [];
  const fallbackCounts = statusCountsFromRecords(rows);
  const labCounts = statusCountsFromLabAssetDiagnostics(
    Array.isArray(labAssetDiagnostics) ? (labAssetDiagnostics as GenericRow[]) : []
  );
  const siteUniverse = Array.isArray((siteSnapshot as GenericRow)?.current_universe)
    ? ((siteSnapshot as GenericRow).current_universe as GenericRow[])
    : [];
  const siteCounts = statusCountsFromRecords(siteUniverse);
  const panelCounts = asObj(asObj(panel).counts);
  let universe = {
    assets: Number(toNum(panelCounts.assets, fallbackCounts.assets || siteCounts.assets) || 0),
    validated: Number(toNum(panelCounts.validated, fallbackCounts.validated || siteCounts.validated) || 0),
    watch: Number(toNum(panelCounts.watch, fallbackCounts.watch || siteCounts.watch) || 0),
    inconclusive: Number(toNum(panelCounts.inconclusive, fallbackCounts.inconclusive || siteCounts.inconclusive) || 0),
  };
  if (universe.assets <= 0 && labCounts.assets > 0) {
    universe = labCounts;
  }
  if (universe.assets <= 0 && siteCounts.assets > 0) {
    universe = siteCounts;
  }

  const playbookRows = Array.isArray(playbook) ? (playbook as GenericRow[]) : [];
  const latestPlay = playbookRows.length ? playbookRows[playbookRows.length - 1] : {};
  const latestState = asObj(labTs?.latest);
  const alertObj = asObj(alerts);

  const shadowModelB = asObj(shadow?.model_b);
  const shadowModelC = asObj(shadow?.model_c);
  const shadowFusion = asObj(shadow?.fusion);
  const shadowRun = asObj(shadow?.run);
  const opPayload = asObj(opBrief?.payload);
  const opRunContext = asObj(opPayload.run_context);
  const opRunDir = toText(opRunContext.run_dir, "");
  const opRunId = runIdFromRunDir(opRunDir);
  const opFreshness = asObj(opPayload.freshness);
  const opSignal = asObj(opPayload.operational_signal);
  const opAlloc = asObj(opPayload.allocation_policy);
  const opAllocSignals = asObj(opAlloc.signals);
  const opSnapshot = asObj(opPayload.state_snapshot);
  const opTopSectors = Array.isArray(opSnapshot.top_sectors_global_mode)
    ? (opSnapshot.top_sectors_global_mode as GenericRow[])
    : [];
  const opTopAssets = Array.isArray(opSnapshot.top_assets_global_mode)
    ? (opSnapshot.top_assets_global_mode as GenericRow[])
    : [];
  const opTopSector = opTopSectors.length ? toText(asObj(opTopSectors[0]).sector, "--") : "--";
  const opTopAsset = opTopAssets.length
    ? toText(asObj(opTopAssets[0]).ticker || asObj(opTopAssets[0]).asset_id, "--")
    : "--";
  const opInsights = Array.isArray(opPayload.insights) ? (opPayload.insights as GenericRow[]) : [];
  const opInsightHeadlines = opInsights
    .map((row) => toText(asObj(row).message, "").trim())
    .filter((txt) => txt.length > 0)
    .slice(0, 4);
  const siteFinance = asObj(asObj(siteSnapshot).finance);
  const sitePlaybook = asObj(siteFinance.latest_playbook);
  const siteResearch = asObj(asObj(siteSnapshot).profit_research);
  const siteTopCandidate = asObj(siteResearch.top_candidate);
  const siteLayered = asObj(asObj(siteSnapshot).layered_engine);
  const siteAttack = asObj(siteLayered.best_meta_candidate);
  const sitePatterns = Array.isArray(asObj(siteResearch.patterns).pattern_headlines)
    ? (asObj(siteResearch.patterns).pattern_headlines as unknown[]).map((v) => String(v)).slice(0, 4)
    : Array.isArray(siteResearch.pattern_headlines)
      ? (siteResearch.pattern_headlines as unknown[]).map((v) => String(v)).slice(0, 4)
      : [];

  const publishBlockers = Array.isArray(shadowFusion.publish_blockers)
    ? shadowFusion.publish_blockers.map((v) => String(v))
    : [];

  const improvementBacklog = [
    "manter baseline estrutural causal para finanças e cripto com orçamento de risco fixo",
    "comparar novos sleeves e overlays só com benchmark explícito, custo e delay",
    "promover alpha novo apenas se sobreviver em walk-forward e shadow",
  ];
  const profitResearchObj = asObj(profitResearch);
  const profitTop = asObj(profitResearchObj.top_candidate);
  const profitOos = asObj(profitResearchObj.oos_best_consistent);
  const profitStatusCounts = asObj(profitResearchObj.status_counts);
  const profitAudit = asObj(profitMethodAudit);
  const profitAuditFindings = Array.isArray(profitAudit.findings)
    ? profitAudit.findings
        .map((v) => asObj(v))
        .map((row) => toText(row.message, "").trim())
        .filter((txt) => txt.length > 0)
        .slice(0, 3)
    : [];
  const profitInsights = Array.isArray(profitResearchObj.insights)
    ? profitResearchObj.insights.map((v) => String(v)).slice(0, 4)
    : [];
  const profitPatterns = asObj(profitResearchObj.patterns);
  const profitPatternHeadlines = Array.isArray(profitPatterns.pattern_headlines)
    ? profitPatterns.pattern_headlines.map((v) => String(v)).slice(0, 4)
    : [];

  const context: CopilotContext = {
    generated_at_utc: new Date().toISOString(),
    assistant: {
      name: toText(instruction.name, "Eigen Engine Assistant"),
      role: "copiloto_tecnico_investimentos",
    },
    run: {
      id: run?.runId || opRunId || toText(shadowRun.run_id, "no_valid_run"),
      status: run ? "ok" : opRunId ? "latest_lab_context" : toText(shadowRun.status, "missing"),
      gate_blocked: runBlocked,
      gate_reasons: runReasons,
      policy: toText(runSummary.policy_path, toText(shadowRun.policy_path, "production_policy_lock.json")),
      window_days: toNum(runSummary.official_window, toNum(shadowRun.official_window)),
    },
    universe,
    lab: {
      run_id: labRun?.runId || "no_lab_corr_run",
      regime: toText(latestPlay.regime, toText(sitePlaybook.regime, "--")),
      signal_tier: toText(latestPlay.signal_tier, "--"),
      signal_reliability: toNum(latestPlay.signal_reliability, toNum(sitePlaybook.signal_reliability)),
      structure_score: toNum(latestState.structure_score),
      n_used: toNum(latestState.N_used, toNum(sitePlaybook.N_used)),
      n_events_60d: Number(toNum(alertObj.n_events_last_60d, 0) || 0),
    },
    model_b: {
      status: shadow ? "shadow_ativo" : "fallback",
      detail: shadow
        ? "Modelo B em shadow mode com artefato operacional por run."
        : "Shadow de B não encontrado para este run; usando fallback.",
      regime: toText(shadowModelB.predicted_regime, "transition"),
      risk_score: toNum(shadowModelB.risk_score),
      confidence: toNum(shadowModelB.probability),
      mode: toText(shadowModelB.mode, shadow ? "shadow" : "fallback"),
    },
    model_c: {
      status: shadow ? toText(shadowModelC.status, "shadow") : "fallback",
      detail: shadow
        ? "Modelo C acoplado ao mesmo fluxo de gate (shadow proxy)."
        : "Shadow de C não encontrado para este run; usando fallback.",
      regime: toText(shadowModelC.regime, "indefinido"),
      risk_score: toNum(shadowModelC.risk_score),
      confidence: toNum(shadowModelC.confidence),
      mode: toText(shadowModelC.mode, shadow ? "shadow" : "fallback"),
      publish_ready: shadowModelC.publish_ready === true,
      reasons: Array.isArray(shadowModelC.reasons) ? shadowModelC.reasons.map((v) => String(v)) : [],
    },
    governance: {
      publishable: shadow ? shadowFusion.publishable === true && !runBlocked : !runBlocked,
      risk_structural: toNum(shadowFusion.risk_structural),
      confidence: toNum(shadowFusion.confidence),
      risk_level: toText(shadowFusion.risk_level, "indefinido"),
      publish_blockers: shadow
        ? [...publishBlockers, ...runReasons.filter((r) => !publishBlockers.includes(r))]
        : [...runReasons],
    },
    instruction_core: instruction,
    platform_db: {
      status: toText(asObj(platformSnapshot).status, "missing"),
      run_id: toText(asObj(platformSnapshot).run_id, ""),
      rows_for_run: Number(toNum(asObj(asObj(platformSnapshot).counts).asset_rows_for_run, 0) || 0),
      runs_total: Number(toNum(asObj(asObj(platformSnapshot).counts).runs_total, 0) || 0),
      db_path: toText(asObj(platformSnapshot).db_path, ""),
      copilot_row_exists: asObj(asObj(platformSnapshot).copilot).row_exists === true,
    },
    operational_brief: {
      brief_available: !!opBrief,
      brief_path: toText(opBrief?.path, ""),
      data_last_date: toText(opPayload.data_last_date, ""),
      freshness_status: toText(opFreshness.status, "unknown"),
      freshness_days_lag: toNum(opFreshness.days_lag),
      risk_level_next_month: toText(opSignal.risk_level_next_month, toText(siteFinance.risk_level_next_month, "unknown")),
      operational_state: toText(opSignal.operational_state, toText(siteFinance.operational_state, "monitoramento_normal")),
      action_hint: toText(opSignal.action_hint, "manter monitoramento estrutural"),
      confidence_score: toNum(opSignal.confidence_score, toNum(siteFinance.confidence_score)),
      allocation_mode: toText(opAlloc.mode, toText(opSignal.allocation_mode, "equilibrado")),
      target_exposure: toNum(opAlloc.target_exposure, toNum(opSignal.target_exposure, toNum(sitePlaybook.exposure))),
      target_exposure_min: toNum(opAlloc.range_min, toNum(opSignal.target_exposure_min)),
      target_exposure_max: toNum(opAlloc.range_max, toNum(opSignal.target_exposure_max)),
      profit_reinforcement_enabled:
        opAlloc.profit_reinforcement_enabled === true || (toNum(opAllocSignals.alpha_recent6, 0) ?? 0) > 0,
      top_sector_global: opTopSector,
      top_asset_global: opTopAsset,
      insight_headlines: opInsightHeadlines.length ? opInsightHeadlines : sitePatterns,
    },
    domain_scenarios: {
      finance: {
        available: financeReady.available,
        overall_readiness: financeReady.overall_readiness,
        data_last_date: financeReady.data_last_date || toText(opPayload.data_last_date, ""),
        risk_level_next_month: financeReady.risk_level_next_month || toText(opSignal.risk_level_next_month, ""),
        operational_state: financeReady.operational_state || toText(opSignal.operational_state, "monitoramento_normal"),
        confidence_score: financeReady.confidence_score ?? toNum(opSignal.confidence_score),
        warnings: financeReady.warnings,
        report_path: financeReady.report_path,
      },
    },
    profit_research: {
      available: toText(profitResearchObj.status, "missing") === "ok",
      top_candidate: toText(profitTop.candidate_id, toText(siteTopCandidate.candidate_id, "--")),
      top_methodology: toText(profitTop.methodology, toText(siteTopCandidate.methodology, "--")),
      top_net_ann_return: toNum(profitTop.net_ann_return, toNum(siteTopCandidate.net_ann_return)),
      oos_candidate: toText(profitOos.candidate_id, "--"),
      oos_mean_test_net_ann_return: toNum(profitOos.mean_test_net_ann_return),
      promotable_now: asObj(profitAudit.verdict).promotable_now === true,
      audit_findings: profitAuditFindings,
      keep_count: Number(toNum(profitStatusCounts.keep, 0) || 0),
      watch_count: Number(toNum(profitStatusCounts.watch, 0) || 0),
      kill_count: Number(toNum(profitStatusCounts.kill, 0) || 0),
      registry_path: toText(profitResearchObj.registry_path, ""),
      insight_headlines: profitInsights.length ? profitInsights : sitePatterns,
      pattern_headlines: profitPatternHeadlines.length ? profitPatternHeadlines : sitePatterns,
      event_count: Number(toNum(profitPatterns.event_count, 0) || 0),
    },
    improvement_backlog: improvementBacklog,
    watch_assets: sampleAssets(rows.length ? rows : siteUniverse, "watch", 6),
    inconclusive_assets: sampleAssets(rows.length ? rows : siteUniverse, "inconclusive", 6),
    sources: shadow
      ? [
          ...(Array.isArray(shadow.sources) ? shadow.sources.map((v) => String(v)) : []),
          ...(opBrief?.path ? [String(opBrief.path)] : []),
        ]
      : [
          `results/ops/snapshots/${run?.runId || "N_A"}/summary.json`,
          `results/ops/snapshots/${run?.runId || "N_A"}/api_snapshot.jsonl`,
          "results/validation/risk_truth_panel.json",
          `results/lab_corr_macro/${labRun?.runId || "N_A"}/summary.json`,
          ...(opBrief?.path ? [String(opBrief.path)] : []),
        ],
  };

  return context;
}

function withPublishGuard(message: string, ctx: CopilotContext): string {
  if (!ctx.run.gate_blocked) return message;
  const reasons = ctx.run.gate_reasons.length ? ctx.run.gate_reasons.join(", ") : "gate_or_integrity";
  return `Aviso importante: a publicação oficial está bloqueada por gate.\nMotivos atuais: ${reasons}.\n\n${message}`;
}

function renderSaudacao(ctx: CopilotContext): string {
  return withPublishGuard(
    [
      "Oi. Eu sou o copiloto do Eigen Engine.",
      "",
      "Eu te ajudo a entender o que o motor está vendo agora, quanto risco ele suporta, quando faz mais sentido olhar finanças ou cripto e quando a resposta certa é simplesmente operar menor.",
      "",
      "Se quiser, você pode me perguntar assim:",
      "- o que você faz",
      "- como usar o motor hoje",
      "- quanto vai para risco",
      "- finanças ou cripto agora",
    ].join("\n"),
    ctx
  );
}

function renderCapabilities(ctx: CopilotContext): string {
  const exposure =
    ctx.operational_brief.target_exposure == null
      ? "sem faixa limpa agora"
      : `${Math.round(ctx.operational_brief.target_exposure * 100)}% de exposição alvo`;
  return withPublishGuard(
    [
      "Eu não sou um chat genérico. Eu sou um copiloto focado no motor.",
      "",
      "Na prática, eu faço 4 coisas:",
      "1. Traduzo o estado do Eigen Engine para português normal.",
      "2. Mostro se hoje o contexto está mais favorável ou mais perigoso.",
      "3. Explico como usar a faixa de risco e exposição sem teatrinho quantitativo.",
      "4. Te digo quando algo ainda é só pesquisa e não merece confiança operacional.",
      "",
      `Hoje, por exemplo, a leitura operacional está em ${humanizeEngineState(ctx.operational_brief.operational_state || "monitoramento_normal")} com ${exposure}.`,
    ].join("\n"),
    ctx
  );
}

function renderHowToUse(ctx: CopilotContext): string {
  const exposure =
    ctx.operational_brief.target_exposure == null
      ? null
      : Math.round(ctx.operational_brief.target_exposure * 100);
  const risk = humanizeRiskLevel(ctx.operational_brief.risk_level_next_month || "unknown");
  const confidence = pctText(ctx.operational_brief.confidence_score);
  const action = ctx.operational_brief.action_hint || "manter monitoramento estrutural";
  return withPublishGuard(
    [
      "Use o app nesta ordem:",
      "1. Veja o estado do motor.",
      "2. Veja o risco do próximo mês.",
      "3. Só depois olhe a exposição sugerida.",
      "4. Se a confiança estiver fraca ou o gate estiver bloqueado, opere menor ou não opere.",
      "",
      `Hoje eu resumiria assim: risco ${risk}, confiança ${confidence}, ação prática '${action}'.`,
      exposure == null
        ? "Hoje eu não vejo uma faixa limpa de exposição publicada."
        : `Hoje a faixa central publicada gira em torno de ${exposure}% em risco.`,
      "",
      "Se você quiser, eu também posso transformar isso num exemplo com R$ 1 mil, R$ 10 mil ou R$ 50 mil.",
    ].join("\n"),
    ctx
  );
}

function renderResumo(ctx: CopilotContext): string {
  const blocked = ctx.run.gate_blocked;
  const exposure =
    ctx.operational_brief.target_exposure == null
      ? "--"
      : `${Math.round(ctx.operational_brief.target_exposure * 100)}%`;
  const topResearch = humanizeStrategyName(ctx.profit_research.top_candidate || "--");
  const regime = humanizeEngineState(ctx.lab.regime || "monitoramento_normal");
  const risk = humanizeRiskLevel(ctx.operational_brief.risk_level_next_month || "unknown");
  const intro = blocked
    ? "Hoje eu usaria o app em modo diagnóstico. O motor ainda ajuda a entender o contexto, mas não está em trilha limpa para confiar cegamente."
    : "Hoje a trilha do motor está mais íntegra. Ainda não é certeza de lucro, mas dá para usar a leitura como apoio de decisão.";
  return withPublishGuard(
    [
      "Resumo rápido do motor:",
      "",
      intro,
      "",
      `O quadro de agora é este: risco para o próximo mês em ${risk}, exposição alvo perto de ${exposure} e estado ${humanizeEngineState(ctx.operational_brief.operational_state || "monitoramento_normal")}.`,
      `No laboratório, o modo mais forte segue ${topResearch} e o regime principal continua em ${regime}.`,
      `Temos ${ctx.universe.assets} ativos no universo atual, com ${ctx.universe.validated} validados e ${ctx.universe.watch} em observação.`,
      "",
      "Se você quiser, eu posso te responder de forma bem direta sobre risco, exposição, finanças, cripto ou pesquisa.",
    ].join("\n"),
    ctx
  );
}

function renderProfitResearch(ctx: CopilotContext): string {
  const insights = ctx.profit_research.insight_headlines.length
    ? ctx.profit_research.insight_headlines.map((line) => `  - ${line}`).join("\n")
    : "  - sem insights consolidados";
  const patterns = ctx.profit_research.pattern_headlines.length
    ? ctx.profit_research.pattern_headlines.map((line) => `  - ${line}`).join("\n")
    : "  - sem padroes consolidados";
  const audit = ctx.profit_research.audit_findings.length
    ? ctx.profit_research.audit_findings.map((line) => `  - ${line}`).join("\n")
    : "  - sem findings de auditoria";
  return withPublishGuard(
    [
      "Pesquisa de lucro consolidada:",
      `- Disponível: ${yesNo(ctx.profit_research.available)}.`,
      `- Topo atual: ${humanizeStrategyName(ctx.profit_research.top_candidate)}.`,
      `- Método: ${humanizeMethodology(ctx.profit_research.top_methodology)}.`,
      `- Retorno líquido anual do topo: ${pctText(ctx.profit_research.top_net_ann_return)}.`,
      `- Candidato OOS mais consistente: ${humanizeStrategyName(ctx.profit_research.oos_candidate)} com média de ${pctText(ctx.profit_research.oos_mean_test_net_ann_return)} ao ano.`,
      `- Promovível agora: ${yesNo(ctx.profit_research.promotable_now)}.`,
      `- Keep: ${ctx.profit_research.keep_count} | observação: ${ctx.profit_research.watch_count} | matar: ${ctx.profit_research.kill_count}.`,
      `- Eventos registrados: ${ctx.profit_research.event_count}.`,
      `- Registry: ${ctx.profit_research.registry_path || "--"}.`,
      "- Insights:",
      insights,
      "- Padrões recentes:",
      patterns,
      "- Auditoria:",
      audit,
      "- Leitura correta: pesquisa de alpha e alocação, com custos e impostos em proxy explícita; não é promessa de retorno.",
    ].join("\n"),
    ctx
  );
}

function renderGate(ctx: CopilotContext): string {
  const reasons = ctx.governance.publish_blockers.length ? ctx.governance.publish_blockers.join(", ") : "nenhum";
  return withPublishGuard(
    [
      "Diagnóstico de publicação:",
      `- Gate bloqueado: ${yesNo(ctx.run.gate_blocked)}.`,
      `- Publicável: ${yesNo(ctx.governance.publishable)}.`,
      `- Motivos: ${reasons}.`,
      `- Janela oficial: ${ctx.run.window_days ?? "--"} dias; política ativa: ${ctx.run.policy}.`,
      "- Regra operacional: se a publicação não passa, a resposta fica em modo diagnóstico.",
    ].join("\n"),
    ctx
  );
}

function renderOperationalBrief(ctx: CopilotContext): string {
  const lag = ctx.operational_brief.freshness_days_lag == null ? "--" : String(ctx.operational_brief.freshness_days_lag);
  const insight = ctx.operational_brief.insight_headlines[0] || "sem destaque sintético publicado agora";
  const exposure =
    ctx.operational_brief.target_exposure == null
      ? "sem faixa limpa"
      : `${Math.round(ctx.operational_brief.target_exposure * 100)}%`;
  return withPublishGuard(
    [
      `Hoje o motor está em ${humanizeEngineState(ctx.operational_brief.operational_state || "monitoramento_normal")}, com risco do próximo mês em ${humanizeRiskLevel(ctx.operational_brief.risk_level_next_month || "unknown")}.`,
      `A faixa de exposição alvo está perto de ${exposure}, com dados até ${ctx.operational_brief.data_last_date || "--"} e frescor ${ctx.operational_brief.freshness_status} (${lag} dias).`,
      `A ação prática sugerida pelo motor é: ${ctx.operational_brief.action_hint}.`,
      `O destaque do momento é ${ctx.operational_brief.top_asset_global || "sem ativo dominante limpo"} no setor ${ctx.operational_brief.top_sector_global || "sem setor dominante limpo"}.`,
      `Insight curto: ${insight}.`,
      `Confiança publicada agora: ${pctText(ctx.operational_brief.confidence_score)}.`,
      "",
      "Leitura correta: usar isso como contexto para tamanho de posição e disciplina de risco, não como promessa direcional de preço.",
    ].join("\n"),
    ctx
  );
}

function renderCausal(ctx: CopilotContext): string {
  return withPublishGuard(
    [
      "Checagem causal e integridade:",
      "- O copiloto lê artefatos do run e do painel de validação, sem recalibrar histórico durante a resposta.",
      `- O brief operacional e lido de results/ops/ai_knowledge/latest_operational_brief.json (quando disponivel).`,
      `- Política declarada no run: ${ctx.run.policy}.`,
      `- Nucleo de instrucoes ativo: ${ctx.instruction_core.version}.`,
      "- Se houver falha de gate ou integridade, o copiloto assume modo diagnóstico.",
    ].join("\n"),
    ctx
  );
}

function renderAssets(ctx: CopilotContext): string {
  const watch = ctx.watch_assets.length
    ? ctx.watch_assets.map((x) => `${x.asset} (c=${x.confidence.toFixed(3)}, q=${x.quality.toFixed(3)})`).join(", ")
    : "sem ativos em watch";
  const inc = ctx.inconclusive_assets.length
    ? ctx.inconclusive_assets
        .map((x) => `${x.asset} (c=${x.confidence.toFixed(3)}, q=${x.quality.toFixed(3)})`)
        .join(", ")
    : "sem ativos inconclusive";
  return withPublishGuard(
    [
      "Amostra de ativos para monitorar:",
      `- Watch: ${watch}.`,
      `- Inconclusive: ${inc}.`,
      "- Priorize queda de confiança e mudança de regime para acionar revisão operacional.",
    ].join("\n"),
    ctx
  );
}

function renderModels(ctx: CopilotContext): string {
  return withPublishGuard(
    [
      "Status dos modelos B e C:",
      `- B: status=${ctx.model_b.status}, modo=${ctx.model_b.mode}, regime=${ctx.model_b.regime}, risco=${ctx.model_b.risk_score ?? "--"}, conf=${ctx.model_b.confidence ?? "--"}.`,
      `- C: status=${ctx.model_c.status}, modo=${ctx.model_c.mode}, regime=${ctx.model_c.regime}, risco=${ctx.model_c.risk_score ?? "--"}, conf=${ctx.model_c.confidence ?? "--"}, publish_ready=${yesNo(ctx.model_c.publish_ready)}.`,
      "- Fluxo: A em produção + B/C em shadow com bloqueio de publicação por gate e integridade.",
    ].join("\n"),
    ctx
  );
}

function renderDomainScenarios(ctx: CopilotContext): string {
  const f = ctx.domain_scenarios.finance;
  return withPublishGuard(
    [
      `Finanças estão em ${humanizeEngineState(f.operational_state || "monitoramento_normal")} com risco ${humanizeRiskLevel(f.risk_level_next_month || "unknown")} e data-base ${f.data_last_date || "--"}.`,
      `Na pesquisa, o topo atual é ${humanizeStrategyName(ctx.profit_research.top_candidate || "--")} com método ${humanizeMethodology(ctx.profit_research.top_methodology || "--")} e retorno líquido anual de ${pctText(ctx.profit_research.top_net_ann_return)}.`,
      `O estado operacional combina isso com exposição alvo de ${pctText(ctx.operational_brief.target_exposure)} e ativo líder ${ctx.operational_brief.top_asset_global || "--"}.`,
      "",
      "Uso correto: entender qual sleeve está mais forte e qual faixa de risco faz sentido hoje. Não é um oráculo de preço.",
    ].join("\n"),
    ctx
  );
}

function renderImprovementPlan(ctx: CopilotContext): string {
  const items = ctx.improvement_backlog.length
    ? ctx.improvement_backlog.map((line, idx) => `- ${idx + 1}. ${line}`).join("\n")
    : "- backlog vazio";
  return withPublishGuard(
    [
      "Plano de melhoria (ML/DL com rigor causal):",
      "- Treino em janela fixa, teste só no futuro, walk-forward por blocos de tempo.",
      "- Sempre comparar contra alerta aleatório com mesma taxa de alertas.",
      "- Promover modelo novo só com ganho estável em vários blocos; se não houver, manter baseline estrutural.",
      items,
    ].join("\n"),
    ctx
  );
}

function renderGreetingFallback(ctx: CopilotContext): string {
  return withPublishGuard(
    [
      "Entendi a pergunta, mas vou te responder do jeito mais útil para o produto.",
      "",
      `Hoje o motor está em ${humanizeEngineState(ctx.operational_brief.operational_state || "monitoramento_normal")} e a faixa de risco gira em torno de ${pctText(ctx.operational_brief.target_exposure)}.`,
      "Se você quiser, me pergunta em linguagem simples: 'o que eu faço hoje', 'quanto vai para risco', 'finanças ou cripto', 'o que você faz' ou 'me explica sem economês'.",
    ].join("\n"),
    ctx
  );
}

export function buildCopilotReply(question: string, ctx: CopilotContext): string {
  const q = question.trim().toLowerCase();
  if (!q) return renderSaudacao(ctx);

  if (
    q === "oi" ||
    q === "olá" ||
    q === "ola" ||
    q === "bom dia" ||
    q === "boa tarde" ||
    q === "boa noite" ||
    q === "e ai" ||
    q === "e aí"
  ) {
    return renderSaudacao(ctx);
  }

  if (
    q.includes("o que você faz") ||
    q.includes("o que vc faz") ||
    q.includes("quem é você") ||
    q.includes("quem e voce") ||
    q.includes("quem é vc") ||
    q.includes("quem e vc")
  ) {
    return renderCapabilities(ctx);
  }

  if (
    q.includes("como usar") ||
    q.includes("como eu uso") ||
    q.includes("como usar o app") ||
    q.includes("como usar o motor") ||
    q.includes("me ajuda a usar")
  ) {
    return renderHowToUse(ctx);
  }

  if (
    q.includes("estado") ||
    q.includes("operac") ||
    q.includes("acao") ||
    q.includes("brief") ||
    q.includes("insight") ||
    q.includes("expos") ||
    q.includes("lucro") ||
    q.includes("perda")
  )
    return renderOperationalBrief(ctx);
  if (q.includes("alpha") || q.includes("benchmark") || q.includes("grupo") || q.includes("pesquisa") || q.includes("metodologia"))
    return renderProfitResearch(ctx);
  if (q.includes("gate") || q.includes("public") || q.includes("bloque")) return renderGate(ctx);
  if (q.includes("causal") || q.includes("look") || q.includes("futuro") || q.includes("leak")) return renderCausal(ctx);
  if (q.includes("ativo") || q.includes("watch") || q.includes("inconclusive") || q.includes("setor")) return renderAssets(ctx);
  if (q.includes("finan") || q.includes("cripto") || q.includes("mercado") || q.includes("dominio")) return renderDomainScenarios(ctx);
  if (q.includes("ml") || q.includes("dl") || q.includes("deep learning") || q.includes("melhor") || q.includes("padrao"))
    return renderImprovementPlan(ctx);
  if (q.includes("modelo b") || q.includes("modelo c") || q.includes("gnn") || q.includes("rede neural"))
    return renderModels(ctx);

  return renderGreetingFallback(ctx);
}
