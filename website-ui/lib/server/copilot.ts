import { promises as fs } from "fs";
import path from "path";
import {
  dataDirs,
  findLatestLabCorrRun,
  findLatestValidRun,
  readLatestLabCorrActionPlaybook,
  readLatestLabCorrOperationalAlerts,
  readLatestLabCorrTimeseries,
  readLatestSnapshot,
  readPlatformDbSnapshot,
  readRiskTruthPanel,
} from "@/lib/server/data";

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
  return value ? "sim" : "nao";
}

async function readJsonFile<T>(target: string, fallback: T): Promise<T> {
  try {
    const raw = await fs.readFile(target, "utf-8");
    return JSON.parse(raw) as T;
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

type CopilotContext = {
  generated_at_utc: string;
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
    top_sector_global: string;
    top_asset_global: string;
    insight_headlines: string[];
  };
  watch_assets: Array<{ asset: string; confidence: number; quality: number }>;
  inconclusive_assets: Array<{ asset: string; confidence: number; quality: number }>;
  sources: string[];
};

export async function buildCopilotContext(): Promise<CopilotContext> {
  const [run, snap, panel, labRun, labTs, playbook, alerts, instruction, platformSnapshot, opBrief] = await Promise.all([
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
  ]);

  const shadow = await readCopilotShadow(run?.runId || null);
  const runSummary = asObj(run?.summary);
  const runGate = asObj(runSummary.deployment_gate);
  const runReasons = Array.isArray(runGate.reasons) ? runGate.reasons.map((v) => String(v)) : [];
  const runBlocked = runGate.blocked === true;

  const rows = Array.isArray(snap?.records) ? (snap.records as GenericRow[]) : [];
  const fallbackCounts = statusCountsFromRecords(rows);
  const panelCounts = asObj(asObj(panel).counts);
  const universe = {
    assets: Number(toNum(panelCounts.assets, fallbackCounts.assets) || 0),
    validated: Number(toNum(panelCounts.validated, fallbackCounts.validated) || 0),
    watch: Number(toNum(panelCounts.watch, fallbackCounts.watch) || 0),
    inconclusive: Number(toNum(panelCounts.inconclusive, fallbackCounts.inconclusive) || 0),
  };

  const playbookRows = Array.isArray(playbook) ? (playbook as GenericRow[]) : [];
  const latestPlay = playbookRows.length ? playbookRows[playbookRows.length - 1] : {};
  const latestState = asObj(labTs?.latest);
  const alertObj = asObj(alerts);

  const shadowModelB = asObj(shadow?.model_b);
  const shadowModelC = asObj(shadow?.model_c);
  const shadowFusion = asObj(shadow?.fusion);
  const shadowRun = asObj(shadow?.run);
  const opPayload = asObj(opBrief?.payload);
  const opFreshness = asObj(opPayload.freshness);
  const opSignal = asObj(opPayload.operational_signal);
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

  const publishBlockers = Array.isArray(shadowFusion.publish_blockers)
    ? shadowFusion.publish_blockers.map((v) => String(v))
    : [];

  const context: CopilotContext = {
    generated_at_utc: new Date().toISOString(),
    run: {
      id: run?.runId || toText(shadowRun.run_id, "no_valid_run"),
      status: run ? "ok" : toText(shadowRun.status, "missing"),
      gate_blocked: runBlocked,
      gate_reasons: runReasons,
      policy: toText(runSummary.policy_path, toText(shadowRun.policy_path, "production_policy_lock.json")),
      window_days: toNum(runSummary.official_window, toNum(shadowRun.official_window)),
    },
    universe,
    lab: {
      run_id: labRun?.runId || "no_lab_corr_run",
      regime: toText(latestPlay.regime, "--"),
      signal_tier: toText(latestPlay.signal_tier, "--"),
      signal_reliability: toNum(latestPlay.signal_reliability),
      structure_score: toNum(latestState.structure_score),
      n_used: toNum(latestState.N_used),
      n_events_60d: Number(toNum(alertObj.n_events_last_60d, 0) || 0),
    },
    model_b: {
      status: shadow ? "shadow_ativo" : "fallback",
      detail: shadow
        ? "Modelo B em shadow mode com artefato operacional por run."
        : "Shadow de B nao encontrado para este run; usando fallback.",
      regime: toText(shadowModelB.predicted_regime, "transition"),
      risk_score: toNum(shadowModelB.risk_score),
      confidence: toNum(shadowModelB.probability),
      mode: toText(shadowModelB.mode, shadow ? "shadow" : "fallback"),
    },
    model_c: {
      status: shadow ? toText(shadowModelC.status, "shadow") : "fallback",
      detail: shadow
        ? "Modelo C acoplado ao mesmo fluxo de gate (shadow proxy)."
        : "Shadow de C nao encontrado para este run; usando fallback.",
      regime: toText(shadowModelC.regime, "indefinido"),
      risk_score: toNum(shadowModelC.risk_score),
      confidence: toNum(shadowModelC.confidence),
      mode: toText(shadowModelC.mode, shadow ? "shadow" : "fallback"),
      publish_ready: shadowModelC.publish_ready === true,
      reasons: Array.isArray(shadowModelC.reasons) ? shadowModelC.reasons.map((v) => String(v)) : [],
    },
    governance: {
      publishable: shadowFusion.publishable === true && !runBlocked,
      risk_structural: toNum(shadowFusion.risk_structural),
      confidence: toNum(shadowFusion.confidence),
      risk_level: toText(shadowFusion.risk_level, "indefinido"),
      publish_blockers: shadow
        ? [...publishBlockers, ...runReasons.filter((r) => !publishBlockers.includes(r))]
        : [...runReasons, "shadow_artifact_missing"],
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
      risk_level_next_month: toText(opSignal.risk_level_next_month, "unknown"),
      operational_state: toText(opSignal.operational_state, "monitoramento_normal"),
      action_hint: toText(opSignal.action_hint, "manter monitoramento estrutural"),
      confidence_score: toNum(opSignal.confidence_score),
      top_sector_global: opTopSector,
      top_asset_global: opTopAsset,
      insight_headlines: opInsightHeadlines,
    },
    watch_assets: sampleAssets(rows, "watch", 6),
    inconclusive_assets: sampleAssets(rows, "inconclusive", 6),
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
  if (ctx.governance.publishable) return message;
  const reasons = ctx.governance.publish_blockers.length
    ? ctx.governance.publish_blockers.join(", ")
    : "gate_or_integrity";
  return `STATUS: NAO PUBLICAVEL\nMotivos: ${reasons}\n\n${message}`;
}

function renderResumo(ctx: CopilotContext): string {
  const opFresh =
    ctx.operational_brief.freshness_days_lag == null
      ? ctx.operational_brief.freshness_status
      : `${ctx.operational_brief.freshness_status} (${ctx.operational_brief.freshness_days_lag}d)`;
  const lines = [
    "Leitura estrutural (fisica matematica) do run atual:",
    `- Run: ${ctx.run.id} | gate bloqueado: ${yesNo(ctx.run.gate_blocked)} | politica: ${ctx.run.policy}.`,
    `- Universo: ${ctx.universe.assets} ativos (${ctx.universe.validated} validated, ${ctx.universe.watch} watch, ${ctx.universe.inconclusive} inconclusive).`,
    `- IA operacional: estado=${ctx.operational_brief.operational_state}, risco_mes=${ctx.operational_brief.risk_level_next_month}, data_base=${ctx.operational_brief.data_last_date || "--"}, freshness=${opFresh}.`,
    `- Macro estrutural: regime=${ctx.lab.regime}, tier=${ctx.lab.signal_tier}, confianca=${ctx.lab.signal_reliability ?? "--"}.`,
    `- B: regime=${ctx.model_b.regime}, risco=${ctx.model_b.risk_score ?? "--"}, conf=${ctx.model_b.confidence ?? "--"} (${ctx.model_b.mode}).`,
    `- C: regime=${ctx.model_c.regime}, risco=${ctx.model_c.risk_score ?? "--"}, conf=${ctx.model_c.confidence ?? "--"} (${ctx.model_c.mode}).`,
    `- Fusao: risco=${ctx.governance.risk_structural ?? "--"} (${ctx.governance.risk_level}), conf=${ctx.governance.confidence ?? "--"}, publishable=${yesNo(ctx.governance.publishable)}.`,
    `- Banco: status=${ctx.platform_db.status}, run_indexado=${ctx.platform_db.run_id || "--"}, rows=${ctx.platform_db.rows_for_run}, copilot_row=${yesNo(ctx.platform_db.copilot_row_exists)}.`,
    "- Limite formal: diagnostico estrutural, sem recomendacao de compra/venda e sem promessa de retorno.",
  ];
  return withPublishGuard(lines.join("\n"), ctx);
}

function renderGate(ctx: CopilotContext): string {
  const reasons = ctx.governance.publish_blockers.length ? ctx.governance.publish_blockers.join(", ") : "nenhum";
  return withPublishGuard(
    [
      "Diagnostico de publicacao (gate):",
      `- gate_blocked=${yesNo(ctx.run.gate_blocked)}.`,
      `- publishable=${yesNo(ctx.governance.publishable)}.`,
      `- blockers: ${reasons}.`,
      `- janela oficial: ${ctx.run.window_days ?? "--"} dias; politica ativa: ${ctx.run.policy}.`,
      "- Regra operacional: se publishable=false, resposta fica em modo diagnostico.",
    ].join("\n"),
    ctx
  );
}

function renderOperationalBrief(ctx: CopilotContext): string {
  const lag =
    ctx.operational_brief.freshness_days_lag == null ? "--" : String(ctx.operational_brief.freshness_days_lag);
  const insights = ctx.operational_brief.insight_headlines.length
    ? ctx.operational_brief.insight_headlines.map((line) => `  - ${line}`).join("\n")
    : "  - sem insights sintetizados no brief atual";
  return withPublishGuard(
    [
      "Brief operacional unificado (IA):",
      `- data_last_date=${ctx.operational_brief.data_last_date || "--"} | freshness=${ctx.operational_brief.freshness_status} | lag_dias=${lag}.`,
      `- estado=${ctx.operational_brief.operational_state} | risco_proximo_mes=${ctx.operational_brief.risk_level_next_month} | confianca=${ctx.operational_brief.confidence_score ?? "--"}.`,
      `- acao sugerida: ${ctx.operational_brief.action_hint}.`,
      `- top setor global=${ctx.operational_brief.top_sector_global} | top ativo global=${ctx.operational_brief.top_asset_global}.`,
      "- insights chave:",
      insights,
      "- Limite formal: estado estrutural e dependencia/relacao; nao e promessa direcional de preco.",
    ].join("\n"),
    ctx
  );
}

function renderCausal(ctx: CopilotContext): string {
  return withPublishGuard(
    [
      "Checagem causal e integridade:",
      "- O copiloto le artefatos do run e do painel de validacao, sem recalibrar historico durante resposta.",
      `- O brief operacional e lido de results/ops/ai_knowledge/latest_operational_brief.json (quando disponivel).`,
      `- Politica declarada no run: ${ctx.run.policy}.`,
      `- Nucleo de instrucoes ativo: ${ctx.instruction_core.version}.`,
      "- Se houver falha de gate/integridade, status vira NAO PUBLICAVEL.",
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
      "- Priorize queda de confianca e mudanca de regime para acionar revisao operacional.",
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
      "- Fluxo: A em producao + B/C em shadow com bloqueio de publicacao por gate/integridade.",
    ].join("\n"),
    ctx
  );
}

export function buildCopilotReply(question: string, ctx: CopilotContext): string {
  const q = question.trim().toLowerCase();
  if (!q) return renderResumo(ctx);

  if (
    q.includes("estado") ||
    q.includes("operac") ||
    q.includes("acao") ||
    q.includes("brief") ||
    q.includes("insight")
  )
    return renderOperationalBrief(ctx);
  if (q.includes("gate") || q.includes("public") || q.includes("bloque")) return renderGate(ctx);
  if (q.includes("causal") || q.includes("look") || q.includes("futuro") || q.includes("leak")) return renderCausal(ctx);
  if (q.includes("ativo") || q.includes("watch") || q.includes("inconclusive") || q.includes("setor")) return renderAssets(ctx);
  if (q.includes("modelo b") || q.includes("modelo c") || q.includes("gnn") || q.includes("rede neural"))
    return renderModels(ctx);

  return renderResumo(ctx);
}
