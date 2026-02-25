import { promises as fs } from "fs";
import path from "path";
import { readAssetStatusMap } from "@/lib/server/validated";
import { existsSync } from "node:fs";

function repoRoot() {
  return path.resolve(process.cwd(), "..");
}

function resolveResultsDir() {
  if (process.env.RESULTS_DIR) return process.env.RESULTS_DIR;
  const candidates = [path.join(process.cwd(), "results"), path.join(repoRoot(), "results")];
  for (const candidate of candidates) {
    if (existsSync(candidate)) return candidate;
  }
  return candidates[1];
}

export function dataDirs() {
  const root = repoRoot();
  return {
    latest: process.env.DATA_DIR || path.join(root, "results", "latest"),
    publicLatest: path.join(process.cwd(), "public", "data", "latest"),
    results: resolveResultsDir(),
  };
}

export async function listLatestFiles() {
  const { latest } = dataDirs();
  const dir = latest;
  await fs.access(dir);
  const files = await fs.readdir(dir);
  return files.filter((f) => f.endsWith(".json"));
}

export async function readLatestFile(file: string) {
  const { latest } = dataDirs();
  const target = path.join(latest, file);
  const text = await fs.readFile(target, "utf-8");
  return JSON.parse(text);
}

export async function findLatestApiRecords() {
  const run = await findLatestValidRun();
  return run?.snapshotPath || null;
}

function sanitizeJsonLine(line: string) {
  return line
    .replace(/\bNaN\b/g, "null")
    .replace(/\bInfinity\b/g, "null")
    .replace(/\b-Infinity\b/g, "null");
}

function parseJsonLine(line: string): Record<string, unknown> {
  try {
    return JSON.parse(line);
  } catch {
    const fixed = sanitizeJsonLine(line);
    return JSON.parse(fixed);
  }
}

export async function readJsonl(pathFile: string): Promise<Record<string, unknown>[]> {
  const text = await fs.readFile(pathFile, "utf-8");
  const out: Record<string, unknown>[] = [];
  for (const raw of text.split("\n")) {
    const line = raw.trim();
    if (!line) continue;
    try {
      out.push(parseJsonLine(line));
    } catch {
      // ignore malformed line instead of breaking entire snapshot read
    }
  }
  return out;
}

function repairMojibake(value: string) {
  return value
    .replace(/ÃƒÂ§/g, "ç")
    .replace(/ÃƒÂ£/g, "ã")
    .replace(/ÃƒÂ¡/g, "á")
    .replace(/ÃƒÂ©/g, "é")
    .replace(/ÃƒÂª/g, "ê")
    .replace(/ÃƒÂ­/g, "í")
    .replace(/ÃƒÂ³/g, "ó")
    .replace(/ÃƒÂ´/g, "ô")
    .replace(/ÃƒÂº/g, "ú")
    .replace(/Ãƒâ€°/g, "É")
    .replace(/Ãƒâ€œ/g, "Ó")
    .replace(/Ãƒ/g, "à")
    .replace(/Ã‚/g, "");
}

function sanitizeEncoding<T>(input: T): T {
  if (typeof input === "string") {
    return repairMojibake(input) as T;
  }
  if (Array.isArray(input)) {
    return input.map((item) => sanitizeEncoding(item)) as T;
  }
  if (input && typeof input === "object") {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(input as Record<string, unknown>)) {
      out[k] = sanitizeEncoding(v);
    }
    return out as T;
  }
  return input;
}

export type LatestRunInfo = {
  runId: string;
  snapshotPath: string;
  summaryPath: string;
  summary: Record<string, unknown>;
};

function isRunValid(summary: Record<string, unknown>) {
  const status = String(summary?.status || "").toLowerCase();
  const gate = (summary?.deployment_gate || {}) as Record<string, unknown>;
  const blocked = gate?.blocked === true;
  return status === "ok" && !blocked;
}

export async function findLatestValidRun(): Promise<LatestRunInfo | null> {
  const { results } = dataDirs();
  const snapshotsRoot = path.join(results, "ops", "snapshots");
  let runDirs: string[] = [];
  try {
    runDirs = (await fs.readdir(snapshotsRoot, { withFileTypes: true }))
      .filter((ent) => ent.isDirectory())
      .map((ent) => ent.name)
      .sort()
      .reverse();
  } catch {
    runDirs = [];
  }

  for (const runId of runDirs) {
    const summaryPath = path.join(snapshotsRoot, runId, "summary.json");
    const snapshotPath = path.join(snapshotsRoot, runId, "api_snapshot.jsonl");
    try {
      const [summaryText, snapshotStat] = await Promise.all([
        fs.readFile(summaryPath, "utf-8"),
        fs.stat(snapshotPath),
      ]);
      if (!snapshotStat.size) continue;
      const summary = JSON.parse(summaryText) as Record<string, unknown>;
      if (!isRunValid(summary)) continue;
      return { runId, summaryPath, snapshotPath, summary };
    } catch {
      // ignore invalid run and keep scanning older runs
    }
  }
  return null;
}

export async function readLatestSnapshot() {
  const run = await findLatestValidRun();
  if (!run) return null;
  const records = await readJsonl(run.snapshotPath);
  return {
    runId: run.runId,
    summary: sanitizeEncoding(run.summary),
    records: sanitizeEncoding(records),
  };
}

export async function readRiskTruthPanel() {
  const { results } = dataDirs();
  const target = path.join(results, "validation", "risk_truth_panel.json");
  try {
    const text = await fs.readFile(target, "utf-8");
    return sanitizeEncoding(JSON.parse(text));
  } catch {
    return {
      status: "empty",
      counts: { assets: 0, validated: 0, watch: 0, inconclusive: 0 },
      entries: [],
    };
  }
}

export async function readLatestValidationSummary() {
  const { results } = dataDirs();
  const target = path.join(results, "validation", "latest_validation.json");
  try {
    const raw = await fs.readFile(target, "utf-8");
    return sanitizeEncoding(JSON.parse(raw));
  } catch {
    return {
      schema_version: "latest_validation_v1",
      status: "missing",
      as_of_date: "",
      evidence: {
        event_rate: null,
        alert_rate: null,
        lift: null,
      },
      validation_gate: {
        status: "unknown",
      },
    };
  }
}

export async function readGlobalStatus() {
  const run = await findLatestValidRun();
  if (run) {
    const gate = (run.summary?.deployment_gate || {}) as Record<string, unknown>;
    const blocked = gate?.blocked === true;
    return {
      status: blocked ? "blocked" : "ok",
      source: "latest_run_summary",
      deployment_gate: gate,
      checks: (run.summary?.checks || {}) as Record<string, unknown>,
      scores: (run.summary?.scores || {}) as Record<string, unknown>,
    };
  }
  return { status: "unknown", source: "no_valid_run", checks: {}, scores: {} };
}

export async function readJsonlWithValidationGate(pathFile: string) {
  const records = await readJsonl(pathFile);
  let statusMap: Record<string, Record<string, string>> = {};
  try {
    statusMap = await readAssetStatusMap();
  } catch {
    return records;
  }
  return records.map((record: Record<string, unknown>) => {
    const key = `${record.asset || ""}__${record.timeframe || ""}`;
    const gate = statusMap[key];
    if (!gate || (gate.status || "").toLowerCase() === "validated") {
      return record;
    }
    const reason = gate.reason || "gate_not_validated";
    const warnings = Array.isArray(record.warnings) ? [...record.warnings] : [];
    if (!warnings.includes("INCONCLUSIVE_SIGNAL")) {
      warnings.push("INCONCLUSIVE_SIGNAL");
    }
    return {
      ...record,
      signal_status: "inconclusive",
      use_forecast_bool: false,
      action: "DIAGNOSTICO_INCONCLUSIVO",
      regime_label: "INCONCLUSIVE",
      confidence_level: "LOW",
      warnings,
      gate_reason: reason,
    };
  });
}

export async function readDashboardOverview() {
  const { results } = dataDirs();
  const overviewPath = path.join(results, "dashboard", "overview.json");
  const text = await fs.readFile(overviewPath, "utf-8");
  return JSON.parse(text);
}

export type LabCorrRunInfo = {
  runId: string;
  runDir: string;
  summary: Record<string, unknown>;
  summaryCompact: string;
};

function publicLabCorrLatestDir() {
  return path.join(process.cwd(), "public", "data", "lab_corr_macro", "latest");
}

export type LabCorrTimeseriesRow = {
  date: string;
  N_used: number;
  p1: number;
  deff: number;
  top5: number | null;
  cluster_count: number | null;
  largest_share: number | null;
  entropy: number | null;
  turnover_pair_frac: number | null;
  structure_score: number | null;
  p1_shuffle: number | null;
  deff_shuffle: number | null;
};

export type LabCorrCaseStudy = {
  case_regime: string;
  date: string;
  N_used: number;
  p1: number;
  deff: number;
  lambda1: number;
  lambda2: number;
  top5: number;
  exposure: number;
  horizon_days: number;
  future_days_used: number;
  bench_cum_return: number;
  strategy_cum_return: number;
  alpha_cum: number;
  bench_max_drawdown: number;
  strategy_max_drawdown: number;
  dd_improvement: number;
  honest_verdict: string;
};

function toFiniteNumber(value: unknown): number | null {
  if (typeof value === "number") return Number.isFinite(value) ? value : null;
  if (typeof value !== "string") return null;
  const trimmed = value.trim();
  if (!trimmed) return null;
  const n = Number(trimmed);
  return Number.isFinite(n) ? n : null;
}

function parseCsvLine(line: string): string[] {
  const out: string[] = [];
  let current = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i += 1) {
    const ch = line[i];
    if (ch === "\"") {
      if (inQuotes && line[i + 1] === "\"") {
        current += "\"";
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
      continue;
    }
    if (ch === "," && !inQuotes) {
      out.push(current);
      current = "";
      continue;
    }
    current += ch;
  }
  out.push(current);
  return out.map((item) => item.trim());
}

function parseCsvRecords(text: string): Record<string, string>[] {
  const lines = text.split(/\r?\n/).filter((line) => line.trim().length > 0);
  if (lines.length < 2) return [];
  const headers = parseCsvLine(lines[0]);
  if (!headers.length) return [];
  return lines.slice(1).map((line) => {
    const cols = parseCsvLine(line);
    const row: Record<string, string> = {};
    headers.forEach((h, idx) => {
      row[h] = cols[idx] || "";
    });
    return row;
  });
}

function normalizeLabTimeseriesRow(row: Record<string, string>): LabCorrTimeseriesRow | null {
  const date = String(row.date || "").trim();
  const nUsed = toFiniteNumber(row.N_used);
  const p1 = toFiniteNumber(row.p1);
  const deff = toFiniteNumber(row.deff);
  if (!date || nUsed == null || p1 == null || deff == null) return null;
  return {
    date,
    N_used: nUsed,
    p1,
    deff,
    top5: toFiniteNumber(row.top5),
    cluster_count: toFiniteNumber(row.cluster_count),
    largest_share: toFiniteNumber(row.largest_share),
    entropy: toFiniteNumber(row.entropy),
    turnover_pair_frac: toFiniteNumber(row.turnover_pair_frac),
    structure_score: toFiniteNumber(row.structure_score),
    p1_shuffle: toFiniteNumber(row.p1_shuffle),
    deff_shuffle: toFiniteNumber(row.deff_shuffle),
  };
}

function normalizeLabCaseStudy(row: Record<string, string>): LabCorrCaseStudy | null {
  const requiredNumbers = [
    "N_used",
    "p1",
    "deff",
    "lambda1",
    "lambda2",
    "top5",
    "exposure",
    "horizon_days",
    "future_days_used",
    "bench_cum_return",
    "strategy_cum_return",
    "alpha_cum",
    "bench_max_drawdown",
    "strategy_max_drawdown",
    "dd_improvement",
  ];
  const parsed: Record<string, number> = {};
  for (const key of requiredNumbers) {
    const val = toFiniteNumber(row[key]);
    if (val == null) return null;
    parsed[key] = val;
  }
  const caseRegime = String(row.case_regime || "").trim();
  const date = String(row.date || "").trim();
  if (!caseRegime || !date) return null;
  return {
    case_regime: caseRegime,
    date,
    N_used: parsed.N_used,
    p1: parsed.p1,
    deff: parsed.deff,
    lambda1: parsed.lambda1,
    lambda2: parsed.lambda2,
    top5: parsed.top5,
    exposure: parsed.exposure,
    horizon_days: parsed.horizon_days,
    future_days_used: parsed.future_days_used,
    bench_cum_return: parsed.bench_cum_return,
    strategy_cum_return: parsed.strategy_cum_return,
    alpha_cum: parsed.alpha_cum,
    bench_max_drawdown: parsed.bench_max_drawdown,
    strategy_max_drawdown: parsed.strategy_max_drawdown,
    dd_improvement: parsed.dd_improvement,
    honest_verdict: String(row.honest_verdict || "").trim(),
  };
}

export async function findLatestLabCorrRun(): Promise<LabCorrRunInfo | null> {
  const { results } = dataDirs();
  const labRoot = path.join(results, "lab_corr_macro");
  const pointerPath = path.join(labRoot, "latest_release.json");

  const inspectCandidate = async (runId: string, runDir: string): Promise<LabCorrRunInfo | null> => {
    const summaryPath = path.join(runDir, "summary.json");
    const compactPath = path.join(runDir, "summary_compact.txt");
    try {
      const summaryRaw = await fs.readFile(summaryPath, "utf-8");
      const summary = JSON.parse(summaryRaw) as Record<string, unknown>;
      if (!isRunValid(summary)) return null;
      let summaryCompact = "";
      try {
        summaryCompact = await fs.readFile(compactPath, "utf-8");
      } catch {
        summaryCompact = "";
      }
      return { runId, runDir, summary, summaryCompact };
    } catch {
      return null;
    }
  };

  try {
    const pointerRaw = await fs.readFile(pointerPath, "utf-8");
    const pointer = JSON.parse(pointerRaw) as Record<string, unknown>;
    const runId = String(pointer.run_id || "").trim();
    const runDirFromPointer = String(pointer.run_dir || "").trim();
    const runDir = runDirFromPointer || (runId ? path.join(labRoot, runId) : "");
    if (runId && runDir) {
      const hit = await inspectCandidate(runId, runDir);
      if (hit) return hit;
    }
  } catch {
    // fallback scan
  }

  try {
    const dirs = await fs.readdir(labRoot, { withFileTypes: true });
    const runs = dirs
      .filter((d) => d.isDirectory())
      .map((d) => d.name)
      .filter((name) => /^\d{8}T\d{6}Z$/i.test(name))
      .sort()
      .reverse();
    for (const runId of runs) {
      const candidate = await inspectCandidate(runId, path.join(labRoot, runId));
      if (candidate) return candidate;
    }
  } catch {
    // fallback to bundled public artifacts
  }

  try {
    const pubDir = publicLabCorrLatestDir();
    const summaryPath = path.join(pubDir, "summary.json");
    const compactPath = path.join(pubDir, "summary_compact.txt");
    const summaryRaw = await fs.readFile(summaryPath, "utf-8");
    const summary = JSON.parse(summaryRaw) as Record<string, unknown>;
    const runId = String(summary.run_id || "public_lab_corr_latest");
    let summaryCompact = "";
    try {
      summaryCompact = await fs.readFile(compactPath, "utf-8");
    } catch {
      summaryCompact = "";
    }
    return { runId, runDir: pubDir, summary, summaryCompact };
  } catch {
    return null;
  }
}

export async function readLatestLabCorrTimeseries(window = 120) {
  const run = await findLatestLabCorrRun();
  if (!run) return null;
  const win = Number(window);
  if (!Number.isFinite(win) || win <= 0) return null;
  const filePath = path.join(run.runDir, `macro_timeseries_T${Math.trunc(win)}.csv`);
  try {
    const raw = await fs.readFile(filePath, "utf-8");
    const parsed = parseCsvRecords(raw)
      .map(normalizeLabTimeseriesRow)
      .filter((row): row is LabCorrTimeseriesRow => row != null);
    if (!parsed.length) return null;
    const latest = parsed[parsed.length - 1];
    const refIndex = Math.max(0, parsed.length - 21);
    const ref20d = parsed[refIndex];
    const nValues = parsed.map((r) => r.N_used).filter((v) => Number.isFinite(v));
    const nMean = nValues.length ? nValues.reduce((acc, v) => acc + v, 0) / nValues.length : null;
    return {
      runId: run.runId,
      runDir: run.runDir,
      window: Math.trunc(win),
      start: parsed[0].date,
      end: latest.date,
      n_used_stats: {
        min: nValues.length ? Math.min(...nValues) : null,
        max: nValues.length ? Math.max(...nValues) : null,
        mean: nMean,
      },
      latest,
      delta_20d: {
        p1: latest.p1 - ref20d.p1,
        deff: latest.deff - ref20d.deff,
      },
      rows: parsed,
    };
  } catch {
    return null;
  }
}

export async function readLatestLabCorrCaseStudies(window = 120) {
  const run = await findLatestLabCorrRun();
  if (!run) return null;
  const win = Number(window);
  if (!Number.isFinite(win) || win <= 0) return null;
  const filePath = path.join(run.runDir, `case_studies_T${Math.trunc(win)}.csv`);
  try {
    const raw = await fs.readFile(filePath, "utf-8");
    const records = parseCsvRecords(raw);
    const cases = records
      .map(normalizeLabCaseStudy)
      .filter((row): row is LabCorrCaseStudy => row != null);
    return {
      runId: run.runId,
      runDir: run.runDir,
      window: Math.trunc(win),
      count_raw: records.length,
      count_valid: cases.length,
      dropped_rows: Math.max(0, records.length - cases.length),
      cases,
    };
  } catch {
    return null;
  }
}

async function readLatestLabCorrJsonArtifact(window: number, fileStem: string, fallback: unknown) {
  const run = await findLatestLabCorrRun();
  if (!run) return fallback;
  const win = Number(window);
  if (!Number.isFinite(win) || win <= 0) return fallback;
  const filePath = path.join(run.runDir, `${fileStem}_T${Math.trunc(win)}.json`);
  try {
    const raw = await fs.readFile(filePath, "utf-8");
    try {
      return JSON.parse(raw);
    } catch {
      const repaired = raw
        .replace(/\bNaN\b/g, "null")
        .replace(/\bInfinity\b/g, "null")
        .replace(/\b-Infinity\b/g, "null");
      return JSON.parse(repaired);
    }
  } catch {
    return fallback;
  }
}

export async function readLatestLabCorrOperationalAlerts(window = 120) {
  return readLatestLabCorrJsonArtifact(window, "operational_alerts", {
    latest_date: "",
    latest_events: [],
    n_events_total: 0,
    n_events_last_60d: 0,
    event_counts: {},
    latest_event_rows: [],
  });
}

export async function readLatestLabCorrEraEvaluation(window = 120) {
  const payload = await readLatestLabCorrJsonArtifact(window, "era_evaluation", []);
  return Array.isArray(payload) ? payload : [];
}

export async function readLatestLabCorrActionPlaybook(window = 120) {
  const payload = await readLatestLabCorrJsonArtifact(window, "action_playbook", []);
  return Array.isArray(payload) ? payload : [];
}

export async function readLatestLabCorrUiViewModel(window = 120) {
  return readLatestLabCorrJsonArtifact(window, "ui_view_model", {
    schema_version: "lab_corr_view_v1",
    latest_state: {},
    latest_regime: {},
    alerts: { latest_events: [] },
    playbook_latest: {},
    case_preview: [],
    era_summary: [],
  });
}

export async function readLatestLabCorrBacktestSummary(window = 120) {
  const run = await findLatestLabCorrRun();
  if (!run) return null;
  const win = Number(window);
  if (!Number.isFinite(win) || win <= 0) return null;
  const filePath = path.join(run.runDir, `backtest_summary_T${Math.trunc(win)}.json`);
  try {
    const raw = await fs.readFile(filePath, "utf-8");
    return JSON.parse(raw) as Record<string, unknown>;
  } catch {
    return null;
  }
}

export async function readLatestLabCorrQaChecks() {
  const run = await findLatestLabCorrRun();
  if (!run) return null;
  const filePath = path.join(run.runDir, "qa_checks.json");
  try {
    const raw = await fs.readFile(filePath, "utf-8");
    return JSON.parse(raw) as Record<string, unknown>;
  } catch {
    return null;
  }
}

export async function readLatestLabCorrRegimeSeries(window = 120, limit = 365) {
  const run = await findLatestLabCorrRun();
  if (!run) return [];
  const win = Number(window);
  if (!Number.isFinite(win) || win <= 0) return [];
  const filePath = path.join(run.runDir, `regime_series_T${Math.trunc(win)}.csv`);
  try {
    const raw = await fs.readFile(filePath, "utf-8");
    const rows = parseCsvRecords(raw)
      .map((row) => {
        const date = String(row.date || "").trim();
        const regime = String(row.regime || "").trim();
        if (!date || !regime) return null;
        return {
          date,
          regime,
          regime_raw: String(row.regime_raw || "").trim(),
          exposure: toFiniteNumber(row.exposure),
          p1: toFiniteNumber(row.p1),
          deff: toFiniteNumber(row.deff),
          dp1_5: toFiniteNumber(row.dp1_5),
          ddeff_5: toFiniteNumber(row.ddeff_5),
          transition_score: toFiniteNumber(row.transition_score),
        };
      })
      .filter((row): row is NonNullable<typeof row> => row != null);
    const k = Math.max(1, Math.trunc(limit));
    return rows.slice(-k);
  } catch {
    return [];
  }
}

export async function readLatestLabCorrAlertLevels(window = 120, limit = 365) {
  const run = await findLatestLabCorrRun();
  if (!run) return [];
  const win = Number(window);
  if (!Number.isFinite(win) || win <= 0) return [];
  const filePath = path.join(run.runDir, `alert_levels_T${Math.trunc(win)}.csv`);
  try {
    const raw = await fs.readFile(filePath, "utf-8");
    const rows = parseCsvRecords(raw)
      .map((row) => {
        const date = String(row.date || "").trim();
        const level = String(row.alert_level || "").trim().toLowerCase();
        if (!date || !level) return null;
        return {
          date,
          alert_level: level,
          alert_level_raw: String(row.alert_level_raw || "").trim().toLowerCase(),
          regime: String(row.regime || "").trim(),
          regime_raw: String(row.regime_raw || "").trim(),
          risk_score: toFiniteNumber(row.risk_score),
          signal_confidence: toFiniteNumber(row.signal_confidence),
          transition_score: toFiniteNumber(row.transition_score),
        };
      })
      .filter((row): row is NonNullable<typeof row> => row != null);
    return rows.slice(-Math.max(1, Math.trunc(limit)));
  } catch {
    return [];
  }
}

export async function readLatestLabCorrSignificanceSummary() {
  const run = await findLatestLabCorrRun();
  if (!run) return [];
  const filePath = path.join(run.runDir, "significance_summary_by_window.csv");
  try {
    const raw = await fs.readFile(filePath, "utf-8");
    return parseCsvRecords(raw).map((row) => ({
      window: toFiniteNumber(row.window),
      metric: String(row.metric || "").trim(),
      n: toFiniteNumber(row.n),
      mean_delta: toFiniteNumber(row.mean_delta),
      std_delta: toFiniteNumber(row.std_delta),
      significant_share_p_lt_0_05: toFiniteNumber(row.significant_share_p_lt_0_05),
      mean_pvalue_vs_zero: toFiniteNumber(row.mean_pvalue_vs_zero),
      latest_pvalue: toFiniteNumber(row.latest_pvalue),
    }));
  } catch {
    return [];
  }
}

export async function readLatestLabCorrAssetDiagnostics(limit = 500) {
  const run = await findLatestLabCorrRun();
  if (!run) return [];
  const filePath = path.join(run.runDir, "asset_regime_diagnostics.csv");
  try {
    const raw = await fs.readFile(filePath, "utf-8");
    const rows = parseCsvRecords(raw).map((row) => ({
      ticker: String(row.ticker || "").trim(),
      sector: String(row.sector || "").trim(),
      risk_score: toFiniteNumber(row.risk_score),
      confidence_score: toFiniteNumber(row.confidence_score),
      regime_asset: String(row.regime_asset || "").trim(),
      switches_30d: toFiniteNumber(row.switches_30d),
      switches_90d: toFiniteNumber(row.switches_90d),
      switches_180d: toFiniteNumber(row.switches_180d),
      vol60_latest: toFiniteNumber(row.vol60_latest),
      corr120_latest: toFiniteNumber(row.corr120_latest),
      sensitivity_score: toFiniteNumber(row.sensitivity_score),
      stability_score: toFiniteNumber(row.stability_score),
    }));
    return rows
      .filter((row) => row.ticker.length > 0)
      .slice(0, Math.max(1, Math.trunc(limit)));
  } catch {
    return [];
  }
}

export async function readLatestLabCorrSectorDiagnostics() {
  const run = await findLatestLabCorrRun();
  if (!run) return [];
  const filePath = path.join(run.runDir, "sector_regime_diagnostics.csv");
  try {
    const raw = await fs.readFile(filePath, "utf-8");
    return parseCsvRecords(raw).map((row) => ({
      sector: String(row.sector || "").trim(),
      n_assets: toFiniteNumber(row.n_assets),
      risk_mean: toFiniteNumber(row.risk_mean),
      confidence_mean: toFiniteNumber(row.confidence_mean),
      pct_instavel: toFiniteNumber(row.pct_instavel),
      pct_transicao: toFiniteNumber(row.pct_transicao),
      alerta_setor: String(row.alerta_setor || "").trim().toLowerCase(),
      plano_acao: String(row.plano_acao || "").trim(),
    }));
  } catch {
    return [];
  }
}

export async function readLatestLabCorrAssetSectorSummary() {
  const run = await findLatestLabCorrRun();
  if (!run) return {};
  const filePath = path.join(run.runDir, "asset_sector_summary.json");
  try {
    const raw = await fs.readFile(filePath, "utf-8");
    return JSON.parse(raw) as Record<string, unknown>;
  } catch {
    return {};
  }
}

export async function readPlatformDbSnapshot() {
  const { results } = dataDirs();
  const target = path.join(results, "platform", "latest_db_snapshot.json");
  try {
    const raw = await fs.readFile(target, "utf-8");
    return sanitizeEncoding(JSON.parse(raw));
  } catch {
    return {
      status: "missing",
      run_id: "",
      generated_at_utc: "",
      db_path: path.join(results, "platform", "assyntrax_platform.db"),
      counts: {
        runs_total: 0,
        asset_rows_total: 0,
        asset_rows_for_run: 0,
      },
      run: {
        status: "unknown",
        gate_blocked: true,
        n_assets: 0,
        validated_signals: 0,
        watch_signals: 0,
        inconclusive_signals: 0,
        validated_ratio: 0,
      },
      domains: [],
      signal_status: [],
      copilot: {
        row_exists: false,
        publishable: false,
        risk_structural: null,
        confidence: null,
        risk_level: "indefinido",
      },
    };
  }
}

export async function readPlatformDbRelease() {
  const { results } = dataDirs();
  const target = path.join(results, "platform", "latest_release.json");
  try {
    const raw = await fs.readFile(target, "utf-8");
    return sanitizeEncoding(JSON.parse(raw));
  } catch {
    return {
      updated_at_utc: "",
      run_id: "",
      db_path: path.join(results, "platform", "assyntrax_platform.db"),
      latest_db_snapshot: "",
    };
  }
}
