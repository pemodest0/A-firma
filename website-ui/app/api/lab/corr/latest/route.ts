import { NextResponse } from "next/server";
import {
  readLatestLabCorrAlertLevels,
  findLatestLabCorrRun,
  readLatestLabCorrActionPlaybook,
  readLatestLabCorrAssetDiagnostics,
  readLatestLabCorrAssetSectorSummary,
  readLatestLabCorrEraEvaluation,
  readLatestLabCorrQaChecks,
  readLatestLabCorrRegimeSeries,
  readLatestLabCorrSectorDiagnostics,
  readLatestLabCorrSignificanceSummary,
  readLatestLabCorrOperationalAlerts,
  readLatestLabCorrCaseStudies,
  readLatestLabCorrTimeseries,
  readLatestLabCorrUiViewModel,
} from "@/lib/server/data";

const SCHEMA_VERSION = "lab_corr_api_v1";

function emptyContract(windowValue: number) {
  return {
    schema_version: SCHEMA_VERSION,
    generated_at_utc: new Date().toISOString(),
    window: Math.trunc(windowValue),
    run: { id: "", dir: "", status: "missing" },
    metrics: {
      period: { start: "", end: "" },
      latest_state: null,
      delta_20d: null,
      n_used_stats: null,
    },
    alerts: {
      operational: {
        latest_date: "",
        latest_events: [],
        n_events_total: 0,
        n_events_last_60d: 0,
        event_counts: {},
        latest_event_rows: [],
      },
    },
    era_evaluation: { count: 0, items: [] },
    case_studies: { count_raw: 0, count_valid: 0, dropped_rows: 0, items: [] },
    playbook: { count: 0, latest: null, items_recent: [] },
    view_model: null,
    summary: null,
    summary_compact: "",
    qa_checks: null,
    qa_failed_checks: [],
    period_days: 180,
    regime_history: [],
    alert_levels: [],
    significance: [],
    asset_diagnostics: { count: 0, items: [] },
    sector_diagnostics: { count: 0, items: [] },
    asset_sector_summary: {},
    limits: {
      what_it_does: "",
      what_it_does_not: "",
      methodological_limits: [],
    },
    contract_checks: { case_rows_dropped: 0, has_view_model: false },
    timeseries: [],
  };
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const windowParam = Number(searchParams.get("window") || "120");
  const includeRows = searchParams.get("include_rows") === "1";
  const periodDays = Math.max(7, Math.min(1500, Number(searchParams.get("period_days") || "180")));
  const assetFilter = String(searchParams.get("asset") || "").trim().toUpperCase();
  const sectorFilter = String(searchParams.get("sector") || "").trim().toLowerCase();

  if (!Number.isFinite(windowParam) || windowParam <= 0) {
    return NextResponse.json(
      {
        ...emptyContract(120),
        run: { id: "", dir: "", status: "error" },
        error: "invalid_window",
      },
      { status: 400 }
    );
  }

  const [run, ts, cases, opAlerts, eraEval, playbook, viewModel, qaChecks, regimeHistoryRaw, alertLevelsRaw, significance, assetDiagRaw, sectorDiagRaw, assetSectorSummary] = await Promise.all([
    findLatestLabCorrRun(),
    readLatestLabCorrTimeseries(windowParam),
    readLatestLabCorrCaseStudies(windowParam),
    readLatestLabCorrOperationalAlerts(windowParam),
    readLatestLabCorrEraEvaluation(windowParam),
    readLatestLabCorrActionPlaybook(windowParam),
    readLatestLabCorrUiViewModel(windowParam),
    readLatestLabCorrQaChecks(),
    readLatestLabCorrRegimeSeries(windowParam, 2000),
    readLatestLabCorrAlertLevels(windowParam, 2000),
    readLatestLabCorrSignificanceSummary(),
    readLatestLabCorrAssetDiagnostics(2000),
    readLatestLabCorrSectorDiagnostics(),
    readLatestLabCorrAssetSectorSummary(),
  ]);

  if (!run) {
    return NextResponse.json(
      {
        ...emptyContract(windowParam),
        run: { id: "", dir: "", status: "missing" },
        error: "no_valid_lab_run",
        message: "Nenhum run valido de lab_corr_macro foi encontrado.",
      },
      { status: 503 }
    );
  }

  const playbookRows = Array.isArray(playbook) ? playbook : [];
  const latestPlaybook = playbookRows.length ? playbookRows[playbookRows.length - 1] : null;
  const nowDate = ts?.end ? new Date(`${ts.end}T00:00:00Z`) : new Date();
  const cutoff = new Date(nowDate.getTime() - periodDays * 24 * 3600 * 1000);

  // Mantemos o histórico completo de regime para evitar truncamento inesperado na timeline da UI.
  // O recorte visual por período é feito no front.
  const regimeHistory = Array.isArray(regimeHistoryRaw)
    ? (regimeHistoryRaw as Record<string, unknown>[])
    : [];
  const alertLevelsBase = Array.isArray(alertLevelsRaw)
    ? (alertLevelsRaw as Record<string, unknown>[])
    : [];
  const sectorDiagBase = Array.isArray(sectorDiagRaw)
    ? (sectorDiagRaw as Record<string, unknown>[])
    : [];
  const assetDiagBase = Array.isArray(assetDiagRaw)
    ? (assetDiagRaw as Record<string, unknown>[])
    : [];
  const alertLevels = alertLevelsBase.filter(
    (row: Record<string, unknown>) => new Date(`${String(row.date)}T00:00:00Z`) >= cutoff
  );
  const sectorDiag = sectorDiagBase.filter((row: Record<string, unknown>) =>
    sectorFilter ? String(row.sector || "").toLowerCase().includes(sectorFilter) : true
  );
  const assetDiag = assetDiagBase.filter((row: Record<string, unknown>) => {
    const t = String(row.ticker || "");
    const sec = String(row.sector || "").toLowerCase();
    if (assetFilter && !t.includes(assetFilter)) return false;
    if (sectorFilter && !sec.includes(sectorFilter)) return false;
    return true;
  });
  const qaObj = qaChecks && typeof qaChecks === "object" ? (qaChecks as Record<string, unknown>) : {};
  const failedChecks = Array.isArray(qaObj.failed_checks) ? qaObj.failed_checks : [];
  const limits = {
    what_it_does:
      "Le risco estrutural do mercado e oferece interpretacao estatistica por nivel de alerta.",
    what_it_does_not:
      "Nao prevê dia exato de crash, nao garante retorno e nao substitui decisao humana.",
    methodological_limits: [
      "Sensivel a qualidade e cobertura dos dados de entrada.",
      "Pode atrasar em choque totalmente exogeno.",
      "Funciona melhor como monitor de risco do que gerador puro de alpha.",
    ],
  };

  const payload = {
    schema_version: SCHEMA_VERSION,
    generated_at_utc: new Date().toISOString(),
    window: Math.trunc(windowParam),
    run: { id: run.runId, dir: run.runDir, status: "ok" },
    metrics: {
      period: ts ? { start: ts.start, end: ts.end } : { start: "", end: "" },
      latest_state: ts?.latest || null,
      delta_20d: ts?.delta_20d || null,
      n_used_stats: ts?.n_used_stats || null,
    },
    alerts: {
      operational: opAlerts,
    },
    era_evaluation: {
      count: Array.isArray(eraEval) ? eraEval.length : 0,
      items: Array.isArray(eraEval) ? eraEval : [],
    },
    case_studies: {
      count_raw: cases?.count_raw ?? 0,
      count_valid: cases?.count_valid ?? 0,
      dropped_rows: cases?.dropped_rows ?? 0,
      items: cases?.cases || [],
    },
    playbook: {
      count: playbookRows.length,
      latest: latestPlaybook,
      items_recent: playbookRows.slice(-30),
    },
    view_model: viewModel || null,
    summary: run.summary,
    summary_compact: run.summaryCompact || "",
    qa_checks: qaObj,
    qa_failed_checks: failedChecks,
    period_days: periodDays,
    regime_history: regimeHistory,
    alert_levels: alertLevels,
    significance,
    asset_diagnostics: {
      count: assetDiag.length,
      items: assetDiag.slice(0, 500),
    },
    sector_diagnostics: {
      count: sectorDiag.length,
      items: sectorDiag,
    },
    asset_sector_summary: assetSectorSummary,
    limits,
    contract_checks: {
      case_rows_dropped: cases?.dropped_rows ?? 0,
      has_view_model: Boolean(viewModel && typeof viewModel === "object"),
    },
    timeseries: includeRows ? ts?.rows || [] : [],
  };

  return NextResponse.json(payload);
}
