import { NextResponse } from "next/server";
import { listLatestFiles, readLatestFile, readLatestSnapshot, readRiskTruthPanel, readSiteFinanceSnapshot } from "@/lib/server/data";
import { readAssetStatusMap } from "@/lib/server/validated";

function inferDomain(group?: string, asset?: string) {
  const g = (group || "").toLowerCase();
  const a = (asset || "").toLowerCase();
  if (g.includes("crypto") || a.endsWith("-usd")) return "crypto";
  if (g) return "finance";
  return "finance";
}

function normalizeDomain(value?: string, group?: string, asset?: string) {
  const raw = (value || "").toLowerCase().trim();
  if (!raw) return inferDomain(group, asset);
  if (raw.includes("crypto")) {
    return "crypto";
  }
  if (raw.includes("fin") || raw.includes("equit") || raw.includes("macro") || raw.includes("energy") || raw.includes("realestate")) {
    return "finance";
  }
  return inferDomain(group, asset);
}

function normalizeRecord(record: Record<string, unknown>, riskTruthStatus?: string) {
  const regimeFromState = (record.state as Record<string, unknown> | undefined)?.label;
  const warnings = Array.isArray(record.warnings) ? record.warnings.map((w) => String(w)) : [];
  const inferredRegime =
    String(record.regime_label || record.regime || regimeFromState || "").toUpperCase() ||
    (warnings.includes("REGIME_INSTAVEL") ? "UNSTABLE" : "TRANSITION");
  const baseConfidence =
    Number(
      record.confidence ??
        (record.metrics as Record<string, unknown> | undefined)?.confidence ??
        record.forecast_confidence ??
        (record.state as Record<string, unknown> | undefined)?.confidence ??
        0
    ) || 0;
  const mase = Number(record.mase_6m ?? Number.NaN);
  const baseQuality =
    Number(record.quality ?? (record.metrics as Record<string, unknown> | undefined)?.quality ?? Number.NaN);
  const quality = Number.isFinite(baseQuality) ? baseQuality : Number.isFinite(mase) ? Math.max(0, Math.min(1, 1 - mase * 0.25)) : 0.5;
  const signalStatus = String(record.status || record.signal_status || "unknown").toLowerCase();
  const fallbackRiskStatus =
    signalStatus === "validated" || signalStatus === "watch" || signalStatus === "inconclusive"
      ? signalStatus
      : "unknown";
  return {
    asset: String(record.asset || ""),
    domain: normalizeDomain(String(record.domain || ""), String(record.group || ""), String(record.asset || "")),
    timestamp: String(record.timestamp || ""),
    run_id: String(record.run_id || ""),
    data_adequacy: String(record.data_adequacy || "unknown"),
    source_type: String(record.source_type || "proxy"),
    regime: inferredRegime,
    confidence: baseConfidence,
    quality,
    instability_score: Number(record.instability_score ?? 0),
    status: signalStatus,
    signal_status: signalStatus,
    reason: String(record.reason || record.warning_reason || ""),
    risk_truth_status: (riskTruthStatus || fallbackRiskStatus || "unknown").toLowerCase(),
    group: String(record.group || ""),
  };
}

type NormalizedAssetRecord = ReturnType<typeof normalizeRecord>;

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const file = searchParams.get("file");
  if (file) {
    try {
      const data = await readLatestFile(file);
      const m = file.match(/^(.+?)_(daily|weekly)\.json$/i);
      if (m) {
        try {
          const statusMap = await readAssetStatusMap();
          const key = `${m[1]}__${m[2].toLowerCase()}`;
          const gate = statusMap[key];
          if (gate && (gate.status || "").toLowerCase() !== "validated") {
            return NextResponse.json({
              ...data,
              signal_status: "inconclusive",
              regime_label: "INCONCLUSIVE",
              action: "DIAGNOSTICO_INCONCLUSIVO",
              gate_reason: gate.reason || "gate_not_validated",
            });
          }
        } catch {
          // no validated map, keep legacy payload
        }
      }
      return NextResponse.json(data);
    } catch {
      return NextResponse.json({ error: "file_not_found", file }, { status: 404 });
    }
  }

  const domain = (searchParams.get("domain") || "").toLowerCase();
  const statusParam = searchParams.get("status") || "validated,watch";
  const includeInconclusive = searchParams.get("include_inconclusive") === "1";
  const allowedStatus = statusParam
    .split(",")
    .map((s) => s.trim().toLowerCase())
    .filter(Boolean);
  const allowed = new Set(allowedStatus.length ? allowedStatus : ["validated", "watch"]);

  const [snap, riskTruth, siteSnapshot] = await Promise.all([readLatestSnapshot(), readRiskTruthPanel(), readSiteFinanceSnapshot()]);
  const truthMap = new Map<string, string>(
    Array.isArray(riskTruth?.entries) ? riskTruth.entries.map((e: Record<string, unknown>) => [String(e.asset_id || ""), String(e.risk_truth_status || "unknown")]) : []
  );

  const siteNormalized = (Array.isArray(siteSnapshot?.current_universe) ? siteSnapshot.current_universe : [])
    .map((r: Record<string, unknown>) => normalizeRecord(r, String(r.risk_truth_status || "")))
    .filter((r) => r.asset);

  const latestNormalized = (Array.isArray(snap?.records) ? snap.records : [])
    .map((r: Record<string, unknown>) => normalizeRecord(r, truthMap.get(String(r.asset || ""))))
    .filter((r) => r.asset);

  const normalized = [...siteNormalized, ...latestNormalized]
    .filter((r) => (domain ? r.domain === domain : true))
    .filter((r) => {
      const rt = String(r.risk_truth_status || "unknown").toLowerCase();
      if (rt === "unknown") return includeInconclusive || allowed.has("watch");
      if (rt === "inconclusive") return includeInconclusive;
      return allowed.has(rt);
    });
  const dedup = new Map<string, NormalizedAssetRecord>();

  for (const row of normalized) {
    const key = `${row.asset}__${row.domain}`;
    const prev = dedup.get(key);
    const preferCurrent = String(row.source_type || "").includes("lab_corr") || String(row.source_type || "").includes("crypto_research");
    if (!prev || preferCurrent || String(row.timestamp || "") >= String(prev.timestamp || "")) {
      dedup.set(key, row);
    }
  }
  const records = Array.from(dedup.values()).sort((a, b) => a.asset.localeCompare(b.asset));

  if (!records.length) {
    const files = await listLatestFiles();
    return NextResponse.json({
      files,
      records: [],
      run_id: snap?.runId || siteSnapshot?.finance?.lab_run_id || "",
      summary: snap?.summary || siteSnapshot || {},
    });
  }

  return NextResponse.json({
    run_id: snap?.runId || siteSnapshot?.finance?.lab_run_id || "",
    summary: snap?.summary || siteSnapshot || {},
    records,
  });
}
