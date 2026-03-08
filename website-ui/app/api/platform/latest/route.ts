import { NextResponse } from "next/server";
import {
  readPlatformDbRelease,
  readPlatformDbSnapshot,
  readPlatformHierarchicalStateLatest,
  readPlatformRankingsLatest,
  readSiteFinanceSnapshot,
} from "@/lib/server/data";

export const dynamic = "force-dynamic";

function toNumber(value: unknown) {
  const num = Number(value);
  return Number.isFinite(num) ? num : null;
}

function buildPayloadFromSiteSnapshot(siteSnapshot: Record<string, unknown>) {
  const universe = Array.isArray(siteSnapshot.current_universe)
    ? (siteSnapshot.current_universe as Record<string, unknown>[])
    : [];
  const sectorPressure = Array.isArray((siteSnapshot.charts as Record<string, unknown> | undefined)?.sector_pressure)
    ? (((siteSnapshot.charts as Record<string, unknown>).sector_pressure as Record<string, unknown>[]))
    : [];
  const asOfDate = String(siteSnapshot.as_of_date || "");
  const confidenceScore = toNumber((siteSnapshot.finance as Record<string, unknown> | undefined)?.confidence_score);
  const riskLevel = String((siteSnapshot.finance as Record<string, unknown> | undefined)?.risk_level_next_month || "indefinido");
  const gateBlocked = Boolean((siteSnapshot.finance as Record<string, unknown> | undefined)?.gate_blocked);
  const statusCounts = universe.reduce(
    (acc, row) => {
      const status = String(row.signal_status || "inconclusive").toLowerCase();
      if (status === "validated") acc.validated += 1;
      else if (status === "watch") acc.watch += 1;
      else acc.inconclusive += 1;
      return acc;
    },
    { validated: 0, watch: 0, inconclusive: 0 }
  );
  const domainCounts = universe.reduce(
    (acc, row) => {
      const domain = String(row.domain || "finance");
      acc[domain] = (acc[domain] || 0) + 1;
      return acc;
    },
    {} as Record<string, number>
  );

  return {
    ok: true,
    snapshot: {
      status: "ok",
      source: "site_finance_snapshot",
      generated_at_utc: String(siteSnapshot.generated_at_utc || ""),
      run_id: String((siteSnapshot.finance as Record<string, unknown> | undefined)?.lab_run_id || ""),
      counts: {
        runs_total: 1,
        asset_rows_total: universe.length,
        asset_rows_for_run: universe.length,
      },
      run: {
        status: gateBlocked ? "diagnostic" : "ok",
        gate_blocked: gateBlocked,
        n_assets: universe.length,
        validated_signals: statusCounts.validated,
        watch_signals: statusCounts.watch,
        inconclusive_signals: statusCounts.inconclusive,
        validated_ratio: universe.length ? statusCounts.validated / universe.length : 0,
      },
      domains: Object.entries(domainCounts).map(([domain, count]) => ({ domain, count })),
      signal_status: [
        { status: "validated", count: statusCounts.validated },
        { status: "watch", count: statusCounts.watch },
        { status: "inconclusive", count: statusCounts.inconclusive },
      ],
      copilot: {
        row_exists: true,
        publishable: !gateBlocked,
        risk_structural: null,
        confidence: confidenceScore,
        risk_level: riskLevel,
      },
    },
    release: {
      updated_at_utc: String(siteSnapshot.generated_at_utc || ""),
      run_id: String((siteSnapshot.finance as Record<string, unknown> | undefined)?.lab_run_id || ""),
      db_path: "site_finance_snapshot",
      latest_db_snapshot: "site_finance_snapshot",
    },
    rankings: {
      status: "ok",
      date: asOfDate,
      top_assets_global_mode: universe
        .slice()
        .sort((a, b) => (toNumber(b.confidence) ?? 0) - (toNumber(a.confidence) ?? 0))
        .slice(0, 12),
      top_sectors_global_mode: sectorPressure.map((row) => ({
        sector: String(row.sector || ""),
        sector_kind: "gics",
        impact: toNumber(row.impact_score),
        risk_mean: toNumber(row.risk_mean),
        confidence_mean: toNumber(row.confidence_mean),
      })),
      sector_global_overlap: [],
      global_state: {
        as_of_date: asOfDate,
        risk_level_next_month: riskLevel,
      },
    },
    hierarchicalState: {
      status: "ok",
      date: asOfDate,
      global_score: confidenceScore,
      top_sectors_by_score: sectorPressure.map((row) => ({
        sector: String(row.sector || ""),
        score: toNumber(row.impact_score),
      })),
      top_sectors_by_loading: [],
      top_sectors_by_overlap: [],
    },
  };
}

export async function GET() {
  const [siteSnapshot, snapshot, release, rankings, hierarchicalState] = await Promise.all([
    readSiteFinanceSnapshot(),
    readPlatformDbSnapshot(),
    readPlatformDbRelease(),
    readPlatformRankingsLatest(),
    readPlatformHierarchicalStateLatest(),
  ]);
  if (String(siteSnapshot?.status || "") === "ok") {
    return NextResponse.json(buildPayloadFromSiteSnapshot(siteSnapshot as Record<string, unknown>), {
      headers: { "Cache-Control": "no-store" },
    });
  }
  return NextResponse.json(
    {
      ok: true,
      snapshot,
      release,
      rankings,
      hierarchicalState,
    },
    { headers: { "Cache-Control": "no-store" } }
  );
}
