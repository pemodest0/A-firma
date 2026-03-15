import { NextResponse } from "next/server";
import { readInvestmentShadowLatest, readSiteFinanceSnapshot } from "@/lib/server/data";

export const dynamic = "force-dynamic";

function asNumber(value: unknown) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

export async function GET() {
  const [payload, siteSnapshot] = (await Promise.all([
    readInvestmentShadowLatest(),
    readSiteFinanceSnapshot(),
  ])) as [Record<string, unknown>, Record<string, unknown>];
  const shadowGods = (((siteSnapshot as Record<string, unknown>)?.shadow_gods as Record<string, unknown> | undefined) ||
    {}) as Record<string, unknown>;
  const shadowGodsOverview = (((siteSnapshot as Record<string, unknown>)?.shadow_gods_overview as
    | Record<string, unknown>
    | undefined) || {}) as Record<string, unknown>;
  const shadowGodsAvailable = Array.isArray(shadowGods.gods) && shadowGods.gods.length > 0;
  if (String(payload.status || "") === "missing") {
    const shadow = ((siteSnapshot as Record<string, unknown>)?.shadow || {}) as Record<string, unknown>;
    if (
      String((siteSnapshot as Record<string, unknown>)?.status || "") === "ok" &&
      (Object.keys(shadow).length || shadowGodsAvailable)
    ) {
      const latest = (shadow.latest || {}) as Record<string, unknown>;
      const live = (shadow.live || {}) as Record<string, unknown>;
      const historical = (shadow.historical_proxy_replay || {}) as Record<string, unknown>;
      const livePortfolio = (live.portfolio || {}) as Record<string, unknown>;
      return NextResponse.json(
        {
          ok: true,
          status: "site_snapshot",
          run_id: String(shadow.run_id || ""),
          generated_at_utc: String(siteSnapshot.generated_at_utc || ""),
          proxies: {
            risk_proxy: "BTC-USD",
            defensive_proxy: "SHY",
          },
          latest: {
            price_date: String(latest.price_date || siteSnapshot.as_of_date || ""),
            signal_date: String(latest.signal_date || ""),
            effective_date: String(latest.effective_date || ""),
            regime: String(latest.regime || ""),
            target_exposure: asNumber(latest.target_exposure),
            gate_blocked: latest.gate_blocked === true,
            freshness_days: asNumber(latest.freshness_days),
          },
          live: {
            status: String(live.status || "ok"),
            capital_start: asNumber(live.capital_start),
            capital_end: asNumber(live.capital_end),
            n_days: asNumber(live.n_days),
            latest_target_exposure: asNumber(live.latest_target_exposure),
            latest_executed_exposure: asNumber(live.latest_executed_exposure),
            latest_regime: String(live.latest_regime || latest.regime || ""),
            edge_vs_benchmark_total_return: asNumber(live.edge_vs_benchmark_total_return),
            portfolio: {
              total_return: asNumber(livePortfolio.total_return),
              ann_return: asNumber(livePortfolio.ann_return),
              ann_vol: asNumber(livePortfolio.ann_vol),
              sharpe: asNumber(livePortfolio.sharpe),
              max_drawdown: asNumber(livePortfolio.max_drawdown),
            },
          },
          historical_proxy_replay: {
            status: String(historical.status || "ok"),
            edge_vs_benchmark_total_return: asNumber(historical.edge_vs_benchmark_total_return),
            portfolio: {
              total_return: asNumber(historical.total_return),
              ann_return: asNumber(historical.ann_return),
              ann_vol: null,
              sharpe: asNumber(historical.sharpe),
              max_drawdown: asNumber(historical.max_drawdown),
            },
          },
          refresh_prices: { ok: null, failed: null },
          shadow_gods_available: shadowGodsAvailable,
          shadow_gods_overview: shadowGodsAvailable ? shadowGodsOverview : {},
          shadow_gods: shadowGodsAvailable ? shadowGods : {},
          source: "site_finance_snapshot",
        },
        { headers: { "Cache-Control": "no-store" } }
      );
    }
    return NextResponse.json(
      {
        ok: false,
        status: "missing",
      },
      { status: 503, headers: { "Cache-Control": "no-store" } }
    );
  }

  const latest = (payload.latest || {}) as Record<string, unknown>;
  const live = (payload.live || {}) as Record<string, unknown>;
  const livePortfolio = (live.portfolio || {}) as Record<string, unknown>;
  const historical = (payload.historical_proxy_replay || {}) as Record<string, unknown>;
  const historicalPortfolio = (historical.portfolio || {}) as Record<string, unknown>;

  return NextResponse.json(
    {
      ok: true,
      status: String(payload.status || "ok"),
      run_id: String(payload.run_id || ""),
      generated_at_utc: String(payload.generated_at_utc || ""),
      proxies: payload.proxies || {},
      latest: {
        price_date: String(latest.price_date || ""),
        signal_date: String(latest.signal_date || ""),
        effective_date: String(latest.effective_date || ""),
        regime: String(latest.regime || ""),
        target_exposure: asNumber(latest.target_exposure),
        gate_blocked: latest.gate_blocked === true,
        freshness_days: asNumber(latest.freshness_days),
      },
      live: {
        status: String(live.status || ""),
        capital_start: asNumber(live.capital_start),
        capital_end: asNumber(live.capital_end),
        n_days: asNumber(live.n_days),
        latest_target_exposure: asNumber(live.latest_target_exposure),
        latest_executed_exposure: asNumber(live.latest_executed_exposure),
        latest_regime: String(live.latest_regime || ""),
        edge_vs_benchmark_total_return: asNumber(live.edge_vs_benchmark_total_return),
        portfolio: {
          total_return: asNumber(livePortfolio.total_return),
          ann_return: asNumber(livePortfolio.ann_return),
          ann_vol: asNumber(livePortfolio.ann_vol),
          sharpe: asNumber(livePortfolio.sharpe),
          max_drawdown: asNumber(livePortfolio.max_drawdown),
        },
      },
      historical_proxy_replay: {
        status: String(historical.status || ""),
        edge_vs_benchmark_total_return: asNumber(historical.edge_vs_benchmark_total_return),
        portfolio: {
          total_return: asNumber(historicalPortfolio.total_return),
          ann_return: asNumber(historicalPortfolio.ann_return),
          ann_vol: asNumber(historicalPortfolio.ann_vol),
          sharpe: asNumber(historicalPortfolio.sharpe),
          max_drawdown: asNumber(historicalPortfolio.max_drawdown),
        },
      },
      refresh_prices: payload.refresh_prices || {},
      shadow_gods_available: shadowGodsAvailable,
      shadow_gods_overview: shadowGodsAvailable ? shadowGodsOverview : {},
      shadow_gods: shadowGodsAvailable ? shadowGods : {},
    },
    { headers: { "Cache-Control": "no-store" } }
  );
}
