import { NextResponse } from "next/server";
import {
  readOverfitGuardrailsLatest,
  readPortfolioSimulationLatest,
  readPortfolioSystematicLatest,
  readSiteFinanceSnapshot,
} from "@/lib/server/data";
import { humanizeStrategyName } from "@/lib/enginePresentation";

export const dynamic = "force-dynamic";

function asNumber(value: unknown) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function asBool(value: unknown) {
  return value === true;
}

export async function GET() {
  const [guardrails, simulation, systematic, siteSnapshot] = await Promise.all([
    readOverfitGuardrailsLatest(),
    readPortfolioSimulationLatest(),
    readPortfolioSystematicLatest(),
    readSiteFinanceSnapshot(),
  ]);

  const missing: string[] = [];
  if (String((guardrails as Record<string, unknown>)?.status || "") === "missing") {
    missing.push("overfit_guardrails_summary_missing");
  }
  if (String((simulation as Record<string, unknown>)?.status || "") === "missing") {
    missing.push("portfolio_simulation_summary_missing");
  }
  if (String((systematic as Record<string, unknown>)?.status || "") === "missing") {
    missing.push("portfolio_systematic_summary_missing");
  }
  if (missing.length > 0) {
    const finance = ((siteSnapshot as Record<string, unknown>)?.finance || {}) as Record<string, unknown>;
    const layered = ((siteSnapshot as Record<string, unknown>)?.layered_engine || {}) as Record<string, unknown>;
    const charts = ((siteSnapshot as Record<string, unknown>)?.charts || {}) as Record<string, unknown>;
    const playbook = (finance.latest_playbook || {}) as Record<string, unknown>;
    const attack = (layered.best_meta_candidate || {}) as Record<string, unknown>;
    const robust = (layered.drawdown_best_balanced || {}) as Record<string, unknown>;
    const shadow = (((siteSnapshot as Record<string, unknown>)?.shadow || {}) as Record<string, unknown>);
    const historicalShadow = ((shadow.historical_proxy_replay || {}) as Record<string, unknown>);
    const watchlist = Array.isArray(charts.asset_watchlist) ? (charts.asset_watchlist as Record<string, unknown>[]) : [];
    if (String((siteSnapshot as Record<string, unknown>)?.status || "") === "ok") {
      const exposure = asNumber(playbook.exposure);
      const topAssets = watchlist.slice(0, 10).map((row) => ({
        asset_id: String(row.asset || ""),
        ticker: String(row.asset || ""),
        sector_gics: String(row.sector || row.group || ""),
        weight: exposure != null && watchlist.length ? exposure / Math.min(10, watchlist.length) : null,
        amount_1000: exposure != null && watchlist.length ? (1000 * exposure) / Math.min(10, watchlist.length) : null,
        amount_10000: exposure != null && watchlist.length ? (10000 * exposure) / Math.min(10, watchlist.length) : null,
        amount_100000: exposure != null && watchlist.length ? (100000 * exposure) / Math.min(10, watchlist.length) : null,
      }));
      const publishable = finance.gate_blocked !== true;
      const advisoryReady = Boolean(attack.candidate_id || robust.candidate_id);
      return NextResponse.json(
        {
          ok: true,
          status: "site_snapshot",
          guardrails: {
            publishable,
            advisory_ready: advisoryReady,
            step_status: {
              dados_publicados: Boolean(siteSnapshot.as_of_date),
              regime_publicado: Boolean(playbook.regime),
              pesquisa_publicada: Boolean((siteSnapshot as Record<string, unknown>)?.profit_research),
              shadow_publicado: Boolean((siteSnapshot as Record<string, unknown>)?.shadow),
            },
          },
          simulation: {
            run_id: String(finance.lab_run_id || ""),
            test_start: String(historicalShadow.start_date || ""),
            test_end: String(siteSnapshot.as_of_date || ""),
            latest_rebalance: {
              date: String(playbook.date || siteSnapshot.as_of_date || ""),
              regime: String(playbook.regime || ""),
              risk_bucket: String(finance.risk_level_next_month || ""),
              cash_weight: exposure != null ? Math.max(0, 1 - exposure) : null,
              signal_reliability: asNumber(playbook.signal_reliability),
            },
            top_assets: topAssets,
            performance: {
              ann_strategy: asNumber(robust.net_ann_return ?? attack.net_ann_return),
              ann_eqw: null,
              ann_edge: null,
              max_drawdown_strategy: asNumber(robust.net_max_drawdown ?? attack.net_max_drawdown),
              max_drawdown_eqw: null,
              drawdown_edge: null,
              signal_reliability: asNumber(playbook.signal_reliability),
            },
          },
          systematic: {
            run_id: String(finance.lab_run_id || ""),
            years_tested: [],
            worth_it_rate_vs_eqw: null,
            monthly_alpha_prob_positive_vs_eqw: null,
          },
          strategy_state: publishable ? "operacional" : "advisory_controlado",
          guidance: [
            publishable
              ? "Contexto publicado e íntegro. Ainda não é promessa de retorno, mas a trilha operacional está consistente."
              : "O gate do motor está bloqueado. Use esta tela como apoio de leitura, não como execução cega.",
            `Modo ataque atual: ${humanizeStrategyName(String(attack.candidate_id || "n/d"))}.`,
            `Modo robusto atual: ${humanizeStrategyName(String(robust.candidate_id || "n/d"))}.`,
          ],
          source: "site_finance_snapshot",
        },
        { headers: { "Cache-Control": "no-store" } }
      );
    }
    return NextResponse.json(
      {
        ok: false,
        status: "not_ready",
        missing,
      },
      { status: 503, headers: { "Cache-Control": "no-store" } }
    );
  }

  const guard = guardrails as Record<string, unknown>;
  const sim = simulation as Record<string, unknown>;
  const sys = systematic as Record<string, unknown>;
  const finalGate = (guard.final_gate || {}) as Record<string, unknown>;
  const stepsObj = (guard.steps || {}) as Record<string, unknown>;
  const stepStatus = Object.fromEntries(
    Object.entries(stepsObj).map(([k, v]) => [k, asBool((v as Record<string, unknown>)?.pass)])
  ) as Record<string, boolean>;

  const simSummary = (sim.summary || {}) as Record<string, unknown>;
  const perf = (simSummary.performance || {}) as Record<string, unknown>;
  const strategy = (perf.test_strategy || {}) as Record<string, unknown>;
  const eqw = (perf.test_eqw_universe || {}) as Record<string, unknown>;
  const systematicSummary = (sys.summary || {}) as Record<string, unknown>;

  const annStrategy = asNumber(strategy.ann_return);
  const annEqw = asNumber(eqw.ann_return);
  const mddStrategy = asNumber(strategy.max_drawdown);
  const mddEqw = asNumber(eqw.max_drawdown);
  const annEdge = annStrategy != null && annEqw != null ? annStrategy - annEqw : null;
  const drawdownEdge = mddStrategy != null && mddEqw != null ? mddStrategy - mddEqw : null;

  const publishable = asBool(finalGate.publishable);
  const advisoryReady = asBool(finalGate.advisory_ready);

  const strategyLevel = (() => {
    if (publishable && (annEdge ?? -1) >= 0 && (drawdownEdge ?? -1) >= 0) return "operacional";
    if (advisoryReady && (drawdownEdge ?? -1) >= -0.03) return "advisory_controlado";
    return "restrito";
  })();

  const guidance: string[] = [];
  if (strategyLevel === "operacional") {
    guidance.push("Condições mínimas atendidas para uso operacional com monitoramento contínuo.");
  } else if (strategyLevel === "advisory_controlado") {
    guidance.push("Usar somente como apoio de decisão, sem automatizar execução.");
  } else {
    guidance.push("Não usar para execução. Manter apenas como laboratório de hipótese.");
  }
  if ((annEdge ?? -1) < 0) {
    guidance.push("Retorno anual da estratégia ficou abaixo do portfólio equiponderado no teste mais recente.");
  }
  if ((drawdownEdge ?? -1) < 0) {
    guidance.push("Queda máxima da estratégia ficou pior que o portfólio equiponderado no teste mais recente.");
  }
  if (!publishable) {
    guidance.push("Gate de produção reprovado: revisar baseline e testes de causalidade antes de publicar.");
  }

  return NextResponse.json(
    {
      ok: true,
      status: "ok",
      guardrails: {
        publishable,
        advisory_ready: advisoryReady,
        step_status: stepStatus,
      },
      simulation: {
        run_id: sim.run_id || "",
        test_start: simSummary.test_start || "",
        test_end: simSummary.test_end || "",
        latest_rebalance: simSummary.latest_rebalance || {},
        top_assets: Array.isArray(sim.top_assets) ? sim.top_assets : [],
        performance: {
          ann_strategy: annStrategy,
          ann_eqw: annEqw,
          ann_edge: annEdge,
          max_drawdown_strategy: mddStrategy,
          max_drawdown_eqw: mddEqw,
          drawdown_edge: drawdownEdge,
        },
      },
      systematic: {
        run_id: sys.run_id || "",
        years_tested: systematicSummary.years_tested || [],
        worth_it_rate_vs_eqw: asNumber(systematicSummary.worth_it_rate_vs_eqw),
        monthly_alpha_prob_positive_vs_eqw: asNumber(systematicSummary.monthly_alpha_prob_positive_vs_eqw),
      },
      strategy_state: strategyLevel,
      guidance,
    },
    { headers: { "Cache-Control": "no-store" } }
  );
}
