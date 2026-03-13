import { humanizeEngineState, humanizeRiskLevel } from "@/lib/enginePresentation";
import { readSiteFinanceSnapshot } from "@/lib/server/data";

function formatPct(value: number | null | undefined, digits = 0) {
  return typeof value === "number" && Number.isFinite(value) ? `${(value * 100).toFixed(digits)}%` : "n/d";
}

function formatMoney(value: number | null | undefined) {
  return typeof value === "number" && Number.isFinite(value)
    ? value.toLocaleString("pt-BR", { style: "currency", currency: "BRL", maximumFractionDigits: 0 })
    : "n/d";
}

function buildAllocationRows(exposure: number | null | undefined) {
  const bases = [1000, 10000, 50000];
  return bases.map((capital) => {
    const risk = typeof exposure === "number" && Number.isFinite(exposure) ? Math.max(0, capital * exposure) : null;
    const cash = typeof risk === "number" ? Math.max(0, capital - risk) : null;
    return { capital, risk, cash };
  });
}

function meaningFromExposure(exposure: number | null | undefined) {
  if (typeof exposure !== "number" || !Number.isFinite(exposure)) {
    return "Hoje o motor não está te dando uma faixa limpa de exposição. Isso normalmente significa operar pequeno ou esperar.";
  }
  if (exposure <= 0.25) {
    return "O motor está bem defensivo. Em linguagem humana: mais caixa, menos coragem.";
  }
  if (exposure <= 0.6) {
    return "O motor está em modo moderado. Dá para ter risco, mas não é dia de heroísmo.";
  }
  if (exposure <= 1) {
    return "O motor está confortável em carregar risco. Isso não é certeza de alta, só contexto menos hostil.";
  }
  return "O motor está agressivo. Só faz sentido para quem aceita variação forte e sabe que pode tomar susto grande.";
}

function meaningFromGate(blocked: boolean, publishable: boolean) {
  if (blocked || !publishable) {
    return "Quando o gate trava, a leitura continua útil para aprender, mas não para agir como se fosse verdade absoluta.";
  }
  return "Quando o gate está limpo, o motor está mais confiável operacionalmente. Ainda assim, ele não vira garantia.";
}

function simpleActionHint(riskLevel: string, exposure: number | null | undefined, gateBlocked: boolean) {
  const risk = String(riskLevel || "").toLowerCase();
  if (gateBlocked) return "Usar a tela como diagnóstico. Se operar, operar menor do que a vontade inicial.";
  if (risk.includes("alto") || risk.includes("high")) {
    return "Reduzir risco e evitar pressa.";
  }
  if (risk.includes("baixo") || risk.includes("low")) {
    return "Operar com mais liberdade, mas mantendo disciplina.";
  }
  if (typeof exposure === "number" && Number.isFinite(exposure)) {
    if (exposure >= 0.8) return "Pode carregar mais risco, mas sem esquecer caixa e regra de saída.";
    if (exposure >= 0.4) return "Operar com tamanho médio e rebalancear sem impulso.";
  }
  return "Seguir com mão moderada e reavaliar com frequência.";
}

export default async function AplicacoesPage() {
  const snapshot = (await readSiteFinanceSnapshot()) as Record<string, unknown>;
  const finance = ((snapshot.finance as Record<string, unknown> | undefined) || {}) as Record<string, unknown>;
  const playbook = ((finance.latest_playbook as Record<string, unknown> | undefined) || {}) as Record<string, unknown>;
  const shadow = ((snapshot.shadow as Record<string, unknown> | undefined) || {}) as Record<string, unknown>;
  const shadowLatest = ((shadow.latest as Record<string, unknown> | undefined) || {}) as Record<string, unknown>;
  const forecastHorizons = ((snapshot.forecast_horizons as Record<string, unknown> | undefined) || {}) as Record<string, unknown>;
  const dataQuality = ((snapshot.data_quality as Record<string, unknown> | undefined) || {}) as Record<string, unknown>;
  const forecastDaily = ((forecastHorizons.daily as Record<string, unknown> | undefined) || {}) as Record<string, unknown>;
  const forecastWeekly = ((forecastHorizons.weekly as Record<string, unknown> | undefined) || {}) as Record<string, unknown>;
  const forecastMonthly = ((forecastHorizons.monthly as Record<string, unknown> | undefined) || {}) as Record<string, unknown>;
  const playbookStale = finance.latest_playbook_stale === true;
  const playbookStaleDays =
    typeof finance.latest_playbook_stale_days === "number" ? finance.latest_playbook_stale_days : null;
  const ingestionStaleDays =
    typeof dataQuality.ingestion_stale_days === "number" ? dataQuality.ingestion_stale_days : null;
  const ingestionFatalReason = String(dataQuality.ingestion_fatal_reason || "").trim();
  const exposure =
    !playbookStale && typeof playbook.exposure === "number"
      ? playbook.exposure
      : typeof shadowLatest.target_exposure === "number"
        ? shadowLatest.target_exposure
        : typeof forecastMonthly.exposure_target === "number"
          ? forecastMonthly.exposure_target
        : null;
  const rows = buildAllocationRows(exposure);
  const gateBlocked = finance.gate_blocked === true;
  const publishable = !gateBlocked;
  const riskLevel = humanizeRiskLevel(finance.risk_level_next_month);
  const actionHint = simpleActionHint(riskLevel, exposure, gateBlocked);
  const operationalState = humanizeEngineState(finance.operational_state || playbook.regime || "monitoramento_normal");
  const dataLastDate = String(snapshot.as_of_date || finance.data_last_date || shadowLatest.price_date || "n/d");
  const signalReliability =
    typeof playbook.signal_reliability === "number"
      ? playbook.signal_reliability
      : typeof finance.confidence_score === "number"
        ? finance.confidence_score
        : null;

  return (
    <div className="p-5 md:p-6 lg:p-8 space-y-6">
      <section className="rounded-2xl border border-zinc-800 bg-zinc-950/55 p-5">
        <p className="text-xs tracking-[0.14em] uppercase text-zinc-500">Como usar</p>
        <h1 className="mt-2 text-2xl md:text-3xl font-semibold text-zinc-100">Onboarding simples para gente normal</h1>
        <p className="mt-3 max-w-3xl text-sm text-zinc-300">
          Esta tela traduz o Eigen Engine para a vida real. A ideia não é te transformar em trader. É te mostrar como
          usar o motor sem se enrolar, sem all-in e sem fingir que risco não existe.
        </p>
      </section>

      <section className="grid gap-4 md:grid-cols-3">
        <article className="rounded-2xl border border-zinc-800 bg-zinc-950/55 p-5">
          <div className="text-xs uppercase tracking-[0.16em] text-zinc-500">Passo 1</div>
          <h2 className="mt-2 text-lg font-semibold text-zinc-100">Olhe o contexto</h2>
          <p className="mt-3 text-sm text-zinc-300">
            Primeiro entenda se o motor está defensivo, moderado ou agressivo. Isso importa mais do que tentar adivinhar
            o próximo ativo da moda.
          </p>
        </article>

        <article className="rounded-2xl border border-zinc-800 bg-zinc-950/55 p-5">
          <div className="text-xs uppercase tracking-[0.16em] text-zinc-500">Passo 2</div>
          <h2 className="mt-2 text-lg font-semibold text-zinc-100">Decida o tamanho</h2>
          <p className="mt-3 text-sm text-zinc-300">
            A pergunta principal é “quanto do meu dinheiro vai para risco hoje?”. O motor ajuda nisso antes de falar de
            ativo específico.
          </p>
        </article>

        <article className="rounded-2xl border border-zinc-800 bg-zinc-950/55 p-5">
          <div className="text-xs uppercase tracking-[0.16em] text-zinc-500">Passo 3</div>
          <h2 className="mt-2 text-lg font-semibold text-zinc-100">Respeite o freio</h2>
          <p className="mt-3 text-sm text-zinc-300">
            Se o gate travar ou o risco subir, a função do motor é te impedir de fazer besteira por impulso.
          </p>
        </article>
      </section>

      <section className="grid gap-4 md:grid-cols-2">
        <article className="rounded-2xl border border-zinc-800 bg-zinc-950/55 p-5">
          <div className="text-xs uppercase tracking-[0.16em] text-zinc-500">Leitura de hoje</div>
          <div className="mt-2 text-xl font-semibold text-zinc-100">{operationalState}</div>
          <p className="mt-3 text-sm text-zinc-300">{meaningFromExposure(exposure)}</p>
          <p className="mt-3 text-sm text-zinc-400">
            Risco próximo mês: <span className="text-zinc-200">{riskLevel}</span>
            {" · "}
            Exposição alvo: <span className="text-zinc-200">{formatPct(exposure)}</span>
          </p>
          <p className="mt-2 text-sm text-zinc-400">
            Confiabilidade publicada: <span className="text-zinc-200">{signalReliability == null ? "sem dado publicado" : `${Math.round(signalReliability * 100)}%`}</span>
          </p>
          <p className="mt-3 text-sm text-zinc-400">
            Ação prática em linguagem simples: <span className="text-zinc-200">{actionHint}</span>
          </p>
        </article>

        <article className="rounded-2xl border border-zinc-800 bg-zinc-950/55 p-5">
          <div className="text-xs uppercase tracking-[0.16em] text-zinc-500">Quando ignorar o entusiasmo</div>
          <div className="mt-2 text-xl font-semibold text-zinc-100">
            {publishable && !gateBlocked ? "Pode confiar mais" : "Modo estudo"}
          </div>
          <p className="mt-3 text-sm text-zinc-300">{meaningFromGate(gateBlocked, publishable)}</p>
          <p className="mt-3 text-sm text-zinc-400">
            Gate bloqueado: <span className="text-zinc-200">{gateBlocked ? "sim" : "não"}</span>
            {" · "}
            Data-base: <span className="text-zinc-200">{dataLastDate}</span>
          </p>
          <p className="mt-2 text-sm text-zinc-400">
            Última ingestão válida: <span className="text-zinc-200">{String(dataQuality.last_ingestion_data_date || dataLastDate || "n/d")}</span>
          </p>
          {playbookStale ? (
            <p className="mt-2 text-sm text-amber-300">
              A leitura estrutural detalhada ficou {playbookStaleDays == null ? "desatualizada" : `${playbookStaleDays} dias`} atrás.
              As faixas desta página estão usando a operação diária mais recente.
            </p>
          ) : null}
          <p className="mt-2 text-sm text-zinc-400">
            Dias de atraso: <span className="text-zinc-200">{ingestionStaleDays == null ? "n/d" : String(ingestionStaleDays)}</span>
          </p>
          {ingestionFatalReason ? (
            <p className="mt-2 text-sm text-amber-300">Motivo do alerta: {ingestionFatalReason}</p>
          ) : null}
        </article>
      </section>

      <section className="rounded-2xl border border-zinc-800 bg-zinc-950/55 p-5">
        <div className="text-xs uppercase tracking-[0.16em] text-zinc-500">Horizontes do motor</div>
        <h2 className="mt-2 text-lg font-semibold text-zinc-100">Como a leitura muda do dia para a semana e para o mês</h2>
        <div className="mt-4 grid gap-3 md:grid-cols-3">
          {[forecastDaily, forecastWeekly, forecastMonthly].map((horizon, index) => {
            const horizonExposure =
              typeof horizon.exposure_target === "number" ? horizon.exposure_target : null;
            return (
              <article key={`horizon-${index}`} className="rounded-xl border border-zinc-800 bg-black/20 p-4">
                <div className="text-xs uppercase tracking-[0.16em] text-zinc-500">
                  {String(horizon.label || "Horizonte")}
                </div>
                <div className="mt-2 text-base font-semibold text-zinc-100">
                  {humanizeEngineState(String(horizon.mode || "monitoramento_normal"))}
                </div>
                <p className="mt-3 text-sm text-zinc-300">
                  {String(horizon.summary || "Sem leitura adicional publicada para este horizonte.")}
                </p>
                <div className="mt-4 space-y-1 text-sm text-zinc-400">
                  <p>
                    Confiança: <span className="text-zinc-200">{String(horizon.confidence_level || "n/d")}</span>
                  </p>
                  <p>
                    Risco: <span className="text-zinc-200">{String(horizon.risk_level || "n/d")}</span>
                  </p>
                  <p>
                    Exposição: <span className="text-zinc-200">{formatPct(horizonExposure)}</span>
                  </p>
                </div>
              </article>
            );
          })}
        </div>
      </section>

      <section className="rounded-2xl border border-zinc-800 bg-zinc-950/55 p-5">
        <div className="text-xs uppercase tracking-[0.16em] text-zinc-500">Exemplo didático</div>
        <h2 className="mt-2 text-lg font-semibold text-zinc-100">Quanto vai para risco e quanto fica defensivo</h2>
        <div className="mt-4 grid gap-3 md:grid-cols-3">
          {rows.map((row) => (
            <article key={row.capital} className="rounded-xl border border-zinc-800 bg-black/20 p-4">
              <div className="text-sm font-semibold text-zinc-100">{formatMoney(row.capital)}</div>
              <p className="mt-3 text-sm text-zinc-300">
                Em risco: <span className="text-zinc-100">{formatMoney(row.risk)}</span>
              </p>
              <p className="mt-1 text-sm text-zinc-400">
                Em caixa/defensivo: <span className="text-zinc-200">{formatMoney(row.cash)}</span>
              </p>
            </article>
          ))}
        </div>
      </section>

      <section className="rounded-2xl border border-cyan-900/40 bg-cyan-950/15 p-5">
        <div className="text-xs uppercase tracking-[0.16em] text-cyan-300/80">Regras simples para não fazer burrada</div>
        <div className="mt-3 space-y-2 text-sm text-zinc-200">
          <p>- Não use dinheiro de emergência para seguir o motor.</p>
          <p>- Não aumente posição só porque uma semana foi boa.</p>
          <p>- Se você não aguenta ver queda, use menos risco do que o motor sugere, não mais.</p>
          <p>- Se o sinal ficou confuso, a resposta adulta é reduzir tamanho, não adivinhar.</p>
        </div>
      </section>
    </div>
  );
}
