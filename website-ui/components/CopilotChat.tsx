"use client";

import { useCallback, useEffect, useRef, useState } from "react";

type AssetSample = {
  asset: string;
  confidence: number;
  quality: number;
};

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
  watch_assets: AssetSample[];
  inconclusive_assets: AssetSample[];
  sources: string[];
};

type CopilotResponse = {
  ok: boolean;
  mode: string;
  context: CopilotContext;
  answer: string;
};

type Message = {
  id: string;
  role: "assistant" | "user";
  text: string;
};

const quickPrompts = [
  "Me de um resumo do run atual",
  "Qual o estado operacional e a acao sugerida?",
  "Como esta o gate de publicacao?",
  "Quais ativos estao em watch e inconclusive?",
  "Qual status dos modelos B e C?",
  "Explique a causalidade usada aqui",
];

function fmt(value: number | null, digits = 3) {
  if (value == null) return "--";
  return value.toFixed(digits);
}

export default function CopilotChat() {
  const [context, setContext] = useState<CopilotContext | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const bootstrapped = useRef(false);

  const refresh = useCallback(async () => {
    setRefreshing(true);
    setError(null);
    try {
      const res = await fetch("/api/copilot", { cache: "no-store" });
      const payload = (await res.json()) as CopilotResponse;
      if (!res.ok || !payload?.ok) throw new Error("copilot_api_unavailable");
      setContext(payload.context);
      if (!bootstrapped.current) {
        setMessages([
          {
            id: `assistant-${Date.now()}`,
            role: "assistant",
            text:
              payload.answer +
              "\n\nPergunte sobre gate, causalidade, ativos em watch ou status dos modelos B/C.",
          },
        ]);
        bootstrapped.current = true;
      }
    } catch {
      setError("Nao foi possivel carregar o copiloto agora.");
    } finally {
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
    const timer = setInterval(() => {
      void refresh();
    }, 15000);
    return () => clearInterval(timer);
  }, [refresh]);

  const sendQuestion = useCallback(
    async (questionRaw?: string) => {
      const question = (questionRaw ?? input).trim();
      if (!question || loading) return;

      const userMsg: Message = { id: `user-${Date.now()}`, role: "user", text: question };
      setMessages((prev) => [...prev, userMsg]);
      setInput("");
      setLoading(true);
      setError(null);

      try {
        const res = await fetch("/api/copilot", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question }),
        });
        const payload = (await res.json()) as CopilotResponse;
        if (!res.ok || !payload?.ok) throw new Error("copilot_api_error");
        setContext(payload.context);
        setMessages((prev) => [
          ...prev,
          {
            id: `assistant-${Date.now()}`,
            role: "assistant",
            text: payload.answer,
          },
        ]);
      } catch {
        setError("Falha ao responder. Tente novamente.");
      } finally {
        setLoading(false);
      }
    },
    [input, loading]
  );

  return (
    <div className="p-5 md:p-6 lg:p-8 space-y-6">
      <section className="rounded-2xl border border-zinc-800 bg-zinc-950/50 p-5">
        <p className="text-xs tracking-[0.14em] uppercase text-zinc-500">Copiloto</p>
        <h1 className="mt-2 text-2xl md:text-3xl font-semibold text-zinc-100">
          IA fisico-matematica para leitura do Eigen Engine
        </h1>
        <p className="mt-3 text-sm text-zinc-300">
          Chat operacional em tempo real com base nos artefatos do run, sem dependencia externa.
        </p>
      </section>

      <div className="grid grid-cols-1 xl:grid-cols-[320px_1fr] gap-4">
        <aside className="rounded-2xl border border-zinc-800 bg-zinc-950/45 p-4 space-y-4">
          <div className={`rounded-xl border p-3 text-xs ${context?.governance.publishable ? "border-emerald-700/60 text-emerald-300" : "border-amber-700/60 text-amber-300"}`}>
            publicacao: {context?.governance.publishable ? "publicavel" : "nao publicavel"}
          </div>

          <div>
            <div className="text-xs uppercase tracking-[0.12em] text-zinc-500">Run</div>
            <div className="mt-1 text-sm text-zinc-100">{context?.run.id || "--"}</div>
            <div className="mt-1 text-xs text-zinc-400">
              gate: {context?.run.gate_blocked ? "nao verde" : "verde"} | politica: {context?.run.policy || "--"}
            </div>
          </div>

          <div>
            <div className="text-xs uppercase tracking-[0.12em] text-zinc-500">Brief IA</div>
            <div className="mt-1 text-xs text-zinc-300">
              estado {context?.operational_brief.operational_state || "--"} | risco mes{" "}
              {context?.operational_brief.risk_level_next_month || "--"}
            </div>
            <div className="mt-1 text-xs text-zinc-400">
              data base {context?.operational_brief.data_last_date || "--"} | freshness{" "}
              {context?.operational_brief.freshness_status || "--"}
              {context?.operational_brief.freshness_days_lag == null ? "" : ` (${context?.operational_brief.freshness_days_lag}d)`}
            </div>
          </div>

          <div>
            <div className="text-xs uppercase tracking-[0.12em] text-zinc-500">Risco x Confianca</div>
            <div className="mt-1 text-xs text-zinc-300">
              risco {fmt(context?.governance.risk_structural ?? null)} ({context?.governance.risk_level || "--"})
            </div>
            <div className="mt-1 text-xs text-zinc-300">confianca {fmt(context?.governance.confidence ?? null)}</div>
          </div>

          <div>
            <div className="text-xs uppercase tracking-[0.12em] text-zinc-500">Universo</div>
            <div className="mt-1 text-xs text-zinc-300">
              ativos {context?.universe.assets ?? 0} | validated {context?.universe.validated ?? 0} | watch{" "}
              {context?.universe.watch ?? 0} | inconclusive {context?.universe.inconclusive ?? 0}
            </div>
          </div>

          <div>
            <div className="text-xs uppercase tracking-[0.12em] text-zinc-500">Macro</div>
            <div className="mt-1 text-xs text-zinc-300">
              regime {context?.lab.regime || "--"} | tier {context?.lab.signal_tier || "--"} | conf{" "}
              {fmt(context?.lab.signal_reliability ?? null)}
            </div>
            <div className="mt-1 text-xs text-zinc-400">
              n_used {fmt(context?.lab.n_used ?? null, 0)} | structure_score {fmt(context?.lab.structure_score ?? null)}
            </div>
          </div>

          <div>
            <div className="text-xs uppercase tracking-[0.12em] text-zinc-500">Modelos</div>
            <div className="mt-1 text-xs text-zinc-300">
              B: {context?.model_b.status || "--"} | risco {fmt(context?.model_b.risk_score ?? null)} | conf{" "}
              {fmt(context?.model_b.confidence ?? null)}
            </div>
            <div className="mt-1 text-xs text-zinc-300">
              C: {context?.model_c.status || "--"} | risco {fmt(context?.model_c.risk_score ?? null)} | conf{" "}
              {fmt(context?.model_c.confidence ?? null)}
            </div>
            <div className="mt-1 text-xs text-zinc-400">
              C publish_ready: {context?.model_c.publish_ready ? "sim" : "nao"}
            </div>
          </div>

          <div>
            <div className="text-xs uppercase tracking-[0.12em] text-zinc-500">Nucleo</div>
            <div className="mt-1 text-xs text-zinc-300">versao {context?.instruction_core.version || "--"}</div>
          </div>

          <div>
            <div className="text-xs uppercase tracking-[0.12em] text-zinc-500">Banco</div>
            <div className="mt-1 text-xs text-zinc-300">
              status {context?.platform_db.status || "--"} | run {context?.platform_db.run_id || "--"}
            </div>
            <div className="mt-1 text-xs text-zinc-400">
              rows {context?.platform_db.rows_for_run ?? 0} | runs {context?.platform_db.runs_total ?? 0} | copiloto row{" "}
              {context?.platform_db.copilot_row_exists ? "sim" : "nao"}
            </div>
          </div>

          <div className="text-xs text-zinc-500">
            Atualizacao automatica a cada 15s. Estado: {refreshing ? "sincronizando" : "ok"}.
          </div>
        </aside>

        <section className="rounded-2xl border border-zinc-800 bg-zinc-950/45 p-4 flex flex-col gap-3">
          <div className="flex flex-wrap gap-2">
            {quickPrompts.map((prompt) => (
              <button
                key={prompt}
                type="button"
                className="rounded-full border border-zinc-700 px-3 py-1 text-xs text-zinc-300 hover:border-cyan-500/60 hover:text-cyan-200 transition"
                onClick={() => void sendQuestion(prompt)}
              >
                {prompt}
              </button>
            ))}
          </div>

          <div className="rounded-xl border border-zinc-800 bg-black/35 p-3 h-[420px] overflow-y-auto space-y-3">
            {messages.map((msg) => (
              <div
                key={msg.id}
                className={`max-w-[92%] rounded-xl border px-3 py-2 text-sm whitespace-pre-wrap ${
                  msg.role === "assistant"
                    ? "border-cyan-900/60 bg-cyan-950/20 text-zinc-100"
                    : "ml-auto border-zinc-700 bg-zinc-900/80 text-zinc-200"
                }`}
              >
                {msg.text}
              </div>
            ))}
            {!messages.length ? <div className="text-sm text-zinc-500">Carregando copiloto...</div> : null}
          </div>

          <form
            className="flex gap-2"
            onSubmit={(event) => {
              event.preventDefault();
              void sendQuestion();
            }}
          >
            <input
              value={input}
              onChange={(event) => setInput(event.target.value)}
              className="flex-1 rounded-xl border border-zinc-700 bg-zinc-950/80 px-3 py-2 text-sm text-zinc-100 outline-none focus:border-cyan-500/70"
              placeholder="Pergunte sobre regime, gate, causalidade, ativos ou modelos B/C..."
            />
            <button
              type="submit"
              disabled={loading}
              className="rounded-xl bg-zinc-100 text-black px-4 py-2 text-sm font-medium hover:bg-white transition disabled:opacity-60"
            >
              {loading ? "Enviando..." : "Enviar"}
            </button>
          </form>

          {error ? <div className="text-xs text-red-300">{error}</div> : null}
          <div className="text-xs text-zinc-500">
            Modo: diagnostico estrutural. Nao recomenda compra/venda e nao promete retorno.
          </div>
        </section>
      </div>
    </div>
  );
}
