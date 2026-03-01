"use client";

import { FormEvent, useEffect, useRef, useState } from "react";

type CopilotResponse = {
  ok: boolean;
  answer: string;
};

type Message = {
  id: string;
  role: "assistant" | "user";
  text: string;
};

export default function EigenEngineAssistantFloatingChat() {
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const initialized = useRef(false);

  useEffect(() => {
    if (!open || initialized.current) return;
    initialized.current = true;
    void bootstrap();
  }, [open]);

  async function bootstrap() {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/copilot", { cache: "no-store" });
      const payload = (await res.json()) as CopilotResponse;
      if (!res.ok || !payload?.ok) throw new Error("copilot_unavailable");
      setMessages([
        {
          id: `assistant-${Date.now()}`,
          role: "assistant",
          text: payload.answer,
        },
      ]);
    } catch {
      setError("Nao foi possivel carregar o Eigen Engine Assistant agora.");
    } finally {
      setLoading(false);
    }
  }

  async function sendQuestion(event: FormEvent) {
    event.preventDefault();
    const question = input.trim();
    if (!question || loading) return;
    setInput("");
    setError(null);
    setMessages((prev) => [...prev, { id: `user-${Date.now()}`, role: "user", text: question }]);
    setLoading(true);
    try {
      const res = await fetch("/api/copilot", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      const payload = (await res.json()) as CopilotResponse;
      if (!res.ok || !payload?.ok) throw new Error("copilot_error");
      setMessages((prev) => [...prev, { id: `assistant-${Date.now()}`, role: "assistant", text: payload.answer }]);
    } catch {
      setError("Falha ao responder. Tente novamente.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="fixed bottom-4 right-4 z-50">
      {open ? (
        <section className="w-[360px] max-w-[92vw] h-[520px] rounded-2xl border border-zinc-700 bg-zinc-950/95 shadow-2xl backdrop-blur flex flex-col overflow-hidden">
          <header className="flex items-center justify-between border-b border-zinc-800 px-4 py-3">
            <div>
              <div className="text-sm font-semibold text-zinc-100">Eigen Engine Assistant</div>
              <div className="text-[11px] text-zinc-400">Copiloto do projeto Assyntrax</div>
            </div>
            <button
              type="button"
              onClick={() => setOpen(false)}
              className="rounded-lg border border-zinc-700 px-2 py-1 text-xs text-zinc-300 hover:text-zinc-100"
            >
              fechar
            </button>
          </header>

          <div className="flex-1 overflow-y-auto p-3 space-y-2">
            {!messages.length && loading ? <div className="text-xs text-zinc-500">Carregando contexto...</div> : null}
            {messages.map((msg) => (
              <div
                key={msg.id}
                className={`rounded-xl border px-3 py-2 text-sm whitespace-pre-wrap ${
                  msg.role === "assistant"
                    ? "border-cyan-900/60 bg-cyan-950/30 text-zinc-100"
                    : "ml-6 border-zinc-700 bg-zinc-900 text-zinc-200"
                }`}
              >
                {msg.text}
              </div>
            ))}
          </div>

          <form onSubmit={sendQuestion} className="border-t border-zinc-800 p-3 space-y-2">
            <input
              value={input}
              onChange={(event) => setInput(event.target.value)}
              placeholder="Pergunte sobre risco, regime, gate..."
              className="w-full rounded-xl border border-zinc-700 bg-zinc-900/70 px-3 py-2 text-sm text-zinc-100 outline-none focus:border-cyan-500/60"
            />
            <button
              type="submit"
              disabled={loading}
              className="w-full rounded-xl bg-zinc-100 text-black py-2 text-sm font-medium hover:bg-white disabled:opacity-60"
            >
              {loading ? "Enviando..." : "Enviar"}
            </button>
            {error ? <div className="text-xs text-rose-300">{error}</div> : null}
            <div className="text-[11px] text-zinc-500">
              Diagnostico estrutural. Sem recomendacao de compra/venda.
            </div>
          </form>
        </section>
      ) : null}

      {!open ? (
        <button
          type="button"
          onClick={() => setOpen(true)}
          className="rounded-full border border-cyan-600/50 bg-cyan-950/80 px-4 py-2 text-sm font-medium text-cyan-100 shadow-lg hover:bg-cyan-900/80"
        >
          Eigen Engine Assistant
        </button>
      ) : null}
    </div>
  );
}

