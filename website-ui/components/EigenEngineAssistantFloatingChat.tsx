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

type ChatMode = "floating" | "embedded";

export default function EigenEngineAssistantFloatingChat({
  mode = "floating",
  className = "",
}: {
  mode?: ChatMode;
  className?: string;
}) {
  const quickPrompts = [
    "Oi",
    "O que você faz?",
    "Como usar o motor hoje?",
    "Finanças ou cripto agora?",
  ];
  const embedded = mode === "embedded";
  const [open, setOpen] = useState(embedded);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const initialized = useRef(false);

  useEffect(() => {
    const shouldLoad = embedded || open;
    if (!shouldLoad || initialized.current) return;
    initialized.current = true;
    void bootstrap();
  }, [embedded, open]);

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
      setError("Não foi possível carregar o copiloto agora.");
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

  async function askQuickPrompt(question: string) {
    if (loading) return;
    setInput(question);
    setMessages((prev) => [...prev, { id: `user-${Date.now()}`, role: "user", text: question }]);
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/copilot", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      const payload = (await res.json()) as CopilotResponse;
      if (!res.ok || !payload?.ok) throw new Error("copilot_error");
      setInput("");
      setMessages((prev) => [...prev, { id: `assistant-${Date.now()}`, role: "assistant", text: payload.answer }]);
    } catch {
      setError("Falha ao responder. Tente novamente.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className={embedded ? className : "fixed bottom-4 right-4 z-50"}>
      {open ? (
        <section
          className={`rounded-2xl border border-zinc-700 bg-zinc-950/95 shadow-2xl backdrop-blur flex flex-col overflow-hidden ${
            embedded ? "h-[650px] w-full" : "h-[520px] w-[360px] max-w-[92vw]"
          }`}
        >
          <header className="flex items-center justify-between border-b border-zinc-800 px-4 py-3">
            <div>
              <div className="text-sm font-semibold text-zinc-100">Copiloto do Eigen Engine</div>
              <div className="text-[11px] text-zinc-400">Ajuda prática para entender risco, exposição e sleeves</div>
            </div>
            {!embedded ? (
              <button
                type="button"
                onClick={() => setOpen(false)}
                className="rounded-lg border border-zinc-700 px-2 py-1 text-xs text-zinc-300 hover:text-zinc-100"
              >
                fechar
              </button>
            ) : null}
          </header>

          <div className="flex-1 overflow-y-auto p-3 space-y-2">
            {!messages.length && loading ? <div className="text-xs text-zinc-500">Carregando contexto...</div> : null}
            {!messages.length && !loading ? (
              <div className="rounded-2xl border border-zinc-800 bg-black/20 p-4 text-sm leading-7 text-zinc-300">
                Pode falar comigo como se estivesse pedindo ajuda para uma pessoa normal. Eu traduzo o motor para risco,
                exposição, finanças, cripto e uso prático do app.
              </div>
            ) : null}
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
            <div className="flex flex-wrap gap-2">
              {quickPrompts.map((prompt) => (
                <button
                  key={prompt}
                  type="button"
                  onClick={() => void askQuickPrompt(prompt)}
                  className="rounded-full border border-zinc-700 bg-zinc-900/60 px-3 py-1.5 text-xs text-zinc-300 hover:border-cyan-500/40 hover:text-zinc-100"
                >
                  {prompt}
                </button>
              ))}
            </div>
            <input
              value={input}
              onChange={(event) => setInput(event.target.value)}
              placeholder="Pergunte do seu jeito: risco hoje, exposição, cripto ou finanças..."
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
              Diagnóstico estrutural em linguagem simples. Sem recomendação de compra ou venda.
            </div>
          </form>
        </section>
      ) : null}

      {!embedded && !open ? (
        <button
          type="button"
          onClick={() => setOpen(true)}
          className="rounded-full border border-cyan-600/50 bg-cyan-950/80 px-4 py-2 text-sm font-medium text-cyan-100 shadow-lg hover:bg-cyan-900/80"
        >
          Conversar com o copiloto
        </button>
      ) : null}
    </div>
  );
}
