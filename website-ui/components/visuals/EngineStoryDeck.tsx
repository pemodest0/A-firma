"use client";

import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useMemo, useState } from "react";

const FRAMES = [
  {
    id: "dados",
    kicker: "01 · Séries limpas",
    title: "Primeiro limpamos o que pode distorcer a leitura",
    body:
      "O motor começa com retornos diários por ativo, checa cobertura temporal, corta extremos da janela e evita misturar buraco de dado com mudança real de mercado.",
    points: ["Cobertura mínima por ativo", "Janela causal", "Outliers tratados antes do espectro"],
    caption: "Sem dado limpo, a matriz só repete ruído com aparência de ciência.",
  },
  {
    id: "matriz",
    kicker: "02 · Matriz viva",
    title: "A correlação evolui como um organismo, não como uma foto parada",
    body:
      "A matriz de correlação mostra quando o mercado deixa de andar disperso e passa a se mover em bloco. É aí que a física estatística começa a ficar útil.",
    points: ["Mais bloco = mais sincronização", "Menos breadth = mais fragilidade", "Mudança estrutural aparece antes da manchete"],
    caption: "A matriz não adivinha preço. Ela mede organização coletiva.",
  },
  {
    id: "espectro",
    kicker: "03 · Autovalores",
    title: "O espectro separa estrutura de coincidência",
    body:
      "Quando poucos autovalores crescem demais, um fator dominante começa a engolir o resto. Isso costuma sinalizar risco sistêmico e queda de diversidade real.",
    points: ["Autovetor dominante", "Dimensão efetiva", "Ruído filtrado por banda espectral"],
    caption: "O mercado fica previsível não por magia, mas porque perde graus de liberdade.",
  },
  {
    id: "regime",
    kicker: "04 · Regime",
    title: "A leitura estrutural vira regra de risco",
    body:
      "Depois do espectro, o motor classifica o contexto em leitura estável, transição ou estresse. Essa camada decide o orçamento de risco antes da escolha do sleeve.",
    points: ["Walk-forward sem look-ahead", "Histerese para evitar flip-flop", "Gate bloqueia publicação ruim"],
    caption: "Não é previsão de candle. É controle do tamanho do erro.",
  },
  {
    id: "insights",
    kicker: "05 · Ação prática",
    title: "No fim, a matemática vira decisão humana",
    body:
      "A leitura estrutural conversa com ranking, sleeves e execução. O resultado final é uma faixa de exposição, um modo do motor e um caderno de insights por ativo.",
    points: ["Modo ataque", "Modo robusto", "Insights simples por ativo e setor"],
    caption: "O valor do sistema está em ligar contexto, risco e decisão numa trilha auditável.",
  },
] as const;

function DataGraphic() {
  return (
    <svg viewBox="0 0 560 300" className="h-full w-full">
      <defs>
        <linearGradient id="seriesA" x1="0%" x2="100%">
          <stop offset="0%" stopColor="#38bdf8" />
          <stop offset="100%" stopColor="#22d3ee" />
        </linearGradient>
        <linearGradient id="seriesB" x1="0%" x2="100%">
          <stop offset="0%" stopColor="#f59e0b" />
          <stop offset="100%" stopColor="#fb7185" />
        </linearGradient>
      </defs>
      <rect x="0" y="0" width="560" height="300" rx="28" fill="rgba(3,7,18,0.75)" />
      <g opacity="0.16" stroke="#94a3b8">
        {[40, 90, 140, 190, 240].map((y) => (
          <line key={y} x1="28" y1={y} x2="532" y2={y} />
        ))}
      </g>
      <motion.path
        d="M32 196 C86 174, 126 160, 180 152 S274 124, 326 136 S418 178, 528 98"
        stroke="url(#seriesA)"
        strokeWidth="4"
        fill="none"
        initial={{ pathLength: 0, opacity: 0.45 }}
        animate={{ pathLength: 1, opacity: 1 }}
        transition={{ duration: 1.1 }}
      />
      <motion.path
        d="M32 214 C88 206, 132 188, 180 172 S272 132, 326 164 S418 208, 528 152"
        stroke="url(#seriesB)"
        strokeWidth="3"
        fill="none"
        initial={{ pathLength: 0, opacity: 0.35 }}
        animate={{ pathLength: 1, opacity: 0.95 }}
        transition={{ duration: 1.15, delay: 0.15 }}
      />
      {[86, 180, 274, 418, 528].map((x, index) => (
        <motion.circle
          key={x}
          cx={x}
          cy={[174, 152, 124, 178, 98][index]}
          r="5"
          fill="#e2e8f0"
          initial={{ scale: 0.4, opacity: 0.2 }}
          animate={{ scale: [0.9, 1.15, 0.9], opacity: 1 }}
          transition={{ repeat: Infinity, duration: 2.8, delay: index * 0.2 }}
        />
      ))}
      <g fontSize="12" fill="#94a3b8">
        <text x="34" y="28">Dados de retorno por ativo</text>
        <text x="372" y="28">Cobertura + consistência</text>
        <text x="34" y="282">janela temporal ativa</text>
      </g>
    </svg>
  );
}

function MatrixGraphic() {
  const cells = useMemo(
    () =>
      Array.from({ length: 7 }, (_, row) =>
        Array.from({ length: 7 }, (_, col) => {
          const strong = Math.abs(row - col) <= 1;
          const base = strong ? 0.82 : Math.max(0.1, 0.58 - Math.abs(row - col) * 0.12);
          return { row, col, base };
        })
      ).flat(),
    []
  );

  return (
    <svg viewBox="0 0 560 300" className="h-full w-full">
      <rect x="0" y="0" width="560" height="300" rx="28" fill="rgba(3,7,18,0.75)" />
      <g transform="translate(72 42)">
        {cells.map((cell, idx) => (
          <motion.rect
            key={`${cell.row}-${cell.col}`}
            x={cell.col * 48}
            y={cell.row * 28}
            width="38"
            height="20"
            rx="4"
            fill={`rgba(34,211,238,${cell.base})`}
            stroke="rgba(255,255,255,0.06)"
            initial={{ opacity: 0.25 }}
            animate={{ opacity: [0.35, 0.9, 0.45] }}
            transition={{ duration: 3.2, repeat: Infinity, delay: idx * 0.03 }}
          />
        ))}
      </g>
      <g fontSize="12" fill="#94a3b8">
        <text x="74" y="24">Matriz de correlação</text>
        <text x="340" y="278">mais brilho = mais sincronização</text>
      </g>
      <motion.path
        d="M80 242 C170 214, 250 214, 338 242"
        stroke="#38bdf8"
        strokeWidth="2"
        fill="none"
        opacity="0.6"
        animate={{ pathLength: [0.15, 1, 0.3] }}
        transition={{ repeat: Infinity, duration: 3.4 }}
      />
    </svg>
  );
}

function SpectrumGraphic() {
  const bars = [0.92, 0.56, 0.31, 0.18, 0.13, 0.09, 0.07];
  return (
    <svg viewBox="0 0 560 300" className="h-full w-full">
      <rect x="0" y="0" width="560" height="300" rx="28" fill="rgba(3,7,18,0.75)" />
      <g transform="translate(70 36)">
        {bars.map((value, idx) => {
          const height = value * 170;
          return (
            <motion.rect
              key={idx}
              x={idx * 54}
              y={188 - height}
              width="28"
              height={height}
              rx="8"
              fill={idx === 0 ? "#22d3ee" : idx === 1 ? "#60a5fa" : "#64748b"}
              initial={{ height: 8, y: 180 }}
              animate={{ height, y: 188 - height }}
              transition={{ duration: 0.7, delay: idx * 0.08 }}
            />
          );
        })}
      </g>
      <motion.path
        d="M72 246 C150 210, 216 180, 292 152 S424 110, 494 76"
        stroke="#f97316"
        strokeWidth="3"
        fill="none"
        initial={{ pathLength: 0.15, opacity: 0.4 }}
        animate={{ pathLength: 1, opacity: 1 }}
        transition={{ duration: 1 }}
      />
      <g fontSize="12" fill="#94a3b8">
        <text x="72" y="24">Autovalores e concentração de risco</text>
        <text x="352" y="278">λ₁ sobe quando um fator domina tudo</text>
      </g>
    </svg>
  );
}

function RegimeGraphic() {
  const bands = [
    { x: 56, width: 90, label: "estável", color: "#10b981" },
    { x: 156, width: 86, label: "transição", color: "#f59e0b" },
    { x: 252, width: 96, label: "estresse", color: "#f43f5e" },
    { x: 358, width: 110, label: "dispersão", color: "#38bdf8" },
  ];
  return (
    <svg viewBox="0 0 560 300" className="h-full w-full">
      <rect x="0" y="0" width="560" height="300" rx="28" fill="rgba(3,7,18,0.75)" />
      <g transform="translate(0 72)">
        {bands.map((band, idx) => (
          <g key={band.label}>
            <rect x={band.x} y="62" width={band.width} height="28" rx="14" fill={`${band.color}22`} stroke={band.color} />
            <text x={band.x + 18} y="80" fontSize="12" fill="#e2e8f0">
              {band.label}
            </text>
            <motion.circle
              cx={band.x + band.width / 2}
              cy="48"
              r="7"
              fill={band.color}
              animate={{ cy: [48, 36, 48], opacity: [0.55, 1, 0.55] }}
              transition={{ duration: 2.4, repeat: Infinity, delay: idx * 0.25 }}
            />
          </g>
        ))}
      </g>
      <motion.path
        d="M74 118 C140 86, 188 86, 244 122 S348 176, 470 132"
        stroke="#e2e8f0"
        strokeWidth="3"
        fill="none"
        initial={{ pathLength: 0.2 }}
        animate={{ pathLength: 1 }}
        transition={{ duration: 1.1 }}
      />
      <g fontSize="12" fill="#94a3b8">
        <text x="72" y="30">Regime estrutural ao longo do tempo</text>
        <text x="334" y="272">histerese + gate evitam flip-flop por ruído</text>
      </g>
    </svg>
  );
}

function InsightsGraphic() {
  const labels = [
    { x: 86, y: 60, text: "Cripto: atacar" },
    { x: 252, y: 92, text: "Ações: manter" },
    { x: 392, y: 148, text: "Caixa: aumentar" },
  ];
  return (
    <svg viewBox="0 0 560 300" className="h-full w-full">
      <rect x="0" y="0" width="560" height="300" rx="28" fill="rgba(3,7,18,0.75)" />
      <motion.path
        d="M64 214 C120 194, 176 130, 230 144 S332 208, 384 170 S458 102, 504 112"
        stroke="#22d3ee"
        strokeWidth="3"
        fill="none"
        initial={{ pathLength: 0 }}
        animate={{ pathLength: 1 }}
        transition={{ duration: 1.15 }}
      />
      <motion.path
        d="M64 238 C128 226, 188 184, 238 198 S336 236, 388 214 S458 168, 504 184"
        stroke="#f97316"
        strokeWidth="2.5"
        fill="none"
        initial={{ pathLength: 0 }}
        animate={{ pathLength: 1 }}
        transition={{ duration: 1.15, delay: 0.2 }}
      />
      {labels.map((label, index) => (
        <motion.g key={label.text} animate={{ y: [0, -4, 0] }} transition={{ duration: 2.8, repeat: Infinity, delay: index * 0.22 }}>
          <rect x={label.x} y={label.y} width="112" height="30" rx="10" fill="rgba(8,15,30,0.92)" stroke="rgba(56,189,248,0.4)" />
          <text x={label.x + 14} y={label.y + 19} fontSize="12" fill="#e2e8f0">
            {label.text}
          </text>
        </motion.g>
      ))}
      <g fontSize="12" fill="#94a3b8">
        <text x="72" y="28">Insights por sleeve e ativo</text>
        <text x="296" y="274">diagnóstico → orçamento de risco → decisão</text>
      </g>
    </svg>
  );
}

function FrameGraphic({ id }: { id: string }) {
  if (id === "dados") return <DataGraphic />;
  if (id === "matriz") return <MatrixGraphic />;
  if (id === "espectro") return <SpectrumGraphic />;
  if (id === "regime") return <RegimeGraphic />;
  return <InsightsGraphic />;
}

export default function EngineStoryDeck() {
  const [active, setActive] = useState(0);

  useEffect(() => {
    const timer = window.setInterval(() => {
      setActive((current) => (current + 1) % FRAMES.length);
    }, 7600);
    return () => window.clearInterval(timer);
  }, []);

  const current = FRAMES[active];

  return (
    <section className="overflow-hidden rounded-[32px] border border-zinc-800 bg-zinc-950/70">
      <div className="border-b border-zinc-800/80 p-6 md:p-8">
        <div className="max-w-4xl">
          <div className="text-xs uppercase tracking-[0.28em] text-cyan-300/80">Storyboard do motor</div>
          <AnimatePresence mode="wait">
            <motion.div
              key={current.id}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.42 }}
              className="mt-5"
            >
              <div className="text-[11px] uppercase tracking-[0.18em] text-zinc-500">{current.kicker}</div>
              <h2 className="mt-3 text-3xl font-semibold tracking-tight text-zinc-100 md:text-[2.2rem]">
                {current.title}
              </h2>
              <p className="mt-4 max-w-xl text-sm leading-7 text-zinc-300 md:text-[15px]">
                {current.body}
              </p>
              <div className="mt-5 space-y-2">
                {current.points.map((point) => (
                  <div key={point} className="flex items-start gap-3 rounded-2xl border border-zinc-800 bg-black/20 px-3 py-2.5 text-sm text-zinc-200">
                    <span className="mt-1 h-2 w-2 rounded-full bg-cyan-300" />
                    <span>{point}</span>
                  </div>
                ))}
              </div>
              <div className="mt-5 rounded-2xl border border-emerald-900/40 bg-emerald-950/20 px-4 py-3 text-sm text-emerald-100">
                {current.caption}
              </div>
            </motion.div>
          </AnimatePresence>

          <div className="mt-6 flex flex-wrap items-center gap-2">
            {FRAMES.map((frame, index) => (
              <button
                key={frame.id}
                type="button"
                onClick={() => setActive(index)}
                className={`rounded-full border px-3 py-1.5 text-xs transition ${
                  active === index
                    ? "border-cyan-300/60 bg-cyan-400/10 text-cyan-100"
                    : "border-zinc-800 bg-black/20 text-zinc-400 hover:border-zinc-600 hover:text-zinc-200"
                }`}
              >
                {frame.kicker}
              </button>
            ))}
            <button
              type="button"
              onClick={() => setActive((currentIndex) => (currentIndex + 1) % FRAMES.length)}
              className="rounded-full border border-zinc-700 px-3 py-1.5 text-xs text-zinc-300 hover:border-zinc-500 hover:text-zinc-100"
            >
              Próximo quadro
            </button>
          </div>
        </div>
      </div>

      <div className="relative min-h-[340px] bg-[radial-gradient(circle_at_top,_rgba(34,211,238,0.08),_transparent_58%),radial-gradient(circle_at_bottom,_rgba(59,130,246,0.08),_transparent_62%)] p-4 md:min-h-[420px] md:p-6">
        <AnimatePresence mode="wait">
          <motion.div
            key={current.id}
            initial={{ opacity: 0, scale: 0.985 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.992 }}
            transition={{ duration: 0.42 }}
            className="h-full"
          >
            <FrameGraphic id={current.id} />
          </motion.div>
        </AnimatePresence>
      </div>
    </section>
  );
}
