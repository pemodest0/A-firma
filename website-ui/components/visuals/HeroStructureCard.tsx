export default function HeroStructureCard() {
  return (
    <div className="ax-hero-structure-card rounded-3xl border border-cyan-200/25 bg-[#06122A]/78 p-5 md:p-6">
      <div className="flex items-center justify-between gap-3">
        <div className="text-xs uppercase tracking-[0.2em] text-cyan-100/75">Núcleo estrutural</div>
        <div className="text-[11px] text-zinc-300">Mercado financeiro</div>
      </div>
      <h3 className="mt-2 text-xl font-semibold text-zinc-100">Setores em correlação dinâmica</h3>
      <p className="mt-2 text-sm text-zinc-300/90">
        Os clusters setoriais se separam, reforçam correlações e convergem para o sistema financeiro global.
      </p>

      <div className="mt-4 rounded-2xl border border-zinc-700/70 bg-black/20 p-2">
        <svg viewBox="0 0 320 300" className="h-[300px] w-full">
          <defs>
            <marker id="ax-hero-fin-arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
              <path d="M 0 0 L 10 5 L 0 10 z" fill="rgba(150, 238, 255, 0.86)" />
            </marker>
          </defs>
          <rect x="0" y="0" width="320" height="300" fill="transparent" />

          <g className="ax-hero-fin-correlation-layer">
            <path d="M 112 102 C 146 90, 194 94, 228 108" className="ax-hero-fin-corr-edge" />
            <path d="M 126 200 C 146 172, 196 172, 214 194" className="ax-hero-fin-corr-edge" />
            <path d="M 112 102 C 102 136, 112 172, 126 200" className="ax-hero-fin-corr-edge" />
            <path d="M 228 108 C 236 140, 228 170, 214 194" className="ax-hero-fin-corr-edge" />
            {Array.from({ length: 10 }).map((_, idx) => (
              <circle key={`corr-packet-${idx}`} r="2.2" className="ax-hero-fin-corr-packet">
                <animateMotion
                  begin={`${(idx % 6) * 0.24}s`}
                  dur={`${6.1 + (idx % 3) * 0.9}s`}
                  path={
                    [
                      "M 112 102 C 146 90, 194 94, 228 108",
                      "M 126 200 C 146 172, 196 172, 214 194",
                      "M 112 102 C 102 136, 112 172, 126 200",
                      "M 228 108 C 236 140, 228 170, 214 194",
                    ][idx % 4]
                  }
                  repeatCount="indefinite"
                />
              </circle>
            ))}
          </g>

          <g className="ax-hero-sector-cluster ax-hero-sector-a">
            <line x1="52" y1="82" x2="74" y2="62" markerEnd="url(#ax-hero-fin-arrow)" className="ax-hero-fin-sector-edge" />
            <line x1="74" y1="62" x2="100" y2="86" markerEnd="url(#ax-hero-fin-arrow)" className="ax-hero-fin-sector-edge" />
            <line x1="52" y1="82" x2="98" y2="88" markerEnd="url(#ax-hero-fin-arrow)" className="ax-hero-fin-sector-edge" />
            <circle cx="52" cy="82" r="3.2" className="ax-hero-fin-node" />
            <circle cx="74" cy="62" r="3.2" className="ax-hero-fin-node" />
            <circle cx="98" cy="88" r="3.4" className="ax-hero-fin-node" />
            <circle cx="112" cy="102" r="4.4" className="ax-hero-fin-hub" />
            <text x="44" y="52" className="ax-hero-fin-label">Ações</text>
          </g>

          <g className="ax-hero-sector-cluster ax-hero-sector-b">
            <line x1="244" y1="70" x2="264" y2="92" markerEnd="url(#ax-hero-fin-arrow)" className="ax-hero-fin-sector-edge" />
            <line x1="264" y1="92" x2="286" y2="74" markerEnd="url(#ax-hero-fin-arrow)" className="ax-hero-fin-sector-edge" />
            <line x1="244" y1="70" x2="286" y2="74" markerEnd="url(#ax-hero-fin-arrow)" className="ax-hero-fin-sector-edge" />
            <circle cx="244" cy="70" r="3.2" className="ax-hero-fin-node" />
            <circle cx="264" cy="92" r="3.2" className="ax-hero-fin-node" />
            <circle cx="286" cy="74" r="3.2" className="ax-hero-fin-node" />
            <circle cx="228" cy="108" r="4.4" className="ax-hero-fin-hub" />
            <text x="234" y="52" className="ax-hero-fin-label">Juros</text>
          </g>

          <g className="ax-hero-sector-cluster ax-hero-sector-c">
            <line x1="64" y1="236" x2="86" y2="214" markerEnd="url(#ax-hero-fin-arrow)" className="ax-hero-fin-sector-edge" />
            <line x1="86" y1="214" x2="114" y2="238" markerEnd="url(#ax-hero-fin-arrow)" className="ax-hero-fin-sector-edge" />
            <line x1="64" y1="236" x2="112" y2="240" markerEnd="url(#ax-hero-fin-arrow)" className="ax-hero-fin-sector-edge" />
            <circle cx="64" cy="236" r="3.2" className="ax-hero-fin-node" />
            <circle cx="86" cy="214" r="3.2" className="ax-hero-fin-node" />
            <circle cx="112" cy="240" r="3.2" className="ax-hero-fin-node" />
            <circle cx="126" cy="200" r="4.4" className="ax-hero-fin-hub" />
            <text x="34" y="260" className="ax-hero-fin-label">Commodities</text>
          </g>

          <g className="ax-hero-sector-cluster ax-hero-sector-d">
            <line x1="228" y1="224" x2="252" y2="206" markerEnd="url(#ax-hero-fin-arrow)" className="ax-hero-fin-sector-edge" />
            <line x1="252" y1="206" x2="278" y2="228" markerEnd="url(#ax-hero-fin-arrow)" className="ax-hero-fin-sector-edge" />
            <line x1="228" y1="224" x2="278" y2="228" markerEnd="url(#ax-hero-fin-arrow)" className="ax-hero-fin-sector-edge" />
            <circle cx="228" cy="224" r="3.2" className="ax-hero-fin-node" />
            <circle cx="252" cy="206" r="3.2" className="ax-hero-fin-node" />
            <circle cx="278" cy="228" r="3.2" className="ax-hero-fin-node" />
            <circle cx="214" cy="194" r="4.4" className="ax-hero-fin-hub" />
            <text x="238" y="260" className="ax-hero-fin-label">Cripto</text>
          </g>

          <line x1="112" y1="102" x2="160" y2="150" markerEnd="url(#ax-hero-fin-arrow)" className="ax-hero-fin-global-edge" />
          <line x1="228" y1="108" x2="160" y2="150" markerEnd="url(#ax-hero-fin-arrow)" className="ax-hero-fin-global-edge" />
          <line x1="126" y1="200" x2="160" y2="150" markerEnd="url(#ax-hero-fin-arrow)" className="ax-hero-fin-global-edge" />
          <line x1="214" y1="194" x2="160" y2="150" markerEnd="url(#ax-hero-fin-arrow)" className="ax-hero-fin-global-edge" />

          {Array.from({ length: 16 }).map((_, idx) => (
            <circle key={`packet-${idx}`} r="2.5" className="ax-hero-fin-packet">
              <animateMotion
                begin={`${(idx % 8) * 0.24}s`}
                dur={`${5.2 + (idx % 4) * 0.8}s`}
                path={
                  [
                    "M 112 102 L 160 150",
                    "M 228 108 L 160 150",
                    "M 126 200 L 160 150",
                    "M 214 194 L 160 150",
                  ][idx % 4]
                }
                repeatCount="indefinite"
              />
            </circle>
          ))}

          <circle cx="160" cy="150" r="6.5" className="ax-hero-fin-global" />
          <circle cx="160" cy="150" r="17" className="ax-hero-fin-global-ring" />
          <circle cx="160" cy="150" r="30" className="ax-hero-fin-global-ring ax-hero-fin-global-ring-wide" />
          <text x="146" y="174" className="ax-hero-fin-global-label">Global</text>
        </svg>
      </div>
    </div>
  );
}
