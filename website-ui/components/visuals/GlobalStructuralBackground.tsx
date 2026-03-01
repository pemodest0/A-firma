type Node = {
  id: string;
  x: number;
  y: number;
  hub?: boolean;
  delay: number;
};

type Edge = {
  from: string;
  to: string;
  delay: number;
};

const nodes: Node[] = [
  { id: "f1", x: 110, y: 160, delay: 0.1 },
  { id: "f2", x: 190, y: 120, delay: 0.2 },
  { id: "f3", x: 210, y: 220, delay: 0.25 },
  { id: "f4", x: 300, y: 150, delay: 0.35 },
  { id: "f5", x: 320, y: 260, delay: 0.45 },
  { id: "f6", x: 410, y: 190, delay: 0.55 },
  { id: "h1", x: 510, y: 210, hub: true, delay: 0.65 },

  { id: "e1", x: 170, y: 430, delay: 0.15 },
  { id: "e2", x: 260, y: 380, delay: 0.22 },
  { id: "e3", x: 270, y: 500, delay: 0.32 },
  { id: "e4", x: 360, y: 430, delay: 0.41 },
  { id: "e5", x: 450, y: 470, delay: 0.5 },
  { id: "h2", x: 550, y: 440, hub: true, delay: 0.72 },

  { id: "s1", x: 220, y: 700, delay: 0.2 },
  { id: "s2", x: 300, y: 640, delay: 0.27 },
  { id: "s3", x: 340, y: 760, delay: 0.33 },
  { id: "s4", x: 430, y: 680, delay: 0.42 },
  { id: "s5", x: 470, y: 780, delay: 0.5 },
  { id: "h3", x: 590, y: 700, hub: true, delay: 0.76 },

  { id: "c1", x: 760, y: 240, delay: 0.2 },
  { id: "c2", x: 820, y: 360, delay: 0.27 },
  { id: "c3", x: 760, y: 520, delay: 0.35 },
  { id: "c4", x: 860, y: 620, delay: 0.42 },
  { id: "h4", x: 940, y: 420, hub: true, delay: 0.9 },

  { id: "r1", x: 1100, y: 210, delay: 0.2 },
  { id: "r2", x: 1210, y: 150, delay: 0.3 },
  { id: "r3", x: 1190, y: 300, delay: 0.4 },
  { id: "r4", x: 1300, y: 240, delay: 0.48 },
  { id: "r5", x: 1320, y: 360, delay: 0.56 },
  { id: "r6", x: 1230, y: 460, delay: 0.62 },
  { id: "r7", x: 1320, y: 540, delay: 0.68 },
  { id: "sink", x: 1450, y: 350, hub: true, delay: 1.0 },
];

const edges: Edge[] = [
  { from: "f1", to: "f2", delay: 0.1 }, { from: "f2", to: "f4", delay: 0.18 }, { from: "f3", to: "f5", delay: 0.25 },
  { from: "f4", to: "f6", delay: 0.32 }, { from: "f5", to: "f6", delay: 0.4 }, { from: "f6", to: "h1", delay: 0.48 },

  { from: "e1", to: "e2", delay: 0.14 }, { from: "e1", to: "e3", delay: 0.21 }, { from: "e2", to: "e4", delay: 0.3 },
  { from: "e3", to: "e4", delay: 0.37 }, { from: "e4", to: "e5", delay: 0.45 }, { from: "e5", to: "h2", delay: 0.52 },

  { from: "s1", to: "s2", delay: 0.12 }, { from: "s1", to: "s3", delay: 0.2 }, { from: "s2", to: "s4", delay: 0.3 },
  { from: "s3", to: "s4", delay: 0.36 }, { from: "s4", to: "s5", delay: 0.46 }, { from: "s5", to: "h3", delay: 0.54 },

  { from: "h1", to: "c1", delay: 0.2 }, { from: "h2", to: "c2", delay: 0.3 }, { from: "h3", to: "c3", delay: 0.4 },
  { from: "c1", to: "h4", delay: 0.5 }, { from: "c2", to: "h4", delay: 0.58 }, { from: "c3", to: "h4", delay: 0.66 },
  { from: "c4", to: "h4", delay: 0.74 }, { from: "c3", to: "c4", delay: 0.82 },

  { from: "h4", to: "r1", delay: 0.1 }, { from: "h4", to: "r3", delay: 0.18 }, { from: "h4", to: "r6", delay: 0.27 },
  { from: "r1", to: "r2", delay: 0.36 }, { from: "r1", to: "r3", delay: 0.42 }, { from: "r3", to: "r4", delay: 0.5 },
  { from: "r3", to: "r5", delay: 0.57 }, { from: "r6", to: "r7", delay: 0.64 }, { from: "r4", to: "sink", delay: 0.72 },
  { from: "r5", to: "sink", delay: 0.8 }, { from: "r7", to: "sink", delay: 0.88 }, { from: "r6", to: "sink", delay: 0.95 },
];

const nodeMap = new Map(nodes.map((node) => [node.id, node] as const));

const packetPaths = edges
  .map((edge) => {
    const from = nodeMap.get(edge.from);
    const to = nodeMap.get(edge.to);
    if (!from || !to) return "";
    return `M ${from.x} ${from.y} L ${to.x} ${to.y}`;
  })
  .filter(Boolean);

const convergingCurves = [
  "M 510 210 C 680 180, 780 230, 940 420",
  "M 550 440 C 700 420, 820 390, 940 420",
  "M 590 700 C 710 620, 830 540, 940 420",
  "M 940 420 C 1120 410, 1280 420, 1450 350",
];

export default function GlobalStructuralBackground() {
  return (
    <div aria-hidden className="ax-global-directed-bg pointer-events-none absolute inset-0">
      <svg viewBox="0 0 1600 900" preserveAspectRatio="xMidYMid slice" className="h-full w-full">
        <defs>
          <marker id="ax-global-arrow" viewBox="0 0 10 10" refX="8.4" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="rgba(167, 238, 255, 0.82)" />
          </marker>
        </defs>

        <g className="ax-global-directed-network">
          {edges.map((edge, idx) => {
            const from = nodeMap.get(edge.from);
            const to = nodeMap.get(edge.to);
            if (!from || !to) return null;
            return (
              <line
                key={`edge-${edge.from}-${edge.to}-${idx}`}
                x1={from.x}
                y1={from.y}
                x2={to.x}
                y2={to.y}
                markerEnd="url(#ax-global-arrow)"
                className="ax-global-directed-edge"
                style={{ animationDelay: `${edge.delay}s` }}
              />
            );
          })}
        </g>

        <g className="ax-global-directed-convergence">
          {convergingCurves.map((curve, idx) => (
            <path key={`curve-${idx}`} d={curve} className="ax-global-directed-curve" style={{ animationDelay: `${idx * 0.35}s` }} />
          ))}
        </g>

        <g className="ax-global-directed-nodes">
          {nodes.map((node) => (
            <circle
              key={node.id}
              cx={node.x}
              cy={node.y}
              r={node.hub ? 5 : 3}
              className={node.hub ? "ax-global-directed-hub" : "ax-global-directed-node"}
              style={{ animationDelay: `${node.delay}s` }}
            />
          ))}
        </g>

        <g className="ax-global-directed-packets">
          {Array.from({ length: 42 }).map((_, idx) => (
            <circle key={`packet-${idx}`} r="2.6" className="ax-global-directed-packet">
              <animateMotion
                begin={`${(idx % 11) * 0.22}s`}
                dur={`${7 + (idx % 6) * 1.1}s`}
                path={packetPaths[idx % packetPaths.length]}
                repeatCount="indefinite"
              />
            </circle>
          ))}
          {Array.from({ length: 16 }).map((_, idx) => (
            <circle key={`curve-packet-${idx}`} r="2.4" className="ax-global-directed-packet">
              <animateMotion
                begin={`${(idx % 8) * 0.31}s`}
                dur={`${8.8 + (idx % 4) * 0.9}s`}
                path={convergingCurves[idx % convergingCurves.length]}
                repeatCount="indefinite"
              />
            </circle>
          ))}
        </g>
      </svg>
    </div>
  );
}
