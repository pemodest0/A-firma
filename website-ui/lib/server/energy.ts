import { promises as fs } from "fs";
import path from "path";
import { existsSync } from "node:fs";

type JsonObject = Record<string, unknown>;

export class EnergyArtifactError extends Error {
  statusCode: number;

  constructor(message: string, statusCode = 503) {
    super(message);
    this.name = "EnergyArtifactError";
    this.statusCode = statusCode;
  }
}

type SeriesPoint = { date: string; r: number };

type SubmarketDef = {
  id: "N" | "NE" | "SECO" | "S";
  label: string;
  onsTicker: string;
  cmoTicker: string;
  states: string[];
};

const SUBMARKETS: SubmarketDef[] = [
  {
    id: "N",
    label: "Norte",
    onsTicker: "ONS_N",
    cmoTicker: "CMO_N",
    states: ["AC", "AP", "AM", "PA", "RO", "RR", "TO"],
  },
  {
    id: "NE",
    label: "Nordeste",
    onsTicker: "ONS_NE",
    cmoTicker: "CMO_NE",
    states: ["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"],
  },
  {
    id: "SECO",
    label: "Sudeste/Centro-Oeste",
    onsTicker: "ONS_SE",
    cmoTicker: "CMO_SE",
    states: ["DF", "ES", "GO", "MG", "MS", "MT", "RJ", "SP"],
  },
  {
    id: "S",
    label: "Sul",
    onsTicker: "ONS_S",
    cmoTicker: "CMO_S",
    states: ["PR", "RS", "SC"],
  },
];

const UF_NAME: Record<string, string> = {
  AC: "Acre",
  AL: "Alagoas",
  AP: "Amapá",
  AM: "Amazonas",
  BA: "Bahia",
  CE: "Ceará",
  DF: "Distrito Federal",
  ES: "Espírito Santo",
  GO: "Goiás",
  MA: "Maranhão",
  MT: "Mato Grosso",
  MS: "Mato Grosso do Sul",
  MG: "Minas Gerais",
  PA: "Pará",
  PB: "Paraíba",
  PR: "Paraná",
  PE: "Pernambuco",
  PI: "Piauí",
  RJ: "Rio de Janeiro",
  RN: "Rio Grande do Norte",
  RS: "Rio Grande do Sul",
  RO: "Rondônia",
  RR: "Roraima",
  SC: "Santa Catarina",
  SP: "São Paulo",
  SE: "Sergipe",
  TO: "Tocantins",
};

function repoRoot() {
  return path.resolve(process.cwd(), "..");
}

function energyRoot() {
  const candidates = [path.join(process.cwd(), "results", "energy_br"), path.join(repoRoot(), "results", "energy_br")];
  for (const c of candidates) {
    if (existsSync(c)) return c;
  }
  return candidates[1];
}

async function readJsonFile(filePath: string): Promise<JsonObject> {
  const raw = await fs.readFile(filePath, "utf-8");
  const parsed = JSON.parse(raw);
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new EnergyArtifactError(`artifact_not_object:${path.basename(filePath)}`);
  }
  return parsed as JsonObject;
}

function resolvePackDir(root: string, release: JsonObject) {
  if (process.env.ENERGY_PACK_DIR) return process.env.ENERGY_PACK_DIR;
  const fromRelease = String(release.pack_dir || "").trim();
  if (fromRelease && existsSync(fromRelease)) return fromRelease;
  const localPacks = (() => {
    try {
      return fs.readdir(root);
    } catch {
      return Promise.resolve<string[]>([]);
    }
  })();
  return localPacks.then((dirs) => {
    const pick = dirs
      .filter((d) => d.startsWith("local_pack_"))
      .sort()
      .reverse()[0];
    if (!pick) throw new EnergyArtifactError("energy_pack_missing");
    return path.join(root, pick);
  });
}

function clamp(x: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, x));
}

function mean(values: number[]) {
  if (!values.length) return 0;
  return values.reduce((acc, v) => acc + v, 0) / values.length;
}

function std(values: number[]) {
  if (values.length < 2) return 0;
  const m = mean(values);
  const variance = values.reduce((acc, v) => acc + (v - m) ** 2, 0) / values.length;
  return Math.sqrt(variance);
}

function latestZ(series: SeriesPoint[], window = 60) {
  if (series.length < 8) return { z: 0, latest: 0 };
  const tail = series.slice(-Math.max(8, window + 1));
  const latest = Number(tail[tail.length - 1]?.r || 0);
  const base = tail.slice(0, -1).map((p) => Number(p.r || 0));
  const s = std(base);
  if (!Number.isFinite(s) || s < 1e-9) return { z: 0, latest };
  const z = (latest - mean(base)) / s;
  return { z, latest };
}

function corr(a: number[], b: number[]) {
  if (a.length !== b.length || a.length < 3) return 0;
  const ma = mean(a);
  const mb = mean(b);
  let num = 0;
  let da = 0;
  let db = 0;
  for (let i = 0; i < a.length; i += 1) {
    const xa = a[i] - ma;
    const xb = b[i] - mb;
    num += xa * xb;
    da += xa * xa;
    db += xb * xb;
  }
  if (da <= 1e-12 || db <= 1e-12) return 0;
  return num / Math.sqrt(da * db);
}

function riskLevel(probability: number) {
  if (probability >= 0.78) return "critico";
  if (probability >= 0.62) return "alto";
  if (probability >= 0.42) return "moderado";
  return "baixo";
}

async function readPanelSeries(packDir: string) {
  const panelPath = path.join(packDir, "panel_long_energy_br.csv");
  if (!existsSync(panelPath)) {
    throw new EnergyArtifactError("panel_long_energy_br_missing");
  }
  const raw = await fs.readFile(panelPath, "utf-8");
  const lines = raw.split(/\r?\n/).filter((line) => line.trim().length > 0);
  if (lines.length <= 1) {
    throw new EnergyArtifactError("panel_long_energy_br_empty");
  }
  const header = lines[0].split(",");
  const iDate = header.indexOf("date");
  const iTicker = header.indexOf("ticker");
  const iR = header.indexOf("r");
  if (iDate < 0 || iTicker < 0 || iR < 0) {
    throw new EnergyArtifactError("panel_long_energy_br_schema_invalid");
  }

  const seriesByTicker = new Map<string, SeriesPoint[]>();
  let asOfDate = "";
  for (let i = 1; i < lines.length; i += 1) {
    const parts = lines[i].split(",");
    if (parts.length <= Math.max(iDate, iTicker, iR)) continue;
    const date = parts[iDate]?.trim();
    const ticker = parts[iTicker]?.trim();
    const r = Number(parts[iR]);
    if (!date || !ticker || !Number.isFinite(r)) continue;
    const arr = seriesByTicker.get(ticker) || [];
    arr.push({ date, r });
    seriesByTicker.set(ticker, arr);
    if (!asOfDate || date > asOfDate) asOfDate = date;
  }

  for (const [ticker, arr] of seriesByTicker.entries()) {
    arr.sort((a, b) => a.date.localeCompare(b.date));
    seriesByTicker.set(ticker, arr);
  }

  return { seriesByTicker, asOfDate };
}

export async function readEnergyGridMapState() {
  const root = energyRoot();
  const releasePath = path.join(root, "latest_release_energy_br.json");
  if (!existsSync(releasePath)) {
    throw new EnergyArtifactError("energy_latest_release_missing");
  }
  const release = await readJsonFile(releasePath);
  const packDir = await resolvePackDir(root, release);

  const latestDir = path.join(root, "latest");
  const stateLatestPath = path.join(latestDir, "hierarchical_state_latest_energy_br.json");
  const evidencePath = path.join(latestDir, "historical_structure_summary_energy_br.json");
  const stateLatest = existsSync(stateLatestPath) ? await readJsonFile(stateLatestPath) : {};
  const evidence = existsSync(evidencePath) ? await readJsonFile(evidencePath) : {};

  const { seriesByTicker, asOfDate } = await readPanelSeries(packDir);
  if (!asOfDate) throw new EnergyArtifactError("energy_as_of_date_missing");

  const global = latestZ(seriesByTicker.get("ONS_BR") || []);
  const globalPressure = clamp(Math.abs(global.z) / 3, 0, 1);

  const submarkets = SUBMARKETS.map((sm) => {
    const zLoad = latestZ(seriesByTicker.get(sm.onsTicker) || []);
    const zCmo = latestZ(seriesByTicker.get(sm.cmoTicker) || []);
    const loadTerm = clamp(Math.abs(zLoad.z) / 3, 0, 1);
    const cmoTerm = clamp(Math.abs(zCmo.z) / 3, 0, 1);
    const surgeBonus = zCmo.latest > 0 && cmoTerm > 0.35 ? 0.05 : 0;
    const probability = clamp(0.12 + 0.5 * loadTerm + 0.28 * cmoTerm + 0.1 * globalPressure + surgeBonus, 0.03, 0.97);
    return {
      id: sm.id,
      label: sm.label,
      ons_ticker: sm.onsTicker,
      cmo_ticker: sm.cmoTicker,
      probability,
      risk_level: riskLevel(probability),
      drivers: {
        load_z: zLoad.z,
        cmo_z: zCmo.z,
        load_return_latest: zLoad.latest,
        cmo_return_latest: zCmo.latest,
      },
      states: sm.states,
    };
  });

  const subById = new Map(submarkets.map((s) => [s.id, s]));
  const states = SUBMARKETS.flatMap((sm) =>
    sm.states.map((uf) => {
      const sub = subById.get(sm.id)!;
      return {
        uf,
        name: UF_NAME[uf] || uf,
        submarket_id: sm.id,
        submarket_label: sm.label,
        probability: sub.probability,
        risk_level: sub.risk_level,
      };
    })
  ).sort((a, b) => b.probability - a.probability || a.uf.localeCompare(b.uf));

  const loadIds = SUBMARKETS.map((sm) => ({ id: sm.id, ticker: sm.onsTicker, label: sm.label }));
  const edges: Array<{ from: string; to: string; correlation: number; abs_correlation: number; critical: boolean }> = [];
  for (let i = 0; i < loadIds.length; i += 1) {
    for (let j = i + 1; j < loadIds.length; j += 1) {
      const aSeries = (seriesByTicker.get(loadIds[i].ticker) || []).slice(-120);
      const bSeries = (seriesByTicker.get(loadIds[j].ticker) || []).slice(-120);
      const bMap = new Map(bSeries.map((p) => [p.date, p.r]));
      const aVals: number[] = [];
      const bVals: number[] = [];
      for (const p of aSeries) {
        const bv = bMap.get(p.date);
        if (!Number.isFinite(bv)) continue;
        aVals.push(p.r);
        bVals.push(Number(bv));
      }
      const c = corr(aVals, bVals);
      const abs = Math.abs(c);
      edges.push({
        from: loadIds[i].id,
        to: loadIds[j].id,
        correlation: c,
        abs_correlation: abs,
        critical: abs >= 0.62,
      });
    }
  }
  edges.sort((a, b) => b.abs_correlation - a.abs_correlation);

  const nationalProb = mean(submarkets.map((s) => s.probability));
  const modelScoreLatest = Number((stateLatest as JsonObject).global_score);
  const modelDate = String((stateLatest as JsonObject).date || "");
  const evidenceObj = (evidence as JsonObject).evidence as JsonObject | undefined;

  return {
    status: "ok",
    generated_at_utc: new Date().toISOString(),
    as_of_date: asOfDate,
    source: {
      pack_dir: packDir,
      release_run_dir: String(release.run_dir || ""),
      model_state_date: modelDate || "",
    },
    national: {
      bottleneck_probability: nationalProb,
      risk_level: riskLevel(nationalProb),
      global_load_z: global.z,
      global_load_return_latest: global.latest,
      model_score_latest: Number.isFinite(modelScoreLatest) ? modelScoreLatest : null,
      model_score_date: modelDate || null,
    },
    submarkets,
    states,
    network: {
      nodes: submarkets.map((s) => ({ id: s.id, label: s.label, probability: s.probability })),
      edges,
    },
    evidence: {
      pre_signal_rate: Number((evidenceObj || {}).pre_signal_rate || 0),
      events_valid: Number((evidenceObj || {}).events_valid || 0),
      interpretation: String((evidenceObj || {}).interpretation || "indisponivel"),
    },
  };
}
