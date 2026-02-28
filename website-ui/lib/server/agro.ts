import { promises as fs } from "fs";
import path from "path";
import { existsSync } from "node:fs";

type JsonObject = Record<string, unknown>;

export class AgroArtifactError extends Error {
  statusCode: number;

  constructor(message: string, statusCode = 503) {
    super(message);
    this.name = "AgroArtifactError";
    this.statusCode = statusCode;
  }
}

function repoRoot() {
  return path.resolve(process.cwd(), "..");
}

function resolveAgroLatestDir() {
  if (process.env.AGRO_RESULTS_DIR) return process.env.AGRO_RESULTS_DIR;
  const candidates = [
    path.join(process.cwd(), "results", "agro_br", "latest"),
    path.join(repoRoot(), "results", "agro_br", "latest"),
  ];
  for (const c of candidates) {
    if (existsSync(c)) return c;
  }
  return candidates[1];
}

async function readJsonFile(filePath: string): Promise<JsonObject> {
  const raw = await fs.readFile(filePath, "utf-8");
  const parsed = JSON.parse(raw);
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new AgroArtifactError(`artifact_not_object:${path.basename(filePath)}`);
  }
  return parsed as JsonObject;
}

async function readAgroArtifact(fileName: string, requiredKeys: string[]) {
  const dir = resolveAgroLatestDir();
  const target = path.join(dir, fileName);
  if (!existsSync(target)) {
    throw new AgroArtifactError(`artifact_missing:${fileName}`);
  }
  const parsed = await readJsonFile(target);
  const missing = requiredKeys.filter((key) => !(key in parsed));
  if (missing.length) {
    throw new AgroArtifactError(`artifact_invalid_schema:${fileName}:missing=${missing.join(",")}`);
  }
  return parsed;
}

export async function readAgroStateLatest() {
  return readAgroArtifact("hierarchical_state_latest_agro_br.json", [
    "date",
    "global_score",
    "top_sectors_by_score",
    "top_sectors_by_loading",
    "top_sectors_by_overlap",
  ]);
}

export async function readAgroRankingsLatest() {
  return readAgroArtifact("rankings_latest_agro_br.json", [
    "date",
    "top_assets_global_mode",
    "top_sectors_global_mode",
    "sector_global_overlap",
    "global_state",
  ]);
}

export async function readAgroEvidenceLatest() {
  return readAgroArtifact("historical_structure_summary_agro_br.json", [
    "schema_version",
    "status",
    "last_date",
    "evidence",
    "events",
  ]);
}

