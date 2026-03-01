import { promises as fs } from "fs";
import path from "path";
import { existsSync } from "node:fs";

function repoRoot() {
  return path.resolve(process.cwd(), "..");
}

export function validatedRoot() {
  if (process.env.VALIDATED_DIR) return process.env.VALIDATED_DIR;
  const candidates = [
    path.join(process.cwd(), "public", "data", "latest"),
    path.join(repoRoot(), "results", "validated", "latest"),
  ];
  for (const candidate of candidates) {
    if (existsSync(candidate)) return candidate;
  }
  return candidates[0];
}

export async function readValidatedUniverse(tf: string) {
  const file = path.join(validatedRoot(), `universe_${tf}.json`);
  const text = await fs.readFile(file, "utf-8");
  return JSON.parse(text);
}

export async function readAssetStatusMap() {
  const file = path.join(validatedRoot(), "asset_status.csv");
  const text = await fs.readFile(file, "utf-8");
  const lines = text.trim().split("\n");
  const header = (lines.shift() || "").split(",").map((h) => h.trim());
  const out: Record<string, Record<string, string>> = {};
  for (const line of lines) {
    const parts = line.split(",");
    const row: Record<string, string> = {};
    header.forEach((h, i) => {
      row[h] = (parts[i] || "").trim();
    });
    const key = `${row.asset}__${row.timeframe}`;
    out[key] = row;
  }
  return out;
}
