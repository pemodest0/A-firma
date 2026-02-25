import { NextResponse } from "next/server";
import { promises as fs } from "fs";
import path from "path";
import { resultsRoot } from "@/lib/server/results";

function parseCsv(text: string) {
  const lines = text.trim().split("\n");
  const header = (lines.shift()?.split(",") || []).map((h) => h.trim());
  return lines.map((line) => {
    const parts = line.split(",").map((p) => p.trim());
    const row: Record<string, string> = {};
    header.forEach((h, idx) => {
      row[h] = parts[idx];
    });
    return row;
  });
}

function pickNumber(row: Record<string, string>, keys: string[]) {
  for (const key of keys) {
    const raw = row[key];
    const n = Number(raw);
    if (Number.isFinite(n)) return n;
  }
  return null;
}

function toWeeklyIndices(dates: string[]) {
  const out: number[] = [];
  let lastKey = "";
  for (let i = 0; i < dates.length; i += 1) {
    const d = dates[i];
    const dt = new Date(d + "T00:00:00Z");
    const year = dt.getUTCFullYear();
    const week = Math.ceil(((dt.getTime() - Date.UTC(year, 0, 1)) / 86400000 + 1) / 7);
    const key = `${year}-W${week}`;
    if (key !== lastKey) {
      out.push(i);
      lastKey = key;
    } else {
      out[out.length - 1] = i;
    }
  }
  return out;
}

function std(values: number[]) {
  if (!values.length) return 0;
  const avg = values.reduce((a, b) => a + b, 0) / values.length;
  const varSum = values.reduce((a, b) => a + (b - avg) ** 2, 0) / values.length;
  return Math.sqrt(varSum);
}

function quantile(values: number[], q: number) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const pos = (sorted.length - 1) * q;
  const base = Math.floor(pos);
  const rest = pos - base;
  if (sorted[base + 1] !== undefined) {
    return sorted[base] + rest * (sorted[base + 1] - sorted[base]);
  }
  return sorted[base];
}

function computeRegime(vol: number, q1: number, q2: number) {
  if (vol <= q1) return "STABLE";
  if (vol <= q2) return "TRANSITION";
  return "UNSTABLE";
}

function computeConfidence(vol: number, minVol: number, maxVol: number) {
  if (!Number.isFinite(vol) || maxVol <= minVol) return 0.6;
  const rel = (vol - minVol) / (maxVol - minVol);
  const conf = 0.6 + 0.3 * Math.abs(rel - 0.5) * 2;
  return Math.max(0.55, Math.min(0.9, conf));
}

async function resolvePriceFile(asset: string) {
  const candidates = [
    path.join(resultsRoot(), "..", "data", "raw", "finance", "yfinance_daily", `${asset}.csv`),
    path.join(process.cwd(), "public", "data", "raw", "finance", "yfinance_daily", `${asset}.csv`),
  ];
  for (const candidate of candidates) {
    try {
      await fs.access(candidate);
      return candidate;
    } catch {
      // keep trying
    }
  }
  throw new Error("price_file_not_found");
}

async function loadFallbackSeries(asset: string, tf: string, limit: number, step: number) {
  const priceFile = await resolvePriceFile(asset);
  const rawPrice = await fs.readFile(priceFile, "utf-8");
  const priceRows = parseCsv(rawPrice);
  const dates = priceRows.map((r) => r.date).filter(Boolean);
  const returns = priceRows.map((r) => Number(r.r)).filter((v) => Number.isFinite(v));

  const window = 20;
  const volSeries: number[] = [];
  for (let i = window; i < returns.length; i += 1) {
    const slice = returns.slice(i - window, i);
    volSeries.push(std(slice));
  }

  const q1 = quantile(volSeries, 0.33);
  const q2 = quantile(volSeries, 0.66);
  const minVol = Math.min(...volSeries, q1);
  const maxVol = Math.max(...volSeries, q2);

  const seriesRaw = priceRows.map((row, idx) => {
    const vol = volSeries[idx - window] ?? volSeries[volSeries.length - 1] ?? 0;
    const regime = computeRegime(vol, q1, q2);
    const confidence = computeConfidence(vol, minVol, maxVol);
    return {
      date: row.date,
      confidence,
      regime,
      price: Number(row.price ?? Number.NaN) || null,
      volume: pickNumber(row, ["volume", "Volume", "VOL", "vol", "qty"]),
    };
  });

  const indices = tf === "weekly" ? toWeeklyIndices(dates) : seriesRaw.map((_, idx) => idx);
  const sliced = indices.map((i) => seriesRaw[i]).filter((r) => r && r.date);
  const total = sliced.length;
  const n = limit ? Math.min(limit, total) : total;
  return sliced.slice(-n).filter((_, idx) => idx % step === 0);
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const assetsParam = searchParams.get("assets");
  const tf = searchParams.get("tf") || "weekly";
  const limitParam = searchParams.get("limit");
  const stepParam = searchParams.get("step");
  const limit = limitParam ? Math.max(1, Number(limitParam)) : 0;
  const step = stepParam ? Math.max(1, Number(stepParam)) : 1;
  if (!assetsParam) {
    return NextResponse.json({ error: "missing_assets" }, { status: 400 });
  }
  const assets = assetsParam
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
  const out: Record<
    string,
    { date: string; confidence: number; regime: string; price: number | null; volume: number | null }[]
  > = {};

  await Promise.all(
    assets.map(async (asset) => {
      try {
        out[asset] = await loadFallbackSeries(asset, tf, limit, step);
      } catch {
        out[asset] = [];
      }
    })
  );

  return NextResponse.json(out);
}
