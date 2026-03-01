import { NextResponse } from "next/server";
import {
  readPlatformDbRelease,
  readPlatformDbSnapshot,
  readPlatformHierarchicalStateLatest,
  readPlatformRankingsLatest,
} from "@/lib/server/data";

export const dynamic = "force-dynamic";

export async function GET() {
  const [snapshot, release, rankings, hierarchicalState] = await Promise.all([
    readPlatformDbSnapshot(),
    readPlatformDbRelease(),
    readPlatformRankingsLatest(),
    readPlatformHierarchicalStateLatest(),
  ]);
  return NextResponse.json(
    {
      ok: true,
      snapshot,
      release,
      rankings,
      hierarchicalState,
    },
    { headers: { "Cache-Control": "no-store" } }
  );
}
