import { NextResponse } from "next/server";
import { AgroArtifactError, readAgroStateLatest } from "@/lib/server/agro";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const payload = await readAgroStateLatest();
    return NextResponse.json(payload, { headers: { "Cache-Control": "no-store" } });
  } catch (err) {
    const status = err instanceof AgroArtifactError ? err.statusCode : 503;
    return NextResponse.json(
      {
        error: "agro_state_unavailable",
        reason: err instanceof Error ? err.message : "unknown_error",
      },
      { status, headers: { "Cache-Control": "no-store" } }
    );
  }
}

