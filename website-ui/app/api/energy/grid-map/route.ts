import { NextResponse } from "next/server";
import { EnergyArtifactError, readEnergyGridMapState } from "@/lib/server/energy";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const payload = await readEnergyGridMapState();
    return NextResponse.json(payload, { headers: { "Cache-Control": "no-store" } });
  } catch (error) {
    const statusCode = error instanceof EnergyArtifactError ? error.statusCode : 503;
    return NextResponse.json(
      {
        status: "error",
        error: "energy_grid_map_unavailable",
        detail: error instanceof Error ? error.message : "unknown_error",
      },
      { status: statusCode, headers: { "Cache-Control": "no-store" } }
    );
  }
}
