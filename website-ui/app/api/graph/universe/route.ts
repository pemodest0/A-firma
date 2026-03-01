import { NextResponse } from "next/server";
import { readValidatedUniverse } from "@/lib/server/validated";

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const tf = searchParams.get("tf") || "weekly";
  try {
    const validated = await readValidatedUniverse(tf);
    if (Array.isArray(validated) && validated.length) {
      return NextResponse.json(validated);
    }
    return NextResponse.json(
      { error: "validated_universe_empty", message: `Universe validado (${tf}) sem registros publicáveis.` },
      { status: 503 }
    );
  } catch {
    return NextResponse.json(
      { error: "validated_universe_unavailable", message: `Universe validado (${tf}) não encontrado nos artefatos publicados.` },
      { status: 503 }
    );
  }
}
