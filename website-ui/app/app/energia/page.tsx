import EnergyAnomalyDashboard from "@/components/EnergyAnomalyDashboard";
import { readEnergyGridMapState } from "@/lib/server/energy";

export const dynamic = "force-dynamic";

export default async function EnergiaPage() {
  const result = await readEnergyGridMapState()
    .then((payload) => ({ payload, errorMessage: "" }))
    .catch((error: unknown) => ({
      payload: null,
      errorMessage: error instanceof Error ? error.message : "energy_grid_map_unavailable",
    }));

  if (!result.payload) {
    return (
      <section className="p-5 md:p-6 lg:p-8">
        <div className="rounded-2xl border border-rose-900/50 bg-rose-950/30 p-5">
          <p className="text-xs uppercase tracking-[0.14em] text-rose-300/90">Energia BR</p>
          <h1 className="mt-2 text-2xl font-semibold text-zinc-100 md:text-3xl">Dados indisponíveis para mapa estrutural</h1>
          <p className="mt-3 text-sm text-zinc-300">
            O dashboard de anomalias da rede elétrica exige artefatos válidos do domínio energia.
          </p>
          <p className="mt-2 text-xs text-zinc-400">Erro: {result.errorMessage}</p>
        </div>
      </section>
    );
  }

  return <EnergyAnomalyDashboard payload={result.payload} />;
}
