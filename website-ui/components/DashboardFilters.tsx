"use client";

import { useMemo, useState } from "react";

type UniverseAsset = {
  asset: string;
  group?: string;
};

type Props = {
  assets: UniverseAsset[];
  selected: string[];
  onSelectedChange: (assets: string[]) => void;
  sector: string;
  onSectorChange: (value: string) => void;
  timeframe: string;
  onTimeframeChange: (value: string) => void;
  rangePreset: string;
  onRangePresetChange: (value: string) => void;
  normalize: boolean;
  onNormalizeChange: (value: boolean) => void;
  showRegimeBands: boolean;
  onShowRegimeBandsChange: (value: boolean) => void;
  regimeBandsLabel?: string;
  regimeBandsTitle?: string;
  smoothing: "none" | "ema_short" | "ema_long";
  onSmoothingChange: (value: "none" | "ema_short" | "ema_long") => void;
  sectors?: Array<{ value: string; label: string }>;
};

const groupLabels: Record<string, string> = {
  crypto: "Cripto",
  volatility: "Volatilidade",
  commodities_broad: "Commodities",
  metals: "Metais",
  bonds_rates: "Juros e Bonds",
  fx: "Moedas",
  equities_us_broad: "Ações EUA - índice amplo",
  equities_us_sectors: "Ações EUA - setores",
  equities_international: "Ações internacionais",
};

const helperText = {
  ativos: "Escolha os ativos que entram no gráfico e na tabela. Limite atual: até 40 ativos.",
  grupos: "Filtra o universo por setor/grupo de mercado para leitura mais limpa.",
  frequencia: "Diário mostra mais detalhe; semanal suaviza ruído de curto prazo.",
  periodo: "Define o trecho do histórico exibido no gráfico.",
  suavizacao: "A suavização só melhora leitura visual; não muda o dado bruto.",
  normalizar: "Base 100 facilita comparar ativos com preços em escalas diferentes.",
  destaques: "Mostra faixas de estado estrutural no fundo do gráfico.",
};

function labelForSelected(selected: string[]) {
  if (!selected.length) return "Selecionar ativos";
  if (selected.length === 1) return selected[0];
  return `${selected[0]} +${selected.length - 1}`;
}

export default function DashboardFilters(props: Props) {
  const {
    assets,
    selected,
    onSelectedChange,
    sector,
    onSectorChange,
    timeframe,
    onTimeframeChange,
    rangePreset,
    onRangePresetChange,
    normalize,
    onNormalizeChange,
    showRegimeBands,
    onShowRegimeBandsChange,
    regimeBandsLabel = "Destaques",
    regimeBandsTitle = "Exibir ou ocultar destaques no fundo do gráfico",
    smoothing,
    onSmoothingChange,
    sectors,
  } = props;

  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return assets
      .filter((a) => (sectors && sectors.length ? true : sector === "all" || (a.group || "") === sector))
      .filter((a) => !q || a.asset.toLowerCase().includes(q) || (a.group || "").toLowerCase().includes(q))
      .slice(0, 250);
  }, [assets, query, sector, sectors]);

  const sectorsFromAssets = useMemo(() => {
    const set = new Set<string>();
    assets.forEach((a) => {
      if (a.group) set.add(a.group);
    });
    return ["all", ...Array.from(set).sort()];
  }, [assets]);

  const toggleAsset = (asset: string) => {
    if (selected.includes(asset)) {
      onSelectedChange(selected.filter((a) => a !== asset));
      return;
    }
    if (selected.length >= 40) return;
    onSelectedChange([...selected, asset]);
  };

  return (
    <div className="sticky top-2 z-20 rounded-xl border border-zinc-800 bg-zinc-950/95 p-4 md:p-5 backdrop-blur-sm">
      <div className="mb-3 flex flex-wrap items-center gap-2 text-xs text-zinc-500">
        <span className="rounded-full border border-zinc-700 px-2 py-1">Ativos: {selected.length}</span>
        <span className="rounded-full border border-zinc-700 px-2 py-1">Seleção: {labelForSelected(selected)}</span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-7 gap-3">
        <div className="xl:col-span-2 relative">
          <FilterLabel label="Ativos selecionados" helper={helperText.ativos} />
          <button
            className="w-full rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-2 text-left text-sm"
            onClick={() => setOpen((v) => !v)}
            title={helperText.ativos}
            aria-label="Abrir seletor de ativos"
          >
            {selected.length ? `${selected.length} ativos | ${labelForSelected(selected)}` : "Selecionar ativos"}
          </button>
          {open ? (
            <div className="absolute mt-2 w-full max-h-80 overflow-auto rounded-lg border border-zinc-700 bg-zinc-950 p-2 shadow-xl">
              <input
                className="mb-2 w-full rounded border border-zinc-700 bg-zinc-900 px-2 py-1.5 text-xs"
                placeholder="Buscar ativo"
                aria-label="Buscar ativo"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
              />
              <div className="space-y-1">
                {filtered.map((a) => (
                  <label key={a.asset} className="flex items-center gap-2 rounded px-1 py-1 text-xs hover:bg-zinc-900" title={`Grupo: ${groupLabels[a.group || ""] || a.group || "Sem grupo"}`}>
                    <input
                      type="checkbox"
                      checked={selected.includes(a.asset)}
                      onChange={() => toggleAsset(a.asset)}
                      className="h-3 w-3 accent-cyan-400"
                    />
                    <span className="font-medium">{a.asset}</span>
                    <span className="text-zinc-500">{groupLabels[a.group || ""] || a.group || "Sem grupo"}</span>
                  </label>
                ))}
              </div>
              <div className="mt-2 text-[11px] text-zinc-500">Máximo de 40 ativos por leitura.</div>
            </div>
          ) : null}
        </div>

        <div>
          <FilterLabel label="Setor" helper={helperText.grupos} />
          <select
            value={sector}
            onChange={(e) => onSectorChange(e.target.value)}
            aria-label="Filtrar setor"
            className="w-full rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm"
            title={helperText.grupos}
          >
            {(sectors && sectors.length
              ? sectors
              : sectorsFromAssets.map((s) => ({ value: s, label: s === "all" ? "Todos os grupos" : groupLabels[s] || s }))
            ).map((s) => (
              <option key={s.value} value={s.value}>
                {s.label}
              </option>
            ))}
          </select>
        </div>

        <div>
          <FilterLabel label="Frequência" helper={helperText.frequencia} />
          <select
            value={timeframe}
            onChange={(e) => onTimeframeChange(e.target.value)}
            aria-label="Selecionar frequência"
            className="w-full rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm"
            title={helperText.frequencia}
          >
            <option value="daily">Diário (detalhe maior)</option>
            <option value="weekly">Semanal (mais estável)</option>
          </select>
        </div>

        <div>
          <FilterLabel label="Período" helper={helperText.periodo} />
          <select
            value={rangePreset}
            onChange={(e) => onRangePresetChange(e.target.value)}
            aria-label="Selecionar janela temporal"
            className="w-full rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm"
            title={helperText.periodo}
          >
            <option value="30d">Últimos 30 dias</option>
            <option value="90d">Últimos 90 dias</option>
            <option value="180d">Últimos 180 dias</option>
            <option value="1y">Último 1 ano</option>
            <option value="all">Histórico completo</option>
          </select>
        </div>

        <div>
          <FilterLabel label="Suavização" helper={helperText.suavizacao} />
          <select
            value={smoothing}
            onChange={(e) => onSmoothingChange(e.target.value as "none" | "ema_short" | "ema_long")}
            aria-label="Selecionar suavização"
            className="w-full rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm"
            title={helperText.suavizacao}
          >
            <option value="none">Sem suavização</option>
            <option value="ema_short">EMA curta</option>
            <option value="ema_long">EMA longa</option>
          </select>
        </div>

        <div className="space-y-2">
          <button
            className={`w-full rounded-lg border px-3 py-2 text-sm ${normalize ? "border-cyan-400 text-cyan-300" : "border-zinc-700"}`}
            onClick={() => onNormalizeChange(!normalize)}
            title={helperText.normalizar}
            aria-label={normalize ? "Desativar normalização" : "Ativar normalização"}
          >
            {normalize ? "Normalizar: ON" : "Normalizar: OFF"}
          </button>
          <button
            className={`w-full rounded-lg border px-3 py-2 text-sm ${showRegimeBands ? "border-cyan-400 text-cyan-300" : "border-zinc-700"}`}
            onClick={() => onShowRegimeBandsChange(!showRegimeBands)}
            title={regimeBandsTitle}
            aria-label={showRegimeBands ? "Ocultar faixas de regime" : "Mostrar faixas de regime"}
          >
            {showRegimeBands ? `${regimeBandsLabel}: ON` : `${regimeBandsLabel}: OFF`}
          </button>
        </div>
      </div>

      <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-2 text-[11px] text-zinc-500">
        <div>Normalizar: {helperText.normalizar}</div>
        <div>{regimeBandsLabel}: {helperText.destaques}</div>
      </div>
    </div>
  );
}

function FilterLabel({ label, helper }: { label: string; helper: string }) {
  return (
    <div className="mb-1 flex items-center gap-1 text-[11px] uppercase tracking-[0.12em] text-zinc-500">
      <span>{label}</span>
      <span
        className="inline-flex h-4 w-4 items-center justify-center rounded-full border border-zinc-700 text-[10px] text-zinc-400"
        title={helper}
        aria-label={helper}
      >
        ?
      </span>
    </div>
  );
}
