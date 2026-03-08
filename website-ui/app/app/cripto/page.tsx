import SectorDashboard from "@/components/SectorDashboard";

export default function CriptoPage() {
  return (
    <SectorDashboard
      title="Eigen Engine | Cripto"
      showTable
      initialDomain="crypto"
      initialGroupFilter="crypto"
      headline="Painel cripto por ativo"
      description="Leitura de preço, risco, estabilidade e contexto estrutural para moedas líquidas do universo ativo."
    />
  );
}
