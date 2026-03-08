import type { Metadata } from "next";
import HeroSection from "@/components/sections/HeroSection";
import ProblemSection from "@/components/sections/ProblemSection";
import HowItWorksSection from "@/components/sections/HowItWorksSection";
import SectorCoverageSection from "@/components/sections/SectorCoverageSection";
import SignalSnapshotSection from "@/components/site/SignalSnapshotSection";
import { buildPageMetadata } from "@/lib/site/metadata";
import { readSiteFinanceSnapshot } from "@/lib/server/data";

export const metadata: Metadata = buildPageMetadata({
  title: "Eigen Engine para investimentos com risco controlado",
  description:
    "Plataforma pessoal de diagnóstico estrutural para finanças e cripto. O Eigen Engine organiza regime, orçamento de risco e evidências auditáveis sem look-ahead.",
  path: "/",
  locale: "pt-BR",
  keywords: [
    "eigen engine",
    "diagnóstico estrutural",
    "risco controlado",
    "regime de mercado",
    "finanças",
    "cripto",
    "orçamento de risco",
    "análise espectral",
    "mercado financeiro",
  ],
});

export default async function HomePage() {
  const snapshot = await readSiteFinanceSnapshot();
  const softwareJsonLd = {
    "@context": "https://schema.org",
    "@type": "SoftwareApplication",
    name: "Eigen Engine (Assyntrax)",
    applicationCategory: "BusinessApplication",
    operatingSystem: "Web",
    description:
      "Diagnóstico estrutural para finanças e cripto com gate auditável, shadow e controle de risco.",
    offers: {
      "@type": "Offer",
      price: "0",
      priceCurrency: "USD",
      availability: "https://schema.org/InStock",
    },
  };

  const organizationJsonLd = {
    "@context": "https://schema.org",
    "@type": "Organization",
    name: "Assyntrax",
    url: "https://assyntrax.vercel.app",
    logo: "https://assyntrax.vercel.app/assets/og/eigen-engine-og.svg",
  };

  return (
    <div className="py-10 md:py-12 lg:py-14 xl:py-16">
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(softwareJsonLd) }}
      />
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(organizationJsonLd) }}
      />
      <HeroSection />
      <SignalSnapshotSection snapshot={snapshot as Record<string, unknown>} />
      <ProblemSection />
      <SectorCoverageSection />
      <HowItWorksSection />
    </div>
  );
}
