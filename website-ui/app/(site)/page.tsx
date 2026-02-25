import type { Metadata } from "next";
import HeroSection from "@/components/sections/HeroSection";
import ProblemSection from "@/components/sections/ProblemSection";
import HowItWorksSection from "@/components/sections/HowItWorksSection";
import ProductSection from "@/components/sections/ProductSection";
import CTASection from "@/components/sections/CTASection";
import { buildPageMetadata } from "@/lib/site/metadata";

export const metadata: Metadata = buildPageMetadata({
  title: "Diagnóstico causal de regime e risco estrutural",
  description:
    "Monitor de risco estrutural para equipes de risco e governança. Entrega leitura causal de regime, sem promessa de retorno.",
  path: "/",
  locale: "pt-BR",
  keywords: [
    "diagnóstico de regime",
    "risco estrutural",
    "monitor de risco",
    "análise espectral",
    "mercado financeiro",
    "event study",
  ],
});

export default function HomePage() {
  const softwareJsonLd = {
    "@context": "https://schema.org",
    "@type": "SoftwareApplication",
    name: "Eigen Engine (Assyntrax)",
    applicationCategory: "BusinessApplication",
    operatingSystem: "Web",
    description:
      "Diagnóstico causal de regime e risco estrutural com gate auditável para uso institucional.",
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
      <ProblemSection />
      <HowItWorksSection />
      <ProductSection />
      <CTASection />
    </div>
  );
}
