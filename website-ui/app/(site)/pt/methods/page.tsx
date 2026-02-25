import type { Metadata } from "next";
import MethodsPageClient from "@/components/pages/MethodsPageClient";
import { buildPageMetadata } from "@/lib/site/metadata";

export const metadata: Metadata = buildPageMetadata({
  title: "Eigen Engine: metodologia e guias operacionais",
  description: "Versão espelhada em /pt. Conteúdo principal publicado em /methods.",
  path: "/pt/methods",
  locale: "pt-BR",
  noIndex: true,
  canonicalPath: "/methods",
});

export default function MethodsPagePT() {
  return <MethodsPageClient />;
}
