import type { Metadata } from "next";
import MethodsPage from "@/app/(site)/pt/methods/page";
import { buildPageMetadata } from "@/lib/site/metadata";

export const metadata: Metadata = buildPageMetadata({
  title: "Eigen Engine: metodologia e guias operacionais",
  description:
    "Página única do Eigen Engine com metodologia atual, limites, guias de operação e integração de piloto.",
  path: "/methods",
  locale: "pt-BR",
});

export default MethodsPage;
