import type { Metadata } from "next";
import AboutPage from "@/app/(site)/pt/about/page";
import { buildPageMetadata } from "@/lib/site/metadata";

export const metadata: Metadata = buildPageMetadata({
  title: "Sobre o Eigen Engine",
  description:
    "Contexto do projeto, direção do produto e foco em diagnóstico estrutural para decisão com governança.",
  path: "/about",
  locale: "pt-BR",
});

export default AboutPage;
