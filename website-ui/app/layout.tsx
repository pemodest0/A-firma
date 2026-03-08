import type { Metadata } from "next";
import "./globals.css";
import { siteMetadataBase } from "@/lib/site/metadata";

export const metadata: Metadata = {
  metadataBase: siteMetadataBase(),
  title: {
    default: "Assyntrax | Plataforma de investimentos com risco controlado",
    template: "%s | Assyntrax",
  },
  description: "Plataforma pessoal para finanças e cripto com Eigen Engine, diagnóstico estrutural, orçamento de risco e trilha auditável.",
  icons: {
    icon: "/assets/brand/assyntrax-mark.svg",
    shortcut: "/assets/brand/assyntrax-mark.svg",
    apple: "/assets/brand/assyntrax-mark.svg",
  },
  openGraph: {
    type: "website",
    siteName: "Assyntrax | Plataforma de investimentos com risco controlado",
    title: "Assyntrax | Plataforma de investimentos com risco controlado",
    description: "Plataforma pessoal para finanças e cripto com Eigen Engine, diagnóstico estrutural, orçamento de risco e trilha auditável.",
    images: [
      {
        url: "/assets/og/eigen-engine-og.svg",
        width: 1200,
        height: 630,
        alt: "Assyntrax - Plataforma de investimentos com risco controlado",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "Assyntrax | Plataforma de investimentos com risco controlado",
    description: "Plataforma pessoal para finanças e cripto com Eigen Engine, diagnóstico estrutural, orçamento de risco e trilha auditável.",
    images: ["/assets/og/eigen-engine-og.svg"],
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="pt-BR" suppressHydrationWarning>
      <body suppressHydrationWarning className="antialiased">
        {children}
      </body>
    </html>
  );
}
