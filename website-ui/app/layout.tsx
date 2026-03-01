import type { Metadata } from "next";
import "./globals.css";
import { siteMetadataBase } from "@/lib/site/metadata";

export const metadata: Metadata = {
  metadataBase: siteMetadataBase(),
  title: {
    default: "Assyntrax | Diagnóstico de Regime",
    template: "%s | Assyntrax",
  },
  description: "Diagnóstico causal de regime e risco estrutural com trilha auditável e governança de publicação.",
  icons: {
    icon: "/assets/brand/assyntrax-mark.svg",
    shortcut: "/assets/brand/assyntrax-mark.svg",
    apple: "/assets/brand/assyntrax-mark.svg",
  },
  openGraph: {
    type: "website",
    siteName: "Assyntrax | Diagnóstico de Regime",
    title: "Assyntrax | Diagnóstico de Regime",
    description: "Diagnóstico causal de regime e risco estrutural com trilha auditável e governança de publicação.",
    images: [
      {
        url: "/assets/og/eigen-engine-og.svg",
        width: 1200,
        height: 630,
        alt: "Assyntrax - Diagnóstico de Regime",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "Assyntrax | Diagnóstico de Regime",
    description: "Diagnóstico causal de regime e risco estrutural com trilha auditável e governança de publicação.",
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
