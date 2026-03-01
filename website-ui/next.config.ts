import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  turbopack: {
    root: __dirname,
  },
  typescript: {
    ignoreBuildErrors: true,
  },
  allowedDevOrigins: ["http://192.168.0.71:3000", "http://localhost:3000"],
  async redirects() {
    return [
      { source: "/ativos", destination: "/app/dashboard", permanent: false },
      { source: "/setores", destination: "/app/dashboard", permanent: false },
      { source: "/benchmark", destination: "/app/dashboard", permanent: false },
      { source: "/simulador", destination: "/app/dashboard", permanent: false },
      { source: "/forecast-check", destination: "/app/dashboard", permanent: false },
      { source: "/api-docs", destination: "/app/dashboard", permanent: false },
      { source: "/sobre", destination: "/about", permanent: false },
      { source: "/dashboard", destination: "/app/dashboard", permanent: false },
      { source: "/imoveis", destination: "/app/dashboard", permanent: false },
      { source: "/real-estate", destination: "/app/dashboard", permanent: false },
      { source: "/macro", destination: "/app/dashboard", permanent: false },
      { source: "/metodologia", destination: "/methods", permanent: false },
      { source: "/guia", destination: "/", permanent: false },
      { source: "/product", destination: "/methods", permanent: false },
      { source: "/proposta", destination: "/methods", permanent: false },
      { source: "/operacao", destination: "/app/dashboard", permanent: false },
      { source: "/app/imoveis", destination: "/app/dashboard", permanent: false },
      { source: "/app/real-estate", destination: "/app/dashboard", permanent: false },
      { source: "/app/macro", destination: "/app/dashboard", permanent: false },
      { source: "/app/setores", destination: "/app/dashboard", permanent: false },
      { source: "/app/finance", destination: "/app/financas", permanent: false },
      { source: "/app/aplicacoes", destination: "/app/dashboard", permanent: false },
      { source: "/app/sobre", destination: "/about", permanent: false },
      { source: "/pt", destination: "/", permanent: false },
      { source: "/pt/guia", destination: "/", permanent: false },
      { source: "/pt/about", destination: "/about", permanent: false },
      { source: "/pt/contact", destination: "/contact", permanent: false },
      { source: "/pt/methods", destination: "/methods", permanent: false },
      { source: "/pt/product", destination: "/methods", permanent: false },
      { source: "/pt/proposta", destination: "/methods", permanent: false },
    ];
  },
  async headers() {
    return [
      {
        source: "/(.*)",
        headers: [
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "X-Frame-Options", value: "DENY" },
          { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
          { key: "Permissions-Policy", value: "camera=(), microphone=(), geolocation=()" },
          { key: "Cross-Origin-Opener-Policy", value: "same-origin" },
          { key: "Cross-Origin-Resource-Policy", value: "same-origin" },
          {
            key: "Content-Security-Policy",
            value:
              "default-src 'self'; base-uri 'self'; frame-ancestors 'none'; object-src 'none'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self' data:; connect-src 'self' https:;",
          },
        ],
      },
    ];
  },
};

export default nextConfig;
