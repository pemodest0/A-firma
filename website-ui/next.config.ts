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
      { source: "/setores", destination: "/app/finance", permanent: false },
      { source: "/benchmark", destination: "/app/dashboard", permanent: false },
      { source: "/simulador", destination: "/app/dashboard", permanent: false },
      { source: "/forecast-check", destination: "/app/dashboard", permanent: false },
      { source: "/api-docs", destination: "/app/dashboard", permanent: false },
      { source: "/sobre", destination: "/about", permanent: false },
      { source: "/dashboard", destination: "/app/dashboard", permanent: false },
      { source: "/imoveis", destination: "/app/finance", permanent: false },
      { source: "/real-estate", destination: "/app/finance", permanent: false },
      { source: "/macro", destination: "/app/finance", permanent: false },
      { source: "/metodologia", destination: "/methods", permanent: false },
      { source: "/product", destination: "/methods", permanent: false },
      { source: "/proposta", destination: "/methods", permanent: false },
      { source: "/operacao", destination: "/app/dashboard", permanent: false },
      { source: "/app/imoveis", destination: "/app/finance", permanent: false },
      { source: "/app/real-estate", destination: "/app/finance", permanent: false },
      { source: "/app/macro", destination: "/app/finance", permanent: false },
      { source: "/app/setores", destination: "/app/finance", permanent: false },
      { source: "/app/sobre", destination: "/about", permanent: false },
      { source: "/pt", destination: "/", permanent: false },
      { source: "/pt/about", destination: "/about", permanent: false },
      { source: "/pt/contact", destination: "/contact", permanent: false },
      { source: "/pt/methods", destination: "/methods", permanent: false },
      { source: "/pt/product", destination: "/methods", permanent: false },
      { source: "/pt/proposta", destination: "/methods", permanent: false },
    ];
  },
};

export default nextConfig;
