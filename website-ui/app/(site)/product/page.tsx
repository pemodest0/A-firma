import ProductMock from "@/components/visuals/ProductMock";
import SignalWeave from "@/components/visuals/SignalWeave";
import ProofMatrix from "@/components/visuals/ProofMatrix";
import type { Metadata } from "next";
import { buildPageMetadata } from "@/lib/site/metadata";

export const metadata: Metadata = buildPageMetadata({
  title: "Produto: painel e API com gates auditáveis",
  description:
    "Painel + API com regime, confiança, qualidade, status operacional e motivo do gate. Integração direta para operação.",
  path: "/product",
  locale: "pt-BR",
  keywords: ["dashboard de risco", "api de risco", "gate operacional", "trilha auditável"],
});

export default function ProductPage() {
  const productJsonLd = {
    "@context": "https://schema.org",
    "@type": "SoftwareApplication",
    name: "Eigen Engine (Assyntrax)",
    applicationCategory: "BusinessApplication",
    operatingSystem: "Web",
    description:
      "Painel e API para diagnóstico causal de regime e risco estrutural com trilha auditável.",
    url: "https://assyntrax.vercel.app/product",
  };

  return (
    <div className="space-y-16">
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(productJsonLd) }}
      />
      <section className="grid grid-cols-1 lg:grid-cols-[1.1fr_0.9fr] gap-10 items-center">
        <div className="space-y-4">
          <div className="text-xs uppercase tracking-[0.3em] text-zinc-400">Produto</div>
          <h1 className="text-4xl md:text-5xl font-semibold tracking-tight">
            Estrutura, métricas e diagnóstico em um fluxo único
          </h1>
          <p className="text-zinc-300 max-w-3xl text-lg">
            O Eigen Engine combina analise espectral de correlacoes, classificacao de regime e governanca de publicacao.
            O foco é leitura de risco estrutural com transparência técnica.
          </p>
        </div>
        <SignalWeave />
      </section>

      <section className="grid grid-cols-1 lg:grid-cols-[1.05fr_0.95fr] gap-10 items-center">
        <div className="space-y-4">
          <h2 className="text-3xl font-semibold">Dashboard operacional</h2>
          <p className="text-zinc-300 text-lg">
            Leitura de regime com confiança, qualidade, status do gate e motivo explícito.
            Quando o sistema está inconclusivo, a publicação permanece em modo diagnóstico.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-zinc-300">
            <div className="rounded-2xl border border-zinc-800 bg-zinc-950/60 p-5">
              Estado atual: o que mudou no sistema hoje.
            </div>
            <div className="rounded-2xl border border-zinc-800 bg-zinc-950/60 p-5">
              Governança operacional: quando a publicação é bloqueada por qualidade.
            </div>
          </div>
        </div>
        <ProductMock />
      </section>

      <section id="saida-json" className="grid grid-cols-1 lg:grid-cols-[0.9fr_1.1fr] gap-10 items-center">
        <ProofMatrix />
        <div className="space-y-4">
          <div className="text-xs uppercase tracking-[0.3em] text-zinc-400">API e integração</div>
          <h2 className="text-3xl font-semibold">Saídas prontas para produção</h2>
          <p className="text-zinc-300 text-lg">
            Respostas em JSON com regime, confiança, qualidade, status, motivo e id de execução para auditoria.
            Integração direta com BI, alertas e sistemas internos.
          </p>
          <pre className="rounded-2xl border border-zinc-800 bg-black/60 p-4 text-xs text-zinc-200 overflow-x-auto">
{`{
  "regime": "transition",
  "confidence": 0.84,
  "quality": 0.78,
  "status": "watch",
  "reason": "transição persistente",
  "id_execucao": "20260210_xxx"
}`}
          </pre>
        </div>
      </section>

      <section className="space-y-4">
        <h2 className="text-2xl font-semibold">Escopo de aplicação</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Card title="Finanças">
            Regime diário para ativos líquidos, com leitura causal e rastreamento por id de execução.
            O produto está em fase simplificada com foco exclusivo em Finanças.
          </Card>
          <Card title="Operações e risco institucional">
            Base para comitês e governança: status, motivo do gate, limites de uso
            e trilha auditável por execução.
          </Card>
          <Card title="Limites do produto">
            Sem promessa de retorno e sem recomendação de compra ou venda.
            Uso focado em risco estrutural, governança e auditoria.
          </Card>
          <Card title="Garantias técnicas">
            Cálculo causal sem look-ahead, artefatos completos por execução
            e bloqueio automático de publicação quando critérios mínimos falham.
          </Card>
        </div>
      </section>
    </div>
  );
}

function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-2xl border border-zinc-800 bg-zinc-900/40 p-6">
      <div className="text-lg font-semibold">{title}</div>
      <div className="mt-3 text-sm text-zinc-300">{children}</div>
    </div>
  );
}
