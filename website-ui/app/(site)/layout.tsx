import "@/app/globals.css";
import SiteHeader from "@/components/SiteHeader";
import Link from "next/link";

export default function SiteLayout({ children }: { children: React.ReactNode }) {
  const year = 2026;
  return (
    <div className="relative min-h-screen overflow-hidden ax-site-bg text-zinc-100">
      <div aria-hidden className="pointer-events-none absolute inset-0 ax-site-grid-layer" />
      <div aria-hidden className="pointer-events-none absolute inset-0 ax-site-physics-layer" />
      <div aria-hidden className="pointer-events-none absolute inset-0 ax-site-vignette" />
      <div className="relative z-10">
        <SiteHeader />
        <main className="mx-auto max-w-7xl px-4 md:px-6 lg:px-8 pb-10 md:pb-12 lg:pb-14 xl:pb-16">{children}</main>
        <footer className="border-t border-zinc-900/80 bg-black/40">
          <div className="mx-auto max-w-7xl px-4 md:px-6 lg:px-8 py-10 grid grid-cols-1 md:grid-cols-4 gap-8 text-sm">
            <div className="space-y-3">
              <div className="text-zinc-100 font-semibold tracking-wide">Assyntrax</div>
              <p className="text-zinc-400 text-xs leading-relaxed">
                Sistema de diagnóstico de regime e risco estrutural para finanças, com cálculo causal e trilha auditável.
              </p>
            </div>
            <div className="space-y-2">
              <div className="text-zinc-200 text-xs uppercase tracking-[0.2em]">Produto</div>
              <Link className="block text-zinc-400 hover:text-zinc-200" href="/methods">Eigen Engine</Link>
              <Link className="block text-zinc-400 hover:text-zinc-200" href="/guia">Leonardo (guia)</Link>
              <Link className="block text-zinc-400 hover:text-zinc-200" href="/app/dashboard">Dashboard</Link>
            </div>
            <div className="space-y-2">
              <div className="text-zinc-200 text-xs uppercase tracking-[0.2em]">Suporte</div>
              <Link className="block text-zinc-400 hover:text-zinc-200" href="/contact">Contato</Link>
              <Link className="block text-zinc-400 hover:text-zinc-200" href="/privacy">Política de Privacidade</Link>
              <Link className="block text-zinc-400 hover:text-zinc-200" href="/about">Sobre</Link>
            </div>
            <div className="space-y-2">
              <div className="text-zinc-200 text-xs uppercase tracking-[0.2em]">Conformidade</div>
              <p className="text-zinc-400 text-xs">Sem promessa de retorno. Sem recomendação de compra ou venda.</p>
              <p className="text-zinc-400 text-xs">Uso focado em risco, governança e auditoria.</p>
              <p className="text-zinc-500 text-xs">Garantias técnicas: cálculo causal, artefatos auditáveis e bloqueio de publicação por gate.</p>
            </div>
          </div>
          <div className="mx-auto max-w-7xl px-4 md:px-6 lg:px-8 pb-6 pt-1 border-t border-zinc-900/70 text-xs text-zinc-500 flex flex-wrap items-center justify-between gap-2">
            <span>© {year} Assyntrax Labs. Todos os direitos reservados.</span>
            <div className="flex items-center gap-4">
              <a href="https://github.com/pemodest0/Assyntrax" className="hover:text-zinc-300 transition">GitHub</a>
              <a href="https://www.linkedin.com/in/pedro-henrique-gesualdo-modesto-39a135272/" className="hover:text-zinc-300 transition">LinkedIn</a>
              <a href="https://pemodest0.github.io/" className="hover:text-zinc-300 transition">Portfólio</a>
              <span>Eigen Engine</span>
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
}
