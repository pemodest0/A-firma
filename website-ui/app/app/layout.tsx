import "@/app/globals.css";
import Link from "next/link";
import GlobalStructuralBackground from "@/components/visuals/GlobalStructuralBackground";

export default function AppLayout({ children }: { children: React.ReactNode }) {
  const navItems = [
    { href: "/app/dashboard", label: "Dashboard", enabled: true },
    { href: "/app/aplicacoes", label: "Como usar", enabled: true },
    { href: "/app/financas", label: "Finanças", enabled: true },
    { href: "/app/cripto", label: "Cripto", enabled: true },
    { href: "/app/shadow-mode", label: "Shadow mode", enabled: true },
    { href: "/app/copiloto", label: "Copiloto", enabled: true },
    { href: "/app/teoria", label: "Teoria", enabled: true },
  ].filter((item) => item.enabled);

  return (
    <div className="relative min-h-screen overflow-hidden ax-site-bg text-zinc-100">
      <GlobalStructuralBackground />
      <div aria-hidden className="pointer-events-none absolute inset-0 ax-site-vignette" />
      <div className="relative z-10 mx-auto max-w-none w-full px-4 py-6">
        <div className="grid grid-cols-12 gap-6">
          <aside className="col-span-12 lg:col-span-3">
            <div className="rounded-2xl border border-zinc-800 bg-zinc-900/60 backdrop-blur shadow-lg p-4">
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-lg font-semibold tracking-tight">Assyntrax</div>
                  <div className="text-xs text-zinc-400">Eigen Engine | finanças e cripto</div>
                </div>
                <span className="text-[10px] px-2 py-1 rounded-full border border-zinc-700 text-zinc-300">
                  Plataforma pessoal
                </span>
              </div>

              <nav className="mt-6 space-y-1 text-sm">
                {navItems.map((item) => (
                  <NavItem key={item.href} href={item.href} label={item.label} />
                ))}
              </nav>

              <div className="mt-6 border-t border-zinc-800 pt-4 text-xs text-zinc-400">
                <div className="flex items-center justify-between">
                  <span>Website</span>
                  <Link className="underline hover:text-zinc-200" href="/">
                    Home
                  </Link>
                </div>
              </div>
            </div>
          </aside>

          <main className="col-span-12 lg:col-span-9">
            <div className="rounded-2xl border border-zinc-800 bg-zinc-900/40 backdrop-blur shadow-lg w-full">
              {children}
            </div>
          </main>
        </div>
      </div>
    </div>
  );
}

function NavItem({ href, label }: { href: string; label: string }) {
  return (
    <Link
      href={href}
      className="block rounded-xl px-3 py-2 hover:bg-zinc-800/60 transition border border-transparent hover:border-zinc-700"
    >
      {label}
    </Link>
  );
}
