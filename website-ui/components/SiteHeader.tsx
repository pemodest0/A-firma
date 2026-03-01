"use client";

import Image from "next/image";
import Link from "next/link";

export default function SiteHeader() {
  return (
    <header className="mx-auto max-w-7xl px-4 md:px-6 lg:px-8 py-4 md:py-5 flex flex-wrap items-center justify-between gap-4">
      <Link href="/" className="flex items-center gap-3">
        <Image
          src="/assets/brand/assyntrax-mark.svg"
          alt="Assyntrax"
          width={40}
          height={40}
          className="h-10 w-10"
          priority
        />
        <span className="text-zinc-100 text-base md:text-lg tracking-[0.24em] font-medium">ASSYNTRAX</span>
      </Link>
      <nav className="flex items-center gap-3 md:gap-5 text-sm text-zinc-300">
        <Link className="hover:text-white transition" href="/financas">
          Finanças
        </Link>
        <Link className="hover:text-white transition" href="/energia">
          Energia
        </Link>
        <Link className="hover:text-white transition" href="/agro">
          Agro
        </Link>
        <Link className="hover:text-white transition" href="/evidencias">
          Evidências
        </Link>
        <Link className="hover:text-white transition" href="/methods">
          Eigen Engine
        </Link>
        <Link className="rounded-xl border border-cyan-300/45 bg-white/5 px-3 py-2 font-medium text-zinc-100 hover:border-cyan-200 hover:bg-white/10 transition" href="/contact">
          Solicitar demonstração
        </Link>
      </nav>
    </header>
  );
}
