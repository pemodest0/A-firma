"use client";

import Link from "next/link";

export default function SiteHeader() {
  return (
    <header className="mx-auto max-w-7xl px-4 md:px-6 lg:px-8 py-4 md:py-5 flex items-center justify-between">
      <Link href="/" className="font-semibold tracking-tight text-lg text-zinc-100">
        Assyntrax
      </Link>
      <nav className="flex items-center gap-4 text-sm text-zinc-300">
        <Link className="hover:text-white transition" href="/methods">
          Eigen Engine
        </Link>
        <Link className="hover:text-white transition" href="/guia">
          Leonardo
        </Link>
        <Link className="hover:text-white transition" href="/about">
          Sobre
        </Link>
        <Link className="hover:text-white transition" href="/contact">
          Contato
        </Link>
        <Link className="rounded-xl bg-zinc-100 text-black px-3 py-2 font-medium hover:bg-white transition" href="/contact">
          Pedir demo
        </Link>
      </nav>
    </header>
  );
}
