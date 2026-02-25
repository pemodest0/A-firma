import type { MetadataRoute } from "next";
import { siteMetadataBase } from "@/lib/site/metadata";

const base = siteMetadataBase().toString().replace(/\/$/, "");

export default function sitemap(): MetadataRoute.Sitemap {
  const now = new Date();
  const pages = [
    "/",
    "/methods",
    "/about",
    "/contact",
    "/privacy",
  ];

  return pages.map((path) => ({
    url: `${base}${path}`,
    lastModified: now,
    changeFrequency: path === "/" ? "weekly" : "monthly",
    priority: path === "/" ? 1 : 0.7,
  }));
}
