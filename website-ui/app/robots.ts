import type { MetadataRoute } from "next";
import { siteMetadataBase } from "@/lib/site/metadata";

export default function robots(): MetadataRoute.Robots {
  const base = siteMetadataBase().toString().replace(/\/$/, "");
  return {
    rules: [
      {
        userAgent: "*",
        allow: ["/", "/methods", "/about", "/contact", "/privacy"],
        disallow: ["/app/", "/api/", "/pt/"],
      },
    ],
    sitemap: `${base}/sitemap.xml`,
    host: base,
  };
}
