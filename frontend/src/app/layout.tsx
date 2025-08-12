import type { Metadata } from "next";
import "./globals.css";
import Providers from "./providers";

export const metadata: Metadata = {
  title: "Market Prediction AI",
  description: "Daily signal & chart by Robin Dudeja",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <title>{metadata.title?.toString()}</title>
        <meta name="description" content={metadata.description || ""} />
      </head>
      <body className="antialiased bg-[radial-gradient(1200px_800px_at_70%_-10%,rgba(99,102,241,0.25),transparent),radial-gradient(800px_600px_at_0%_0%,rgba(16,185,129,0.15),transparent)]">
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
