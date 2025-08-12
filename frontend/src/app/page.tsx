import Image from "next/image";
import { getPrediction, getPriceSeries } from "@/lib/api";
import PredictionCard from "@/components/PredictionCard";
import PriceChart from "@/components/PriceChart";

export const dynamic = "force-dynamic";

type SP = Promise<Record<string, string | string[] | undefined>>;

export default async function Page({
  searchParams,
}: {
  searchParams: SP;
}) {
  // Await searchParams because it's dynamic in Next.js 15+
  const sp = await searchParams;
  const raw = sp?.symbol;
  const symbol = (Array.isArray(raw) ? raw[0] : (raw ?? "AAPL")).toUpperCase();

  let pred: any = null;
  let series: any = { series: [] };

  try {
    [pred, series] = await Promise.all([
      getPrediction(symbol),
      getPriceSeries(symbol, 60),
    ]);
  } catch (error) {
    pred = null;
    series = { series: [] };
    console.error("API fetch failed:", error);
  }

  return (
    <main className="min-h-screen text-white px-6 md:px-10 py-8">
      <header className="mb-8">
        <div className="flex items-center gap-4">
          <Image
            className="dark:invert"
            src="/next.svg"
            alt="Next.js logo"
            width={140}
            height={30}
            priority
          />
          <h1 className="text-3xl md:text-4xl font-bold tracking-tight">
            Market Prediction AI
          </h1>
        </div>
        <p className="text-zinc-300 mt-1">
          Daily signal & chart. Built by Robin Dudeja · CS Grad, University of Dayton
        </p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        <div className="lg:col-span-2">
          {pred ? (
            <PredictionCard
              symbol={pred.symbol}
              date={pred.date}
              prob_up={pred.prob_up}
              signal={pred.signal}
              model={pred.model}
            />
          ) : (
            <div className="rounded-xl border border-white/10 bg-white/5 p-4 backdrop-blur text-rose-300">
              Failed to load prediction data.
            </div>
          )}
        </div>
        <div className="lg:col-span-3">
          {series?.series?.length ? (
            <PriceChart series={series.series} />
          ) : (
            <div className="rounded-xl border border-white/10 bg-white/5 p-4 backdrop-blur text-rose-300">
              Failed to load price series.
            </div>
          )}
        </div>
      </div>

      <footer className="mt-12 text-sm text-zinc-400">
        © {new Date().getFullYear()} Robin Dudeja — All rights reserved.
      </footer>
    </main>
  );
}
