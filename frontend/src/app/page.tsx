import Header from "@/components/Header";
import { getPrediction, getPriceSeries, getSymbols } from "@/lib/api";
import PredictionCard from "@/components/PredictionCard";
import PriceChart from "@/components/PriceChart";

export const dynamic = "force-dynamic";

type SP = Promise<Record<string, string | string[] | undefined>>;

export default async function Page({
  searchParams,
}: {
  searchParams: SP;
}) {
  const sp = await searchParams;
  const raw = sp?.symbol;
  const symbol = (Array.isArray(raw) ? raw[0] : raw ?? "AAPL").toUpperCase();

  let pred: any = null;
  let series: any = { series: [] };
  let symlist: { symbols: string[] } = { symbols: [] };

  try {
    [pred, series, symlist] = await Promise.all([
      getPrediction(symbol),
      getPriceSeries(symbol, 60),
      getSymbols(),
    ]);
  } catch (error) {
    pred = null;
    series = { series: [] };
    symlist = { symbols: [] };
    console.error("API fetch failed:", error);
  }

  return (
    <main className="min-h-screen text-white">
      <Header symbols={symlist.symbols} />
      <div className="px-6 md:px-10 py-8">
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
      </div>
    </main>
  );
}
