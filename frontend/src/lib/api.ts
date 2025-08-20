const isServer = typeof window === "undefined";

// API base URL depends on environment (server or browser)
const API = isServer
  ? process.env.API_BASE_INTERNAL || "http://web:8000"  // SSR inside Docker container
  : process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";  // Browser

// Prediction API response type
export type PredictionJSON = {
  symbol: string;
  date?: string;
  prob_up?: number;
  signal?: string;
  model?: string;
  trained_at?: string;
  message?: string;
};

// Price series API response type
export type SeriesJSON = {
  symbol: string;
  series: { date: string; close: number }[];
};

// Fetch prediction for given symbol
export async function getPrediction(symbol: string): Promise<PredictionJSON> {
  const response = await fetch(`${API}/api/prediction?symbol=${encodeURIComponent(symbol)}`, {
    cache: "no-store",
  });
  if (!response.ok) {
    throw new Error(`Prediction fetch failed: ${response.status}`);
  }
  return response.json();
}

// Fetch price series for given symbol and days
export async function getPriceSeries(symbol: string, days = 60): Promise<SeriesJSON> {
  const response = await fetch(
    `${API}/api/price-series?symbol=${encodeURIComponent(symbol)}&days=${days}`,
    {
      cache: "no-store",
    }
  );
  if (!response.ok) {
    throw new Error(`Series fetch failed: ${response.status}`);
  }
  return response.json();
}


export async function getSymbols(): Promise<{ symbols: string[] }> {
  const r = await fetch(`${API}/api/symbols`, { cache: "no-store" });
  if (!r.ok) throw new Error(`Symbols fetch failed: ${r.status}`);
  return r.json();
}
