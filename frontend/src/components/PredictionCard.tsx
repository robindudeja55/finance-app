"use client";

import { motion } from "framer-motion";

export default function PredictionCard({
  symbol,
  date,
  prob_up,
  signal,
  model,
}: {
  symbol: string;
  date?: string;
  prob_up?: number;
  signal?: string;
  model?: string;
}) {
  const color =
    signal === "UP"
      ? "bg-emerald-500/20 text-emerald-300"
      : signal === "DOWN"
      ? "bg-rose-500/20 text-rose-300"
      : "bg-zinc-500/20 text-zinc-200";

  const pct = prob_up != null ? (prob_up * 100).toFixed(1) + "%" : "—";

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="rounded-xl border border-white/10 bg-white/5 p-4 backdrop-blur"
    >
      <div className="flex items-center justify-between">
        <h3 className="text-white text-xl font-semibold">{symbol}</h3>
        <span
          className={`px-3 py-1 rounded-full text-sm ${color}`}
        >
          {signal ?? "—"}
        </span>
      </div>
      <div className="mt-3 text-zinc-300">Probability Up</div>
      <div className="text-4xl font-bold text-white">{pct}</div>
      <div className="mt-3 text-xs text-zinc-400">
        Model: {model ?? "—"} · Date: {date ?? "—"}
      </div>
    </motion.div>
  );
}
