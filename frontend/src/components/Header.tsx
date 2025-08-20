"use client";

import Image from "next/image";
import { useRouter, useSearchParams } from "next/navigation";
import { useEffect, useState, useTransition } from "react";

export default function Header({ symbols }: { symbols: string[] }) {
  const router = useRouter();
  const sp = useSearchParams();
  const current = (sp.get("symbol") || symbols[0] || "AAPL").toUpperCase();
  const [value, setValue] = useState(current);
  const [isPending, startTransition] = useTransition();

  useEffect(() => setValue(current), [current]);

  return (
    <div className="sticky top-0 z-50 backdrop-blur border-b border-white/10 bg-black/30">
      <div className="max-w-6xl mx-auto px-6 md:px-10 py-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Image src="/logo.jpeg" alt="Robin" width={24} height={24} priority />
          <span className="text-white font-semibold tracking-wide">Robin Markets AI</span>
        </div>
        <div className="flex items-center gap-3">
          <label className="text-sm text-zinc-300">Symbol</label>
          <select
            value={value}
            onChange={(e) => {
              const s = e.target.value;
              setValue(s);
              startTransition(() => router.push(`/?symbol=${s}`));
            }}
            className="bg-white/10 text-white rounded-md px-3 py-1.5 border border-white/20 focus:outline-none"
          >
            {symbols.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </div>
      </div>
    </div>
  );
}
