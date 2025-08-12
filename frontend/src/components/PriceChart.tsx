"use client";

import ReactECharts from "echarts-for-react";

export default function PriceChart({
  series,
}: {
  series: { date: string; close: number }[];
}) {
  const dates = series.map((d) => d.date);
  const closes = series.map((d) => d.close);

  const option = {
    backgroundColor: "transparent",
    tooltip: { trigger: "axis" },
    grid: { left: 40, right: 20, top: 20, bottom: 40 },
    xAxis: {
      type: "category",
      data: dates,
      boundaryGap: false,
      axisLabel: { color: "#a3a3a3" },
    },
    yAxis: {
      type: "value",
      axisLabel: { color: "#a3a3a3" },
      splitLine: {
        lineStyle: { color: "rgba(255,255,255,0.08)" },
      },
    },
    series: [
      {
        type: "line",
        data: closes,
        smooth: true,
        symbol: "none",
        lineStyle: { width: 2, color: "#60a5fa" },
        areaStyle: {
          color: {
            type: "linear",
            x: 0,
            y: 0,
            x2: 0,
            y2: 1,
            colorStops: [
              { offset: 0, color: "rgba(96,165,250,0.35)" },
              { offset: 1, color: "rgba(96,165,250,0.00)" },
            ],
          },
        },
      },
    ],
  };

  return (
    <div className="rounded-xl border border-white/10 bg-white/5 p-4 backdrop-blur">
      <ReactECharts option={option} style={{ height: 320, width: "100%" }} />
    </div>
  );
}
