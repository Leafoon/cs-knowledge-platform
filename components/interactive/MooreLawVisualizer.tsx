"use client";

import { useState } from "react";
import { motion } from "framer-motion";

type Metric = "density" | "performance" | "power" | "cost";

const metrics: Record<Metric, { label: string; unit: string; data: { year: number; value: number }[]; color: string; desc: string }> = {
  density: {
    label: "晶体管密度",
    unit: "晶体管/mm²",
    color: "#667eea",
    desc: "每两年翻一倍 — 摩尔定律的核心预测",
    data: [
      { year: 1971, value: 230 },
      { year: 1978, value: 2900 },
      { year: 1985, value: 27500 },
      { year: 1989, value: 120000 },
      { year: 1993, value: 310000 },
      { year: 2000, value: 2300000 },
      { year: 2006, value: 11000000 },
      { year: 2012, value: 88000000 },
      { year: 2020, value: 1200000000 },
      { year: 2024, value: 3500000000 },
    ],
  },
  performance: {
    label: "单核性能",
    unit: "相对值",
    color: "#10b981",
    desc: "早期快速提升，近年因功耗墙增速放缓",
    data: [
      { year: 1971, value: 0.01 },
      { year: 1978, value: 0.1 },
      { year: 1985, value: 1 },
      { year: 1989, value: 5 },
      { year: 1993, value: 15 },
      { year: 2000, value: 100 },
      { year: 2006, value: 500 },
      { year: 2012, value: 1500 },
      { year: 2020, value: 3000 },
      { year: 2024, value: 4500 },
    ],
  },
  power: {
    label: "功耗",
    unit: "W",
    color: "#ef4444",
    desc: "功耗增长成为制约因素，推动多核和低功耗设计",
    data: [
      { year: 1971, value: 0.5 },
      { year: 1978, value: 2 },
      { year: 1985, value: 3 },
      { year: 1989, value: 5 },
      { year: 1993, value: 10 },
      { year: 2000, value: 55 },
      { year: 2006, value: 130 },
      { year: 2012, value: 77 },
      { year: 2020, value: 65 },
      { year: 2024, value: 45 },
    ],
  },
  cost: {
    label: "单位晶体管成本",
    unit: "相对值",
    color: "#f59e0b",
    desc: "晶体管成本持续下降，推动计算民主化",
    data: [
      { year: 1971, value: 1000 },
      { year: 1978, value: 100 },
      { year: 1985, value: 10 },
      { year: 1989, value: 1 },
      { year: 1993, value: 0.1 },
      { year: 2000, value: 0.01 },
      { year: 2006, value: 0.001 },
      { year: 2012, value: 0.0001 },
      { year: 2020, value: 0.00001 },
      { year: 2024, value: 0.000005 },
    ],
  },
};

export function MooreLawVisualizer() {
  const [metric, setMetric] = useState<Metric>("density");
  const [hovered, setHovered] = useState<number | null>(null);
  const d = metrics[metric];
  const maxVal = Math.max(...d.data.map((p) => p.value));

  const chartW = 700;
  const chartH = 280;
  const pad = { top: 20, right: 20, bottom: 40, left: 65 };
  const pw = chartW - pad.left - pad.right;
  const ph = chartH - pad.top - pad.bottom;

  function xPos(i: number) { return pad.left + (i / (d.data.length - 1)) * pw; }
  function yPos(v: number) { return pad.top + ph - (Math.log10(v + 1) / Math.log10(maxVal + 1)) * ph; }
  const path = d.data.map((p, i) => `${i === 0 ? "M" : "L"}${xPos(i)},${yPos(p.value)}`).join(" ");

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">
        摩尔定律可视化
      </h3>
      <p className="text-sm text-text-secondary mb-4">
        选择不同指标观察半导体技术的发展趋势
      </p>

      <div className="flex flex-wrap gap-2 mb-4">
        {(Object.keys(metrics) as Metric[]).map((m) => (
          <button
            key={m}
            onClick={() => { setMetric(m); setHovered(null); }}
            className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all border ${
              metric === m
                ? "text-white border-transparent"
                : "bg-bg-secondary text-text-secondary border-border-subtle"
            }`}
            style={metric === m ? { backgroundColor: metrics[m].color } : {}}
          >
            {metrics[m].label}
          </button>
        ))}
      </div>

      <p className="text-xs text-text-secondary mb-3">{d.desc}</p>

      <div className="overflow-x-auto">
        <svg viewBox={`0 0 ${chartW} ${chartH}`} className="w-full min-w-[500px]">
          {/* grid */}
          {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9].filter(i => {
            const v = Math.pow(10, (i / 9) * Math.log10(maxVal + 1)) - 1;
            return v <= maxVal * 1.1;
          }).map(i => {
            const v = Math.pow(10, (i / 9) * Math.log10(maxVal + 1)) - 1;
            return (
              <g key={i}>
                <line x1={pad.left} y1={yPos(v)} x2={chartW - pad.right} y2={yPos(v)}
                  stroke="#374151" strokeWidth={0.5} strokeDasharray="3 3" />
                <text x={pad.left - 8} y={yPos(v) + 3} textAnchor="end" fontSize={8} className="fill-text-secondary">
                  {v < 1 ? v.toExponential(1) : v >= 1e6 ? `${(v / 1e6).toFixed(0)}M` : v.toLocaleString()}
                </text>
              </g>
            );
          })}
          {/* x axis labels */}
          {d.data.map((p, i) => (
            <text key={i} x={xPos(i)} y={chartH - 8} textAnchor="middle" fontSize={8} className="fill-text-secondary">
              {p.year}
            </text>
          ))}
          {/* line */}
          <motion.path d={path} fill="none" stroke={d.color} strokeWidth={2.5}
            initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: 1 }}
            key={metric} />
          {/* area */}
          <motion.path
            d={`${path} L${xPos(d.data.length - 1)},${pad.top + ph} L${xPos(0)},${pad.top + ph} Z`}
            fill={d.color} opacity={0.08}
            initial={{ opacity: 0 }} animate={{ opacity: 0.08 }} transition={{ duration: 0.5 }}
            key={metric + "-area"}
          />
          {/* dots */}
          {d.data.map((p, i) => (
            <g key={i} onMouseEnter={() => setHovered(i)} onMouseLeave={() => setHovered(null)}>
              <circle cx={xPos(i)} cy={yPos(p.value)} r={hovered === i ? 6 : 3.5}
                fill={hovered === i ? "#fff" : d.color} stroke={d.color} strokeWidth={2}
                className="cursor-pointer" />
            </g>
          ))}
          {/* tooltip */}
          {hovered !== null && (
            <g>
              <rect x={Math.min(xPos(hovered) + 8, chartW - 160)} y={Math.max(yPos(d.data[hovered].value) - 40, 5)}
                width={150} height={35} rx={6} fill="#1e293b" stroke="#374151" />
              <text x={Math.min(xPos(hovered) + 16, chartW - 152)} y={Math.max(yPos(d.data[hovered].value) - 22, 22)}
                fontSize={10} fontWeight="bold" className="fill-text-primary">
                {d.data[hovered].year}年
              </text>
              <text x={Math.min(xPos(hovered) + 16, chartW - 152)} y={Math.max(yPos(d.data[hovered].value) - 10, 32)}
                fontSize={9} className="fill-text-secondary">
                {d.data[hovered].value.toLocaleString()} {d.unit}
              </text>
            </g>
          )}
        </svg>
      </div>
    </div>
  );
}
