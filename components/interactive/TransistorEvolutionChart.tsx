"use client";

import { useState } from "react";
import { motion } from "framer-motion";

const processors = [
  { year: 1971, name: "Intel 4004", transistors: 2300, process: "10μm" },
  { year: 1974, name: "Intel 8080", transistors: 4500, process: "6μm" },
  { year: 1978, name: "Intel 8086", transistors: 29000, process: "3μm" },
  { year: 1982, name: "Intel 80286", transistors: 134000, process: "1.5μm" },
  { year: 1985, name: "Intel 80386", transistors: 275000, process: "1μm" },
  { year: 1989, name: "Intel 80486", transistors: 1200000, process: "1μm" },
  { year: 1993, name: "Pentium", transistors: 3100000, process: "800nm" },
  { year: 1999, name: "Pentium III", transistors: 9500000, process: "250nm" },
  { year: 2000, name: "Pentium 4", transistors: 42000000, process: "180nm" },
  { year: 2006, name: "Core 2 Duo", transistors: 291000000, process: "65nm" },
  { year: 2012, name: "Core i7 (IVB)", transistors: 1400000000, process: "22nm" },
  { year: 2020, name: "Apple M1", transistors: 16000000000, process: "5nm" },
  { year: 2022, name: "Apple M2", transistors: 20000000000, process: "5nm" },
  { year: 2024, name: "Apple M4", transistors: 28000000000, process: "3nm" },
];

function formatNum(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(0)}K`;
  return n.toString();
}

export function TransistorEvolutionChart() {
  const [hovered, setHovered] = useState<number | null>(null);
  const maxT = Math.max(...processors.map((p) => p.transistors));
  const chartW = 800;
  const chartH = 300;
  const padding = { top: 20, right: 30, bottom: 40, left: 60 };
  const plotW = chartW - padding.left - padding.right;
  const plotH = chartH - padding.top - padding.bottom;

  function x(i: number) {
    return padding.left + (i / (processors.length - 1)) * plotW;
  }
  function y(t: number) {
    return padding.top + plotH - (Math.log10(t) / Math.log10(maxT)) * plotH;
  }

  const path = processors.map((p, i) => `${i === 0 ? "M" : "L"}${x(i)},${y(p.transistors)}`).join(" ");

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">
        晶体管数量演进图
      </h3>
      <p className="text-sm text-text-secondary mb-4">
        从 Intel 4004 到现代处理器，晶体管数量呈指数增长（对数坐标）
      </p>

      <div className="overflow-x-auto">
        <svg viewBox={`0 0 ${chartW} ${chartH}`} className="w-full min-w-[600px]">
          {/* grid lines */}
          {[1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11].filter(v => v <= maxT * 2).map((v) => (
            <g key={v}>
              <line
                x1={padding.left} y1={y(v)} x2={chartW - padding.right} y2={y(v)}
                stroke="var(--color-border-subtle, #374151)" strokeWidth={0.5} strokeDasharray="3 3"
              />
              <text x={padding.left - 8} y={y(v) + 4} textAnchor="end" className="fill-text-secondary" fontSize={9}>
                {formatNum(v)}
              </text>
            </g>
          ))}

          {/* curve */}
          <motion.path
            d={path}
            fill="none"
            stroke="#667eea"
            strokeWidth={2.5}
            initial={{ pathLength: 0 }}
            animate={{ pathLength: 1 }}
            transition={{ duration: 1.5, ease: "easeOut" }}
          />

          {/* data points */}
          {processors.map((p, i) => (
            <g key={i} onMouseEnter={() => setHovered(i)} onMouseLeave={() => setHovered(null)}>
              <motion.circle
                cx={x(i)} cy={y(p.transistors)} r={hovered === i ? 6 : 4}
                fill={hovered === i ? "#ef4444" : "#667eea"}
                stroke="white"
                strokeWidth={1.5}
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ delay: 0.8 + i * 0.05 }}
                className="cursor-pointer"
              />
              {/* year label */}
              <text
                x={x(i)} y={chartH - 8}
                textAnchor="middle"
                className="fill-text-secondary"
                fontSize={8}
              >
                {p.year}
              </text>
            </g>
          ))}

          {/* tooltip */}
          {hovered !== null && (
            <g>
              <rect
                x={Math.min(x(hovered) + 10, chartW - 180)}
                y={Math.max(y(processors[hovered].transistors) - 50, 5)}
                width={170} height={45} rx={6}
                fill="var(--color-bg-elevated, #1e293b)"
                stroke="var(--color-border-subtle, #374151)"
                strokeWidth={1}
              />
              <text
                x={Math.min(x(hovered) + 18, chartW - 172)}
                y={Math.max(y(processors[hovered].transistors) - 32, 22)}
                className="fill-text-primary" fontSize={10} fontWeight="bold"
              >
                {processors[hovered].name} ({processors[hovered].year})
              </text>
              <text
                x={Math.min(x(hovered) + 18, chartW - 172)}
                y={Math.max(y(processors[hovered].transistors) - 17, 37)}
                className="fill-text-secondary" fontSize={9}
              >
                {formatNum(processors[hovered].transistors)} 晶体管 | {processors[hovered].process}
              </text>
            </g>
          )}
        </svg>
      </div>
    </div>
  );
}
