"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { TrendingUp } from "lucide-react";

interface Metric {
  name: string;
  risc: number;
  cisc: number;
  unit: string;
  lowerBetter: boolean;
}

const metrics: Metric[] = [
  { name: "CPI", risc: 1.0, cisc: 4.5, unit: "周期/指令", lowerBetter: true },
  { name: "代码大小", risc: 1.4, cisc: 1.0, unit: "相对值", lowerBetter: true },
  { name: "硬件复杂度", risc: 1.0, cisc: 3.5, unit: "相对值", lowerBetter: true },
  { name: "时钟频率", risc: 3.0, cisc: 1.5, unit: "GHz", lowerBetter: false },
  { name: "功耗", risc: 1.0, cisc: 2.5, unit: "W/MIPS", lowerBetter: true },
];

export function InstructionComplexityChart() {
  const [hovered, setHovered] = useState<number | null>(null);

  const maxVal = Math.max(...metrics.flatMap(m => [m.risc, m.cisc]));

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <TrendingUp className="w-5 h-5 text-blue-500" />
        指令复杂度对比图
      </h3>

      <div className="flex gap-4 mb-6">
        <div className="flex items-center gap-2 text-sm">
          <div className="w-3 h-3 rounded bg-green-500" />
          <span>RISC</span>
        </div>
        <div className="flex items-center gap-2 text-sm">
          <div className="w-3 h-3 rounded bg-orange-500" />
          <span>CISC</span>
        </div>
      </div>

      <div className="space-y-3">
        {metrics.map((m, i) => {
          const riscBar = (m.risc / maxVal) * 100;
          const ciscBar = (m.cisc / maxVal) * 100;
          const riscBetter = m.lowerBetter ? m.risc < m.cisc : m.risc > m.cisc;

          return (
            <div key={m.name} className="cursor-pointer" onMouseEnter={() => setHovered(i)} onMouseLeave={() => setHovered(null)}>
              <div className="flex items-center justify-between mb-1">
                <span className="text-sm font-medium">{m.name}</span>
                <span className="text-xs text-text-muted">{m.unit} {m.lowerBetter ? "(↓越小越好)" : "(↑越大越好)"}</span>
              </div>
              <div className="flex gap-1 items-center">
                <motion.div className="h-6 rounded flex items-center px-2"
                  style={{ backgroundColor: riscBetter ? "#10b981" : "#10b981aa" }}
                  initial={{ width: 0 }} animate={{ width: `${riscBar}%` }} transition={{ duration: 0.6, delay: i * 0.1 }}>
                  <span className="text-xs text-white font-mono whitespace-nowrap">{m.risc}</span>
                </motion.div>
              </div>
              <div className="flex gap-1 items-center mt-0.5">
                <motion.div className="h-6 rounded flex items-center px-2"
                  style={{ backgroundColor: !riscBetter ? "#f59e0b" : "#f59e0baa" }}
                  initial={{ width: 0 }} animate={{ width: `${ciscBar}%` }} transition={{ duration: 0.6, delay: i * 0.1 + 0.05 }}>
                  <span className="text-xs text-white font-mono whitespace-nowrap">{m.cisc}</span>
                </motion.div>
              </div>
            </div>
          );
        })}
      </div>

      {hovered !== null && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}
          className="mt-4 p-3 rounded bg-bg-surface border border-border-subtle text-sm text-text-secondary">
          {metrics[hovered].name}: RISC = {metrics[hovered].risc}{metrics[hovered].unit}，
          CISC = {metrics[hovered].cisc}{metrics[hovered].unit}。
          {metrics[hovered].lowerBetter ? "值越小越优。" : "值越大越优。"}
        </motion.div>
      )}
    </div>
  );
}
