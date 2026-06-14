"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { BarChart3 } from "lucide-react";

interface EncodingMethod {
  name: string;
  avgBits: number;
  decodeSpeed: string;
  description: string;
  color: string;
}

const methods: EncodingMethod[] = [
  { name: "定长编码 (4位)", avgBits: 4.0, decodeSpeed: "最快", description: "所有指令固定4位操作码，最多16条指令", color: "#3b82f6" },
  { name: "哈夫曼编码", avgBits: 2.8, decodeSpeed: "较慢", description: "按频率分配变长编码，平均码长最短", color: "#10b981" },
  { name: "扩展操作码", avgBits: 3.2, decodeSpeed: "快", description: "高频用短码，低频用长码，折中方案", color: "#f59e0b" },
];

const barMax = 5;

export function EncodingEfficiencyChart() {
  const [hovered, setHovered] = useState<number | null>(null);

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <BarChart3 className="w-5 h-5 text-orange-500" />
        编码效率对比图
      </h3>
      <p className="text-xs text-text-muted mb-6">对比三种编码方案的平均码长与译码特性</p>

      <div className="space-y-4 mb-6">
        {methods.map((m, i) => (
          <div key={m.name} className="cursor-pointer" onMouseEnter={() => setHovered(i)} onMouseLeave={() => setHovered(null)}>
            <div className="flex items-center justify-between mb-1">
              <span className="text-sm font-medium">{m.name}</span>
              <span className="text-sm font-mono text-text-muted">{m.avgBits.toFixed(1)} bits</span>
            </div>
            <div className="h-6 bg-bg-surface rounded overflow-hidden">
              <motion.div
                className="h-full rounded flex items-center px-2"
                style={{ backgroundColor: m.color }}
                initial={{ width: 0 }}
                animate={{ width: `${(m.avgBits / barMax) * 100}%` }}
                transition={{ duration: 0.8, delay: i * 0.15 }}
              >
                <span className="text-xs text-white font-mono">{m.avgBits.toFixed(1)}</span>
              </motion.div>
            </div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-3 gap-3">
        {methods.map((m, i) => (
          <motion.div
            key={m.name}
            className={`p-3 rounded border text-center transition-colors ${
              hovered === i ? "border-blue-500 bg-blue-500/5" : "border-border-subtle"
            }`}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 + i * 0.1 }}
          >
            <div className="text-xs text-text-muted mb-1">译码速度</div>
            <div className="text-sm font-semibold">{m.decodeSpeed}</div>
          </motion.div>
        ))}
      </div>

      {hovered !== null && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}
          className="mt-4 p-3 rounded bg-bg-surface border border-border-subtle text-sm text-text-secondary">
          {methods[hovered].description}
        </motion.div>
      )}
    </div>
  );
}
