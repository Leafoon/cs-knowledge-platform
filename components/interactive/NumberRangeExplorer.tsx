"use client";

import { useState } from "react";
import { motion } from "framer-motion";

type Encoding = "unsigned" | "signMagnitude" | "onesComplement" | "twosComplement";

interface RangeInfo {
  min: number;
  max: number;
  total: number;
  hasNegZero: boolean;
  label: string;
  color: string;
}

function getRange(n: number, enc: Encoding): RangeInfo {
  switch (enc) {
    case "unsigned":
      return { min: 0, max: (1 << n) - 1, total: 1 << n, hasNegZero: false, label: "无符号", color: "#667eea" };
    case "signMagnitude":
      return { min: -(2 ** (n - 1) - 1), max: 2 ** (n - 1) - 1, total: (1 << n) - 1, hasNegZero: true, label: "原码", color: "#f59e0b" };
    case "onesComplement":
      return { min: -(2 ** (n - 1) - 1), max: 2 ** (n - 1) - 1, total: (1 << n) - 1, hasNegZero: true, label: "反码", color: "#ef4444" };
    case "twosComplement":
      return { min: -(1 << (n - 1)), max: (1 << (n - 1)) - 1, total: 1 << n, hasNegZero: false, label: "补码", color: "#10b981" };
  }
}

export function NumberRangeExplorer() {
  const [bits, setBits] = useState(8);
  const [encoding, setEncoding] = useState<Encoding>("twosComplement");

  const range = getRange(bits, encoding);
  const span = range.max - range.min;
  const zeroPos = span > 0 ? ((0 - range.min) / span) * 100 : 50;

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">
        数值范围探索器
      </h3>
      <p className="text-sm text-text-secondary mb-4">
        选择字长和编码方式，查看可表示的数值范围
      </p>

      <div className="flex flex-wrap gap-4 mb-6">
        <div>
          <label className="block text-xs font-medium text-text-secondary mb-1">字长 (位)</label>
          <div className="flex rounded-lg overflow-hidden border border-border-subtle">
            {([4, 8, 16, 32] as const).map((b) => (
              <button key={b} onClick={() => setBits(b)}
                className={`px-3 py-1.5 text-sm font-medium transition-colors ${bits === b ? "bg-accent-primary text-white" : "bg-bg-secondary text-text-secondary"}`}>
                {b}
              </button>
            ))}
          </div>
        </div>
        <div>
          <label className="block text-xs font-medium text-text-secondary mb-1">编码方式</label>
          <div className="flex flex-wrap rounded-lg overflow-hidden border border-border-subtle">
            {(["unsigned", "signMagnitude", "onesComplement", "twosComplement"] as Encoding[]).map((e) => (
              <button key={e} onClick={() => setEncoding(e)}
                className={`px-3 py-1.5 text-xs font-medium transition-colors ${encoding === e ? "text-white" : "bg-bg-secondary text-text-secondary"}`}
                style={encoding === e ? { backgroundColor: getRange(bits, e).color } : {}}>
                {getRange(bits, e).label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* number line visualization */}
      <div className="mb-6">
        <div className="relative h-12 mx-4">
          {/* line */}
          <div className="absolute top-5 left-0 right-0 h-0.5 bg-border-subtle" />
          {/* zero marker */}
          <div className="absolute top-5 w-0.5 h-3 bg-text-secondary" style={{ left: `${zeroPos}%` }} />
          <span className="absolute top-9 text-[10px] text-text-secondary -translate-x-1/2 font-mono" style={{ left: `${zeroPos}%` }}>0</span>
          {/* min marker */}
          <motion.div initial={{ scale: 0 }} animate={{ scale: 1 }}
            className="absolute top-3 w-3 h-3 rounded-full -translate-x-1/2" style={{ left: "0%", backgroundColor: range.color }} />
          <span className="absolute -top-1 text-xs font-mono font-bold -translate-x-1/2" style={{ left: "0%", color: range.color }}>{range.min}</span>
          {/* max marker */}
          <motion.div initial={{ scale: 0 }} animate={{ scale: 1 }}
            className="absolute top-3 w-3 h-3 rounded-full -translate-x-1/2" style={{ left: "100%", backgroundColor: range.color }} />
          <span className="absolute -top-1 text-xs font-mono font-bold -translate-x-1/2" style={{ left: "100%", color: range.color }}>{range.max}</span>
        </div>
      </div>

      {/* stats cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
        {[
          { label: "最小值", value: String(range.min) },
          { label: "最大值", value: String(range.max) },
          { label: "可表示个数", value: String(range.total) },
          { label: "是否有 -0", value: range.hasNegZero ? "是" : "否" },
        ].map((s) => (
          <div key={s.label} className="rounded-lg border border-border-subtle bg-bg-secondary p-3 text-center">
            <p className="text-xs text-text-secondary mb-1">{s.label}</p>
            <p className="font-mono font-bold text-sm text-text-primary">{s.value}</p>
          </div>
        ))}
      </div>

      {/* formula */}
      <div className="rounded-lg bg-bg-secondary border border-border-subtle p-3">
        <p className="text-xs text-text-secondary">
          {encoding === "unsigned" && <span><span className="font-semibold text-text-primary">无符号：</span>范围 [0, 2<sup>{bits}</sup> - 1] = [0, {(1 << bits) - 1}]</span>}
          {encoding === "signMagnitude" && <span><span className="font-semibold text-text-primary">原码：</span>范围 [-(2<sup>{bits - 1}</sup> - 1), 2<sup>{bits - 1}</sup> - 1]，有 ±0</span>}
          {encoding === "onesComplement" && <span><span className="font-semibold text-text-primary">反码：</span>范围 [-(2<sup>{bits - 1}</sup> - 1), 2<sup>{bits - 1}</sup> - 1]，有 ±0</span>}
          {encoding === "twosComplement" && <span><span className="font-semibold text-text-primary">补码：</span>范围 [-2<sup>{bits - 1}</sup>, 2<sup>{bits - 1}</sup> - 1] = [{-(1 << (bits - 1))}, {(1 << (bits - 1)) - 1}]，无 -0</span>}
        </p>
      </div>
    </div>
  );
}
