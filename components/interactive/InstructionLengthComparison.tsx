"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Ruler, ArrowRight } from "lucide-react";

export function InstructionLengthComparison() {
  const [instrCount, setInstrCount] = useState(64);
  const [avgAddr, setAvgAddr] = useState(2);

  const fixedOpcodeBits = Math.ceil(Math.log2(instrCount));
  const fixedTotalBits = fixedOpcodeBits + avgAddr * 12;

  const variableItems = [
    { name: "高频指令 (60%)", opcodeBits: Math.ceil(Math.log2(instrCount * 0.3)), pct: 60 },
    { name: "中频指令 (30%)", opcodeBits: Math.ceil(Math.log2(instrCount * 0.7)), pct: 30 },
    { name: "低频指令 (10%)", opcodeBits: fixedOpcodeBits + 2, pct: 10 },
  ];
  const variableAvgBits = variableItems.reduce((s, v) => s + (v.opcodeBits + avgAddr * 12) * v.pct, 0) / 100;

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <Ruler className="w-5 h-5 text-rose-500" />
        指令长度对比
      </h3>

      <div className="flex flex-wrap gap-4 mb-6">
        <div>
          <label className="block text-xs text-text-muted mb-1">指令条数</label>
          <input type="number" value={instrCount} onChange={e => setInstrCount(Math.max(2, Number(e.target.value)))}
            className="w-24 px-2 py-1 rounded border border-border-subtle bg-bg-surface text-sm font-mono" />
        </div>
        <div>
          <label className="block text-xs text-text-muted mb-1">平均地址数</label>
          <input type="number" value={avgAddr} onChange={e => setAvgAddr(Math.max(0, Math.min(3, Number(e.target.value))))}
            className="w-20 px-2 py-1 rounded border border-border-subtle bg-bg-surface text-sm font-mono" />
        </div>
      </div>

      <div className="grid grid-cols-2 gap-6">
        <motion.div className="p-4 rounded border border-blue-500/30 bg-blue-500/5"
          initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }}>
          <h4 className="font-medium text-blue-400 mb-3">定长编码</h4>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between"><span>操作码位数</span><span className="font-mono">{fixedOpcodeBits}b</span></div>
            <div className="flex justify-between"><span>每条指令长度</span><span className="font-mono">{fixedTotalBits}b</span></div>
            <div className="flex justify-between"><span>译码速度</span><span className="text-green-400">快（固定切分）</span></div>
            <div className="flex justify-between"><span>空间效率</span><span className="text-yellow-400">一般</span></div>
          </div>
        </motion.div>

        <motion.div className="p-4 rounded border border-emerald-500/30 bg-emerald-500/5"
          initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }}>
          <h4 className="font-medium text-emerald-400 mb-3">变长编码</h4>
          <div className="space-y-2 text-sm">
            {variableItems.map(v => (
              <div key={v.name} className="flex justify-between">
                <span>{v.name}</span>
                <span className="font-mono">{v.opcodeBits + avgAddr * 12}b</span>
              </div>
            ))}
            <div className="flex justify-between border-t border-border-subtle pt-2">
              <span>平均长度</span>
              <span className="font-mono text-emerald-400">{variableAvgBits.toFixed(1)}b</span>
            </div>
            <div className="flex justify-between"><span>译码速度</span><span className="text-yellow-400">较慢（需解析）</span></div>
          </div>
        </motion.div>
      </div>

      <div className="mt-4 flex items-center justify-center gap-4 p-3 rounded bg-bg-surface border border-border-subtle">
        <span className="text-sm">定长 <b>{fixedTotalBits}b</b></span>
        <ArrowRight className="w-4 h-4 text-text-muted" />
        <span className="text-sm">变长平均 <b>{variableAvgBits.toFixed(1)}b</b></span>
        <span className="text-xs text-text-muted ml-2">
          节省 {((1 - variableAvgBits / fixedTotalBits) * 100).toFixed(1)}%
        </span>
      </div>
    </div>
  );
}
