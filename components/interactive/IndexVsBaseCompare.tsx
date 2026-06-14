"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { GitCompare } from "lucide-react";

export function IndexVsBaseCompare() {
  const [indexReg, setIndexReg] = useState(0x004);
  const [baseReg, setBaseReg] = useState(0x1000);
  const [offset, setOffset] = useState(0x020);

  const indexEA = offset + indexReg;
  const baseEA = baseReg + offset;

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <GitCompare className="w-5 h-5 text-amber-500" />
        变址 vs 基址对比
      </h3>

      <div className="flex flex-wrap gap-4 mb-6">
        <div>
          <label className="block text-xs text-text-muted mb-1">偏移量 (A)</label>
          <input type="text" value={`0x${offset.toString(16)}`}
            onChange={e => { const v = parseInt(e.target.value.replace("0x", ""), 16); if (!isNaN(v)) setOffset(v); }}
            className="w-24 px-2 py-1 rounded border border-border-subtle bg-bg-surface text-sm font-mono" />
        </div>
        <div>
          <label className="block text-xs text-text-muted mb-1">变址寄存器</label>
          <input type="text" value={`0x${indexReg.toString(16)}`}
            onChange={e => { const v = parseInt(e.target.value.replace("0x", ""), 16); if (!isNaN(v)) setIndexReg(v); }}
            className="w-24 px-2 py-1 rounded border border-border-subtle bg-bg-surface text-sm font-mono" />
        </div>
        <div>
          <label className="block text-xs text-text-muted mb-1">基址寄存器</label>
          <input type="text" value={`0x${baseReg.toString(16)}`}
            onChange={e => { const v = parseInt(e.target.value.replace("0x", ""), 16); if (!isNaN(v)) setBaseReg(v); }}
            className="w-24 px-2 py-1 rounded border border-border-subtle bg-bg-surface text-sm font-mono" />
        </div>
      </div>

      <div className="grid grid-cols-2 gap-6">
        <motion.div className="p-4 rounded border border-blue-500/30 bg-blue-500/5"
          initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }}>
          <h4 className="font-medium text-blue-400 mb-2">变址寻址 (Indexed)</h4>
          <div className="space-y-2 text-sm">
            <div className="font-mono text-xs bg-bg-surface p-2 rounded">EA = A + (变址寄存器)</div>
            <div className="font-mono text-xs bg-bg-surface p-2 rounded">EA = 0x{offset.toString(16)} + 0x{indexReg.toString(16)}</div>
            <div className="font-mono text-lg text-blue-400 font-bold">EA = 0x{indexEA.toString(16).toUpperCase()}</div>
          </div>
          <div className="mt-3 text-xs text-text-muted">
            <div className="font-medium text-text-secondary mb-1">特点：</div>
            <div>• 地址字段A为基地址，寄存器为变址量</div>
            <div>• 适合数组访问、循环迭代</div>
            <div>• 变址寄存器值动态变化</div>
          </div>
        </motion.div>

        <motion.div className="p-4 rounded border border-emerald-500/30 bg-emerald-500/5"
          initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }}>
          <h4 className="font-medium text-emerald-400 mb-2">基址寻址 (Based)</h4>
          <div className="space-y-2 text-sm">
            <div className="font-mono text-xs bg-bg-surface p-2 rounded">EA = (基址寄存器) + A</div>
            <div className="font-mono text-xs bg-bg-surface p-2 rounded">EA = 0x{baseReg.toString(16)} + 0x{offset.toString(16)}</div>
            <div className="font-mono text-lg text-emerald-400 font-bold">EA = 0x{baseEA.toString(16).toUpperCase()}</div>
          </div>
          <div className="mt-3 text-xs text-text-muted">
            <div className="font-medium text-text-secondary mb-1">特点：</div>
            <div>• 基址寄存器为基地址，A为偏移量</div>
            <div>• 适合程序重定位、多道程序</div>
            <div>• 基址寄存器值固定</div>
          </div>
        </motion.div>
      </div>

      <div className="mt-4 p-3 rounded bg-bg-surface border border-border-subtle text-xs text-text-muted">
        <span className="font-medium text-text-secondary">公式相同 EA = R + A，但语义不同：</span> 变址中A固定、R变化（遍历数组）；
        基址中R固定、A变化（程序内寻址）。
      </div>
    </div>
  );
}
