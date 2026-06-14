"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Gauge, Zap, Clock } from "lucide-react";

export function BusBandwidthCalc() {
  const [width, setWidth] = useState(64);
  const [freq, setFreq] = useState(800);
  const [transferPerCycle, setTransferPerCycle] = useState(1);
  const [overhead, setOverhead] = useState(10);

  const theoreticalBW = (width / 8) * (freq * 1e6) * transferPerCycle;
  const effectiveBW = theoreticalBW * (1 - overhead / 100);

  const formatBW = (bytes: number) => {
    if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(2)} GB/s`;
    if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(2)} MB/s`;
    return `${(bytes / 1e3).toFixed(2)} KB/s`;
  };

  const presets = [
    { name: "DDR4-3200", width: 64, freq: 1600, tpc: 2, ovh: 5 },
    { name: "PCIe 4.0 x16", width: 128, freq: 8000, tpc: 1, ovh: 15 },
    { name: "USB 3.0", width: 2, freq: 5000, tpc: 1, ovh: 20 },
    { name: "SATA III", width: 1, freq: 6000, tpc: 1, ovh: 10 },
  ];

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <Gauge className="w-5 h-5 text-orange-500" />
        总线带宽计算器
      </h3>
      <div className="text-sm text-text-secondary mb-4 font-mono">
        带宽 = 总线宽度 × 频率 × 每周期传输次数 × (1 - 开销%)
      </div>
      <div className="flex gap-2 mb-4 flex-wrap">
        {presets.map(p => (
          <button key={p.name} onClick={() => { setWidth(p.width); setFreq(p.freq); setTransferPerCycle(p.tpc); setOverhead(p.ovh); }}
            className="px-2 py-1 bg-bg-subtle rounded text-xs hover:bg-bg-elevated">
            {p.name}
          </button>
        ))}
      </div>
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div>
          <label className="text-sm text-text-secondary">总线宽度: {width} 位</label>
          <input type="range" min={1} max={512} step={8} value={width}
            onChange={e => setWidth(+e.target.value)} className="w-full" />
        </div>
        <div>
          <label className="text-sm text-text-secondary">频率: {freq} MHz</label>
          <input type="range" min={100} max={16000} step={100} value={freq}
            onChange={e => setFreq(+e.target.value)} className="w-full" />
        </div>
        <div>
          <label className="text-sm text-text-secondary">每周期传输: {transferPerCycle} 次</label>
          <input type="range" min={1} max={4} value={transferPerCycle}
            onChange={e => setTransferPerCycle(+e.target.value)} className="w-full" />
        </div>
        <div>
          <label className="text-sm text-text-secondary">协议开销: {overhead}%</label>
          <input type="range" min={0} max={50} value={overhead}
            onChange={e => setOverhead(+e.target.value)} className="w-full" />
        </div>
      </div>
      <div className="grid grid-cols-2 gap-4">
        <motion.div initial={{ scale: 0.95 }} animate={{ scale: 1 }}
          className="bg-orange-500/10 border border-orange-500 rounded-lg p-4 text-center">
          <div className="text-xs text-text-secondary">理论带宽</div>
          <div className="text-2xl font-bold text-orange-500 font-mono">{formatBW(theoreticalBW)}</div>
        </motion.div>
        <motion.div initial={{ scale: 0.95 }} animate={{ scale: 1 }}
          className="bg-green-500/10 border border-green-500 rounded-lg p-4 text-center">
          <div className="text-xs text-text-secondary">有效带宽</div>
          <div className="text-2xl font-bold text-green-500 font-mono">{formatBW(effectiveBW)}</div>
        </motion.div>
      </div>
      <div className="mt-4 text-xs text-text-secondary">
        开销损失: {formatBW(theoreticalBW - effectiveBW)} ({overhead}%)
      </div>
    </div>
  );
}
