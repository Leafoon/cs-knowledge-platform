"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Calculator, Clock, Zap } from "lucide-react";

export function CachePerformanceCalc() {
  const [hitRate, setHitRate] = useState(95);
  const [hitTime, setHitTime] = useState(1);
  const [missPenalty, setMissPenalty] = useState(100);

  const missRate = 100 - hitRate;
  const amat = hitTime + (missRate / 100) * missPenalty;
  const speedup = missPenalty / amat;

  const scenarios = [
    { label: "当前配置", hitRate, hitTime, missPenalty },
    { label: "提升命中率到99%", hitRate: 99, hitTime, missPenalty },
    { label: "降低缺失惩罚到50", hitRate, hitTime, missPenalty: 50 },
    { label: "降低命中时间到0.5", hitRate, hitTime: 0.5, missPenalty },
  ];

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <Calculator className="w-5 h-5 text-indigo-500" />
        Cache性能计算器
      </h3>
      <div className="text-sm text-text-secondary mb-4 font-mono">
        AMAT = 命中时间 + 缺失率 × 缺失惩罚
      </div>
      <div className="grid grid-cols-3 gap-4 mb-6">
        <div>
          <label className="text-sm text-text-secondary">命中率: {hitRate}%</label>
          <input type="range" min={50} max={99} value={hitRate}
            onChange={e => setHitRate(+e.target.value)} className="w-full" />
        </div>
        <div>
          <label className="text-sm text-text-secondary">命中时间: {hitTime} 周期</label>
          <input type="range" min={0.5} max={5} step={0.5} value={hitTime}
            onChange={e => setHitTime(+e.target.value)} className="w-full" />
        </div>
        <div>
          <label className="text-sm text-text-secondary">缺失惩罚: {missPenalty} 周期</label>
          <input type="range" min={10} max={300} step={10} value={missPenalty}
            onChange={e => setMissPenalty(+e.target.value)} className="w-full" />
        </div>
      </div>
      <motion.div key={`${hitRate}-${hitTime}-${missPenalty}`}
        initial={{ scale: 0.95 }} animate={{ scale: 1 }}
        className="bg-indigo-500/10 border border-indigo-500 rounded-lg p-6 text-center mb-4">
        <div className="text-sm text-text-secondary mb-1">平均访存时间 (AMAT)</div>
        <div className="text-4xl font-bold text-indigo-500 font-mono">{amat.toFixed(2)}</div>
        <div className="text-sm text-text-secondary mt-1">周期</div>
      </motion.div>
      <div className="grid grid-cols-2 gap-3 mb-4">
        <div className="bg-bg-subtle rounded p-3 text-center">
          <div className="text-xs text-text-secondary">缺失率</div>
          <div className="text-lg font-bold text-red-500">{missRate}%</div>
        </div>
        <div className="bg-bg-subtle rounded p-3 text-center">
          <div className="text-xs text-text-secondary">相对加速比</div>
          <div className="text-lg font-bold text-green-500">{speedup.toFixed(2)}×</div>
        </div>
      </div>
      <div className="text-sm font-medium mb-2">场景对比</div>
      <div className="space-y-2">
        {scenarios.map((s, i) => {
          const a = s.hitTime + ((100 - s.hitRate) / 100) * s.missPenalty;
          return (
            <div key={i} className="flex items-center gap-3 text-sm">
              <span className="w-40 text-text-secondary">{s.label}</span>
              <div className="flex-1 h-4 bg-bg-subtle rounded overflow-hidden">
                <motion.div animate={{ width: `${Math.min((a / 30) * 100, 100)}%` }}
                  className={`h-full rounded ${i === 0 ? "bg-indigo-500" : "bg-indigo-500/40"}`} />
              </div>
              <span className="font-mono w-16 text-right">{a.toFixed(2)}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
