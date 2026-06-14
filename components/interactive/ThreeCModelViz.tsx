"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { AlertTriangle, Layers, Grid3X3, Zap } from "lucide-react";

type MissType = "compulsory" | "capacity" | "conflict";

const MISS_INFO: Record<MissType, { name: string; desc: string; color: string; icon: typeof Zap }> = {
  compulsory: {
    name: "强制缺失 (Compulsory)",
    desc: "首次访问某数据块时必然发生的缺失，也称冷启动缺失",
    color: "blue",
    icon: Zap,
  },
  capacity: {
    name: "容量缺失 (Capacity)",
    desc: "Cache容量不足，无法容纳所有需要的数据块",
    color: "orange",
    icon: Layers,
  },
  conflict: {
    name: "冲突缺失 (Conflict)",
    desc: "组相联或直接映射Cache中，多个块竞争同一组/行",
    color: "red",
    icon: Grid3X3,
  },
};

const COLORS: Record<MissType, string> = {
  compulsory: "bg-blue-500",
  capacity: "bg-orange-500",
  conflict: "bg-red-500",
};

interface Access { addr: number; hit: boolean; type?: MissType }

export function ThreeCModelViz() {
  const [cacheSize, setCacheSize] = useState(4);
  const [associativity, setAssociativity] = useState(1);
  const [accesses] = useState<Access[]>([
    { addr: 0, hit: false }, { addr: 1, hit: false }, { addr: 2, hit: false }, { addr: 3, hit: false },
    { addr: 4, hit: false }, { addr: 0, hit: false }, { addr: 1, hit: false }, { addr: 5, hit: false },
    { addr: 0, hit: false }, { addr: 6, hit: false },
  ]);
  const [step, setStep] = useState(0);

  const simulate = (): Access[] => {
    const cache: Map<number, number[]> = new Map();
    const seen = new Set<number>();
    return accesses.map(a => {
      const setIdx = associativity === 1 ? a.addr % cacheSize : 0;
      const set = cache.get(setIdx) || [];
      if (set.includes(a.addr)) return { ...a, hit: true };
      if (!seen.has(a.addr)) {
        seen.add(a.addr);
        set.push(a.addr);
        cache.set(setIdx, set);
        return { ...a, hit: false, type: "compulsory" as MissType };
      }
      if (set.length < associativity || (associativity === cacheSize && set.length < cacheSize)) {
        set.push(a.addr);
        cache.set(setIdx, set);
        return { ...a, hit: false, type: associativity === 1 ? "conflict" as MissType : "capacity" as MissType };
      }
      set.shift();
      set.push(a.addr);
      cache.set(setIdx, set);
      return { ...a, hit: false, type: associativity === 1 ? "conflict" as MissType : "capacity" as MissType };
    });
  };

  const result = simulate();
  const visible = result.slice(0, step + 1);
  const counts = { compulsory: 0, capacity: 0, conflict: 0 };
  visible.forEach(a => { if (a.type) counts[a.type]++; });
  const total = visible.length;
  const hits = visible.filter(a => a.hit).length;

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <AlertTriangle className="w-5 h-5 text-orange-500" />
        3C模型可视化
      </h3>
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <label className="text-sm text-text-secondary">Cache大小: {cacheSize}行</label>
          <input type="range" min={2} max={8} value={cacheSize} onChange={e => { setCacheSize(+e.target.value); setStep(0); }}
            className="w-full" />
        </div>
        <div>
          <label className="text-sm text-text-secondary">相联度: {associativity === cacheSize ? "全相联" : `${associativity}-way`}</label>
          <input type="range" min={1} max={cacheSize} value={associativity}
            onChange={e => { setAssociativity(+e.target.value); setStep(0); }} className="w-full" />
        </div>
      </div>
      <div className="flex gap-2 mb-4">
        <button onClick={() => setStep(s => Math.min(s + 1, result.length - 1))}
          className="px-3 py-1 bg-orange-500 text-white rounded text-sm">下一步</button>
        <button onClick={() => setStep(0)} className="px-3 py-1 bg-bg-subtle rounded text-sm">重置</button>
      </div>
      <div className="flex gap-1 mb-4 flex-wrap">
        {result.map((a, i) => (
          <motion.div key={i} initial={{ scale: 0 }} animate={{ scale: i <= step ? 1 : 0.5, opacity: i <= step ? 1 : 0.3 }}
            className={`w-10 h-10 rounded flex items-center justify-center text-xs font-mono ${i > step ? "bg-bg-subtle" : a.hit ? "bg-green-500/30 border border-green-500" : COLORS[a.type!] + "/30 border border-" + a.type}`}>
            {a.addr}
          </motion.div>
        ))}
      </div>
      <div className="grid grid-cols-4 gap-2 text-center">
        {(["compulsory", "capacity", "conflict"] as MissType[]).map(t => {
          const m = MISS_INFO[t];
          return (
            <div key={t} className={`rounded p-2 ${COLORS[t]}/10`}>
              <div className={`text-lg font-bold text-${m.color}-500`}>{counts[t]}</div>
              <div className="text-xs">{m.name.split("(")[0].trim()}</div>
            </div>
          );
        })}
        <div className="rounded p-2 bg-green-500/10">
          <div className="text-lg font-bold text-green-500">{hits}</div>
          <div className="text-xs">命中</div>
        </div>
      </div>
      <div className="mt-3 text-xs text-text-secondary">
        命中率: {total > 0 ? ((hits / total) * 100).toFixed(1) : 0}% | 缺失率: {total > 0 ? (((total - hits) / total) * 100).toFixed(1) : 0}%
      </div>
    </div>
  );
}
