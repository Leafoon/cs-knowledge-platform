"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Download, Zap, ArrowRight, RotateCcw } from "lucide-react";

type PrefetchType = "stride" | "stream";

const PREFETCH_INFO: Record<PrefetchType, { name: string; desc: string }> = {
  stride: { name: "步长预取", desc: "检测固定步长的访问模式，提前预取后续块" },
  stream: { name: "流预取", desc: "检测连续访问流，顺序预取后续多个块" },
};

export function CachePrefetchDemo() {
  const [type, setType] = useState<PrefetchType>("stride");
  const [step, setStep] = useState(0);

  const strideAccesses = [0, 4, 8, 12, 16, 20, 24, 28];
  const streamAccesses = [0, 1, 2, 3, 4, 5, 6, 7];
  const accesses = type === "stride" ? strideAccesses : streamAccesses;
  const prefetchDistance = 2;

  const getEvents = () => {
    const events: { access: number; prefetch: number[]; hit: boolean }[] = [];
    for (let i = 0; i < accesses.length; i++) {
      const prefetch: number[] = [];
      const hit = i >= prefetchDistance;
      if (type === "stride") {
        for (let d = 1; d <= prefetchDistance; d++) {
          prefetch.push(accesses[i] + d * 4);
        }
      } else {
        for (let d = 1; d <= prefetchDistance; d++) {
          prefetch.push(accesses[i] + d);
        }
      }
      events.push({ access: accesses[i], prefetch, hit });
    }
    return events;
  };

  const events = getEvents();

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <Download className="w-5 h-5 text-teal-500" />
        Cache预取演示
      </h3>
      <div className="flex gap-2 mb-4">
        {(Object.keys(PREFETCH_INFO) as PrefetchType[]).map(t => (
          <button key={t} onClick={() => { setType(t); setStep(0); }}
            className={`px-3 py-1.5 rounded text-sm ${type === t ? "bg-teal-500 text-white" : "bg-bg-subtle"}`}>
            {PREFETCH_INFO[t].name}
          </button>
        ))}
      </div>
      <p className="text-sm text-text-secondary mb-4">{PREFETCH_INFO[type].desc}</p>
      <div className="flex gap-2 mb-4">
        <button onClick={() => setStep(s => Math.min(s + 1, events.length - 1))}
          className="px-3 py-1 bg-teal-500 text-white rounded text-sm flex items-center gap-1">
          <Zap className="w-3 h-3" /> 下一步
        </button>
        <button onClick={() => setStep(0)}
          className="px-3 py-1 bg-bg-subtle rounded text-sm flex items-center gap-1">
          <RotateCcw className="w-3 h-3" /> 重置
        </button>
      </div>
      <div className="space-y-2">
        {events.map((e, i) => (
          <motion.div key={i} initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: i <= step ? 1 : 0.3, x: i <= step ? 0 : -10 }}
            className={`flex items-center gap-3 p-3 rounded border ${i === step ? "border-teal-500 bg-teal-500/10" : "border-border-subtle"}`}>
            <div className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold ${e.hit ? "bg-green-500 text-white" : "bg-red-500 text-white"}`}>
              {i + 1}
            </div>
            <div className="flex-1">
              <div className="text-sm">
                访问块 <span className="font-mono font-bold">{e.access}</span>
                <span className={`ml-2 text-xs ${e.hit ? "text-green-500" : "text-red-500"}`}>
                  {e.hit ? "命中" : "缺失"}
                </span>
              </div>
              {i <= step && (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                  className="flex items-center gap-1 mt-1">
                  <ArrowRight className="w-3 h-3 text-teal-500" />
                  <span className="text-xs text-teal-500">
                    预取: {e.prefetch.join(", ")}
                  </span>
                </motion.div>
              )}
            </div>
          </motion.div>
        ))}
      </div>
      {step > 0 && (
        <div className="mt-4 text-xs text-text-secondary">
          缺失率: {((events.slice(0, step + 1).filter(e => !e.hit).length / (step + 1)) * 100).toFixed(0)}%
          {step >= prefetchDistance && " (预取生效后显著降低)"}
        </div>
      )}
    </div>
  );
}
