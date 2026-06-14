"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { ShieldAlert, ArrowRight, Zap, RotateCcw } from "lucide-react";

interface CacheAccess { id: number; addr: string; type: "hit" | "miss" }

export function NonBlockingCacheViz() {
  const [step, setStep] = useState(0);
  const accesses: CacheAccess[] = [
    { id: 1, addr: "A", type: "miss" },
    { id: 2, addr: "B", type: "hit" },
    { id: 3, addr: "C", type: "hit" },
    { id: 4, addr: "D", type: "miss" },
    { id: 5, addr: "E", type: "hit" },
  ];

  const blockingTimeline = [
    { label: "请求A", status: "miss", cycles: 4 },
    { label: "等待A", status: "wait", cycles: 3 },
    { label: "请求B", status: "hit", cycles: 1 },
    { label: "请求C", status: "hit", cycles: 1 },
    { label: "请求D", status: "miss", cycles: 4 },
    { label: "等待D", status: "wait", cycles: 3 },
    { label: "请求E", status: "hit", cycles: 1 },
  ];

  const nonBlockingTimeline = [
    { label: "请求A", status: "miss", cycles: 4 },
    { label: "请求B（命中Under缺失）", status: "hit", cycles: 1 },
    { label: "请求C（命中Under缺失）", status: "hit", cycles: 1 },
    { label: "等待A完成", status: "wait", cycles: 2 },
    { label: "请求D", status: "miss", cycles: 4 },
    { label: "请求E（命中Under缺失）", status: "hit", cycles: 1 },
    { label: "等待D完成", status: "wait", cycles: 3 },
  ];

  const [mode, setMode] = useState<"blocking" | "non-blocking">("blocking");
  const timeline = mode === "blocking" ? blockingTimeline : nonBlockingTimeline;
  const totalCycles = timeline.reduce((s, t) => s + t.cycles, 0);

  const statusColor: Record<string, string> = {
    miss: "bg-red-500",
    hit: "bg-green-500",
    wait: "bg-yellow-500",
  };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <ShieldAlert className="w-5 h-5 text-cyan-500" />
        非阻塞Cache（Hit Under Miss）
      </h3>
      <div className="flex gap-2 mb-4">
        {(["blocking", "non-blocking"] as const).map(m => (
          <button key={m} onClick={() => { setMode(m); setStep(0); }}
            className={`px-3 py-1.5 rounded text-sm ${mode === m ? "bg-cyan-500 text-white" : "bg-bg-subtle"}`}>
            {m === "blocking" ? "阻塞Cache" : "非阻塞Cache"}
          </button>
        ))}
      </div>
      <p className="text-sm text-text-secondary mb-4">
        {mode === "blocking"
          ? "阻塞Cache在缺失时暂停所有后续请求，直到数据返回"
          : "非阻塞Cache允许在缺失处理期间继续服务命中的请求"}
      </p>
      <div className="flex gap-2 mb-4">
        <button onClick={() => setStep(s => Math.min(s + 1, timeline.length - 1))}
          className="px-3 py-1 bg-cyan-500 text-white rounded text-sm flex items-center gap-1">
          <Zap className="w-3 h-3" /> 下一步
        </button>
        <button onClick={() => setStep(0)}
          className="px-3 py-1 bg-bg-subtle rounded text-sm flex items-center gap-1">
          <RotateCcw className="w-3 h-3" /> 重置
        </button>
      </div>
      <div className="space-y-1 mb-4">
        {timeline.map((t, i) => (
          <motion.div key={i} initial={{ opacity: 0, scaleX: 0 }}
            animate={{ opacity: i <= step ? 1 : 0.3, scaleX: i <= step ? 1 : 0 }}
            style={{ transformOrigin: "left" }}
            className="flex items-center gap-2">
            <div className="w-20 text-xs text-right text-text-secondary truncate">{t.label}</div>
            <motion.div animate={{ width: i <= step ? t.cycles * 40 : 0 }}
              className={`h-6 rounded ${statusColor[t.status]} flex items-center justify-center`}>
              {i <= step && <span className="text-white text-xs">{t.cycles}c</span>}
            </motion.div>
            {i === step && <ArrowRight className="w-4 h-4 text-cyan-500" />}
          </motion.div>
        ))}
      </div>
      <div className="grid grid-cols-3 gap-2 text-center text-xs">
        <div className="bg-green-500/10 rounded p-2">
          <div className="font-bold text-green-500">命中</div>
          <div className="w-4 h-4 bg-green-500 rounded mx-auto mt-1" />
        </div>
        <div className="bg-red-500/10 rounded p-2">
          <div className="font-bold text-red-500">缺失</div>
          <div className="w-4 h-4 bg-red-500 rounded mx-auto mt-1" />
        </div>
        <div className="bg-yellow-500/10 rounded p-2">
          <div className="font-bold text-yellow-500">等待</div>
          <div className="w-4 h-4 bg-yellow-500 rounded mx-auto mt-1" />
        </div>
      </div>
      <div className="mt-3 text-sm">
        总周期: <span className="font-bold text-cyan-500">{timeline.slice(0, step + 1).reduce((s, t) => s + t.cycles, 0)}</span>
        {step === timeline.length - 1 && (
          <span className="text-text-secondary"> / {totalCycles} 周期</span>
        )}
      </div>
    </div>
  );
}
