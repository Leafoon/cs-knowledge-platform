"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Activity, Clock, Zap, RotateCcw } from "lucide-react";

type CommType = "sync" | "async";

export function BusTimingDiagram() {
  const [type, setType] = useState<CommType>("sync");
  const [step, setStep] = useState(0);

  const syncSignals = [
    { name: "CLK", vals: [0, 1, 0, 1, 0, 1, 0, 1] },
    { name: "地址/数据", vals: ["A0", "A0", "A1", "A1", "D0", "D0", "D1", "D1"] },
    { name: "RD/WR", vals: [1, 1, 1, 1, 0, 0, 0, 0] },
    { name: "MEM/IO", vals: [0, 0, 0, 0, 0, 0, 0, 0] },
  ];

  const asyncSignals = [
    { name: "REQ", vals: [0, 1, 1, 1, 0, 0, 0, 0] },
    { name: "ACK", vals: [0, 0, 0, 1, 1, 1, 0, 0] },
    { name: "数据", vals: ["--", "--", "D0", "D0", "D0", "--", "--", "--"] },
    { name: "等待", vals: [0, 0, 0, 0, 0, 0, 0, 0] },
  ];

  const signals = type === "sync" ? syncSignals : asyncSignals;
  const phases = type === "sync"
    ? ["T1:发地址", "T1:稳定", "T2:发数据", "T2:稳定", "T3:读数据", "T3:采样", "T4:结束", "T4:空闲"]
    : ["请求建立", "请求稳定", "数据准备", "应答建立", "应答稳定", "数据撤销", "应答撤销", "空闲"];

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <Activity className="w-5 h-5 text-rose-500" />
        总线时序图
      </h3>
      <div className="flex gap-2 mb-4">
        {(["sync", "async"] as CommType[]).map(t => (
          <button key={t} onClick={() => { setType(t); setStep(0); }}
            className={`px-3 py-1.5 rounded text-sm ${type === t ? "bg-rose-500 text-white" : "bg-bg-subtle"}`}>
            {t === "sync" ? "同步通信" : "异步通信"}
          </button>
        ))}
      </div>
      <p className="text-sm text-text-secondary mb-4">
        {type === "sync" ? "同步通信：所有操作在统一时钟信号控制下进行，按固定时序传输" : "异步通信：使用应答信号协调，无需统一时钟，适应不同速度设备"}
      </p>
      <div className="flex gap-2 mb-4">
        <button onClick={() => setStep(s => Math.min(s + 1, 7))}
          className="px-3 py-1 bg-rose-500 text-white rounded text-sm flex items-center gap-1">
          <Zap className="w-3 h-3" /> 下一步
        </button>
        <button onClick={() => setStep(0)}
          className="px-3 py-1 bg-bg-subtle rounded text-sm flex items-center gap-1">
          <RotateCcw className="w-3 h-3" /> 重置
        </button>
      </div>
      <div className="space-y-3">
        {signals.map((sig, si) => (
          <div key={si} className="flex items-center gap-2">
            <span className="w-16 text-xs text-text-secondary text-right">{sig.name}</span>
            <div className="flex-1 flex gap-0.5">
              {sig.vals.map((v, vi) => {
                const isHigh = v === 1 || v === 0 && si === 0;
                const isCurrent = vi === step;
                return (
                  <motion.div key={vi}
                    animate={{ opacity: vi <= step ? 1 : 0.3, scaleY: vi <= step ? 1 : 0.5 }}
                    className={`flex-1 h-8 flex items-center justify-center text-xs font-mono border ${isCurrent ? "border-rose-500 bg-rose-500/20" : "border-border-subtle"} ${typeof v === "number" ? (v ? "bg-rose-500/30" : "bg-bg-subtle") : "bg-blue-500/20"}`}>
                    {vi <= step ? v : ""}
                  </motion.div>
                );
              })}
            </div>
          </div>
        ))}
      </div>
      <div className="mt-3 flex justify-between text-xs text-text-secondary">
        {phases.map((p, i) => (
          <span key={i} className={i === step ? "text-rose-500 font-bold" : ""}>{p}</span>
        ))}
      </div>
    </div>
  );
}
