"use client";

import React, { useState, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { ArrowLeftRight, Play, RotateCcw } from "lucide-react";

const models = [
  { name: "进程", color: "#ef4444", switchCost: 100, description: "切换内存空间、系统资源、保存/恢复大量状态", overhead: "高" },
  { name: "线程", color: "#3b82f6", switchCost: 10, description: "保存寄存器、切换线程栈、OS 调度", overhead: "中" },
  { name: "协程", color: "#10b981", switchCost: 1, description: "用户态完成，保存执行位置和局部变量", overhead: "低" },
];

export function ContextSwitchCost() {
  const [selected, setSelected] = useState(0);
  const [switches, setSwitches] = useState(0);
  const [running, setRunning] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const start = () => {
    setRunning(true);
    setSwitches(0);
    setElapsed(0);
    const startTime = Date.now();
    const totalSwitches = 100;
    const model = models[selected];
    const interval = model.switchCost;

    timerRef.current = setInterval(() => {
      const e = (Date.now() - startTime) / 1000;
      setElapsed(e);
      setSwitches((s) => {
        const newS = s + 1;
        if (newS >= totalSwitches) {
          setRunning(false);
          if (timerRef.current) clearInterval(timerRef.current);
        }
        return Math.min(newS, totalSwitches);
      });
    }, interval);
  };

  const reset = () => {
    setRunning(false);
    setSwitches(0);
    setElapsed(0);
    if (timerRef.current) clearInterval(timerRef.current);
  };

  return (
    <div className="max-w-6xl mx-auto p-6">
      <h3 className="text-xl font-bold mb-4 text-slate-900 dark:text-slate-100 flex items-center gap-2">
        <ArrowLeftRight className="w-5 h-5" />
        上下文切换成本
      </h3>
      <div className="flex gap-3 mb-4">
        {models.map((m, i) => (
          <button key={i} onClick={() => { setSelected(i); reset(); }}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              selected === i ? "text-white shadow-md" : "bg-slate-100 dark:bg-slate-800"
            }`}
            style={selected === i ? { backgroundColor: m.color } : undefined}>
            {m.name} <span className="text-xs opacity-70">({m.overhead})</span>
          </button>
        ))}
        <button onClick={start} disabled={running}
          className="ml-auto px-4 py-2 bg-indigo-600 text-white rounded-lg text-sm disabled:opacity-50 flex items-center gap-2">
          <Play className="w-4 h-4" /> 开始 100 次切换
        </button>
        <button onClick={reset} className="px-3 py-2 bg-slate-200 dark:bg-slate-700 rounded-lg"><RotateCcw className="w-4 h-4" /></button>
      </div>
      <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-5">
        <div className="grid grid-cols-3 gap-4 mb-4">
          <div className="text-center">
            <div className="text-xs text-slate-500 mb-1">切换次数</div>
            <div className="text-2xl font-bold" style={{ color: models[selected].color }}>{switches}</div>
          </div>
          <div className="text-center">
            <div className="text-xs text-slate-500 mb-1">已用时间</div>
            <div className="text-2xl font-bold text-slate-900 dark:text-slate-100">{elapsed.toFixed(2)}s</div>
          </div>
          <div className="text-center">
            <div className="text-xs text-slate-500 mb-1">单次切换成本</div>
            <div className="text-2xl font-bold" style={{ color: models[selected].color }}>{models[selected].switchCost}ms</div>
          </div>
        </div>
        <div className="h-4 bg-slate-100 dark:bg-slate-900 rounded-full overflow-hidden">
          <motion.div className="h-full rounded-full" style={{ width: `${switches}%`, backgroundColor: models[selected].color }} />
        </div>
        <p className="mt-3 text-xs text-slate-500">{models[selected].description}</p>
      </div>
    </div>
  );
}
