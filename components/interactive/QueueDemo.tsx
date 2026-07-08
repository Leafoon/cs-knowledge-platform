"use client";

import React, { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Play, Pause, RotateCcw, Package, ArrowRight, Gauge } from "lucide-react";

interface QueueItem { id: number; color: string; }
const COLORS = ["bg-blue-500", "bg-green-500", "bg-purple-500", "bg-orange-500", "bg-pink-500", "bg-cyan-500"];

export function QueueDemo() {
  const [queue, setQueue] = useState<QueueItem[]>([]);
  const [producerSpeed, setProducerSpeed] = useState(1000);
  const [consumerSpeed, setConsumerSpeed] = useState(1500);
  const [running, setRunning] = useState(false);
  const [produced, setProduced] = useState(0);
  const [consumed, setConsumed] = useState(0);
  const [lastAction, setLastAction] = useState("");
  const producerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const consumerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const idRef = useRef(0);

  const stop = () => {
    setRunning(false);
    if (producerRef.current) clearInterval(producerRef.current);
    if (consumerRef.current) clearInterval(consumerRef.current);
  };

  const start = () => {
    setRunning(true);
    producerRef.current = setInterval(() => {
      setQueue((q) => {
        if (q.length >= 8) { setLastAction("队列已满! 生产者等待..."); return q; }
        const item = { id: idRef.current++, color: COLORS[idRef.current % COLORS.length] };
        setProduced((p) => p + 1); setLastAction(`生产者 → 放入 #${item.id}`);
        return [...q, item];
      });
    }, producerSpeed);
    consumerRef.current = setInterval(() => {
      setQueue((q) => {
        if (q.length === 0) { setLastAction("队列为空! 消费者等待..."); return q; }
        setConsumed((c) => c + 1); setLastAction(`消费者 ← 取出 #${q[0].id}`); return q.slice(1);
      });
    }, consumerSpeed);
  };

  const reset = () => { stop(); setQueue([]); setProduced(0); setConsumed(0); setLastAction(""); idRef.current = 0; };
  const restartIfRunning = () => { if (running) { stop(); setTimeout(start, 100); } };
  useEffect(() => () => { if (producerRef.current) clearInterval(producerRef.current); if (consumerRef.current) clearInterval(consumerRef.current); }, []);

  return (
    <div className="max-w-6xl mx-auto p-6">
      <h3 className="text-xl font-bold mb-4 text-slate-900 dark:text-slate-100 flex items-center gap-2">
        <Package className="w-5 h-5 text-indigo-500" /> 队列演示 — 生产者/消费者模型
      </h3>
      <div className="flex flex-wrap gap-3 mb-4">
        <button onClick={running ? stop : start} className={`px-4 py-2 rounded-lg font-medium text-sm flex items-center gap-2 ${running ? "bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-300" : "bg-indigo-600 text-white"}`}>
          {running ? <><Pause className="w-4 h-4" /> 暂停</> : <><Play className="w-4 h-4" /> 启动</>}
        </button>
        <button onClick={reset} className="px-4 py-2 rounded-lg bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-200 font-medium text-sm"><RotateCcw className="w-4 h-4" /></button>
      </div>
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <label className="text-xs text-slate-500 flex items-center gap-1 mb-1"><Gauge className="w-3 h-3" /> 生产速度: {producerSpeed}ms</label>
          <input type="range" min={300} max={3000} step={100} value={producerSpeed} onChange={(e) => { setProducerSpeed(Number(e.target.value)); restartIfRunning(); }} className="w-full accent-indigo-600" />
        </div>
        <div>
          <label className="text-xs text-slate-500 flex items-center gap-1 mb-1"><Gauge className="w-3 h-3" /> 消费速度: {consumerSpeed}ms</label>
          <input type="range" min={300} max={3000} step={100} value={consumerSpeed} onChange={(e) => { setConsumerSpeed(Number(e.target.value)); restartIfRunning(); }} className="w-full accent-indigo-600" />
        </div>
      </div>
      <div className="grid grid-cols-3 gap-3 mb-4">
        {[{ v: produced, l: "已生产", c: "blue" }, { v: queue.length, l: "队列长度", c: "indigo" }, { v: consumed, l: "已消费", c: "green" }].map(({ v, l, c }) => (
          <div key={l} className={`rounded-lg bg-${c}-50 dark:bg-${c}-900/20 border border-${c}-200 dark:border-${c}-800 p-3 text-center`}>
            <div className={`text-xl font-bold text-${c}-700 dark:text-${c}-300`}>{v}</div>
            <div className={`text-xs text-${c}-600 dark:text-${c}-400`}>{l}</div>
          </div>
        ))}
      </div>
      <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-4 mb-4">
        <div className="flex items-center gap-4">
          <div className="text-sm font-medium text-slate-500 w-20">生产者</div>
          <ArrowRight className="w-4 h-4 text-slate-400" />
          <div className="flex-1 flex gap-1 items-center min-h-[48px] p-2 rounded-lg bg-slate-50 dark:bg-slate-900 border border-dashed border-slate-300 dark:border-slate-600 overflow-x-auto">
            <AnimatePresence mode="popLayout">
              {queue.map((item) => (
                <motion.div key={item.id} initial={{ scale: 0 }} animate={{ scale: 1 }} exit={{ scale: 0 }}
                  className={`w-10 h-10 rounded-lg ${item.color} flex items-center justify-center text-white text-xs font-bold flex-shrink-0`}>{item.id}</motion.div>
              ))}
            </AnimatePresence>
            {queue.length === 0 && <span className="text-xs text-slate-400">空队列</span>}
          </div>
          <ArrowRight className="w-4 h-4 text-slate-400" />
          <div className="text-sm font-medium text-slate-500 w-20 text-right">消费者</div>
        </div>
      </div>
      {lastAction && <motion.div key={lastAction} initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="text-sm text-slate-600 dark:text-slate-400 font-mono">{lastAction}</motion.div>}
    </div>
  );
}
