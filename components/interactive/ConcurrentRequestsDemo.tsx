"use client";

import React, { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Play, RotateCcw, Zap, CheckCircle } from "lucide-react";

interface Request { id: number; serverTime: number; seqDone: boolean; concDone: boolean; }

export function ConcurrentRequestsDemo() {
  const [requestCount, setRequestCount] = useState(5);
  const [mode, setMode] = useState<"idle" | "sequential" | "concurrent" | "done">("idle");
  const [progress, setProgress] = useState(0);
  const [requests, setRequests] = useState<Request[]>([]);
  const [seqTime, setSeqTime] = useState(0);
  const [concTime, setConcTime] = useState(0);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const generateRequests = () => Array.from({ length: requestCount }, (_, i) => ({
    id: i, serverTime: Math.floor(Math.random() * 1500) + 500, seqDone: false, concDone: false,
  }));

  const reset = () => {
    setMode("idle"); setProgress(0); setSeqTime(0); setConcTime(0);
    setRequests(generateRequests());
    if (intervalRef.current) clearInterval(intervalRef.current);
  };

  useEffect(() => { reset(); }, [requestCount]);

  const startDemo = () => {
    const reqs = generateRequests(); setRequests(reqs); setMode("sequential"); setProgress(0);
    let seqAccum = 0; reqs.forEach((r) => { seqAccum += r.serverTime; });
    setSeqTime(seqAccum);
    setConcTime(Math.max(...reqs.map((r) => r.serverTime)));
    const total = seqAccum + Math.max(...reqs.map((r) => r.serverTime)) + 400;
    const start = Date.now();
    intervalRef.current = setInterval(() => {
      const elapsed = Date.now() - start; setProgress(elapsed);
      let seqAcc = 0;
      setRequests((prev) => prev.map((r) => {
        const s = seqAcc; seqAcc += r.serverTime;
        return { ...r, seqDone: elapsed > s + seqAccum + 200 ? true : elapsed > s && elapsed < seqAccum + 200 ? elapsed > s + r.serverTime : r.seqDone, concDone: elapsed > seqAccum + 200 && elapsed > seqAccum + 200 + r.serverTime };
      }));
      if (elapsed < seqAccum + 200) setMode("sequential");
      else if (elapsed < total) setMode("concurrent");
      else { setMode("done"); if (intervalRef.current) clearInterval(intervalRef.current); }
    }, 50);
  };

  useEffect(() => () => { if (intervalRef.current) clearInterval(intervalRef.current); }, []);

  return (
    <div className="max-w-6xl mx-auto p-6">
      <h3 className="text-xl font-bold mb-4 text-slate-900 dark:text-slate-100 flex items-center gap-2">
        <Zap className="w-5 h-5 text-yellow-500" /> 串行 vs 并发请求对比
      </h3>
      <div className="flex flex-wrap items-center gap-4 mb-4">
        <div className="flex items-center gap-2">
          <label className="text-sm text-slate-600 dark:text-slate-400">请求数:</label>
          {[3, 5, 8, 10].map((n) => (
            <button key={n} onClick={() => setRequestCount(n)}
              className={`w-8 h-8 rounded-lg text-sm font-bold ${n === requestCount ? "bg-yellow-500 text-white" : "bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-300"}`}>{n}</button>
          ))}
        </div>
        <button onClick={startDemo} disabled={mode !== "idle" && mode !== "done"}
          className="px-4 py-2 rounded-lg bg-yellow-500 text-white font-medium text-sm flex items-center gap-2 disabled:opacity-50">
          <Play className="w-4 h-4" /> 开始对比
        </button>
        <button onClick={reset} className="px-4 py-2 rounded-lg bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-200 font-medium text-sm"><RotateCcw className="w-4 h-4" /></button>
      </div>
      {["sequential", "concurrent"].map((m) => (
        <div key={m} className="mb-4">
          <div className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
            {m === "sequential" ? "串行执行" : "并发执行"}
            <span className="text-xs text-slate-400 ml-2">{((m === "sequential" ? seqTime : concTime) / 1000).toFixed(1)}s</span>
          </div>
          <div className="space-y-1">
            {requests.map((r) => {
              const done = m === "sequential" ? r.seqDone : r.concDone;
              const isActive = mode === m;
              return (
                <div key={r.id} className="flex items-center gap-2">
                  <span className="text-xs w-8 text-slate-400">#{r.id + 1}</span>
                  <div className="flex-1 h-5 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden">
                    <motion.div className={`h-full rounded-full ${done ? "bg-green-400" : isActive ? (m === "sequential" ? "bg-blue-500" : "bg-purple-500") : "bg-slate-300 dark:bg-slate-600"}`}
                      animate={{ width: done ? "100%" : isActive ? "80%" : "0%" }} />
                  </div>
                  <span className="text-xs w-14 text-right text-slate-400">{r.serverTime}ms</span>
                </div>
              );
            })}
          </div>
        </div>
      ))}
      {mode === "done" && (
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
          className="rounded-xl border border-green-200 dark:border-green-800 bg-green-50 dark:bg-green-900/20 p-4 flex items-center gap-3">
          <CheckCircle className="w-5 h-5 text-green-500" />
          <span className="text-sm text-green-700 dark:text-green-300">并发快 <strong>{(seqTime / concTime).toFixed(1)}x</strong>! 串行 {(seqTime / 1000).toFixed(1)}s → 并发 {(concTime / 1000).toFixed(1)}s</span>
        </motion.div>
      )}
    </div>
  );
}
