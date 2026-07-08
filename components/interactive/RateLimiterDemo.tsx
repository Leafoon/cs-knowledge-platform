"use client";

import React, { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Play, RotateCcw, Droplets, CheckCircle, XCircle } from "lucide-react";

interface Request { id: number; allowed: boolean; }

export function RateLimiterDemo() {
  const [maxRPS, setMaxRPS] = useState(3);
  const [tokens, setTokens] = useState(3);
  const [running, setRunning] = useState(false);
  const [requests, setRequests] = useState<Request[]>([]);
  const [incomingRate, setIncomingRate] = useState(500);
  const [stats, setStats] = useState({ allowed: 0, rejected: 0 });
  const reqRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const tokenRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const idRef = useRef(0);

  const reset = () => {
    setRunning(false); setTokens(maxRPS); setRequests([]); setStats({ allowed: 0, rejected: 0 }); idRef.current = 0;
    if (reqRef.current) clearInterval(reqRef.current); if (tokenRef.current) clearInterval(tokenRef.current);
  };

  useEffect(() => { reset(); }, [maxRPS]);

  const start = () => {
    if (running) return; setRunning(true);
    tokenRef.current = setInterval(() => { setTokens((t) => Math.min(t + 1, maxRPS)); }, 1000);
    reqRef.current = setInterval(() => {
      setTokens((t) => {
        const id = idRef.current++; const allowed = t > 0;
        setRequests((prev) => [...prev.slice(-20), { id, allowed }]);
        setStats((s) => ({ allowed: s.allowed + (allowed ? 1 : 0), rejected: s.rejected + (allowed ? 0 : 1) }));
        return allowed ? t - 1 : t;
      });
    }, incomingRate);
  };

  const stop = () => { setRunning(false); if (reqRef.current) clearInterval(reqRef.current); if (tokenRef.current) clearInterval(tokenRef.current); };
  useEffect(() => () => { if (reqRef.current) clearInterval(reqRef.current); if (tokenRef.current) clearInterval(tokenRef.current); }, []);
  const total = stats.allowed + stats.rejected;

  return (
    <div className="max-w-6xl mx-auto p-6">
      <h3 className="text-xl font-bold mb-4 text-slate-900 dark:text-slate-100 flex items-center gap-2">
        <Droplets className="w-5 h-5 text-blue-500" /> 令牌桶限流演示
      </h3>
      <div className="flex flex-wrap items-center gap-4 mb-4">
        <div className="flex items-center gap-2">
          <label className="text-sm text-slate-600 dark:text-slate-400">速率限制:</label>
          {[1, 2, 3, 5, 10].map((n) => (
            <button key={n} onClick={() => setMaxRPS(n)}
              className={`w-8 h-8 rounded-lg text-sm font-bold ${n === maxRPS ? "bg-blue-600 text-white" : "bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-300"}`}>{n}</button>
          ))}
          <span className="text-xs text-slate-400">req/s</span>
        </div>
        <div className="flex items-center gap-2">
          <label className="text-sm text-slate-600 dark:text-slate-400">间隔: {incomingRate}ms</label>
          <input type="range" min={100} max={2000} step={100} value={incomingRate} onChange={(e) => setIncomingRate(Number(e.target.value))} className="w-24 accent-blue-500" />
        </div>
      </div>
      <div className="flex gap-3 mb-6">
        <button onClick={running ? stop : start}
          className={`px-4 py-2 rounded-lg font-medium text-sm flex items-center gap-2 ${running ? "bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-300" : "bg-blue-600 text-white"}`}>
          {running ? <><XCircle className="w-4 h-4" /> 停止</> : <><Play className="w-4 h-4" /> 启动</>}
        </button>
        <button onClick={reset} className="px-4 py-2 rounded-lg bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-200 font-medium text-sm"><RotateCcw className="w-4 h-4" /></button>
      </div>
      <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-6 mb-4">
        <div className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-3">令牌桶</div>
        <div className="flex items-end gap-1 justify-center h-20">
          {Array.from({ length: maxRPS }, (_, i) => (
            <motion.div key={i} animate={{ opacity: i < tokens ? 1 : 0.15, scale: i < tokens ? 1 : 0.9 }}
              className={`w-10 rounded-lg flex items-center justify-center ${i < tokens ? "bg-blue-500" : "bg-slate-200 dark:bg-slate-700"}`} style={{ height: "100%" }}>
              {i < tokens && <Droplets className="w-5 h-5 text-white" />}
            </motion.div>
          ))}
        </div>
        <div className="text-center text-sm text-slate-500 mt-2">{tokens} / {maxRPS} 令牌可用</div>
      </div>
      <div className="grid grid-cols-3 gap-3 mb-4">
        {[{ v: total, l: "总请求", c: "blue" }, { v: stats.allowed, l: "通过", c: "green" }, { v: stats.rejected, l: "限流", c: "red" }].map(({ v, l, c }) => (
          <div key={l} className={`rounded-lg bg-${c}-50 dark:bg-${c}-900/20 border border-${c}-200 dark:border-${c}-800 p-3 text-center`}>
            <div className={`text-xl font-bold text-${c}-700 dark:text-${c}-300`}>{v}</div>
            <div className={`text-xs text-${c}-600 dark:text-${c}-400`}>{l}</div>
          </div>
        ))}
      </div>
      <div className="flex gap-1 flex-wrap">
        <AnimatePresence>
          {requests.slice(-30).map((r) => (
            <motion.div key={r.id} initial={{ scale: 0 }} animate={{ scale: 1 }} exit={{ scale: 0 }}
              className={`w-7 h-7 rounded flex items-center justify-center ${r.allowed ? "bg-green-100 dark:bg-green-900/30" : "bg-red-100 dark:bg-red-900/30"}`}>
              {r.allowed ? <CheckCircle className="w-3.5 h-3.5 text-green-600" /> : <XCircle className="w-3.5 h-3.5 text-red-600" />}
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
    </div>
  );
}
