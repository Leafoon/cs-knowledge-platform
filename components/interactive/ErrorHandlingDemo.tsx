"use client";

import React, { useState, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Play, RotateCcw, AlertTriangle, CheckCircle, RefreshCw, ShieldAlert, BarChart3 } from "lucide-react";

interface Attempt { id: number; status: "成功" | "失败" | "重试中"; error?: string; retries: number; }
const ERRORS = ["连接超时", "500 服务器错误", "DNS 解析失败", "TLS 握手失败", "429 限流"];

export function ErrorHandlingDemo() {
  const [attempts, setAttempts] = useState<Attempt[]>([]);
  const [running, setRunning] = useState(false);
  const [maxRetries, setMaxRetries] = useState(3);
  const [failRate, setFailRate] = useState(40);
  const [stats, setStats] = useState({ success: 0, fail: 0, retries: 0 });
  const idRef = useRef(0);

  const sendRequest = async () => {
    const id = idRef.current++; let retries = 0; let success = false;
    setAttempts((prev) => [...prev, { id, status: "重试中", retries: 0 }]);
    while (retries <= maxRetries && !success) {
      await new Promise((r) => setTimeout(r, 600));
      if (Math.random() * 100 >= failRate) { success = true; }
      else { retries++; if (retries > maxRetries) { setAttempts((p) => p.map((a) => a.id === id ? { ...a, status: "失败", error: ERRORS[Math.floor(Math.random() * ERRORS.length)], retries } : a)); } }
    }
    if (success) setAttempts((p) => p.map((a) => a.id === id ? { ...a, status: "成功", retries } : a));
    setStats((s) => ({ success: s.success + (success ? 1 : 0), fail: s.fail + (success ? 0 : 1), retries: s.retries + retries }));
  };

  const startDemo = async () => {
    setRunning(true); setAttempts([]); setStats({ success: 0, fail: 0, retries: 0 }); idRef.current = 0;
    await Promise.all(Array.from({ length: 10 }, (_, i) => new Promise<void>((r) => setTimeout(() => { sendRequest().then(r); }, i * 200))));
    setRunning(false);
  };

  const reset = () => { setAttempts([]); setStats({ success: 0, fail: 0, retries: 0 }); idRef.current = 0; };
  const total = stats.success + stats.fail;

  return (
    <div className="max-w-6xl mx-auto p-6">
      <h3 className="text-xl font-bold mb-4 text-slate-900 dark:text-slate-100 flex items-center gap-2">
        <ShieldAlert className="w-5 h-5 text-red-500" /> 错误处理演示 — 重试逻辑
      </h3>
      <div className="flex flex-wrap items-center gap-4 mb-4">
        <div className="flex items-center gap-2">
          <label className="text-sm text-slate-600 dark:text-slate-400">最大重试:</label>
          {[0, 1, 2, 3, 5].map((n) => (
            <button key={n} onClick={() => setMaxRetries(n)}
              className={`w-8 h-8 rounded-lg text-sm font-bold ${n === maxRetries ? "bg-red-500 text-white" : "bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-300"}`}>{n}</button>
          ))}
        </div>
        <div className="flex items-center gap-2">
          <label className="text-sm text-slate-600 dark:text-slate-400">失败率: {failRate}%</label>
          <input type="range" min={10} max={80} value={failRate} onChange={(e) => setFailRate(Number(e.target.value))} className="w-24 accent-red-500" />
        </div>
        <button onClick={startDemo} disabled={running} className="px-4 py-2 rounded-lg bg-red-600 text-white font-medium text-sm flex items-center gap-2 disabled:opacity-50">
          <Play className="w-4 h-4" /> 发送 10 个请求
        </button>
        <button onClick={reset} className="px-4 py-2 rounded-lg bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-200 font-medium text-sm"><RotateCcw className="w-4 h-4" /></button>
      </div>
      <div className="grid grid-cols-3 gap-3 mb-4">
        {[{ v: stats.success, l: "成功", i: <CheckCircle className="w-4 h-4 mx-auto text-green-500 mb-1" />, c: "green" },
          { v: stats.fail, l: "最终失败", i: <AlertTriangle className="w-4 h-4 mx-auto text-red-500 mb-1" />, c: "red" },
          { v: stats.retries, l: "总重试次数", i: <RefreshCw className="w-4 h-4 mx-auto text-amber-500 mb-1" />, c: "amber" }
        ].map(({ v, l, i, c }) => (
          <div key={l} className={`rounded-lg bg-${c}-50 dark:bg-${c}-900/20 border border-${c}-200 dark:border-${c}-800 p-3 text-center`}>
            {i}<div className={`text-xl font-bold text-${c}-700 dark:text-${c}-300`}>{v}</div><div className={`text-xs text-${c}-600 dark:text-${c}-400`}>{l}</div>
          </div>
        ))}
      </div>
      <div className="space-y-1.5 max-h-48 overflow-y-auto">
        <AnimatePresence>
          {attempts.map((a) => (
            <motion.div key={a.id} initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }}
              className={`flex items-center gap-3 rounded-lg px-3 py-2 text-sm ${a.status === "成功" ? "bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800" : a.status === "失败" ? "bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800" : "bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800"}`}>
              <span className="font-mono text-xs text-slate-400 w-6">#{a.id + 1}</span>
              {a.status === "成功" ? <CheckCircle className="w-4 h-4 text-green-500" /> : a.status === "失败" ? <AlertTriangle className="w-4 h-4 text-red-500" /> : <RefreshCw className="w-4 h-4 text-blue-500 animate-spin" />}
              <span className={a.status === "成功" ? "text-green-700 dark:text-green-300" : a.status === "失败" ? "text-red-700 dark:text-red-300" : "text-blue-700 dark:text-blue-300"}>
                {a.status === "成功" ? "请求成功" : a.status === "失败" ? `失败: ${a.error}` : "重试中..."}
              </span>
              {a.retries > 0 && <span className="ml-auto text-xs text-slate-400">重试 {a.retries} 次</span>}
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
      {total > 0 && (
        <div className="mt-4 flex items-center gap-4">
          <BarChart3 className="w-4 h-4 text-slate-400" />
          <div className="flex-1 h-4 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden flex">
            <div className="bg-green-500 h-full" style={{ width: `${(stats.success / total) * 100}%` }} />
            <div className="bg-red-500 h-full" style={{ width: `${(stats.fail / total) * 100}%` }} />
          </div>
          <span className="text-xs text-slate-500">{((stats.success / total) * 100).toFixed(0)}% 成功率</span>
        </div>
      )}
    </div>
  );
}
