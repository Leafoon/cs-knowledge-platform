"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Shield, Play, RotateCcw, XCircle, CheckCircle } from "lucide-react";

type Mode = "none" | "shield";

export function ShieldDemo() {
  const [mode, setMode] = useState<Mode>("none");
  const [running, setRunning] = useState(false);
  const [outerStatus, setOuterStatus] = useState("等待");
  const [innerStatus, setInnerStatus] = useState("等待");
  const [log, setLog] = useState<string[]>([]);

  const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

  const addLog = (msg: string) => setLog((prev) => [...prev, msg]);

  const run = async (m: Mode) => {
    setMode(m);
    setRunning(true);
    setOuterStatus("运行中");
    setInnerStatus("运行中");
    setLog([]);

    addLog(`模式: ${m === "shield" ? "有 shield()" : "无 shield()"}`);

    await sleep(500);
    addLog("inner 任务开始...");
    setInnerStatus("执行中 (2秒)");

    await sleep(1000);
    addLog("outer 任务被取消！");
    setOuterStatus("已取消 ❌");

    if (m === "shield") {
      addLog("inner 被 shield 保护，继续执行...");
      setInnerStatus("受保护，继续执行");
      await sleep(1500);
      addLog("inner 任务完成 ✓");
      setInnerStatus("完成 ✓");
    } else {
      addLog("inner 也被取消！");
      setInnerStatus("已取消 ❌");
    }

    setRunning(false);
  };

  const reset = () => {
    setRunning(false);
    setOuterStatus("等待");
    setInnerStatus("等待");
    setLog([]);
    setMode("none");
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-emerald-50 dark:from-slate-900 dark:to-emerald-950 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4 text-center flex items-center justify-center gap-2">
        <Shield className="w-7 h-7 text-emerald-600 dark:text-emerald-400" />
        asyncio.shield() 演示
      </h3>

      <p className="text-sm text-slate-600 dark:text-slate-300 text-center mb-4">shield() 保护内部任务不被外部取消</p>

      <div className="flex justify-center gap-3 mb-5">
        <button onClick={() => run("none")} disabled={running} className="px-5 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:opacity-50 dark:bg-red-500">
          <span className="flex items-center gap-1"><XCircle className="w-4 h-4" /> 无 shield</span>
        </button>
        <button onClick={() => run("shield")} disabled={running} className="px-5 py-2 bg-emerald-600 text-white rounded-lg hover:bg-emerald-700 disabled:opacity-50 dark:bg-emerald-500">
          <span className="flex items-center gap-1"><Shield className="w-4 h-4" /> 有 shield</span>
        </button>
        <button onClick={reset} className="px-5 py-2 bg-slate-500 text-white rounded-lg hover:bg-slate-600">
          <span className="flex items-center gap-1"><RotateCcw className="w-4 h-4" /> 重置</span>
        </button>
      </div>

      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="bg-white dark:bg-slate-800 rounded-lg p-4 shadow">
          <h4 className="font-bold text-slate-800 dark:text-slate-100 mb-2">outer 任务</h4>
          <div className={`p-3 rounded text-center font-semibold ${outerStatus.includes("取消") ? "bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300" : outerStatus.includes("运行") || outerStatus.includes("等待") ? "bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300" : "bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-300"}`}>
            {outerStatus}
          </div>
        </div>
        <div className="bg-white dark:bg-slate-800 rounded-lg p-4 shadow">
          <h4 className="font-bold text-slate-800 dark:text-slate-100 mb-2 flex items-center gap-1">
            inner 任务 {mode === "shield" && <Shield className="w-4 h-4 text-emerald-500" />}
          </h4>
          <div className={`p-3 rounded text-center font-semibold ${innerStatus.includes("取消") ? "bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300" : innerStatus.includes("完成") || innerStatus.includes("保护") ? "bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-300" : "bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300"}`}>
            {innerStatus}
          </div>
        </div>
      </div>

      <div className="bg-white dark:bg-slate-800 rounded-lg p-4 shadow mb-4">
        <h4 className="font-bold text-slate-800 dark:text-slate-100 mb-2">执行日志</h4>
        <div className="space-y-1 min-h-[80px]">
          <AnimatePresence>
            {log.map((msg, i) => (
              <motion.div key={i} initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} className="text-sm font-mono text-slate-600 dark:text-slate-300">
                {msg}
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      </div>

      <div className="bg-white dark:bg-slate-800 rounded-lg p-3 text-xs font-mono text-slate-700 dark:text-slate-300 shadow">
        <p>{`# 无 shield: 取消传播到内部`}</p>
        <p>{`await inner_task  # 被一起取消`}</p>
        <p className="mt-2">{`# 有 shield: 内部任务受保护`}</p>
        <p>{`await asyncio.shield(inner_task)  # 继续执行`}</p>
      </div>
    </div>
  );
}
