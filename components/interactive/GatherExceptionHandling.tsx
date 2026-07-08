"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Play, RotateCcw, AlertTriangle, CheckCircle, XCircle } from "lucide-react";

const TASKS = [
  { id: 1, name: "task_1", delay: 500, willFail: false, result: "data_1" },
  { id: 2, name: "task_2", delay: 300, willFail: true, error: "ValueError" },
  { id: 3, name: "task_3", delay: 700, willFail: false, result: "data_3" },
];

export function GatherExceptionHandling() {
  const [mode, setMode] = useState<"raise" | "return" | null>(null);
  const [running, setRunning] = useState(false);
  const [results, setResults] = useState<{ type: "ok" | "err"; value: string }[]>([]);
  const [completed, setCompleted] = useState<Set<number>>(new Set());
  const [exception, setException] = useState<string | null>(null);

  const run = async (m: "raise" | "return") => {
    setMode(m);
    setRunning(true);
    setResults([]);
    setCompleted(new Set());
    setException(null);

    if (m === "raise") {
      for (const t of TASKS) {
        await new Promise((r) => setTimeout(r, t.delay));
        setCompleted((prev) => new Set([...prev, t.id]));
        if (t.willFail) {
          setException(`${t.error}: ${t.name} 失败`);
          setRunning(false);
          return;
        }
      }
    } else {
      const res: { type: "ok" | "err"; value: string }[] = [];
      for (const t of TASKS) {
        await new Promise((r) => setTimeout(r, t.delay));
        setCompleted((prev) => new Set([...prev, t.id]));
        res.push(t.willFail ? { type: "err", value: `${t.error}(${t.name})` } : { type: "ok", value: t.result ?? "" });
        setResults([...res]);
      }
    }
    setRunning(false);
  };

  const reset = () => {
    setRunning(false);
    setMode(null);
    setResults([]);
    setCompleted(new Set());
    setException(null);
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-indigo-50 dark:from-slate-900 dark:to-indigo-950 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4 text-center flex items-center justify-center gap-2">
        <AlertTriangle className="w-7 h-7 text-indigo-600 dark:text-indigo-400" />
        gather() 异常处理对比
      </h3>

      <div className="flex justify-center gap-3 mb-4">
        <button onClick={() => run("raise")} disabled={running} className="px-5 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:opacity-50 dark:bg-red-500">
          return_exceptions=False
        </button>
        <button onClick={() => run("return")} disabled={running} className="px-5 py-2 bg-emerald-600 text-white rounded-lg hover:bg-emerald-700 disabled:opacity-50 dark:bg-emerald-500">
          return_exceptions=True
        </button>
        <button onClick={reset} className="px-5 py-2 bg-slate-500 text-white rounded-lg hover:bg-slate-600">
          <span className="flex items-center gap-1"><RotateCcw className="w-4 h-4" /> 重置</span>
        </button>
      </div>

      <div className="grid grid-cols-3 gap-3 mb-4">
        {TASKS.map((t) => (
          <motion.div key={t.id} animate={{ scale: completed.has(t.id) ? 1.05 : 1 }}
            className={`p-3 rounded-lg text-center ${completed.has(t.id) ? (t.willFail ? "bg-red-100 dark:bg-red-900/30 border-2 border-red-400" : "bg-emerald-100 dark:bg-emerald-900/30 border-2 border-emerald-400") : "bg-white dark:bg-slate-800 border-2 border-slate-200 dark:border-slate-700"} shadow`}>
            <div className="font-bold text-slate-800 dark:text-slate-100">{t.name}</div>
            <div className="text-xs text-slate-500 dark:text-slate-400">{t.delay}ms</div>
            {t.willFail && <div className="text-xs mt-1 text-red-600 dark:text-red-400">会抛异常</div>}
            {completed.has(t.id) && (t.willFail ? <XCircle className="w-4 h-4 text-red-500 mx-auto mt-1" /> : <CheckCircle className="w-4 h-4 text-emerald-500 mx-auto mt-1" />)}
          </motion.div>
        ))}
      </div>

      <AnimatePresence>
        {mode === "raise" && exception && (
          <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="bg-red-50 dark:bg-red-900/30 border-2 border-red-400 rounded-lg p-4 mb-4">
            <h4 className="font-bold text-red-800 dark:text-red-200">return_exceptions=False</h4>
            <p className="text-sm text-red-700 dark:text-red-300 mt-1">第一个异常立即抛出，后续任务结果被丢弃</p>
            <pre className="bg-red-100 dark:bg-red-900/50 p-2 rounded text-xs mt-2 text-red-800 dark:text-red-200">Exception: {exception}</pre>
          </motion.div>
        )}

        {mode === "return" && results.length > 0 && (
          <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="bg-emerald-50 dark:bg-emerald-900/30 border-2 border-emerald-400 rounded-lg p-4 mb-4">
            <h4 className="font-bold text-emerald-800 dark:text-emerald-200">return_exceptions=True</h4>
            <p className="text-sm text-emerald-700 dark:text-emerald-300 mt-1">异常作为结果返回，不会中断其他任务</p>
            <div className="mt-2 space-y-1">
              {results.map((r, i) => (
                <div key={i} className={`text-xs font-mono p-1 rounded ${r.type === "err" ? "bg-red-100 dark:bg-red-900/50 text-red-700 dark:text-red-300" : "bg-emerald-100 dark:bg-emerald-900/50 text-emerald-700 dark:text-emerald-300"}`}>
                  results[{i}] = {r.type === "ok" ? `"${r.value}"` : `${r.value} (exception object)`}
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="bg-white dark:bg-slate-800 rounded-lg p-3 text-xs font-mono text-slate-700 dark:text-slate-300 shadow">
        <p className="text-red-500">{`# raise: 第一个异常就抛出`}</p>
        <p>{`results = await asyncio.gather(*tasks)`}`</p>
        <p className="mt-2 text-emerald-500">{`# return: 异常作为值返回`}</p>
        <p>{`results = await asyncio.gather(*tasks, return_exceptions=True)`}</p>
      </div>
    </div>
  );
}
