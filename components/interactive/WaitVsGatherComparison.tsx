"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { ArrowRight, CheckCircle, Clock, Layers } from "lucide-react";

const TASKS = [
  { id: 1, name: "任务A", delay: 300, color: "bg-blue-500" },
  { id: 2, name: "任务B", delay: 100, color: "bg-emerald-500" },
  { id: 3, name: "任务C", delay: 200, color: "bg-amber-500" },
];

export function WaitVsGatherComparison() {
  const [mode, setMode] = useState<"gather" | "wait" | null>(null);
  const [running, setRunning] = useState(false);
  const [gatherResults, setGatherResults] = useState<number[]>([]);
  const [waitDone, setWaitDone] = useState<number[]>([]);
  const [waitPending, setWaitPending] = useState<number[]>([1, 2, 3]);
  const [completedSet, setCompletedSet] = useState<Set<number>>(new Set());

  const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

  const runGather = async () => {
    setMode("gather");
    setRunning(true);
    setGatherResults([]);
    setCompletedSet(new Set());

    const promises = TASKS.map(async (t) => {
      await sleep(t.delay);
      setCompletedSet((prev) => new Set([...prev, t.id]));
      return t.id;
    });

    const results = await Promise.all(promises);
    setGatherResults(results);
    setRunning(false);
  };

  const runWait = async () => {
    setMode("wait");
    setRunning(true);
    setWaitDone([]);
    setWaitPending([1, 2, 3]);
    setCompletedSet(new Set());

    const remaining = [1, 2, 3];
    const done: number[] = [];

    const promises = TASKS.map(async (t) => {
      await sleep(t.delay);
      setCompletedSet((prev) => new Set([...prev, t.id]));
      return t.id;
    });

    for (const p of promises) {
      const id = await p;
      done.push(id);
      remaining.splice(remaining.indexOf(id), 1);
      setWaitDone([...done]);
      setWaitPending([...remaining]);
    }

    setRunning(false);
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-indigo-50 dark:from-slate-900 dark:to-indigo-950 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4 text-center flex items-center justify-center gap-2">
        <Layers className="w-7 h-7 text-indigo-600 dark:text-indigo-400" />
        gather() vs wait() 对比
      </h3>

      <div className="bg-white dark:bg-slate-800 rounded-lg p-4 mb-4 text-sm text-slate-600 dark:text-slate-300">
        <p><strong>gather()</strong>: 按输入顺序返回结果，即使后面的先完成。</p>
        <p><strong>wait()</strong>: 返回 (done, pending) 集合，按完成顺序填充 done。</p>
      </div>

      <div className="flex justify-center gap-4 mb-4">
        <button onClick={runGather} disabled={running} className="px-5 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50 dark:bg-indigo-500 dark:hover:bg-indigo-600">
          <span className="flex items-center gap-1"><ArrowRight className="w-4 h-4" /> 运行 gather()</span>
        </button>
        <button onClick={runWait} disabled={running} className="px-5 py-2 bg-emerald-600 text-white rounded-lg hover:bg-emerald-700 disabled:opacity-50 dark:bg-emerald-500 dark:hover:bg-emerald-600">
          <span className="flex items-center gap-1"><ArrowRight className="w-4 h-4" /> 运行 wait()</span>
        </button>
      </div>

      <div className="grid grid-cols-3 gap-3 mb-4">
        {TASKS.map((t) => (
          <motion.div key={t.id} animate={{ scale: completedSet.has(t.id) ? 1.05 : 1 }} className={`p-3 rounded-lg text-center text-white font-bold ${t.color} dark:opacity-90`}>
            {t.name} ({t.delay}ms)
            {completedSet.has(t.id) && <CheckCircle className="w-4 h-4 inline ml-1" />}
          </motion.div>
        ))}
      </div>

      {mode === "gather" && (
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="bg-white dark:bg-slate-800 rounded-lg p-4 shadow">
          <h4 className="font-bold text-slate-800 dark:text-slate-100 mb-2 flex items-center gap-1">
            <Clock className="w-4 h-4 text-indigo-600" /> gather() 结果（输入顺序）
          </h4>
          <div className="flex gap-2">
            {gatherResults.length > 0 ? gatherResults.map((id, i) => (
              <motion.span key={i} initial={{ scale: 0 }} animate={{ scale: 1 }} className="px-3 py-1 bg-indigo-100 dark:bg-indigo-900 text-indigo-800 dark:text-indigo-200 rounded font-mono">
                [{i}] = 任务{String.fromCharCode(64 + id)}
              </motion.span>
            )) : <span className="text-slate-400">{running ? "运行中..." : "点击运行"}</span>}
          </div>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-2">返回顺序: [A, B, C]（与输入顺序一致）</p>
        </motion.div>
      )}

      {mode === "wait" && (
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="grid grid-cols-2 gap-4">
          <div className="bg-white dark:bg-slate-800 rounded-lg p-4 shadow">
            <h4 className="font-bold text-emerald-700 dark:text-emerald-400 mb-2">done 集合</h4>
            <div className="flex flex-wrap gap-2 min-h-[2rem]">
              {waitDone.map((id, i) => (
                <motion.span key={id} initial={{ scale: 0 }} animate={{ scale: 1 }} className="px-3 py-1 bg-emerald-100 dark:bg-emerald-900 text-emerald-800 dark:text-emerald-200 rounded font-mono">
                  任务{String.fromCharCode(64 + id)}
                </motion.span>
              ))}
            </div>
          </div>
          <div className="bg-white dark:bg-slate-800 rounded-lg p-4 shadow">
            <h4 className="font-bold text-amber-700 dark:text-amber-400 mb-2">pending 集合</h4>
            <div className="flex flex-wrap gap-2 min-h-[2rem]">
              {waitPending.map((id) => (
                <motion.span key={id} className="px-3 py-1 bg-amber-100 dark:bg-amber-900 text-amber-800 dark:text-amber-200 rounded font-mono">
                  任务{String.fromCharCode(64 + id)}
                </motion.span>
              ))}
            </div>
          </div>
          <p className="col-span-2 text-xs text-slate-500 dark:text-slate-400">done 按完成顺序填充: B(100ms) → C(200ms) → A(300ms)</p>
        </motion.div>
      )}
    </div>
  );
}
