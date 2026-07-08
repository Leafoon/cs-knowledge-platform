"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Layers, Zap, ArrowLeftRight, Info, Play, RotateCcw } from "lucide-react";

const features = [
  { name: "Creation", coroutine: "Calling async def func()", task: "asyncio.create_task(coro)", winner: "tie" as "tie" | "task" },
  { name: "Reusable", coroutine: "No — consumed after await", task: "No — runs once", winner: "tie" as "tie" | "task" },
  { name: "Result storage", coroutine: "No — must await to get value", task: "Yes — stored after completion", winner: "task" as "tie" | "task" },
  { name: "Multiple awaits", coroutine: "Error on second await", task: "Safe — returns cached result", winner: "task" as "tie" | "task" },
  { name: "Concurrency", coroutine: "Sequential — blocks until done", task: "Runs concurrently on event loop", winner: "task" as "tie" | "task" },
  { name: "Cancellation", coroutine: "Not cancellable", task: "Supports task.cancel()", winner: "task" as "tie" | "task" },
  { name: "Exceptions", coroutine: "Raises immediately", task: "Stores exception until awaited", winner: "task" as "tie" | "task" },
  { name: "done() check", coroutine: "Not available", task: "task.done() returns bool", winner: "task" as "tie" | "task" },
];

export function TaskVsCoroutineComparison() {
  const [activeTab, setActiveTab] = useState<"table" | "demo">("table");
  const [demoState, setDemoState] = useState<"idle" | "running" | "done">("idle");
  const [coroutineResult, setCoroutineResult] = useState<string | null>(null);
  const [taskResult, setTaskResult] = useState<string | null>(null);

  const runDemo = () => {
    setDemoState("running"); setCoroutineResult(null); setTaskResult(null);
    setTimeout(() => { setCoroutineResult("Value (consumed!)"); setTaskResult("Value (stored!)"); setDemoState("done"); }, 1500);
  };
  const resetDemo = () => { setDemoState("idle"); setCoroutineResult(null); setTaskResult(null); };

  return (
    <div className="max-w-6xl mx-auto p-6">
      <h3 className="text-xl font-bold mb-2 text-gray-900 dark:text-gray-100 flex items-center gap-2">
        <ArrowLeftRight className="w-5 h-5" />
        Coroutine vs Task
      </h3>
      <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
        Understand the key differences between a raw coroutine object and an asyncio Task.
      </p>

      <div className="flex gap-2 mb-4">
        {(["table", "demo"] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              activeTab === tab
                ? "bg-blue-600 text-white"
                : "bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600"
            }`}
          >
            {tab === "table" ? "Comparison Table" : "Interactive Demo"}
          </button>
        ))}
      </div>

      <AnimatePresence mode="wait">
        {activeTab === "table" ? (
          <motion.div key="table" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
            <div className="overflow-x-auto">
              <table className="w-full text-sm border-collapse">
                <thead>
                  <tr className="bg-gray-100 dark:bg-gray-800">
                    <th className="text-left p-3 font-semibold text-gray-700 dark:text-gray-300 border-b dark:border-gray-700">Feature</th>
                    <th className="text-left p-3 font-semibold text-purple-700 dark:text-purple-300 border-b dark:border-gray-700"><Layers className="w-4 h-4 inline mr-1" />Coroutine</th>
                    <th className="text-left p-3 font-semibold text-blue-700 dark:text-blue-300 border-b dark:border-gray-700"><Zap className="w-4 h-4 inline mr-1" />Task</th>
                  </tr>
                </thead>
                <tbody>
                  {features.map((f) => (
                    <tr key={f.name} className="border-b dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-800/50">
                      <td className="p-3 font-medium text-gray-800 dark:text-gray-200">{f.name}</td>
                      <td className={`p-3 ${f.winner === "tie" ? "text-green-600 dark:text-green-400 font-medium" : "text-gray-600 dark:text-gray-400"}`}>{f.coroutine}</td>
                      <td className={`p-3 ${f.winner === "task" ? "text-green-600 dark:text-green-400 font-medium" : "text-gray-600 dark:text-gray-400"}`}>{f.task}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </motion.div>
        ) : (
          <motion.div key="demo" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-purple-50 dark:bg-purple-900/20 rounded-xl border border-purple-200 dark:border-purple-800 p-5">
                <h4 className="font-semibold text-purple-800 dark:text-purple-200 mb-3 flex items-center gap-2">
                  <Layers className="w-4 h-4" /> Coroutine Object
                </h4>
                <pre className="text-xs font-mono text-purple-700 dark:text-purple-300 mb-3 bg-white dark:bg-gray-900 p-3 rounded-lg">
{`coro = fetch_data()
result = await coro  # 1st await: OK
result = await coro  # 2nd await: ERROR!`}
                </pre>
                <div className="text-sm">
                  {demoState === "idle" && <span className="text-gray-500 dark:text-gray-400">Press Run to try</span>}
                  {demoState === "running" && <span className="text-purple-600 dark:text-purple-400 animate-pulse">Awaiting coroutine...</span>}
                  {demoState === "done" && (
                    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                      <p className="text-green-600 dark:text-green-400">1st await: {coroutineResult}</p>
                      <p className="text-red-600 dark:text-red-400">2nd await: RuntimeError: cannot reuse</p>
                    </motion.div>
                  )}
                </div>
              </div>

              <div className="bg-blue-50 dark:bg-blue-900/20 rounded-xl border border-blue-200 dark:border-blue-800 p-5">
                <h4 className="font-semibold text-blue-800 dark:text-blue-200 mb-3 flex items-center gap-2">
                  <Zap className="w-4 h-4" /> Task Object
                </h4>
                <pre className="text-xs font-mono text-blue-700 dark:text-blue-300 mb-3 bg-white dark:bg-gray-900 p-3 rounded-lg">
{`task = create_task(fetch_data())
result = await task  # 1st await: OK
result = await task  # 2nd await: OK (cached)`}
                </pre>
                <div className="text-sm">
                  {demoState === "idle" && <span className="text-gray-500 dark:text-gray-400">Press Run to try</span>}
                  {demoState === "running" && <span className="text-blue-600 dark:text-blue-400 animate-pulse">Task running...</span>}
                  {demoState === "done" && (
                    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                      <p className="text-green-600 dark:text-green-400">1st await: {taskResult}</p>
                      <p className="text-green-600 dark:text-green-400">2nd await: {taskResult} (cached!)</p>
                    </motion.div>
                  )}
                </div>
              </div>
            </div>

            <div className="flex gap-3 justify-center mt-6">
              {demoState === "idle" ? (
                <motion.button whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }} onClick={runDemo}
                  className="flex items-center gap-2 px-6 py-2 rounded-lg bg-blue-600 hover:bg-blue-700 text-white font-medium shadow">
                  <Play className="w-4 h-4" /> Run Both
                </motion.button>
              ) : (
                <motion.button whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }} onClick={resetDemo}
                  className="flex items-center gap-2 px-6 py-2 rounded-lg bg-gray-600 hover:bg-gray-700 text-white font-medium shadow">
                  <RotateCcw className="w-4 h-4" /> Reset
                </motion.button>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
