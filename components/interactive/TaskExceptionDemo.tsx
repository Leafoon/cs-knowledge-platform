"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { AlertTriangle, Play, RotateCcw, FileWarning, CheckCircle2 } from "lucide-react";

type Phase = "idle" | "running" | "exception_stored" | "caught";

export function TaskExceptionDemo() {
  const [phase, setPhase] = useState<Phase>("idle");

  const runTask = () => { setPhase("running"); setTimeout(() => setPhase("exception_stored"), 1200); };
  const awaitTask = () => setPhase("caught");
  const reset = () => setPhase("idle");

  const phases = [
    { key: "idle", label: "Task Created", desc: "Task created from a coroutine that will raise ValueError." },
    { key: "running", label: "Task Running", desc: "Coroutine is executing on the event loop..." },
    { key: "exception_stored", label: "Exception Stored", desc: "Exception is stored in the task. It won't crash the loop until awaited." },
    { key: "caught", label: "Exception Caught", desc: "awaiting the task re-raises the stored exception. Caught with try/except." },
  ];
  const current = phases.find((p) => p.key === phase);

  return (
    <div className="max-w-6xl mx-auto p-6">
      <h3 className="text-xl font-bold mb-2 text-gray-900 dark:text-gray-100">
        Task Exception Handling
      </h3>
      <p className="text-sm text-gray-600 dark:text-gray-400 mb-6">
        See how exceptions raised inside a task are stored and later re-raised when awaited.
      </p>

      <div className="flex flex-wrap gap-2 mb-6">
        {phases.map((p, i) => (
          <motion.div
            key={p.key}
            animate={{ opacity: phases.findIndex((x) => x.key === phase) >= i ? 1 : 0.3 }}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium ${
              phase === p.key
                ? "bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300 ring-2 ring-red-400"
                : "bg-gray-100 dark:bg-gray-800 text-gray-500 dark:text-gray-400"
            }`}
          >
            {i + 1}. {p.label}
          </motion.div>
        ))}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
        <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-5 shadow-sm">
          <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-3 flex items-center gap-2">
            <FileWarning className="w-4 h-4" />
            Task State
          </h4>
          <div className="space-y-2 text-sm font-mono">
            <div className="flex justify-between">
              <span className="text-gray-500 dark:text-gray-400">task.done():</span>
              <span className={phase !== "idle" && phase !== "running" ? "text-green-600 dark:text-green-400" : "text-gray-400"}>
                {phase === "idle" ? "False" : phase === "running" ? "False" : "True"}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500 dark:text-gray-400">task.exception():</span>
              <span className={phase === "exception_stored" || phase === "caught" ? "text-red-600 dark:text-red-400" : "text-gray-400"}>
                {phase === "exception_stored" || phase === "caught" ? "ValueError" : "None"}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500 dark:text-gray-400">task.result():</span>
              <span className="text-gray-400">
                {phase === "exception_stored" || phase === "caught" ? "raises exception" : "pending"}
              </span>
            </div>
          </div>
        </div>

        <AnimatePresence mode="wait">
          <motion.div
            key={phase}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-5 shadow-sm"
          >
            <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">{current?.label}</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">{current?.desc}</p>
            {phase === "caught" && (
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-3"
              >
                <div className="flex items-center gap-2 text-red-700 dark:text-red-300 text-sm font-semibold mb-1">
                  <AlertTriangle className="w-4 h-4" />
                  Exception Caught
                </div>
                <pre className="text-xs font-mono text-red-600 dark:text-red-400">
{`try:
    await task
except ValueError as e:
    print(f"Caught: {e}")`}
                </pre>
              </motion.div>
            )}
            {phase === "exception_stored" && (
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                className="bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-lg p-3 text-sm text-amber-700 dark:text-amber-300"
              >
                <CheckCircle2 className="w-4 h-4 inline mr-1" />
                Exception is silently stored. The event loop continues running other tasks.
              </motion.div>
            )}
          </motion.div>
        </AnimatePresence>
      </div>

      <div className="flex gap-3 justify-center">
        {phase === "idle" && (
          <motion.button whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }} onClick={runTask}
            className="flex items-center gap-2 px-6 py-2 rounded-lg bg-blue-600 hover:bg-blue-700 text-white font-medium shadow transition-colors">
            <Play className="w-4 h-4" /> Run Task
          </motion.button>
        )}
        {phase === "running" && (
          <div className="text-sm text-gray-500 dark:text-gray-400 animate-pulse">Task executing...</div>
        )}
        {phase === "exception_stored" && (
          <motion.button whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }} onClick={awaitTask}
            className="flex items-center gap-2 px-6 py-2 rounded-lg bg-amber-600 hover:bg-amber-700 text-white font-medium shadow transition-colors">
            <AlertTriangle className="w-4 h-4" /> await task {'->'} catch
          </motion.button>
        )}
        {phase === "caught" && (
          <motion.button whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }} onClick={reset}
            className="flex items-center gap-2 px-6 py-2 rounded-lg bg-gray-600 hover:bg-gray-700 text-white font-medium shadow transition-colors">
            <RotateCcw className="w-4 h-4" /> Reset
          </motion.button>
        )}
      </div>
    </div>
  );
}
