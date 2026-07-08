"use client";

import React, { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Play, RotateCcw, Timer, CheckCircle2, Hourglass } from "lucide-react";

type Phase = "idle" | "awaiting" | "resolved";

export function FutureResultDemo() {
  const [phase, setPhase] = useState<Phase>("idle");
  const [result, setResult] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState(0);

  const startDemo = useCallback(() => {
    setPhase("awaiting"); setResult(null); setElapsed(0);
    let t = 0;
    const iv = setInterval(() => { t += 100; setElapsed(t); }, 100);
    setTimeout(() => { clearInterval(iv); setResult("Hello from Future!"); setPhase("resolved"); }, 2000);
  }, []);

  const reset = useCallback(() => { setPhase("idle"); setResult(null); setElapsed(0); }, []);

  return (
    <div className="max-w-6xl mx-auto p-6">
      <h3 className="text-xl font-bold mb-2 text-gray-900 dark:text-gray-100">
        Future Result Demo
      </h3>
      <p className="text-sm text-gray-600 dark:text-gray-400 mb-6">
        Create a Future, set its result after a delay, and watch it transition from awaiting to resolved.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
        <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-5 shadow-sm">
          <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-3 flex items-center gap-2">
            <Hourglass className="w-4 h-4" />
            Future Object
          </h4>
          <div className="space-y-3">
            <div className="flex items-center gap-2 text-sm">
              <span className="font-mono text-gray-500 dark:text-gray-400 w-20">State:</span>
              <motion.span
                key={phase}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                className={`font-medium ${
                  phase === "resolved"
                    ? "text-green-600 dark:text-green-400"
                    : phase === "awaiting"
                    ? "text-amber-600 dark:text-amber-400"
                    : "text-gray-400 dark:text-gray-500"
                }`}
              >
                {phase === "idle" ? "Not created" : phase === "awaiting" ? "Awaiting..." : "Resolved"}
              </motion.span>
            </div>
            <div className="flex items-center gap-2 text-sm">
              <span className="font-mono text-gray-500 dark:text-gray-400 w-20">result():</span>
              <AnimatePresence mode="wait">
                {result ? (
                  <motion.span
                    key="result"
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="font-mono text-green-600 dark:text-green-400"
                  >
                    &quot;{result}&quot;
                  </motion.span>
                ) : (
                  <motion.span
                    key="none"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="text-gray-400 dark:text-gray-500 italic"
                  >
                    {phase === "awaiting" ? "raises InvalidStateError" : "None (not set)"}
                  </motion.span>
                )}
              </AnimatePresence>
            </div>
            <div className="flex items-center gap-2 text-sm">
              <span className="font-mono text-gray-500 dark:text-gray-400 w-20">done():</span>
              <span className={phase === "resolved" ? "text-green-600 dark:text-green-400" : "text-gray-400 dark:text-gray-500"}>
                {phase === "resolved" ? "True" : "False"}
              </span>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-5 shadow-sm">
          <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-3 flex items-center gap-2">
            <Timer className="w-4 h-4" />
            Elapsed Time
          </h4>
          <div className="text-center">
            <motion.div
              className="text-4xl font-mono font-bold text-blue-600 dark:text-blue-400 mb-2"
              animate={phase === "awaiting" ? { opacity: [1, 0.5, 1] } : {}}
              transition={{ repeat: Infinity, duration: 1 }}
            >
              {(elapsed / 1000).toFixed(1)}s
            </motion.div>
            <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3 overflow-hidden">
              <motion.div
                className="h-full bg-blue-500 rounded-full"
                animate={{ width: phase === "resolved" ? "100%" : `${Math.min((elapsed / 2000) * 100, 95)}%` }}
                transition={{ duration: 0.1 }}
              />
            </div>
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">
              {phase === "resolved" ? "Result set after 2s delay" : phase === "awaiting" ? "Simulating asyncio.sleep(2)..." : "Press Start to create a Future"}
            </p>
          </div>
        </div>
      </div>

      <div className="flex gap-3 justify-center">
        {phase === "idle" ? (
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={startDemo}
            className="flex items-center gap-2 px-6 py-2 rounded-lg bg-blue-600 hover:bg-blue-700 text-white font-medium shadow transition-colors"
          >
            <Play className="w-4 h-4" />
            Create {'->'} await {'->'} set_result
          </motion.button>
        ) : (
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={reset}
            className="flex items-center gap-2 px-6 py-2 rounded-lg bg-gray-600 hover:bg-gray-700 text-white font-medium shadow transition-colors"
          >
            <RotateCcw className="w-4 h-4" />
            Reset
          </motion.button>
        )}
      </div>
    </div>
  );
}
