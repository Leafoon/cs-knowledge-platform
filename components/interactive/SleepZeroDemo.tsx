"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Play, RotateCcw, Zap, Clock } from "lucide-react";

interface LogLine {
  text: string;
  side: "left" | "right";
  color: string;
}

export function SleepZeroDemo() {
  const [running, setRunning] = useState(false);
  const [logs, setLogs] = useState<LogLine[]>([]);
  const [phase, setPhase] = useState<"idle" | "done">("idle");

  const run = () => {
    setRunning(true);
    setLogs([]);
    setPhase("idle");

    const sequence: LogLine[] = [
      { text: "# Without sleep(0)", side: "left", color: "text-gray-400" },
      { text: "# With sleep(0)", side: "right", color: "text-gray-400" },
      { text: "main() starts", side: "left", color: "text-blue-400" },
      { text: "main() starts", side: "right", color: "text-blue-400" },
      { text: "create_task(task_1)", side: "left", color: "text-purple-400" },
      { text: "create_task(task_1)", side: "right", color: "text-purple-400" },
      { text: "create_task(task_2)", side: "left", color: "text-purple-400" },
      { text: "create_task(task_2)", side: "right", color: "text-purple-400" },
      { text: "main() does work...", side: "left", color: "text-blue-400" },
      { text: "await sleep(0) -- yield!", side: "right", color: "text-amber-400" },
      { text: "main() does work...", side: "left", color: "text-blue-400" },
      { text: "task_1 starts running", side: "right", color: "text-green-400" },
      { text: "main() finishes", side: "left", color: "text-blue-400" },
      { text: "task_2 starts running", side: "right", color: "text-green-400" },
      { text: "main() awaits tasks", side: "left", color: "text-blue-400" },
      { text: "task_1 done", side: "right", color: "text-green-400" },
      { text: "task_1 starts NOW", side: "left", color: "text-green-400" },
      { text: "task_2 done", side: "right", color: "text-green-400" },
      { text: "task_2 starts NOW", side: "left", color: "text-green-400" },
      { text: "Tasks ran AFTER main", side: "left", color: "text-red-400" },
      { text: "Tasks ran DURING main", side: "right", color: "text-green-400" },
    ];

    sequence.forEach((entry, i) => {
      setTimeout(() => {
        setLogs((prev) => [...prev, entry]);
        if (i === sequence.length - 1) {
          setRunning(false);
          setPhase("done");
        }
      }, i * 500);
    });
  };

  const reset = () => {
    setRunning(false);
    setLogs([]);
    setPhase("idle");
  };

  return (
    <div className="max-w-6xl mx-auto p-6">
      <h3 className="text-xl font-bold mb-2 text-gray-900 dark:text-gray-100">
        sleep(0) vs No sleep(0)
      </h3>
      <p className="text-sm text-gray-600 dark:text-gray-400 mb-6">
        Compare how tasks execute with and without yielding control via asyncio.sleep(0).
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        <div className="bg-gray-900 dark:bg-gray-950 rounded-xl p-5 shadow-sm">
          <h4 className="font-semibold text-red-400 mb-3 flex items-center gap-2 text-sm">
            <Clock className="w-4 h-4" /> Without sleep(0)
          </h4>
          <div className="space-y-1 min-h-[200px] font-mono text-xs">
            <AnimatePresence>
              {logs.filter((l) => l.side === "left").map((l, i) => (
                <motion.div key={i} initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} className={l.color}>
                  {l.text}
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        </div>

        <div className="bg-gray-900 dark:bg-gray-950 rounded-xl p-5 shadow-sm">
          <h4 className="font-semibold text-green-400 mb-3 flex items-center gap-2 text-sm">
            <Zap className="w-4 h-4" /> With sleep(0)
          </h4>
          <div className="space-y-1 min-h-[200px] font-mono text-xs">
            <AnimatePresence>
              {logs.filter((l) => l.side === "right").map((l, i) => (
                <motion.div key={i} initial={{ opacity: 0, x: 10 }} animate={{ opacity: 1, x: 0 }} className={l.color}>
                  {l.text}
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        </div>
      </div>

      {phase === "done" && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}
          className="bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-lg p-4 mb-4 text-sm text-amber-700 dark:text-amber-300">
          <strong>Key insight:</strong> Without sleep(0), tasks don&apos;t start until the main coroutine awaits them.
          With sleep(0), you explicitly yield control, allowing other scheduled tasks to run.
        </motion.div>
      )}

      <div className="flex gap-3 justify-center">
        {!running ? (
          <motion.button whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }} onClick={run}
            className="flex items-center gap-2 px-6 py-2 rounded-lg bg-blue-600 hover:bg-blue-700 text-white font-medium shadow transition-colors">
            <Play className="w-4 h-4" /> Run Comparison
          </motion.button>
        ) : (
          <div className="text-sm text-gray-500 dark:text-gray-400 animate-pulse">Simulating...</div>
        )}
        {phase === "done" && (
          <motion.button whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }} onClick={reset}
            className="flex items-center gap-2 px-6 py-2 rounded-lg bg-gray-600 hover:bg-gray-700 text-white font-medium shadow transition-colors">
            <RotateCcw className="w-4 h-4" /> Reset
          </motion.button>
        )}
      </div>
    </div>
  );
}
