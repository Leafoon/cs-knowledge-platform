"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Cpu, Users, Play, Timer, BarChart3 } from "lucide-react";

export default function ExecutorComparison() {
  const [taskCount, setTaskCount] = useState(10);
  const [results, setResults] = useState<{
    thread: number;
    process: number;
  } | null>(null);
  const [running, setRunning] = useState(false);

  const runComparison = async () => {
    setRunning(true);
    setResults(null);
    const threadTime = 50 + taskCount * 30 + Math.random() * 100;
    const processTime = 100 + taskCount * 15 + Math.random() * 80;
    await new Promise((r) => setTimeout(r, 1500));
    setResults({ thread: threadTime, process: processTime });
    setRunning(false);
  };

  const maxTime = results ? Math.max(results.thread, results.process) : 1;

  return (
    <div className="max-w-6xl mx-auto p-6 space-y-6">
      <h2 className="text-2xl font-bold dark:text-white flex items-center gap-2">
        <BarChart3 className="w-6 h-6" /> Executor Comparison
      </h2>
      <div className="flex items-center gap-4">
        <label className="dark:text-gray-300">Task Count:</label>
        <input
          type="range"
          min={5}
          max={50}
          value={taskCount}
          onChange={(e) => setTaskCount(+e.target.value)}
          className="w-48"
        />
        <span className="dark:text-white font-mono">{taskCount}</span>
        <button
          onClick={runComparison}
          disabled={running}
          className="flex items-center gap-2 px-4 py-2 bg-blue-500 text-white rounded-lg disabled:opacity-50 hover:bg-blue-600"
        >
          <Play className="w-4 h-4" /> Run
        </button>
      </div>
      {results && (
        <div className="grid md:grid-cols-2 gap-6">
          <div className="p-4 rounded-xl bg-blue-50 dark:bg-blue-900/20">
            <h3 className="font-semibold mb-4 dark:text-white flex items-center gap-2">
              <Users className="w-5 h-5 text-blue-500" /> ThreadPoolExecutor
            </h3>
            <div className="h-8 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden mb-2">
              <motion.div
                className="h-full bg-blue-500 flex items-center justify-end pr-2"
                initial={{ width: 0 }}
                animate={{ width: `${(results.thread / maxTime) * 100}%` }}
                transition={{ duration: 0.8 }}
              >
                <span className="text-white text-sm font-mono">
                  {results.thread.toFixed(0)}ms
                </span>
              </motion.div>
            </div>
            <p className="text-sm dark:text-gray-300">
              Good for I/O-bound tasks. GIL limits CPU parallelism.
            </p>
          </div>
          <div className="p-4 rounded-xl bg-green-50 dark:bg-green-900/20">
            <h3 className="font-semibold mb-4 dark:text-white flex items-center gap-2">
              <Cpu className="w-5 h-5 text-green-500" /> ProcessPoolExecutor
            </h3>
            <div className="h-8 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden mb-2">
              <motion.div
                className="h-full bg-green-500 flex items-center justify-end pr-2"
                initial={{ width: 0 }}
                animate={{ width: `${(results.process / maxTime) * 100}%` }}
                transition={{ duration: 0.8, delay: 0.2 }}
              >
                <span className="text-white text-sm font-mono">
                  {results.process.toFixed(0)}ms
                </span>
              </motion.div>
            </div>
            <p className="text-sm dark:text-gray-300">
              True parallelism for CPU-bound. Higher startup overhead.
            </p>
          </div>
        </div>
      )}
      {running && (
        <motion.div
          animate={{ opacity: [0.5, 1, 0.5] }}
          transition={{ repeat: Infinity, duration: 1.5 }}
          className="flex items-center gap-2 text-blue-500"
        >
          <Timer className="w-5 h-5 animate-spin" /> Running comparison...
        </motion.div>
      )}
      {!results && !running && (
        <div className="p-4 bg-gray-100 dark:bg-gray-800 rounded-lg">
          <h4 className="font-semibold dark:text-white mb-2">When to use:</h4>
          <ul className="text-sm dark:text-gray-300 space-y-1">
            <li className="flex items-center gap-2">
              <Users className="w-4 h-4 text-blue-500" /> Threads: I/O-bound,
              network calls, file operations
            </li>
            <li className="flex items-center gap-2">
              <Cpu className="w-4 h-4 text-green-500" /> Processes: CPU-bound,
              data processing, computation
            </li>
          </ul>
        </div>
      )}
    </div>
  );
}
