"use client";

import { useState, useRef, useCallback } from "react";
import { motion } from "framer-motion";
import { Play, RotateCcw, Lock, Unlock, AlertTriangle, CheckCircle, Shield } from "lucide-react";

export function RaceConditionDemo() {
  const [counter, setCounter] = useState(0);
  const [expected, setExpected] = useState(0);
  const [useLock, setUseLock] = useState(false);
  const [tasks, setTasks] = useState<{ id: number; result: number }[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [log, setLog] = useState<string[]>([]);
  const counterRef = useRef(0);

  const addLog = useCallback((msg: string) => {
    setLog((p) => [msg, ...p].slice(0, 15));
  }, []);

  const reset = useCallback(() => {
    setCounter(0);
    setExpected(0);
    setTasks([]);
    setLog([]);
    counterRef.current = 0;
    setIsRunning(false);
  }, []);

  const runSimulation = useCallback(async () => {
    setIsRunning(true);
    setLog([]);
    counterRef.current = 0;
    setCounter(0);
    setExpected(0);
    setTasks([]);

    const numTasks = 10;
    const increments = 5;
    const lockRef = { held: false };
    const expectedVal = numTasks * increments;
    setExpected(expectedVal);
    addLog(`Starting ${numTasks} tasks, each incrementing ${increments} times`);

    const doIncrement = async (taskId: number) => {
      for (let i = 0; i < increments; i++) {
        if (useLock) {
          // Simulate lock: wait until lock is free
          while (lockRef.held) {
            await new Promise((r) => setTimeout(r, 1));
          }
          lockRef.held = true;
        }
        // Read-modify-write (simulated race condition)
        const current = counterRef.current;
        await new Promise((r) => setTimeout(r, Math.random() * 5));
        counterRef.current = current + 1;
        setCounter(counterRef.current);

        if (useLock) {
          lockRef.held = false;
        }
      }
      setTasks((p) => [...p, { id: taskId, result: counterRef.current }]);
    };

    // Run all tasks concurrently
    await Promise.all(Array.from({ length: numTasks }, (_, i) => doIncrement(i)));

    addLog(`Final counter: ${counterRef.current}, Expected: ${expectedVal}`);
    addLog(counterRef.current === expectedVal ? "No race condition detected!" : "RACE CONDITION detected! Values lost.");
    setIsRunning(false);
  }, [useLock, addLog]);

  const hasRace = counter > 0 && counter !== expected;

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-yellow-50 dark:from-gray-900 dark:to-gray-800 rounded-xl shadow-lg">
      <h3 className="text-xl font-bold text-slate-800 dark:text-gray-100 mb-4 text-center">Race Condition Demo</h3>

      {/* Lock toggle */}
      <div className="flex items-center justify-center gap-4 mb-4">
        <button onClick={() => setUseLock(!useLock)}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg border-2 ${useLock ? "border-emerald-400 bg-emerald-50 dark:bg-emerald-900/20" : "border-red-400 bg-red-50 dark:bg-red-900/20"}`}>
          {useLock ? <Lock className="w-4 h-4 text-emerald-600" /> : <Unlock className="w-4 h-4 text-red-600" />}
          <span className={`text-sm font-bold ${useLock ? "text-emerald-700 dark:text-emerald-300" : "text-red-700 dark:text-red-300"}`}>
            {useLock ? "Lock ON" : "Lock OFF"}
          </span>
        </button>
      </div>

      {/* Counter display */}
      <div className="grid grid-cols-2 gap-4 mb-4">
        <motion.div animate={{ borderColor: hasRace && !useLock ? "#ef4444" : "#e2e8f0" }}
          className="bg-white dark:bg-gray-800 rounded-lg p-4 border-2 text-center">
          <div className="text-xs text-slate-500 dark:text-gray-400 mb-1">Actual Counter</div>
          <motion.div key={counter} initial={{ scale: 1.3 }} animate={{ scale: 1 }}
            className={`text-4xl font-bold ${hasRace && !useLock ? "text-red-600 dark:text-red-400" : "text-blue-600 dark:text-blue-400"}`}>
            {counter}
          </motion.div>
        </motion.div>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-slate-200 dark:border-gray-700 text-center">
          <div className="text-xs text-slate-500 dark:text-gray-400 mb-1">Expected</div>
          <div className="text-4xl font-bold text-emerald-600 dark:text-emerald-400">{expected}</div>
        </div>
      </div>

      {/* Status */}
      {counter > 0 && !isRunning && (
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
          className={`rounded-lg p-3 border mb-4 flex items-center gap-2 ${hasRace && !useLock ? "bg-red-50 dark:bg-red-900/20 border-red-300 dark:border-red-700" : "bg-emerald-50 dark:bg-emerald-900/20 border-emerald-300 dark:border-emerald-700"}`}>
          {hasRace && !useLock ? <AlertTriangle className="w-4 h-4 text-red-500" /> : <CheckCircle className="w-4 h-4 text-emerald-500" />}
          <span className={`text-sm font-medium ${hasRace && !useLock ? "text-red-700 dark:text-red-300" : "text-emerald-700 dark:text-emerald-300"}`}>
            {hasRace && !useLock ? `Race condition! Lost ${expected - counter} increments (${((1 - counter / expected) * 100).toFixed(1)}% loss)` : "All increments accounted for. No race condition."}
          </span>
        </motion.div>
      )}

      {/* Code comparison */}
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-3 border border-red-200 dark:border-red-800">
          <div className="text-xs font-bold text-red-600 dark:text-red-400 mb-1">Without Lock</div>
          <pre className="text-xs font-mono text-red-700 dark:text-red-300 whitespace-pre-wrap">{"val = counter\n# context switch!\ncounter = val + 1"}</pre>
        </div>
        <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-lg p-3 border border-emerald-200 dark:border-emerald-800">
          <div className="text-xs font-bold text-emerald-600 dark:text-emerald-400 mb-1 flex items-center gap-1"><Shield className="w-3 h-3" /> With Lock</div>
          <pre className="text-xs font-mono text-emerald-700 dark:text-emerald-300 whitespace-pre-wrap">{"async with lock:\n    counter += 1\n    # atomic!"}</pre>
        </div>
      </div>

      {/* Log */}
      <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-3 border border-slate-200 dark:border-gray-700 mb-4 max-h-24 overflow-y-auto">
        {log.length === 0 ? <p className="text-xs text-slate-400 text-center">Run to see results...</p> : log.map((l, i) => <div key={i} className="text-xs font-mono text-slate-600 dark:text-gray-300 py-0.5">{l}</div>)}
      </div>

      <div className="flex items-center justify-center gap-3">
        <button onClick={reset} className="p-2 rounded-lg bg-slate-200 hover:bg-slate-300 dark:bg-gray-700 dark:hover:bg-gray-600 text-slate-600 dark:text-gray-300"><RotateCcw className="w-5 h-5" /></button>
        <button onClick={runSimulation} disabled={isRunning} className="px-5 py-2 rounded-lg bg-yellow-500 hover:bg-yellow-600 text-white font-medium flex items-center gap-2 disabled:opacity-40">
          <Play className="w-4 h-4" /> {isRunning ? "Running..." : "Run"}
        </button>
      </div>
    </div>
  );
}
