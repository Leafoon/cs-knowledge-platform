"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { AlertTriangle, Play, RotateCcw, Timer, Shield, ShieldAlert } from "lucide-react";

type Scenario = "fast" | "slow";

export function BlockingDetectionDemo() {
  const [scenario, setScenario] = useState<Scenario | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [warning, setWarning] = useState(false);
  const [done, setDone] = useState(false);
  const [blockedMs, setBlockedMs] = useState(0);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const THRESHOLD = 100;

  const startSim = useCallback((s: Scenario) => {
    setScenario(s); setElapsed(0); setWarning(false); setDone(false); setBlockedMs(0);
    const blockDuration = s === "slow" ? 500 : 30;
    let t = 0;
    const interval = setInterval(() => { t += 50; setElapsed(t); }, 50);
    timerRef.current = interval;
    setTimeout(() => {
      clearInterval(interval); setBlockedMs(blockDuration); setDone(true);
      if (blockDuration > THRESHOLD) setWarning(true);
    }, s === "slow" ? 1200 : 400);
  }, []);

  useEffect(() => () => { if (timerRef.current) clearInterval(timerRef.current); }, []);

  const reset = () => {
    if (timerRef.current) clearInterval(timerRef.current);
    setScenario(null); setElapsed(0); setWarning(false); setDone(false); setBlockedMs(0);
  };

  return (
    <div className="max-w-6xl mx-auto p-6">
      <h3 className="text-xl font-bold mb-2 text-gray-900 dark:text-gray-100 flex items-center gap-2">
        <ShieldAlert className="w-5 h-5" />
        Blocking Call Detection
      </h3>
      <p className="text-sm text-gray-600 dark:text-gray-400 mb-6">
        Simulate blocking calls in async context. Slow callbacks trigger warnings when they exceed the event loop&apos;s tolerance.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
        <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-5 shadow-sm">
          <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">Callback Simulation</h4>
          <div className="space-y-3">
            <div className="flex items-center gap-2 text-sm">
              <Timer className="w-4 h-4 text-gray-500" />
              <span className="text-gray-600 dark:text-gray-400">Elapsed:</span>
              <span className="font-mono text-gray-900 dark:text-gray-100">{elapsed}ms</span>
            </div>
            <div className="flex items-center gap-2 text-sm">
              <span className="text-gray-600 dark:text-gray-400 w-16">Block time:</span>
              <span className={`font-mono ${blockedMs > THRESHOLD ? "text-red-600 dark:text-red-400 font-bold" : "text-green-600 dark:text-green-400"}`}>
                {blockedMs}ms
              </span>
            </div>
            <div className="flex items-center gap-2 text-sm">
              <span className="text-gray-600 dark:text-gray-400 w-16">Threshold:</span>
              <span className="font-mono text-amber-600 dark:text-amber-400">{THRESHOLD}ms</span>
            </div>

            {done && (
              <motion.div initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }}
                className={`p-3 rounded-lg border ${warning ? "bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800" : "bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800"}`}>
                <div className="flex items-center gap-2 text-sm font-semibold">
                  {warning
                    ? <><AlertTriangle className="w-4 h-4 text-red-600" /><span className="text-red-700 dark:text-red-300">Blocking detected!</span></>
                    : <><Shield className="w-4 h-4 text-green-600" /><span className="text-green-700 dark:text-green-300">Callback OK</span></>}
                </div>
                <p className="text-xs mt-1 text-gray-600 dark:text-gray-400">
                  {warning ? `Blocked ${blockedMs}ms (threshold: ${THRESHOLD}ms). Tasks starved!` : `Took ${blockedMs}ms. Within range.`}
                </p>
              </motion.div>
            )}
          </div>
        </div>

        <div className="bg-gray-900 dark:bg-gray-950 rounded-xl p-5 shadow-sm">
          <h4 className="font-semibold text-gray-300 mb-3">Event Loop View</h4>
          <div className="space-y-2 text-xs font-mono min-h-[180px]">
            {scenario && (
              <>
                <div className="text-blue-400">loop.call_soon(callback)</div>
                <div className="text-gray-500">{'>'} Waiting for callback...</div>
                {elapsed > 0 && elapsed < (scenario === "slow" ? 1200 : 400) && (
                  <motion.div animate={{ opacity: [0.3, 1, 0.3] }} transition={{ repeat: Infinity, duration: 0.8 }}
                    className={scenario === "slow" ? "text-red-400" : "text-amber-400"}>
                    {'>'} Callback running ({elapsed}ms)...
                  </motion.div>
                )}
                {done && (
                  <>
                    <div className={warning ? "text-red-400" : "text-green-400"}>
                      {'>'} Callback returned ({blockedMs}ms)
                    </div>
                    {warning && (
                      <div className="text-red-400">{'>'} WARNING: Event loop blocked for {blockedMs}ms!</div>
                    )}
                    <div className="text-green-400">{'>'} Loop continues...</div>
                  </>
                )}
              </>
            )}
            {!scenario && <div className="text-gray-500">{'>'} Select a scenario to begin</div>}
          </div>
        </div>
      </div>

      <div className="flex flex-wrap gap-3 justify-center">
        {!scenario ? (
          <>
            <motion.button whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }} onClick={() => startSim("fast")}
              className="flex items-center gap-2 px-5 py-2 rounded-lg bg-green-600 hover:bg-green-700 text-white font-medium shadow transition-colors">
              <Shield className="w-4 h-4" /> Fast Callback (30ms)
            </motion.button>
            <motion.button whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }} onClick={() => startSim("slow")}
              className="flex items-center gap-2 px-5 py-2 rounded-lg bg-red-600 hover:bg-red-700 text-white font-medium shadow transition-colors">
              <AlertTriangle className="w-4 h-4" /> Slow Callback (500ms)
            </motion.button>
          </>
        ) : (
          <motion.button whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }} onClick={reset}
            className="flex items-center gap-2 px-6 py-2 rounded-lg bg-gray-600 hover:bg-gray-700 text-white font-medium shadow transition-colors">
            <RotateCcw className="w-4 h-4" /> Reset
          </motion.button>
        )}
      </div>
    </div>
  );
}
