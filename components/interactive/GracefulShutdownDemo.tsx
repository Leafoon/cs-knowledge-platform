"use client";

import { useState, useCallback, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Play, RotateCcw, Skull, CheckCircle, Package, ShieldCheck } from "lucide-react";

type Phase = "idle" | "producing" | "sentinel" | "draining" | "shutdown";

interface QueueItem { id: number; isSentinel: boolean; }

export function GracefulShutdownDemo() {
  const [phase, setPhase] = useState<Phase>("idle");
  const [queue, setQueue] = useState<QueueItem[]>([]);
  const [consumed, setConsumed] = useState<number[]>([]);
  const [log, setLog] = useState<string[]>([]);
  const timeouts = useRef<NodeJS.Timeout[]>([]);

  const addLog = useCallback((msg: string) => {
    setLog((p) => [msg, ...p].slice(0, 15));
  }, []);

  const cleanup = useCallback(() => {
    timeouts.current.forEach(clearTimeout);
    timeouts.current = [];
  }, []);

  const reset = useCallback(() => {
    cleanup();
    setPhase("idle");
    setQueue([]);
    setConsumed([]);
    setLog([]);
  }, [cleanup]);

  const runDemo = useCallback(() => {
    cleanup();
    setPhase("producing");
    setQueue([]);
    setConsumed([]);
    setLog([]);
    addLog("Producer: starting to produce items...");

    // Produce items over time
    const items: QueueItem[] = [];
    for (let i = 0; i < 5; i++) {
      const t = setTimeout(() => {
        const item = { id: i, isSentinel: false };
        items.push(item);
        setQueue([...items]);
        addLog(`Producer: produced item ${i}`);
      }, i * 600);
      timeouts.current.push(t);
    }

    // Send sentinel (poison pill)
    const sentinelT = setTimeout(() => {
      setPhase("sentinel");
      const sentinel = { id: -1, isSentinel: true };
      items.push(sentinel);
      setQueue([...items]);
      addLog("Producer: SENTINEL (poison pill) sent!");
    }, 3500);
    timeouts.current.push(sentinelT);

    // Consumer processes items
    const consumeStart = 3800;
    for (let i = 0; i < items.length + 1; i++) {
      const t = setTimeout(() => {
        setQueue((prev) => {
          if (prev.length === 0) return prev;
          const [first, ...rest] = prev;
          if (first.isSentinel) {
            setPhase("draining");
            addLog("Consumer: received sentinel, draining remaining...");
            // Process remaining items
            setTimeout(() => {
              setPhase("shutdown");
              addLog("Consumer: all items processed. SHUTDOWN complete.");
              setQueue([]);
            }, 1500);
            return rest;
          }
          setConsumed((c) => [...c, first.id]);
          addLog(`Consumer: consumed item ${first.id}`);
          return rest;
        });
      }, consumeStart + i * 500);
      timeouts.current.push(t);
    }
  }, [addLog, cleanup]);

  const phaseBg = phase === "producing" ? "bg-blue-50 dark:bg-blue-900/20" : phase === "sentinel" ? "bg-amber-50 dark:bg-amber-900/20" : phase === "draining" ? "bg-purple-50 dark:bg-purple-900/20" : phase === "shutdown" ? "bg-emerald-50 dark:bg-emerald-900/20" : "bg-slate-100 dark:bg-gray-800";

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-teal-50 dark:from-gray-900 dark:to-gray-800 rounded-xl shadow-lg">
      <h3 className="text-xl font-bold text-slate-800 dark:text-gray-100 mb-4 text-center">Graceful Shutdown {'&'} Sentinel Pattern</h3>
      {/* Phase indicator */}
      <div className="flex justify-center gap-2 mb-4">
        {(["producing", "sentinel", "draining", "shutdown"] as Phase[]).map((p) => (
          <motion.div key={p} animate={{ opacity: phase === p ? 1 : 0.3 }} className={`px-3 py-1 rounded-full text-xs font-bold ${phase === p ? "bg-teal-500 text-white" : "bg-slate-200 dark:bg-gray-700 text-slate-500"}`}>
            {p}
          </motion.div>
        ))}
      </div>
      {/* Queue */}
      <div className={`${phaseBg} rounded-lg p-4 border border-slate-200 dark:border-gray-700 mb-4 transition-colors`}>
        <div className="flex items-center gap-2 mb-2"><Package className="w-4 h-4 text-slate-500" /><span className="text-sm font-medium text-slate-700 dark:text-gray-200">Queue</span></div>
        <div className="flex gap-2 min-h-[3rem] flex-wrap">

          <AnimatePresence>
            {queue.map((item) => (
              <motion.div key={item.id + (item.isSentinel ? "s" : "")} initial={{ scale: 0 }} animate={{ scale: 1 }} exit={{ scale: 0 }}
                className={`w-12 h-12 rounded-lg flex items-center justify-center text-white text-xs font-bold shadow ${item.isSentinel ? "bg-red-500" : "bg-blue-400"}`}>
                {item.isSentinel ? <Skull className="w-5 h-5" /> : item.id}
              </motion.div>
            ))}
          </AnimatePresence>
          {queue.length === 0 && <span className="text-xs text-slate-400 self-center">Empty</span>}
        </div>
      </div>

      {/* Consumed items */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-slate-200 dark:border-gray-700 mb-4">
        <div className="flex items-center gap-2 mb-2"><CheckCircle className="w-4 h-4 text-emerald-500" /><span className="text-sm font-medium text-slate-700 dark:text-gray-200">Consumed: {consumed.length}</span></div>
        <div className="flex gap-1 flex-wrap">
          {consumed.map((id, i) => <motion.div key={i} initial={{ scale: 0 }} animate={{ scale: 1 }} className="w-8 h-8 rounded bg-emerald-100 dark:bg-emerald-900/30 flex items-center justify-center text-emerald-700 dark:text-emerald-300 text-xs font-bold">{id}</motion.div>)}
        </div>
      </div>
      {phase === "shutdown" && (
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="bg-emerald-100 dark:bg-emerald-900/30 rounded-lg p-3 border border-emerald-300 dark:border-emerald-700 mb-4 flex items-center gap-2">
          <ShieldCheck className="w-5 h-5 text-emerald-600 dark:text-emerald-400" />
          <span className="text-sm font-medium text-emerald-700 dark:text-emerald-300">Consumer exited gracefully. No data lost.</span>
        </motion.div>
      )}

      {/* Log */}
      <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-3 border border-slate-200 dark:border-gray-700 mb-4 max-h-28 overflow-y-auto">
        {log.length === 0 ? <p className="text-xs text-slate-400 text-center">Run the demo to see events...</p> : log.map((l, i) => <div key={i} className="text-xs font-mono text-slate-600 dark:text-gray-300 py-0.5">{l}</div>)}
      </div>

      <div className="flex items-center justify-center gap-3">
        <button onClick={reset} className="p-2 rounded-lg bg-slate-200 hover:bg-slate-300 dark:bg-gray-700 dark:hover:bg-gray-600 text-slate-600 dark:text-gray-300"><RotateCcw className="w-5 h-5" /></button>
        <button onClick={runDemo} disabled={phase !== "idle"} className="px-5 py-2 rounded-lg bg-teal-500 hover:bg-teal-600 text-white font-medium flex items-center gap-2 disabled:opacity-40">
          <Play className="w-4 h-4" /> Run Demo
        </button>
      </div>
    </div>
  );
}
