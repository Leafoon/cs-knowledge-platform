"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Play, Pause, RotateCcw, AlertTriangle, Lock, Unlock } from "lucide-react";

const CAPACITY = 6;

interface QueueItem { id: number; color: string; }

export function BackpressureDemo() {
  const [queue, setQueue] = useState<QueueItem[]>([]);
  const [producerBlocked, setProducerBlocked] = useState(false);
  const [consumerBlocked, setConsumerBlocked] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [produced, setProduced] = useState(0);
  const [consumed, setConsumed] = useState(0);
  const [log, setLog] = useState<string[]>([]);
  const nextId = useRef(0);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  const addLog = useCallback((msg: string) => {
    setLog((p) => [msg, ...p].slice(0, 12));
  }, []);

  const reset = useCallback(() => {
    setIsRunning(false);
    setQueue([]);
    setProducerBlocked(false);
    setConsumerBlocked(false);
    setProduced(0);
    setConsumed(0);
    setLog([]);
    nextId.current = 0;
  }, []);

  useEffect(() => {
    if (!isRunning) return;
    intervalRef.current = setInterval(() => {
      setQueue((prev) => {
        const q = [...prev];
        const shouldProduce = Math.random() < 0.6;
        const shouldConsume = Math.random() < 0.6;

        if (shouldConsume && q.length > 0) {
          q.shift();
          setConsumed((c) => c + 1);
          setConsumerBlocked(false);
          addLog("Consumer: dequeued item");
        } else if (shouldConsume && q.length === 0) {
          setConsumerBlocked(true);
          addLog("Consumer: BLOCKED (queue empty)");
        }

        if (shouldProduce) {
          if (q.length < CAPACITY) {
            const id = nextId.current++;
            const colors = ["bg-blue-400", "bg-purple-400", "bg-pink-400", "bg-cyan-400"];
            q.push({ id, color: colors[id % colors.length] });
            setProduced((p) => p + 1);
            setProducerBlocked(false);
            addLog(`Producer: enqueued item ${id}`);
          } else {
            setProducerBlocked(true);
            addLog("Producer: BLOCKED (queue full)");
          }
        }

        return q;
      });
    }, 700);
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [isRunning, addLog]);

  const fillPct = (queue.length / CAPACITY) * 100;

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-rose-50 dark:from-gray-900 dark:to-gray-800 rounded-xl shadow-lg">
      <h3 className="text-xl font-bold text-slate-800 dark:text-gray-100 mb-4 text-center">Backpressure Demo (Bounded Queue)</h3>

      <div className="flex justify-center gap-6 mb-4 text-sm">
        <span className="text-blue-600 dark:text-blue-400">Produced: {produced}</span>
        <span className="text-amber-600 dark:text-amber-400">Queue: {queue.length}/{CAPACITY}</span>
        <span className="text-emerald-600 dark:text-emerald-400">Consumed: {consumed}</span>
      </div>

      {/* Queue visualization */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-slate-200 dark:border-gray-700 mb-4">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-slate-700 dark:text-gray-200">Bounded Queue</span>
          <span className={`text-xs font-bold px-2 py-0.5 rounded ${fillPct >= 100 ? "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400" : fillPct > 60 ? "bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400" : "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400"}`}>
            {queue.length}/{CAPACITY}
          </span>
        </div>
        <div className="h-3 bg-slate-100 dark:bg-gray-700 rounded-full overflow-hidden mb-3">
          <motion.div animate={{ width: `${fillPct}%` }} className={`h-full rounded-full ${fillPct >= 100 ? "bg-red-500" : fillPct > 60 ? "bg-amber-500" : "bg-emerald-500"}`} />
        </div>
        <div className="flex gap-2 min-h-[3rem]">
          <AnimatePresence>
            {queue.map((item) => (
              <motion.div key={item.id} initial={{ scale: 0, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0, opacity: 0 }} className={`w-10 h-10 rounded-md ${item.color} flex items-center justify-center text-white text-xs font-bold shadow`}>
                {item.id}
              </motion.div>
            ))}
          </AnimatePresence>
          {queue.length === 0 && <span className="text-xs text-slate-400 dark:text-gray-500 self-center">Empty</span>}
        </div>
      </div>

      {/* Blocking status */}
      <div className="grid grid-cols-2 gap-4 mb-4">
        <motion.div animate={{ borderColor: producerBlocked ? "#ef4444" : "#e2e8f0" }} className={`rounded-lg p-3 border-2 ${producerBlocked ? "bg-red-50 dark:bg-red-900/20" : "bg-blue-50 dark:bg-blue-900/20"}`}>
          <div className="flex items-center gap-2">
            {producerBlocked ? <Lock className="w-4 h-4 text-red-500" /> : <Unlock className="w-4 h-4 text-blue-500" />}
            <span className="text-sm font-medium text-slate-700 dark:text-gray-200">Producer</span>
          </div>
          {producerBlocked && <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex items-center gap-1 mt-1"><AlertTriangle className="w-3 h-3 text-red-500" /><span className="text-xs text-red-600 dark:text-red-400">BLOCKED: Queue full</span></motion.div>}
        </motion.div>
        <motion.div animate={{ borderColor: consumerBlocked ? "#ef4444" : "#e2e8f0" }} className={`rounded-lg p-3 border-2 ${consumerBlocked ? "bg-red-50 dark:bg-red-900/20" : "bg-emerald-50 dark:bg-emerald-900/20"}`}>
          <div className="flex items-center gap-2">
            {consumerBlocked ? <Lock className="w-4 h-4 text-red-500" /> : <Unlock className="w-4 h-4 text-emerald-500" />}
            <span className="text-sm font-medium text-slate-700 dark:text-gray-200">Consumer</span>
          </div>
          {consumerBlocked && <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex items-center gap-1 mt-1"><AlertTriangle className="w-3 h-3 text-red-500" /><span className="text-xs text-red-600 dark:text-red-400">BLOCKED: Queue empty</span></motion.div>}
        </motion.div>
      </div>

      {/* Log */}
      <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-3 border border-slate-200 dark:border-gray-700 mb-4 max-h-28 overflow-y-auto">
        {log.length === 0 ? <p className="text-xs text-slate-400 text-center">Events appear here...</p> : log.map((l, i) => <div key={i} className="text-xs font-mono text-slate-600 dark:text-gray-300 py-0.5">{l}</div>)}
      </div>

      <div className="flex items-center justify-center gap-3">
        <button onClick={reset} className="p-2 rounded-lg bg-slate-200 hover:bg-slate-300 dark:bg-gray-700 dark:hover:bg-gray-600 text-slate-600 dark:text-gray-300"><RotateCcw className="w-5 h-5" /></button>
        <button onClick={() => setIsRunning(!isRunning)} className="px-5 py-2 rounded-lg bg-rose-500 hover:bg-rose-600 text-white font-medium flex items-center gap-2">
          {isRunning ? <><Pause className="w-4 h-4" /> Pause</> : <><Play className="w-4 h-4" /> Run</>}
        </button>
      </div>
    </div>
  );
}
