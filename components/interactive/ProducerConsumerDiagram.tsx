"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Play, Pause, RotateCcw, Package, ArrowRight, Gauge } from "lucide-react";

interface Item {
  id: number;
  x: number;
  stage: "producer" | "queue" | "consumer";
}

export function ProducerConsumerDiagram() {
  const [items, setItems] = useState<Item[]>([]);
  const [queueSize, setQueueSize] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const [speed, setSpeed] = useState(800);
  const [produced, setProduced] = useState(0);
  const [consumed, setConsumed] = useState(0);
  const nextId = useRef(0);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  const reset = useCallback(() => {
    setIsRunning(false);
    setItems([]);
    setQueueSize(0);
    setProduced(0);
    setConsumed(0);
    nextId.current = 0;
  }, []);

  useEffect(() => {
    if (!isRunning) return;
    intervalRef.current = setInterval(() => {
      setItems((prev) => {
        const updated = prev.map((it) => ({ ...it }));
        // Move queue items forward
        const queueItems = updated.filter((i) => i.stage === "queue");
        if (queueItems.length > 0) {
          const oldest = queueItems[0];
          oldest.stage = "consumer";
          oldest.x = 85;
          setConsumed((c) => c + 1);
          setQueueSize((q) => Math.max(0, q - 1));
          setTimeout(() => {
            setItems((p) => p.filter((i) => i.id !== oldest.id));
          }, 300);
        }
        // Produce new item
        const id = nextId.current++;
        updated.push({ id, x: 5, stage: "producer" });
        setProduced((p) => p + 1);
        setTimeout(() => {
          setItems((p) =>
            p.map((i) => (i.id === id ? { ...i, stage: "queue" as const, x: 40 + Math.random() * 20 } : i))
          );
          setQueueSize((q) => q + 1);
        }, 400);
        return updated;
      });
    }, speed);
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [isRunning, speed]);

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-indigo-50 dark:from-gray-900 dark:to-gray-800 rounded-xl shadow-lg">
      <h3 className="text-xl font-bold text-slate-800 dark:text-gray-100 mb-4 text-center">
        Producer {'->'} Queue {'->'} Consumer
      </h3>

      <div className="flex justify-center gap-6 mb-4 text-sm">
        <span className="text-blue-600 dark:text-blue-400 font-medium">Produced: {produced}</span>
        <span className="text-amber-600 dark:text-amber-400 font-medium">Queue: {queueSize}</span>
        <span className="text-emerald-600 dark:text-emerald-400 font-medium">Consumed: {consumed}</span>
      </div>

      <div className="relative h-48 bg-white dark:bg-gray-800 rounded-lg border border-slate-200 dark:border-gray-700 mb-4 overflow-hidden">
        {/* Producer zone */}
        <div className="absolute left-4 top-1/2 -translate-y-1/2 flex flex-col items-center">
          <Package className="w-8 h-8 text-blue-500" />
          <span className="text-xs text-slate-500 dark:text-gray-400 mt-1">Producer</span>
        </div>
        {/* Arrow */}
        <div className="absolute left-[15%] top-1/2 -translate-y-1/2">
          <ArrowRight className="w-5 h-5 text-slate-300 dark:text-gray-600" />
        </div>
        {/* Queue zone */}
        <div className="absolute left-[20%] right-[20%] top-4 bottom-4 border-2 border-dashed border-amber-300 dark:border-amber-700 rounded-lg flex items-center justify-center">
          <span className="text-xs text-amber-500 dark:text-amber-400 font-medium">Queue</span>
        </div>
        <div className="absolute right-[15%] top-1/2 -translate-y-1/2">
          <ArrowRight className="w-5 h-5 text-slate-300 dark:text-gray-600" />
        </div>
        {/* Consumer zone */}
        <div className="absolute right-4 top-1/2 -translate-y-1/2 flex flex-col items-center">
          <Package className="w-8 h-8 text-emerald-500" />
          <span className="text-xs text-slate-500 dark:text-gray-400 mt-1">Consumer</span>
        </div>

        <AnimatePresence>
          {items.map((item) => (
            <motion.div
              key={item.id}
              initial={{ opacity: 0, scale: 0 }}
              animate={{ opacity: 1, scale: 1, x: `${item.x}%` }}
              exit={{ opacity: 0, scale: 0 }}
              transition={{ type: "spring", stiffness: 120 }}
              className="absolute top-1/2 -translate-y-1/2 w-8 h-8 rounded-md bg-gradient-to-br from-blue-400 to-indigo-500 flex items-center justify-center text-white text-xs font-bold shadow"
            >
              {item.id}
            </motion.div>
          ))}
        </AnimatePresence>
      </div>

      <div className="flex items-center justify-center gap-3">
        <button onClick={reset} className="p-2 rounded-lg bg-slate-200 hover:bg-slate-300 dark:bg-gray-700 dark:hover:bg-gray-600 text-slate-600 dark:text-gray-300">
          <RotateCcw className="w-5 h-5" />
        </button>
        <button
          onClick={() => setIsRunning(!isRunning)}
          className="px-5 py-2 rounded-lg bg-indigo-500 hover:bg-indigo-600 text-white font-medium flex items-center gap-2"
        >
          {isRunning ? <><Pause className="w-4 h-4" /> Pause</> : <><Play className="w-4 h-4" /> Run</>}
        </button>
        <div className="ml-4 flex items-center gap-2">
          <Gauge className="w-4 h-4 text-slate-400" />
          <select value={speed} onChange={(e) => setSpeed(Number(e.target.value))} className="text-xs bg-white dark:bg-gray-700 border border-slate-300 dark:border-gray-600 rounded px-2 py-1">
            <option value={1500}>Slow</option>
            <option value={800}>Normal</option>
            <option value={400}>Fast</option>
          </select>
        </div>
      </div>
    </div>
  );
}
