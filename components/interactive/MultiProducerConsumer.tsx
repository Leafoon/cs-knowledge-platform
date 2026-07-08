"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { motion } from "framer-motion";
import { Play, Pause, RotateCcw, Plus, Minus, BarChart3, Users, ShoppingCart } from "lucide-react";

interface Stats { produced: number; consumed: number; throughput: number; }

export function MultiProducerConsumer() {
  const [numProducers, setNumProducers] = useState(1);
  const [numConsumers, setNumConsumers] = useState(1);
  const [stats, setStats] = useState<Stats>({ produced: 0, consumed: 0, throughput: 0 });
  const [isRunning, setIsRunning] = useState(false);
  const [producerActivity, setProducerActivity] = useState<boolean[]>([false]);
  const [consumerActivity, setConsumerActivity] = useState<boolean[]>([false]);
  const [queueSize, setQueueSize] = useState(0);
  const queueRef = useRef(0);
  const totalProduced = useRef(0);
  const totalConsumed = useRef(0);
  const startTime = useRef(0);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  const reset = useCallback(() => {
    setIsRunning(false);
    setStats({ produced: 0, consumed: 0, throughput: 0 });
    queueRef.current = 0;
    setQueueSize(0);
    totalProduced.current = 0;
    totalConsumed.current = 0;
    setProducerActivity(Array(numProducers).fill(false));
    setConsumerActivity(Array(numConsumers).fill(false));
  }, [numProducers, numConsumers]);

  useEffect(() => {
    if (!isRunning) return;
    startTime.current = Date.now();
    intervalRef.current = setInterval(() => {
      const pa = Array(numProducers).fill(false);
      const ca = Array(numConsumers).fill(false);
      // Each producer may produce
      for (let i = 0; i < numProducers; i++) {
        if (Math.random() < 0.6) {
          queueRef.current++;
          totalProduced.current++;
          pa[i] = true;
        }
      }
      // Each consumer may consume
      for (let i = 0; i < numConsumers; i++) {
        if (queueRef.current > 0 && Math.random() < 0.7) {
          queueRef.current--;
          totalConsumed.current++;
          ca[i] = true;
        }
      }
      setQueueSize(queueRef.current);
      setProducerActivity([...pa]);
      setConsumerActivity([...ca]);
      const elapsed = (Date.now() - startTime.current) / 1000;
      setStats({
        produced: totalProduced.current,
        consumed: totalConsumed.current,
        throughput: elapsed > 0 ? Number((totalConsumed.current / elapsed).toFixed(1)) : 0,
      });
    }, 600);
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [isRunning, numProducers, numConsumers]);

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-violet-50 dark:from-gray-900 dark:to-gray-800 rounded-xl shadow-lg">
      <h3 className="text-xl font-bold text-slate-800 dark:text-gray-100 mb-4 text-center">Multi-Producer Consumer Simulation</h3>

      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-3 border border-blue-200 dark:border-blue-800">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-blue-700 dark:text-blue-300 flex items-center gap-1"><Users className="w-4 h-4" /> Producers</span>
            <div className="flex gap-1">
              <button onClick={() => setNumProducers(Math.max(1, numProducers - 1))} className="p-1 rounded bg-blue-200 dark:bg-blue-800"><Minus className="w-3 h-3" /></button>
              <span className="text-sm font-bold w-6 text-center">{numProducers}</span>
              <button onClick={() => setNumProducers(Math.min(3, numProducers + 1))} className="p-1 rounded bg-blue-200 dark:bg-blue-800"><Plus className="w-3 h-3" /></button>
            </div>
          </div>
          <div className="flex gap-2">
            {producerActivity.map((active, i) => (
              <motion.div key={i} animate={{ scale: active ? 1.2 : 1, backgroundColor: active ? "#3b82f6" : "#93c5fd" }} className="w-10 h-10 rounded-full flex items-center justify-center text-white text-xs font-bold">
                P{i}
              </motion.div>
            ))}
          </div>
        </div>
        <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-lg p-3 border border-emerald-200 dark:border-emerald-800">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-emerald-700 dark:text-emerald-300 flex items-center gap-1"><ShoppingCart className="w-4 h-4" /> Consumers</span>
            <div className="flex gap-1">
              <button onClick={() => setNumConsumers(Math.max(1, numConsumers - 1))} className="p-1 rounded bg-emerald-200 dark:bg-emerald-800"><Minus className="w-3 h-3" /></button>
              <span className="text-sm font-bold w-6 text-center">{numConsumers}</span>
              <button onClick={() => setNumConsumers(Math.min(3, numConsumers + 1))} className="p-1 rounded bg-emerald-200 dark:bg-emerald-800"><Plus className="w-3 h-3" /></button>
            </div>
          </div>
          <div className="flex gap-2">
            {consumerActivity.map((active, i) => (
              <motion.div key={i} animate={{ scale: active ? 1.2 : 1, backgroundColor: active ? "#10b981" : "#6ee7b7" }} className="w-10 h-10 rounded-full flex items-center justify-center text-white text-xs font-bold">
                C{i}
              </motion.div>
            ))}
          </div>
        </div>
      </div>

      <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-slate-200 dark:border-gray-700 mb-4">
        <div className="flex items-center gap-2 mb-2"><BarChart3 className="w-4 h-4 text-violet-500" /><span className="text-sm font-bold text-slate-700 dark:text-gray-200">Throughput Stats</span></div>
        <div className="grid grid-cols-3 gap-4 text-center">
          <div><div className="text-2xl font-bold text-blue-600 dark:text-blue-400">{stats.produced}</div><div className="text-xs text-slate-500">Produced</div></div>
          <div><div className="text-2xl font-bold text-amber-600 dark:text-amber-400">{queueSize}</div><div className="text-xs text-slate-500">In Queue</div></div>
          <div><div className="text-2xl font-bold text-emerald-600 dark:text-emerald-400">{stats.consumed}</div><div className="text-xs text-slate-500">Consumed</div></div>
        </div>
        <div className="mt-3 text-center text-sm text-violet-600 dark:text-violet-400 font-medium">Throughput: {stats.throughput} items/s</div>
      </div>

      <div className="flex items-center justify-center gap-3">
        <button onClick={reset} className="p-2 rounded-lg bg-slate-200 hover:bg-slate-300 dark:bg-gray-700 dark:hover:bg-gray-600 text-slate-600 dark:text-gray-300"><RotateCcw className="w-5 h-5" /></button>
        <button onClick={() => setIsRunning(!isRunning)} className="px-5 py-2 rounded-lg bg-violet-500 hover:bg-violet-600 text-white font-medium flex items-center gap-2">
          {isRunning ? <><Pause className="w-4 h-4" /> Pause</> : <><Play className="w-4 h-4" /> Run</>}
        </button>
      </div>
    </div>
  );
}
