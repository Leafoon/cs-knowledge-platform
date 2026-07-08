"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Play, Pause, RotateCcw, Briefcase, Users, Scale } from "lucide-react";

const NUM_WORKERS = 3;

interface Task { id: number; weight: number; }
interface WorkerState { id: number; tasks: number; active: boolean; current: Task | null; }

export function WorkQueueVisualizer() {
  const [queue, setQueue] = useState<Task[]>([]);
  const [workers, setWorkers] = useState<WorkerState[]>(
    Array.from({ length: NUM_WORKERS }, (_, i) => ({ id: i, tasks: 0, active: false, current: null }))
  );
  const [isRunning, setIsRunning] = useState(false);
  const [totalTasks, setTotalTasks] = useState(0);
  const [completedTasks, setCompletedTasks] = useState(0);
  const nextId = useRef(0);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  const reset = useCallback(() => {
    setIsRunning(false);
    setQueue([]);
    setWorkers(Array.from({ length: NUM_WORKERS }, (_, i) => ({ id: i, tasks: 0, active: false, current: null })));
    setTotalTasks(0);
    setCompletedTasks(0);
    nextId.current = 0;
  }, []);

  useEffect(() => {
    if (!isRunning) return;
    intervalRef.current = setInterval(() => {
      // Add tasks randomly
      if (Math.random() < 0.5) {
        const id = nextId.current++;
        const weight = Math.floor(Math.random() * 3) + 1;
        setQueue((p) => [...p, { id, weight }]);
        setTotalTasks((t) => t + 1);
      }

      // Workers pick tasks (least-loaded first)
      setWorkers((prev) => {
        const w = prev.map((w) => ({ ...w }));
        for (const worker of w) {
          if (!worker.active && queue.length > 0) {
            // Find least loaded worker
            const leastLoaded = w.reduce((min, cur) => cur.tasks < min.tasks ? cur : min, w[0]);
            if (leastLoaded.id === worker.id) {
              setQueue((q) => {
                if (q.length === 0) return q;
                const [task, ...rest] = q;
                worker.current = task;
                worker.active = true;
                return rest;
              });
            }
          }
          // Simulate processing
          if (worker.active && worker.current) {
            if (Math.random() < 0.3) {
              worker.active = false;
              worker.tasks++;
              setCompletedTasks((c) => c + 1);
              worker.current = null;
            }
          }
        }
        return w;
      });
    }, 500);
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [isRunning, queue.length]);

  const maxTasks = Math.max(...workers.map((w) => w.tasks), 1);

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-cyan-50 dark:from-gray-900 dark:to-gray-800 rounded-xl shadow-lg">
      <h3 className="text-xl font-bold text-slate-800 dark:text-gray-100 mb-4 text-center">Work Queue {'&'} Load Balancing</h3>

      <div className="flex justify-center gap-6 mb-4 text-sm">
        <span className="text-cyan-600 dark:text-cyan-400">Queued: {queue.length}</span>
        <span className="text-amber-600 dark:text-amber-400">Total: {totalTasks}</span>
        <span className="text-emerald-600 dark:text-emerald-400">Done: {completedTasks}</span>
      </div>

      {/* Task queue */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-slate-200 dark:border-gray-700 mb-4">
        <div className="flex items-center gap-2 mb-2"><Briefcase className="w-4 h-4 text-cyan-500" /><span className="text-sm font-bold text-slate-700 dark:text-gray-200">Task Queue ({queue.length})</span></div>
        <div className="flex gap-1 min-h-[2.5rem] flex-wrap">
          <AnimatePresence>
            {queue.slice(0, 20).map((t) => (
              <motion.div key={t.id} initial={{ scale: 0 }} animate={{ scale: 1 }} exit={{ scale: 0 }}
                className="w-8 h-8 rounded bg-cyan-100 dark:bg-cyan-900/30 border border-cyan-300 dark:border-cyan-700 flex items-center justify-center text-xs font-bold text-cyan-700 dark:text-cyan-300">
                {t.id}
              </motion.div>
            ))}
          </AnimatePresence>
          {queue.length > 20 && <span className="text-xs text-slate-400 self-center">+{queue.length - 20} more</span>}
        </div>
      </div>

      {/* Workers */}
      <div className="grid grid-cols-3 gap-3 mb-4">
        {workers.map((w) => (
          <div key={w.id} className={`rounded-lg p-3 border-2 ${w.active ? "border-cyan-400 bg-cyan-50 dark:bg-cyan-900/20" : "border-slate-200 dark:border-gray-700 bg-white dark:bg-gray-800"}`}>
            <div className="flex items-center gap-2 mb-2">
              <motion.div animate={{ rotate: w.active ? 360 : 0 }} transition={{ repeat: w.active ? Infinity : 0, duration: 1.5 }}>
                <Users className={`w-4 h-4 ${w.active ? "text-cyan-500" : "text-slate-400"}`} />
              </motion.div>
              <span className="text-sm font-bold text-slate-700 dark:text-gray-200">Worker {w.id}</span>
            </div>
            <div className="text-xs text-slate-500 dark:text-gray-400">Completed: {w.tasks}</div>
            <div className="mt-2 h-2 bg-slate-100 dark:bg-gray-700 rounded-full overflow-hidden">
              <motion.div animate={{ width: `${(w.tasks / maxTasks) * 100}%` }} className="h-full bg-cyan-500 rounded-full" />
            </div>
            {w.current && <div className="mt-1 text-xs text-cyan-600 dark:text-cyan-400">Processing: #{w.current.id}</div>}
          </div>
        ))}
      </div>

      {/* Load balance indicator */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-3 border border-slate-200 dark:border-gray-700 mb-4 flex items-center gap-2">
        <Scale className="w-4 h-4 text-amber-500" />
        <span className="text-sm text-slate-600 dark:text-gray-300">
          Load variance: {Math.round(Math.sqrt(workers.reduce((s, w) => s + (w.tasks - workers.reduce((a, b) => a + b.tasks, 0) / NUM_WORKERS) ** 2, 0) / NUM_WORKERS))}
        </span>
      </div>

      <div className="flex items-center justify-center gap-3">
        <button onClick={reset} className="p-2 rounded-lg bg-slate-200 hover:bg-slate-300 dark:bg-gray-700 dark:hover:bg-gray-600 text-slate-600 dark:text-gray-300"><RotateCcw className="w-5 h-5" /></button>
        <button onClick={() => setIsRunning(!isRunning)} className="px-5 py-2 rounded-lg bg-cyan-500 hover:bg-cyan-600 text-white font-medium flex items-center gap-2">
          {isRunning ? <><Pause className="w-4 h-4" /> Pause</> : <><Play className="w-4 h-4" /> Run</>}
        </button>
      </div>
    </div>
  );
}
