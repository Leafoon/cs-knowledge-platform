"use client";

import { useState, useCallback, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { AlertTriangle, CheckCircle, Trash2, Wrench, Play, RotateCcw, Zap } from "lucide-react";

interface LeakedTask {
  id: number;
  name: string;
  awaited: boolean;
}

export function MemoryLeakDetector() {
  const [tasks, setTasks] = useState<LeakedTask[]>([]);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [fixed, setFixed] = useState(false);
  const [stats, setStats] = useState({ total: 0, leaked: 0, fixed: 0 });
  const nextId = useRef(0);

  const addWarning = useCallback((msg: string) => {
    setWarnings((p) => [msg, ...p].slice(0, 15));
  }, []);

  const createLeakedTasks = useCallback(() => {
    const count = 5;
    const newTasks: LeakedTask[] = [];
    for (let i = 0; i < count; i++) {
      const id = nextId.current++;
      newTasks.push({ id, name: `task_${id}`, awaited: false });
    }
    setTasks((p) => [...p, ...newTasks]);
    setStats((s) => ({ ...s, total: s.total + count, leaked: s.leaked + count }));
    addWarning(`Created ${count} tasks WITHOUT awaiting (memory leak!)`);
    for (const t of newTasks) {
      addWarning(`WARNING: coroutine ${t.name}() was never awaited`);
    }
    setFixed(false);
  }, [addWarning]);

  const fixTasks = useCallback(() => {
    setTasks((p) => p.map((t) => ({ ...t, awaited: true })));
    setStats((s) => ({ ...s, leaked: 0, fixed: s.fixed + tasks.filter((t) => !t.awaited).length }));
    addWarning("FIX: All tasks now properly awaited. Memory reclaimed.");
    setFixed(true);
  }, [tasks, addWarning]);

  const clearAll = useCallback(() => {
    setTasks([]);
    setWarnings([]);
    setStats({ total: 0, leaked: 0, fixed: 0 });
    setFixed(false);
    nextId.current = 0;
  }, []);

  const leakedCount = tasks.filter((t) => !t.awaited).length;

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-purple-50 dark:from-gray-900 dark:to-gray-800 rounded-xl shadow-lg">
      <h3 className="text-xl font-bold text-slate-800 dark:text-gray-100 mb-4 text-center flex items-center justify-center gap-2">
        <Zap className="w-5 h-5 text-purple-500" /> Memory Leak Detector
      </h3>

      {/* Stats */}
      <div className="grid grid-cols-3 gap-3 mb-4">
        <div className="bg-white dark:bg-gray-800 rounded-lg p-3 border border-slate-200 dark:border-gray-700 text-center">
          <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">{stats.total}</div>
          <div className="text-xs text-slate-500">Total Created</div>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-3 border border-slate-200 dark:border-gray-700 text-center">
          <motion.div key={leakedCount} animate={{ scale: leakedCount > 0 ? [1, 1.1, 1] : 1 }} className="text-2xl font-bold text-red-600 dark:text-red-400">{leakedCount}</motion.div>
          <div className="text-xs text-slate-500">Leaked</div>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-3 border border-slate-200 dark:border-gray-700 text-center">
          <div className="text-2xl font-bold text-emerald-600 dark:text-emerald-400">{stats.fixed}</div>
          <div className="text-xs text-slate-500">Fixed</div>
        </div>
      </div>

      {/* Task list */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-slate-200 dark:border-gray-700 mb-4">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-bold text-slate-700 dark:text-gray-200">Tasks ({tasks.length})</span>
          {leakedCount > 0 && <span className="text-xs text-red-500 font-medium animate-pulse">{leakedCount} never awaited!</span>}
        </div>
        <div className="grid grid-cols-5 gap-2 max-h-40 overflow-y-auto">
          <AnimatePresence>
            {tasks.slice(-25).map((t) => (
              <motion.div key={t.id} initial={{ scale: 0 }} animate={{ scale: 1 }}
                className={`rounded-lg p-2 text-center border ${t.awaited ? "bg-emerald-50 dark:bg-emerald-900/20 border-emerald-200 dark:border-emerald-800" : "bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800"}`}>
                <div className="text-xs font-mono text-slate-600 dark:text-gray-300 truncate">{t.name}</div>
                {t.awaited ? <CheckCircle className="w-3 h-3 text-emerald-500 mx-auto mt-1" /> : <AlertTriangle className="w-3 h-3 text-red-500 mx-auto mt-1" />}
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      </div>

      {/* Warnings */}
      <div className="bg-slate-900 rounded-lg p-3 mb-4 max-h-32 overflow-y-auto">
        {warnings.length === 0 ? <p className="text-xs text-slate-500 text-center">Create tasks to see warnings...</p> :
          warnings.map((w, i) => (
            <div key={i} className={`text-xs font-mono py-0.5 ${w.startsWith("FIX") ? "text-emerald-400" : w.startsWith("WARNING") ? "text-red-400" : "text-amber-400"}`}>{w}</div>
          ))}
      </div>

      {/* Code comparison */}
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-3 border border-red-200 dark:border-red-800">
          <div className="text-xs font-bold text-red-600 dark:text-red-400 mb-1">Leaky Code</div>
          <pre className="text-xs font-mono text-red-700 dark:text-red-300 whitespace-pre-wrap">{"# Task created but not saved!\nasyncio.create_task(work())\n# GC will destroy it"}</pre>
        </div>
        <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-lg p-3 border border-emerald-200 dark:border-emerald-800">
          <div className="text-xs font-bold text-emerald-600 dark:text-emerald-400 mb-1">Fixed Code</div>
          <pre className="text-xs font-mono text-emerald-700 dark:text-emerald-300 whitespace-pre-wrap">{"# Save reference + await\ntask = asyncio.create_task(work())\nawait task"}</pre>
        </div>
      </div>

      <div className="flex items-center justify-center gap-3">
        <button onClick={clearAll} className="p-2 rounded-lg bg-slate-200 hover:bg-slate-300 dark:bg-gray-700 dark:hover:bg-gray-600 text-slate-600 dark:text-gray-300"><RotateCcw className="w-5 h-5" /></button>
        <button onClick={createLeakedTasks} className="px-4 py-2 rounded-lg bg-red-500 hover:bg-red-600 text-white font-medium flex items-center gap-2 text-sm">
          <Trash2 className="w-4 h-4" /> Create Leaked Tasks
        </button>
        <button onClick={fixTasks} disabled={leakedCount === 0} className="px-4 py-2 rounded-lg bg-emerald-500 hover:bg-emerald-600 text-white font-medium flex items-center gap-2 text-sm disabled:opacity-40">
          <Wrench className="w-4 h-4" /> Fix (Await All)
        </button>
      </div>
    </div>
  );
}
