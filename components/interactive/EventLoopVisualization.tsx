"use client";

import React, { useState, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Play, Pause, RotateCcw, Zap, Clock, Loader2 } from "lucide-react";

interface Task {
  id: number;
  name: string;
  state: "ready" | "waiting" | "running" | "done";
  color: string;
}

const initialTasks: Task[] = [
  { id: 1, name: "fetch(url)", state: "ready", color: "bg-blue-500" },
  { id: 2, name: "read_db()", state: "ready", color: "bg-purple-500" },
  { id: 3, name: "compute()", state: "ready", color: "bg-amber-500" },
  { id: 4, name: "send_msg()", state: "ready", color: "bg-green-500" },
];

export function EventLoopVisualization() {
  const [tasks, setTasks] = useState<Task[]>(initialTasks);
  const [running, setRunning] = useState(false);
  const [currentIdx, setCurrentIdx] = useState(0);
  const [tick, setTick] = useState(0);

  const step = useCallback(() => {
    setTasks((prev) => {
      const copy = [...prev];
      const active = copy.filter((t) => t.state !== "done");
      if (active.length === 0) return copy;

      const current = copy.find((t) => t.state === "running");
      if (current) {
        if (Math.random() > 0.6) {
          current.state = "done";
        } else {
          current.state = "waiting";
        }
      }

      const waiting = copy.filter((t) => t.state === "waiting");
      if (waiting.length > 0 && Math.random() > 0.4) {
        waiting[Math.floor(Math.random() * waiting.length)].state = "ready";
      }

      const ready = copy.find((t) => t.state === "ready");
      if (ready) {
        ready.state = "running";
      }

      return copy;
    });
    setTick((t) => t + 1);
  }, []);

  useEffect(() => {
    if (!running) return;
    const done = tasks.every((t) => t.state === "done");
    if (done) { setRunning(false); return; }
    const interval = setInterval(step, 1200);
    return () => clearInterval(interval);
  }, [running, tasks, step]);

  const reset = () => {
    setRunning(false);
    setTasks(initialTasks.map((t) => ({ ...t, state: "ready" as const })));
    setTick(0);
  };

  const stateGroups = [
    { state: "ready" as const, label: "Ready Queue", icon: <Clock className="w-4 h-4" />, color: "bg-gray-200 dark:bg-gray-700" },
    { state: "running" as const, label: "Running", icon: <Zap className="w-4 h-4" />, color: "bg-blue-100 dark:bg-blue-900/30" },
    { state: "waiting" as const, label: "Waiting (I/O)", icon: <Loader2 className="w-4 h-4" />, color: "bg-amber-100 dark:bg-amber-900/30" },
    { state: "done" as const, label: "Done", icon: <span className="text-green-600">✓</span>, color: "bg-green-100 dark:bg-green-900/30" },
  ];

  return (
    <div className="max-w-6xl mx-auto p-6">
      <h3 className="text-xl font-bold mb-2 text-gray-900 dark:text-gray-100">
        Event Loop Visualization
      </h3>
      <p className="text-sm text-gray-600 dark:text-gray-400 mb-6">
        Watch tasks move between the ready queue, running state, and waiting (I/O) as the event loop ticks.
      </p>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        {stateGroups.map((g) => (
          <div key={g.state} className={`${g.color} rounded-xl p-4 border border-gray-200 dark:border-gray-700`}>
            <div className="flex items-center gap-2 font-semibold text-sm text-gray-700 dark:text-gray-300 mb-2">
              {g.icon} {g.label}
            </div>
            <div className="flex flex-wrap gap-2 min-h-[40px]">
              <AnimatePresence>
                {tasks.filter((t) => t.state === g.state).map((t) => (
                  <motion.div
                    key={t.id}
                    layout
                    initial={{ opacity: 0, scale: 0.5 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.5 }}
                    className={`${t.color} text-white text-xs font-mono px-3 py-1.5 rounded-lg shadow`}
                  >
                    {t.name}
                  </motion.div>
                ))}
              </AnimatePresence>
            </div>
          </div>
        ))}
      </div>

      <div className="flex items-center justify-center gap-2 text-sm text-gray-500 dark:text-gray-400 mb-4">
        <span>Tick: {tick}</span>
      </div>

      <div className="flex gap-3 justify-center">
        {!running ? (
          <motion.button whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }} onClick={() => setRunning(true)}
            className="flex items-center gap-2 px-6 py-2 rounded-lg bg-blue-600 hover:bg-blue-700 text-white font-medium shadow transition-colors">
            <Play className="w-4 h-4" /> Start Loop
          </motion.button>
        ) : (
          <motion.button whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }} onClick={() => setRunning(false)}
            className="flex items-center gap-2 px-6 py-2 rounded-lg bg-amber-600 hover:bg-amber-700 text-white font-medium shadow transition-colors">
            <Pause className="w-4 h-4" /> Pause
          </motion.button>
        )}
        <motion.button whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }} onClick={reset}
          className="flex items-center gap-2 px-6 py-2 rounded-lg bg-gray-600 hover:bg-gray-700 text-white font-medium shadow transition-colors">
          <RotateCcw className="w-4 h-4" /> Reset
        </motion.button>
      </div>
    </div>
  );
}
