"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Plus, Play, RotateCcw, ArrowRight, Zap } from "lucide-react";

interface LogEntry {
  text: string;
  type: "create" | "yield" | "run" | "done";
}

export function TaskSchedulingDemo() {
  const [tasks, setTasks] = useState<string[]>([]);
  const [log, setLog] = useState<LogEntry[]>([]);
  const [phase, setPhase] = useState<"idle" | "created" | "yielded" | "executed">("idle");

  const addTask = () => {
    const name = `task_${tasks.length + 1}`;
    setTasks((prev) => [...prev, name]);
    setLog((prev) => [...prev, { text: `${name} = asyncio.create_task(coro_${tasks.length + 1})`, type: "create" }]);
    if (tasks.length === 0) setPhase("created");
  };

  const yieldControl = () => {
    setLog((prev) => [...prev, { text: "await asyncio.sleep(0)  # yield to event loop", type: "yield" }]);
    setPhase("yielded");
  };

  const executeTasks = () => {
    const newLogs: LogEntry[] = [];
    tasks.forEach((t, i) => {
      newLogs.push({ text: `${t} starts executing`, type: "run" });
      newLogs.push({ text: `${t} completed`, type: "done" });
    });
    setLog((prev) => [...prev, ...newLogs]);
    setPhase("executed");
  };

  const reset = () => {
    setTasks([]);
    setLog([]);
    setPhase("idle");
  };

  const typeColors: Record<string, string> = {
    create: "text-blue-600 dark:text-blue-400",
    yield: "text-amber-600 dark:text-amber-400",
    run: "text-purple-600 dark:text-purple-400",
    done: "text-green-600 dark:text-green-400",
  };

  return (
    <div className="max-w-6xl mx-auto p-6">
      <h3 className="text-xl font-bold mb-2 text-gray-900 dark:text-gray-100">
        Task Scheduling Demo
      </h3>
      <p className="text-sm text-gray-600 dark:text-gray-400 mb-6">
        Create tasks with create_task {'()'} {'->'} yield control with sleep(0) {'->'} watch them execute on the event loop.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
        <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-5 shadow-sm">
          <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">Tasks Created</h4>
          <div className="flex flex-wrap gap-2 min-h-[48px] mb-4">
            <AnimatePresence>
              {tasks.map((t) => (
                <motion.span key={t} initial={{ opacity: 0, scale: 0.5 }} animate={{ opacity: 1, scale: 1 }}
                  className={`px-3 py-1.5 rounded-lg text-xs font-mono text-white shadow ${
                    phase === "executed" ? "bg-green-500" : "bg-blue-500"
                  }`}>
                  {t}
                </motion.span>
              ))}
            </AnimatePresence>
            {tasks.length === 0 && <span className="text-sm text-gray-400 dark:text-gray-500">No tasks yet</span>}
          </div>

          <div className="flex flex-wrap gap-2">
            <motion.button whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }} onClick={addTask}
              disabled={phase !== "idle" && phase !== "created"}
              className="flex items-center gap-1 px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-700 disabled:opacity-40 text-white text-sm font-medium shadow transition-colors">
              <Plus className="w-3 h-3" /> Add Task
            </motion.button>
            <motion.button whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }} onClick={yieldControl}
              disabled={phase !== "created" || tasks.length === 0}
              className="flex items-center gap-1 px-4 py-2 rounded-lg bg-amber-600 hover:bg-amber-700 disabled:opacity-40 text-white text-sm font-medium shadow transition-colors">
              <Zap className="w-3 h-3" /> sleep(0)
            </motion.button>
            <motion.button whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }} onClick={executeTasks}
              disabled={phase !== "yielded"}
              className="flex items-center gap-1 px-4 py-2 rounded-lg bg-green-600 hover:bg-green-700 disabled:opacity-40 text-white text-sm font-medium shadow transition-colors">
              <Play className="w-3 h-3" /> Run Tasks
            </motion.button>
          </div>
        </div>

        <div className="bg-gray-900 dark:bg-gray-950 rounded-xl p-5 shadow-sm overflow-auto max-h-80">
          <h4 className="font-semibold text-gray-300 mb-3">Event Log</h4>
          <div className="space-y-1">
            <AnimatePresence>
              {log.map((entry, i) => (
                <motion.div key={i} initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }}
                  className={`text-xs font-mono ${typeColors[entry.type]}`}>
                  <span className="text-gray-500 mr-2">{">"}</span>
                  {entry.text}
                </motion.div>
              ))}
            </AnimatePresence>
            {log.length === 0 && <span className="text-xs text-gray-500">Waiting for input...</span>}
          </div>
        </div>
      </div>

      <div className="flex items-center justify-center gap-4 text-xs text-gray-500 dark:text-gray-400">
        <span className="flex items-center gap-1"><ArrowRight className="w-3 h-3" /> create_task schedules but doesn&apos;t start</span>
        <span className="flex items-center gap-1"><ArrowRight className="w-3 h-3" /> sleep(0) yields to let tasks run</span>
      </div>

      <div className="flex justify-center mt-4">
        <motion.button whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }} onClick={reset}
          className="flex items-center gap-2 px-5 py-2 rounded-lg bg-gray-600 hover:bg-gray-700 text-white text-sm font-medium shadow transition-colors">
          <RotateCcw className="w-4 h-4" /> Reset
        </motion.button>
      </div>
    </div>
  );
}
