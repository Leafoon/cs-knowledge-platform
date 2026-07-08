"use client";

import { useState, useCallback } from "react";
import { motion, AnimatePresence, Reorder } from "framer-motion";
import { ListOrdered, Play, Plus, Trash2, ArrowUp, ArrowDown } from "lucide-react";

interface Task {
  id: number;
  name: string;
  priority: number;
  status: "pending" | "running" | "done";
}

let nextId = 1;

export default function TaskPriorityDemo() {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [running, setRunning] = useState(false);

  const addTask = () => {
    const priority = Math.floor(Math.random() * 5) + 1;
    setTasks((prev) => [
      ...prev,
      {
        id: nextId++,
        name: `Task ${nextId - 1}`,
        priority,
        status: "pending",
      },
    ]);
  };

  const removeTask = (id: number) => {
    setTasks((prev) => prev.filter((t) => t.id !== id));
  };

  const runTasks = useCallback(async () => {
    setRunning(true);
    const sorted = [...tasks].sort((a, b) => a.priority - b.priority);
    for (const task of sorted) {
      setTasks((prev) =>
        prev.map((t) => (t.id === task.id ? { ...t, status: "running" } : t))
      );
      await new Promise((r) => setTimeout(r, 800));
      setTasks((prev) =>
        prev.map((t) => (t.id === task.id ? { ...t, status: "done" } : t))
      );
    }
    setRunning(false);
  }, [tasks]);

  const reset = () => {
    setTasks([]);
    nextId = 1;
  };

  const getPriorityColor = (p: number) => {
    if (p <= 2) return "red";
    if (p <= 3) return "orange";
    if (p <= 4) return "yellow";
    return "green";
  };

  return (
    <div className="max-w-6xl mx-auto p-6 space-y-6">
      <h2 className="text-2xl font-bold dark:text-white flex items-center gap-2">
        <ListOrdered className="w-6 h-6" /> Priority Queue Visualization
      </h2>
      <div className="flex gap-4">
        <button
          onClick={addTask}
          disabled={running}
          className="flex items-center gap-2 px-4 py-2 bg-blue-500 text-white rounded-lg disabled:opacity-50 hover:bg-blue-600"
        >
          <Plus className="w-4 h-4" /> Add Random Task
        </button>
        <button
          onClick={runTasks}
          disabled={running || tasks.length === 0}
          className="flex items-center gap-2 px-4 py-2 bg-green-500 text-white rounded-lg disabled:opacity-50 hover:bg-green-600"
        >
          <Play className="w-4 h-4" /> Execute by Priority
        </button>
        <button
          onClick={reset}
          className="flex items-center gap-2 px-4 py-2 bg-gray-500 text-white rounded-lg hover:bg-gray-600"
        >
          <Trash2 className="w-4 h-4" /> Reset
        </button>
      </div>
      <div className="p-4 rounded-xl bg-gray-100 dark:bg-gray-800 min-h-[200px]">
        <div className="flex items-center gap-2 mb-3">
          <span className="text-sm font-semibold dark:text-gray-300">Queue:</span>
          <span className="text-xs text-gray-500 dark:text-gray-400">
            Lower number = higher priority
          </span>
        </div>
        <AnimatePresence>
          {tasks.length === 0 && (
            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="text-gray-500 dark:text-gray-400 text-center py-8"
            >
              Add tasks to see priority queue in action
            </motion.p>
          )}
        </AnimatePresence>
        <div className="space-y-2">
          {[...tasks]
            .sort((a, b) => a.priority - b.priority)
            .map((task, index) => (
              <motion.div
                key={task.id}
                layout
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, x: -100 }}
                className={`flex items-center justify-between p-3 rounded-lg border-l-4 ${
                  task.status === "done"
                    ? "bg-green-100 dark:bg-green-900/30 border-green-500"
                    : task.status === "running"
                    ? "bg-blue-100 dark:bg-blue-900/30 border-blue-500"
                    : `bg-white dark:bg-gray-700 border-${getPriorityColor(task.priority)}-500`
                }`}
              >
                <div className="flex items-center gap-3">
                  <span className="text-sm font-mono text-gray-500 dark:text-gray-400 w-8">
                    #{index + 1}
                  </span>
                  <span className="font-semibold dark:text-white">
                    {task.name}
                  </span>
                  <span
                    className={`px-2 py-0.5 rounded text-xs font-semibold bg-${getPriorityColor(task.priority)}-100 dark:bg-${getPriorityColor(task.priority)}-900/30 text-${getPriorityColor(task.priority)}-700 dark:text-${getPriorityColor(task.priority)}-300`}
                  >
                    P{task.priority}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  {task.status === "running" && (
                    <motion.div
                      animate={{ rotate: 360 }}
                      transition={{ repeat: Infinity, duration: 1 }}
                      className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full"
                    />
                  )}
                  {task.status === "done" && (
                    <span className="text-green-500 text-sm">✓</span>
                  )}
                  <button
                    onClick={() => removeTask(task.id)}
                    className="p-1 hover:bg-gray-200 dark:hover:bg-gray-600 rounded"
                  >
                    <Trash2 className="w-4 h-4 text-gray-400" />
                  </button>
                </div>
              </motion.div>
            ))}
        </div>
      </div>
      <div className="text-sm text-gray-500 dark:text-gray-400">
        Tasks execute in priority order: Priority 1 (highest) → Priority 5 (lowest)
      </div>
    </div>
  );
}
