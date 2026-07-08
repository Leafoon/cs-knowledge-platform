"use client";

import { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Lock, Play, RotateCcw, Settings, CheckCircle, Clock, Loader2 } from "lucide-react";

interface Slot {
  id: number;
  status: "empty" | "running" | "done" | "waiting";
  taskId: number | null;
}

export default function ConcurrencyLimitDemo() {
  const [maxConcurrency, setMaxConcurrency] = useState(3);
  const [totalTasks] = useState(10);
  const [slots, setSlots] = useState<Slot[]>([]);
  const [running, setRunning] = useState(false);
  const [completedCount, setCompletedCount] = useState(0);
  const [waitingCount, setWaitingCount] = useState(0);

  const startDemo = useCallback(async () => {
    setRunning(true);
    setCompletedCount(0);
    const initialSlots: Slot[] = Array.from({ length: maxConcurrency }, (_, i) => ({
      id: i,
      status: "empty",
      taskId: null,
    }));
    setSlots(initialSlots);
    setWaitingCount(totalTasks);

    let taskCounter = 0;
    const completed = new Set<number>();

    while (completed.size < totalTasks) {
      const availableSlot = initialSlots.findIndex((s) => s.status === "empty");
      if (availableSlot !== -1 && taskCounter < totalTasks) {
        const taskId = taskCounter++;
        initialSlots[availableSlot] = { ...initialSlots[availableSlot], status: "running", taskId };
        setSlots([...initialSlots]);
        setWaitingCount(Math.max(0, totalTasks - taskCounter));

        const duration = 400 + Math.random() * 600;
        setTimeout(() => {
          initialSlots[availableSlot] = { ...initialSlots[availableSlot], status: "empty", taskId: null };
          setSlots([...initialSlots]);
          completed.add(taskId);
          setCompletedCount(completed.size);
        }, duration);
      }
      await new Promise((r) => setTimeout(r, 100));
    }

    setRunning(false);
  }, [maxConcurrency, totalTasks]);

  const reset = () => {
    setSlots([]);
    setRunning(false);
    setCompletedCount(0);
    setWaitingCount(0);
  };

  return (
    <div className="max-w-6xl mx-auto p-6 space-y-6">
      <h2 className="text-2xl font-bold dark:text-white flex items-center gap-2">
        <Lock className="w-6 h-6" /> Concurrency Limit Demo
      </h2>
      <div className="flex items-center gap-4 flex-wrap">
        <div className="flex items-center gap-2">
          <Settings className="w-4 h-4 dark:text-gray-400" />
          <label className="dark:text-gray-300">Max Concurrency:</label>
          <input
            type="range"
            min={1}
            max={5}
            value={maxConcurrency}
            onChange={(e) => setMaxConcurrency(+e.target.value)}
            disabled={running}
            className="w-32"
          />
          <span className="font-mono dark:text-white">{maxConcurrency}</span>
        </div>
        <button
          onClick={startDemo}
          disabled={running}
          className="flex items-center gap-2 px-4 py-2 bg-blue-500 text-white rounded-lg disabled:opacity-50 hover:bg-blue-600"
        >
          <Play className="w-4 h-4" /> Run 10 Tasks
        </button>
        <button
          onClick={reset}
          className="flex items-center gap-2 px-4 py-2 bg-gray-500 text-white rounded-lg hover:bg-gray-600"
        >
          <RotateCcw className="w-4 h-4" /> Reset
        </button>
      </div>
      <div className="grid grid-cols-3 gap-4">
        <div className="p-4 rounded-xl bg-gray-100 dark:bg-gray-800 text-center">
          <Clock className="w-6 h-6 mx-auto text-yellow-500 mb-1" />
          <p className="text-2xl font-bold dark:text-white">{waitingCount}</p>
          <p className="text-sm text-gray-500 dark:text-gray-400">Waiting</p>
        </div>
        <div className="p-4 rounded-xl bg-gray-100 dark:bg-gray-800 text-center">
          <Loader2 className="w-6 h-6 mx-auto text-blue-500 mb-1" />
          <p className="text-2xl font-bold dark:text-white">
            {slots.filter((s) => s.status === "running").length}
          </p>
          <p className="text-sm text-gray-500 dark:text-gray-400">Running</p>
        </div>
        <div className="p-4 rounded-xl bg-gray-100 dark:bg-gray-800 text-center">
          <CheckCircle className="w-6 h-6 mx-auto text-green-500 mb-1" />
          <p className="text-2xl font-bold dark:text-white">{completedCount}</p>
          <p className="text-sm text-gray-500 dark:text-gray-400">Completed</p>
        </div>
      </div>
      {slots.length > 0 && (
        <div className="p-4 rounded-xl bg-gray-100 dark:bg-gray-800">
          <h3 className="font-semibold mb-3 dark:text-white">Active Slots</h3>
          <div className="flex gap-3">
            {slots.map((slot) => (
              <motion.div
                key={slot.id}
                animate={{
                  scale: slot.status === "running" ? [1, 1.05, 1] : 1,
                }}
                transition={{ repeat: slot.status === "running" ? Infinity : 0, duration: 1 }}
                className={`w-16 h-16 rounded-lg flex items-center justify-center border-2 ${
                  slot.status === "running"
                    ? "border-blue-500 bg-blue-100 dark:bg-blue-900/30"
                    : "border-gray-300 dark:border-gray-600"
                }`}
              >
                {slot.status === "running" ? (
                  <Loader2 className="w-6 h-6 text-blue-500 animate-spin" />
                ) : (
                  <span className="text-gray-400 dark:text-gray-500 text-xs">Empty</span>
                )}
              </motion.div>
            ))}
          </div>
        </div>
      )}
      <AnimatePresence>
        {!running && completedCount === totalTasks && completedCount > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="p-4 bg-green-100 dark:bg-green-900/30 rounded-lg text-green-700 dark:text-green-300 flex items-center gap-2"
          >
            <CheckCircle className="w-5 h-5" /> All {totalTasks} tasks completed with max {maxConcurrency} concurrent!
          </motion.div>
        )}
      </AnimatePresence>
      <p className="text-sm text-gray-500 dark:text-gray-400">
        Excess tasks wait in queue when all {maxConcurrency} slots are occupied.
      </p>
    </div>
  );
}
