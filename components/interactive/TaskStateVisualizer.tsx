"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Play, CheckCircle, XCircle, Clock, RotateCcw } from "lucide-react";

type State = "pending" | "running" | "done" | "cancelled";

interface StateInfo {
  label: string;
  icon: React.ReactNode;
  color: string;
  ring: string;
  description: string;
}

const stateMap: Record<State, StateInfo> = {
  pending: {
    label: "Pending",
    icon: <Clock className="w-6 h-6" />,
    color: "bg-gray-500",
    ring: "ring-gray-400",
    description: "Task has been created but not yet scheduled on the event loop. Waiting for its turn.",
  },
  running: {
    label: "Running",
    icon: <Play className="w-6 h-6" />,
    color: "bg-blue-500",
    ring: "ring-blue-400",
    description: "Task is actively executing on the event loop. It runs until it awaits or completes.",
  },
  done: {
    label: "Done",
    icon: <CheckCircle className="w-6 h-6" />,
    color: "bg-green-500",
    ring: "ring-green-400",
    description: "Task finished successfully. result() returns the value, done() returns True.",
  },
  cancelled: {
    label: "Cancelled",
    icon: <XCircle className="w-6 h-6" />,
    color: "bg-red-500",
    ring: "ring-red-400",
    description: "Task was cancelled via task.cancel(). A CancelledError will be raised on next await.",
  },
};

const transitions: Record<State, { target: State; label: string; button: string }[]> = {
  pending: [
    { target: "running", label: "event loop picks up", button: "Start Task" },
    { target: "cancelled", label: "task.cancel()", button: "Cancel Task" },
  ],
  running: [
    { target: "done", label: "coroutine completes", button: "Complete Task" },
    { target: "cancelled", label: "task.cancel()", button: "Cancel Task" },
  ],
  done: [],
  cancelled: [],
};

export function TaskStateVisualizer() {
  const [state, setState] = useState<State>("pending");

  const reset = () => setState("pending");
  const info = stateMap[state];
  const available = transitions[state];

  return (
    <div className="max-w-6xl mx-auto p-6">
      <h3 className="text-xl font-bold mb-2 text-gray-900 dark:text-gray-100">
        Task State Machine
      </h3>
      <p className="text-sm text-gray-600 dark:text-gray-400 mb-6">
        Explore the lifecycle of an asyncio Task. Click buttons to trigger state transitions.
      </p>

      <div className="flex flex-wrap justify-center gap-4 mb-8">
        {(Object.keys(stateMap) as State[]).map((s) => (
          <motion.div
            key={s}
            animate={state === s ? { scale: 1.1 } : { scale: 0.95 }}
            className={`flex flex-col items-center gap-2 px-6 py-4 rounded-xl border-2 transition-colors ${
              state === s
                ? `${stateMap[s].color} text-white border-transparent shadow-lg ring-4 ${stateMap[s].ring}`
                : "bg-gray-100 dark:bg-gray-800 text-gray-400 dark:text-gray-500 border-gray-200 dark:border-gray-700"
            }`}
          >
            {stateMap[s].icon}
            <span className="font-semibold text-sm">{stateMap[s].label}</span>
          </motion.div>
        ))}
      </div>

      <AnimatePresence mode="wait">
        <motion.div
          key={state}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
          className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-5 mb-4 shadow-sm"
        >
          <p className="text-gray-700 dark:text-gray-300 mb-1">
            <span className="font-semibold">Current state:</span> {info.label}
          </p>
          <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">{info.description}</p>
          <p className="text-sm font-mono text-gray-500 dark:text-gray-400">
            task.done() {'->'} {state === "done" ? "True" : state === "cancelled" ? "True (raises CancelledError)" : "False"}
          </p>
        </motion.div>
      </AnimatePresence>

      <div className="flex flex-wrap gap-3 justify-center">
        {available.map((t) => (
          <motion.button
            key={t.target}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => setState(t.target)}
            className="px-5 py-2 rounded-lg bg-blue-600 hover:bg-blue-700 text-white font-medium shadow transition-colors"
          >
            {t.button}
          </motion.button>
        ))}
        {(state === "done" || state === "cancelled") && (
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={reset}
            className="flex items-center gap-2 px-5 py-2 rounded-lg bg-gray-600 hover:bg-gray-700 text-white font-medium shadow transition-colors"
          >
            <RotateCcw className="w-4 h-4" />
            Reset
          </motion.button>
        )}
      </div>
    </div>
  );
}
