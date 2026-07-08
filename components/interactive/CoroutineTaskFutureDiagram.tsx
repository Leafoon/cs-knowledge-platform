"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ArrowRight, Code2, Layers, Zap, CircleCheck } from "lucide-react";

interface NodeInfo {
  id: string;
  label: string;
  icon: React.ReactNode;
  color: string;
  darkColor: string;
  description: string;
  code: string;
}

const nodes: NodeInfo[] = [
  {
    id: "async_def",
    label: "async def",
    icon: <Code2 className="w-5 h-5" />,
    color: "bg-blue-500",
    darkColor: "dark:bg-blue-600",
    description: "Defines an asynchronous function. Calling it returns a coroutine object, not executing the body.",
    code: "async def fetch_data():\n    await asyncio.sleep(1)\n    return 'data'",
  },
  {
    id: "coroutine",
    label: "Coroutine",
    icon: <Layers className="w-5 h-5" />,
    color: "bg-purple-500",
    darkColor: "dark:bg-purple-600",
    description: "A coroutine object created by calling an async def. It must be awaited or scheduled to run. Cannot be reused after completion.",
    code: "coro = fetch_data()\n# coro is a coroutine object\n# It hasn't started yet!",
  },
  {
    id: "create_task",
    label: "create_task()",
    icon: <Zap className="w-5 h-5" />,
    color: "bg-amber-500",
    darkColor: "dark:bg-amber-600",
    description: "Schedules a coroutine to run on the event loop. Returns a Task object immediately without waiting for completion.",
    code: "task = asyncio.create_task(coro)\n# Scheduled on event loop\n# Returns Task immediately",
  },
  {
    id: "task_future",
    label: "Task (is a Future)",
    icon: <CircleCheck className="w-5 h-5" />,
    color: "bg-green-500",
    darkColor: "dark:bg-green-600",
    description: "A Task IS a Future. It wraps a coroutine, runs it, and stores the result. You can await it, check done(), get result() or exception().",
    code: "await task          # wait for result\nresult = task.result() # get return value\ntask.done()            # check if finished",
  },
];

const connections = [
  { from: 0, to: 1, label: "call()" },
  { from: 1, to: 2, label: "schedule" },
  { from: 2, to: 3, label: "returns" },
];

export function CoroutineTaskFutureDiagram() {
  const [selected, setSelected] = useState<number | null>(null);

  return (
    <div className="max-w-6xl mx-auto p-6">
      <h3 className="text-xl font-bold mb-2 text-gray-900 dark:text-gray-100">
        Coroutine {'->'} Task {'->'} Future Relationship
      </h3>
      <p className="text-sm text-gray-600 dark:text-gray-400 mb-6">
        Click each node to explore how async def becomes a coroutine, gets scheduled as a Task, and stores results as a Future.
      </p>

      <div className="flex flex-wrap items-center justify-center gap-3 mb-6">
        {nodes.map((node, i) => (
          <React.Fragment key={node.id}>
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => setSelected(selected === i ? null : i)}
              className={`flex items-center gap-2 px-5 py-3 rounded-xl text-white font-medium shadow-lg transition-all ${
                node.color
              } ${selected === i ? "ring-4 ring-offset-2 ring-gray-400 dark:ring-offset-gray-900" : ""}`}
            >
              {node.icon}
              {node.label}
            </motion.button>
            {i < connections.length && (
              <motion.div
                animate={{ x: [0, 6, 0] }}
                transition={{ repeat: Infinity, duration: 1.5 }}
                className="flex items-center gap-1 text-gray-400 dark:text-gray-500"
              >
                <span className="text-xs font-mono">{connections[i].label}</span>
                <ArrowRight className="w-4 h-4" />
              </motion.div>
            )}
          </React.Fragment>
        ))}
      </div>

      <AnimatePresence mode="wait">
        {selected !== null && (
          <motion.div
            key={selected}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
            className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6 shadow-md"
          >
            <div className="flex items-center gap-3 mb-3">
              <div className={`p-2 rounded-lg ${nodes[selected].color} text-white`}>
                {nodes[selected].icon}
              </div>
              <h4 className="text-lg font-bold text-gray-900 dark:text-gray-100">
                {nodes[selected].label}
              </h4>
            </div>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              {nodes[selected].description}
            </p>
            <pre className="bg-gray-100 dark:bg-gray-900 rounded-lg p-4 text-sm font-mono overflow-x-auto text-gray-800 dark:text-gray-200">
              {nodes[selected].code}
            </pre>
          </motion.div>
        )}
      </AnimatePresence>

      {selected === null && (
        <p className="text-center text-sm text-gray-400 dark:text-gray-500 mt-4">
          Click a node above to see details
        </p>
      )}
    </div>
  );
}
