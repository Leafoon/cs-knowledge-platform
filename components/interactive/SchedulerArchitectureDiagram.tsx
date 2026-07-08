"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Layers, ListTodo, Users, Lock, Timer, Info, X } from "lucide-react";

interface ComponentInfo {
  id: string;
  name: string;
  icon: typeof Layers;
  color: string;
  description: string;
  details: string[];
}

const components: ComponentInfo[] = [
  {
    id: "scheduler",
    name: "TaskScheduler",
    icon: Layers,
    color: "blue",
    description: "Central coordinator that manages task lifecycle and routing.",
    details: [
      "Submits tasks to the priority queue",
      "Monitors task completion and failures",
      "Handles graceful shutdown",
    ],
  },
  {
    id: "queue",
    name: "Priority Queue",
    icon: ListTodo,
    color: "purple",
    description: "Orders tasks by priority level for execution scheduling.",
    details: [
      "Supports priority levels 1-5",
      "High priority tasks execute first",
      "Thread-safe insertion and removal",
    ],
  },
  {
    id: "workers",
    name: "Worker Pool",
    icon: Users,
    color: "green",
    description: "Executes tasks concurrently with configurable worker count.",
    details: [
      "Async workers for I/O-bound tasks",
      "Thread/process workers for blocking tasks",
      "Auto-scaling based on load",
    ],
  },
  {
    id: "semaphore",
    name: "Concurrency Semaphore",
    icon: Lock,
    color: "orange",
    description: "Limits maximum concurrent task execution to prevent overload.",
    details: [
      "Configurable max concurrency",
      "Queues excess tasks automatically",
      "Prevents resource exhaustion",
    ],
  },
  {
    id: "timer",
    name: "Retry Timer",
    icon: Timer,
    color: "red",
    description: "Manages delayed retries with exponential backoff.",
    details: [
      "Exponential backoff calculation",
      "Configurable max retries",
      "Jitter to prevent thundering herd",
    ],
  },
];

export default function SchedulerArchitectureDiagram() {
  const [selected, setSelected] = useState<string | null>(null);

  const selectedComponent = components.find((c) => c.id === selected);

  return (
    <div className="max-w-6xl mx-auto p-6 space-y-6">
      <h2 className="text-2xl font-bold dark:text-white flex items-center gap-2">
        <Layers className="w-6 h-6" /> Scheduler Architecture
      </h2>
      <p className="text-gray-600 dark:text-gray-400">
        Click any component to see details about its role in the task scheduler.
      </p>
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        {components.map(({ id, name, icon: Icon, color }) => (
          <motion.button
            key={id}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => setSelected(id === selected ? null : id)}
            className={`p-4 rounded-xl border-2 flex flex-col items-center gap-2 transition-colors ${
              selected === id
                ? `border-${color}-500 bg-${color}-50 dark:bg-${color}-900/20`
                : "border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600"
            }`}
          >
            <Icon className={`w-8 h-8 text-${color}-500`} />
            <span className="text-sm font-semibold dark:text-white text-center">
              {name}
            </span>
          </motion.button>
        ))}
      </div>
      <AnimatePresence>
        {selectedComponent && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            className="p-6 rounded-xl bg-white dark:bg-gray-800 border dark:border-gray-700"
          >
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <selectedComponent.icon
                  className={`w-8 h-8 text-${selectedComponent.color}-500`}
                />
                <div>
                  <h3 className="text-lg font-bold dark:text-white">
                    {selectedComponent.name}
                  </h3>
                  <p className="text-gray-600 dark:text-gray-400">
                    {selectedComponent.description}
                  </p>
                </div>
              </div>
              <button
                onClick={() => setSelected(null)}
                className="p-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded"
              >
                <X className="w-5 h-5 dark:text-gray-400" />
              </button>
            </div>
            <div className="space-y-2">
              <h4 className="font-semibold dark:text-white flex items-center gap-2">
                <Info className="w-4 h-4" /> Key Features
              </h4>
              <ul className="space-y-1">
                {selectedComponent.details.map((detail, i) => (
                  <motion.li
                    key={i}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.1 }}
                    className="text-sm text-gray-600 dark:text-gray-400 pl-4 border-l-2 border-gray-300 dark:border-gray-600"
                  >
                    {detail}
                  </motion.li>
                ))}
              </ul>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
