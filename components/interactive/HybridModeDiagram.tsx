"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Activity, Cpu, HardDrive, Wifi, ArrowRight } from "lucide-react";

export default function HybridModeDiagram() {
  const [activeTask, setActiveTask] = useState<string | null>(null);
  const [tasks, setTasks] = useState([
    { id: "io", label: "HTTP Request", type: "io", color: "blue" },
    { id: "cpu", label: "Data Processing", type: "cpu", color: "green" },
    { id: "block", label: "File Read", type: "block", color: "orange" },
  ]);

  useEffect(() => {
    const interval = setInterval(() => {
      setActiveTask((prev) => {
        const idx = tasks.findIndex((t) => t.id === prev);
        return tasks[(idx + 1) % tasks.length].id;
      });
    }, 2000);
    return () => clearInterval(interval);
  }, [tasks]);

  const getRoute = (type: string) => {
    switch (type) {
      case "io":
        return { target: "Event Loop", icon: Activity, color: "blue" };
      case "cpu":
        return { target: "Process Pool", icon: Cpu, color: "green" };
      case "block":
        return { target: "Thread Pool", icon: HardDrive, color: "orange" };
      default:
        return { target: "Event Loop", icon: Activity, color: "blue" };
    }
  };

  const currentTask = tasks.find((t) => t.id === activeTask);
  const route = currentTask ? getRoute(currentTask.type) : null;

  return (
    <div className="max-w-6xl mx-auto p-6 space-y-6">
      <h2 className="text-2xl font-bold dark:text-white flex items-center gap-2">
        <Activity className="w-6 h-6" /> Hybrid Execution Model
      </h2>
      <div className="grid md:grid-cols-3 gap-4">
        {[
          { id: "io", label: "I/O Tasks", icon: Wifi, color: "blue", desc: "Network, DB queries" },
          { id: "cpu", label: "CPU Tasks", icon: Cpu, color: "green", desc: "Computation, processing" },
          { id: "block", label: "Blocking I/O", icon: HardDrive, color: "orange", desc: "File system, sync calls" },
        ].map(({ id, label, icon: Icon, color, desc }) => (
          <motion.div
            key={id}
            animate={{ scale: activeTask === id ? 1.05 : 1 }}
            className={`p-4 rounded-xl border-2 ${
              activeTask === id
                ? `border-${color}-500 bg-${color}-50 dark:bg-${color}-900/20`
                : "border-gray-200 dark:border-gray-700"
            }`}
          >
            <Icon className={`w-6 h-6 text-${color}-500 mb-2`} />
            <h3 className="font-semibold dark:text-white">{label}</h3>
            <p className="text-sm dark:text-gray-300">{desc}</p>
          </motion.div>
        ))}
      </div>
      <AnimatePresence>
        {route && currentTask && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="flex items-center justify-center gap-4 p-4 bg-gray-100 dark:bg-gray-800 rounded-xl"
          >
            <span className="font-semibold dark:text-white">
              {currentTask.label}
            </span>
            <ArrowRight className="w-5 h-5 text-gray-400" />
            <motion.div
              animate={{ x: [0, 10, 0] }}
              transition={{ repeat: Infinity, duration: 1 }}
            >
              <route.icon className={`w-6 h-6 text-${route.color}-500`} />
            </motion.div>
            <span className={`font-semibold text-${route.color}-500`}>
              {route.target}
            </span>
          </motion.div>
        )}
      </AnimatePresence>
      <div className="p-4 bg-gray-100 dark:bg-gray-800 rounded-lg">
        <h4 className="font-semibold dark:text-white mb-2">How it works:</h4>
        <ul className="text-sm dark:text-gray-300 space-y-1">
          <li>• Event loop handles async I/O efficiently</li>
          <li>• Thread pool offloads blocking operations</li>
          <li>• Process pool provides true CPU parallelism</li>
          <li>• All managed by a single async application</li>
        </ul>
      </div>
    </div>
  );
}
