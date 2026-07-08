"use client";

import { useState, useEffect, useCallback } from "react";
import { motion } from "framer-motion";
import { Activity, BarChart3, CheckCircle, AlertTriangle, Clock, Loader2, Play, RotateCcw } from "lucide-react";

interface Stats {
  pending: number;
  running: number;
  done: number;
  failed: number;
  throughput: number[];
}

export default function SchedulerDashboard() {
  const [stats, setStats] = useState<Stats>({
    pending: 0,
    running: 0,
    done: 0,
    failed: 0,
    throughput: [],
  });
  const [simulating, setSimulating] = useState(false);
  const [tick, setTick] = useState(0);

  const startSimulation = useCallback(() => {
    setSimulating(true);
    setTick(0);
    setStats({ pending: 20, running: 0, done: 0, failed: 0, throughput: [] });
  }, []);

  useEffect(() => {
    if (!simulating) return;
    const interval = setInterval(() => {
      setTick((prev) => {
        if (prev >= 30) {
          setSimulating(false);
          return prev;
        }
        return prev + 1;
      });
      setStats((prev) => {
        const newRunning = Math.min(prev.pending, 3 + Math.floor(Math.random() * 2));
        const newDone = Math.floor(Math.random() * 3);
        const newFailed = Math.random() < 0.15 ? 1 : 0;
        const actualDone = Math.min(newDone, prev.running || newRunning);
        const throughput = [...prev.throughput, actualDone].slice(-15);
        return {
          pending: Math.max(0, prev.pending - newRunning + newFailed),
          running: Math.max(0, (prev.running || newRunning) - actualDone - newFailed + Math.min(newRunning, prev.pending)),
          done: prev.done + actualDone,
          failed: prev.failed + newFailed,
          throughput,
        };
      });
    }, 500);
    return () => clearInterval(interval);
  }, [simulating]);

  const reset = () => {
    setSimulating(false);
    setTick(0);
    setStats({ pending: 0, running: 0, done: 0, failed: 0, throughput: [] });
  };

  const maxThroughput = Math.max(...stats.throughput, 1);

  return (
    <div className="max-w-6xl mx-auto p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold dark:text-white flex items-center gap-2">
          <Activity className="w-6 h-6" /> Scheduler Dashboard
        </h2>
        <div className="flex gap-2">
          <button
            onClick={startSimulation}
            disabled={simulating}
            className="flex items-center gap-2 px-4 py-2 bg-blue-500 text-white rounded-lg disabled:opacity-50 hover:bg-blue-600"
          >
            <Play className="w-4 h-4" /> Simulate
          </button>
          <button
            onClick={reset}
            className="flex items-center gap-2 px-4 py-2 bg-gray-500 text-white rounded-lg hover:bg-gray-600"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
        </div>
      </div>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[
          { label: "Pending", value: stats.pending, icon: Clock, color: "yellow" },
          { label: "Running", value: stats.running, icon: Loader2, color: "blue" },
          { label: "Completed", value: stats.done, icon: CheckCircle, color: "green" },
          { label: "Failed", value: stats.failed, icon: AlertTriangle, color: "red" },
        ].map(({ label, value, icon: Icon, color }) => (
          <motion.div
            key={label}
            animate={{ scale: simulating ? [1, 1.02, 1] : 1 }}
            transition={{ repeat: simulating ? Infinity : 0, duration: 2 }}
            className="p-4 rounded-xl bg-white dark:bg-gray-800 border dark:border-gray-700"
          >
            <div className="flex items-center justify-between mb-2">
              <Icon className={`w-5 h-5 text-${color}-500`} />
              {simulating && label === "Running" && (
                <Loader2 className="w-4 h-4 text-blue-500 animate-spin" />
              )}
            </div>
            <p className="text-3xl font-bold dark:text-white">{value}</p>
            <p className="text-sm text-gray-500 dark:text-gray-400">{label}</p>
          </motion.div>
        ))}
      </div>
      <div className="grid md:grid-cols-2 gap-6">
        <div className="p-4 rounded-xl bg-white dark:bg-gray-800 border dark:border-gray-700">
          <h3 className="font-semibold mb-3 dark:text-white flex items-center gap-2">
            <BarChart3 className="w-5 h-5" /> Throughput (tasks/tick)
          </h3>
          <div className="flex items-end gap-1 h-32">
            {stats.throughput.map((val, i) => (
              <motion.div
                key={i}
                initial={{ height: 0 }}
                animate={{ height: `${(val / maxThroughput) * 100}%` }}
                className="flex-1 bg-blue-500 rounded-t min-h-[4px]"
              />
            ))}
            {stats.throughput.length === 0 && (
              <div className="w-full flex items-center justify-center text-gray-400 dark:text-gray-600 text-sm">
                Run simulation to see throughput
              </div>
            )}
          </div>
        </div>
        <div className="p-4 rounded-xl bg-white dark:bg-gray-800 border dark:border-gray-700">
          <h3 className="font-semibold mb-3 dark:text-white">Worker Status</h3>
          <div className="flex gap-2 flex-wrap">
            {Array.from({ length: 5 }).map((_, i) => (
              <motion.div
                key={i}
                animate={{
                  backgroundColor: i < stats.running ? "#3b82f6" : "#374151",
                }}
                className="w-12 h-12 rounded-lg flex items-center justify-center"
              >
                {i < stats.running ? (
                  <Loader2 className="w-5 h-5 text-white animate-spin" />
                ) : (
                  <span className="text-xs text-gray-400">W{i + 1}</span>
                )}
              </motion.div>
            ))}
          </div>
          <div className="mt-3 text-sm text-gray-500 dark:text-gray-400">
            {stats.running} / 5 workers active
          </div>
        </div>
      </div>
      {simulating && (
        <div className="text-center text-sm text-gray-500 dark:text-gray-400">
          Simulation tick: {tick} / 30
        </div>
      )}
    </div>
  );
}
