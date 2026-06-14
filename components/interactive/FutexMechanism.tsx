"use client";

import { useState, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Play, RotateCcw, Zap, Clock, ArrowDown, ArrowUp, Lock, Unlock, Moon } from "lucide-react";

type Phase =
  | "idle"
  | "fast_lock_attempt"
  | "fast_lock_success"
  | "fast_lock_fail"
  | "slow_enter_kernel"
  | "slow_wait_queue"
  | "slow_sleep"
  | "fast_unlock_no_waiters"
  | "slow_unlock_wake"
  | "slow_wake_thread"
  | "slow_return_user";

interface Step {
  phase: Phase;
  label: string;
  description: string;
  path: "fast" | "slow";
  threadAType: "user" | "kernel" | "sleeping" | "running" | "none";
  threadBType: "user" | "kernel" | "sleeping" | "running" | "none";
  lockValue: number;
  waitQueueCount: number;
  timingNs: number;
  cpuMode: "user" | "kernel" | "none";
}

const SCENARIOS: Record<string, Step[]> = {
  "Fast Path Acquire": [
    { phase: "idle", label: "Initial State", description: "Lock is free (0). Thread A wants to acquire it.", path: "fast", threadAType: "user", threadBType: "none", lockValue: 0, waitQueueCount: 0, timingNs: 0, cpuMode: "none" },
    { phase: "fast_lock_attempt", label: "CAS in user space", description: "Thread A does CAS(lock, 0, 1) entirely in user space. No syscall needed!", path: "fast", threadAType: "user", threadBType: "none", lockValue: 0, waitQueueCount: 0, timingNs: 0, cpuMode: "user" },
    { phase: "fast_lock_success", label: "CAS succeeds", description: "CAS returns success. Lock is now 1. Thread A enters critical section. Total: ~10ns", path: "fast", threadAType: "running", threadBType: "none", lockValue: 1, waitQueueCount: 0, timingNs: 10, cpuMode: "user" },
  ],
  "Slow Path Acquire": [
    { phase: "idle", label: "Lock Contended", description: "Thread B holds the lock. Thread A tries CAS -> fails.", path: "fast", threadAType: "user", threadBType: "running", lockValue: 1, waitQueueCount: 0, timingNs: 0, cpuMode: "none" },
    { phase: "fast_lock_fail", label: "CAS fails (lock=1)", description: "Thread A's CAS fails because lock != 0. Must enter slow path.", path: "fast", threadAType: "user", threadBType: "running", lockValue: 1, waitQueueCount: 0, timingNs: 10, cpuMode: "user" },
    { phase: "slow_enter_kernel", label: "Enter kernel (syscall)", description: "Thread A calls futex(FUTEX_WAIT). Traps into kernel mode.", path: "slow", threadAType: "kernel", threadBType: "running", lockValue: 1, waitQueueCount: 0, timingNs: 500, cpuMode: "kernel" },
    { phase: "slow_wait_queue", label: "Add to wait queue", description: "Kernel checks: still locked? Yes. Add thread A to futex wait queue.", path: "slow", threadAType: "kernel", threadBType: "running", lockValue: 1, waitQueueCount: 1, timingNs: 800, cpuMode: "kernel" },
    { phase: "slow_sleep", label: "Thread A sleeps", description: "Thread A is put to sleep. CPU can run other work. Very expensive path.", path: "slow", threadAType: "sleeping", threadBType: "running", lockValue: 1, waitQueueCount: 1, timingNs: 1000, cpuMode: "none" },
  ],
  "Slow Path Unlock & Wake": [
    { phase: "slow_sleep", label: "Thread A sleeping", description: "Thread A is in wait queue. Thread B is in critical section.", path: "slow", threadAType: "sleeping", threadBType: "running", lockValue: 1, waitQueueCount: 1, timingNs: 0, cpuMode: "none" },
    { phase: "fast_unlock_no_waiters", label: "B: futex(FUTEX_WAKE)", description: "Thread B sets lock=0, calls futex(FUTEX_WAKE). Kernel removes waiter from queue.", path: "slow", threadAType: "sleeping", threadBType: "kernel", lockValue: 0, waitQueueCount: 1, timingNs: 500, cpuMode: "kernel" },
    { phase: "slow_wake_thread", label: "Wake thread A", description: "Kernel wakes thread A. It re-enters runnable state. Context switch occurs.", path: "slow", threadAType: "kernel", threadBType: "user", lockValue: 0, waitQueueCount: 0, timingNs: 800, cpuMode: "kernel" },
    { phase: "slow_return_user", label: "A returns to user space", description: "Thread A returns to user space, retries CAS, succeeds. Total wake: ~1us", path: "slow", threadAType: "running", threadBType: "user", lockValue: 1, waitQueueCount: 0, timingNs: 1000, cpuMode: "user" },
  ],
};

export default function FutexMechanism() {
  const [scenario, setScenario] = useState("Fast Path Acquire");
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  const steps = SCENARIOS[scenario];
  const step = steps[Math.min(currentStep, steps.length - 1)];

  const reset = useCallback(() => {
    setCurrentStep(0);
    setIsPlaying(false);
  }, []);

  useEffect(() => {
    reset();
  }, [scenario, reset]);

  useEffect(() => {
    if (!isPlaying) return;
    const timer = setInterval(() => {
      setCurrentStep((s) => {
        if (s >= steps.length - 1) { setIsPlaying(false); return s; }
        return s + 1;
      });
    }, 1800);
    return () => clearInterval(timer);
  }, [isPlaying, steps.length]);

  const getThreadColor = (type: string) => {
    switch (type) {
      case "user": return "bg-blue-100 border-blue-300 text-blue-700 dark:bg-blue-900/30 dark:border-blue-600 dark:text-blue-300";
      case "kernel": return "bg-amber-100 border-amber-300 text-amber-700 dark:bg-amber-900/30 dark:border-amber-600 dark:text-amber-300";
      case "sleeping": return "bg-slate-100 border-slate-300 text-slate-500 dark:bg-slate-800 dark:border-slate-600 dark:text-slate-400";
      case "running": return "bg-green-100 border-green-300 text-green-700 dark:bg-green-900/30 dark:border-green-600 dark:text-green-300";
      default: return "";
    }
  };

  const getThreadIcon = (type: string) => {
    switch (type) {
      case "user": return "User";
      case "kernel": return "Shield";
      case "sleeping": return "Moon";
      case "running": return "Zap";
      default: return "User";
    }
  };

  const formatTiming = (ns: number) => {
    if (ns >= 1000) return `~${(ns / 1000).toFixed(0)}us`;
    return `~${ns}ns`;
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-amber-50 rounded-xl shadow-lg dark:from-gray-900 dark:to-gray-800">
      <h2 className="text-2xl font-bold text-slate-800 dark:text-gray-100 mb-6 text-center">
        Futex Mechanism: Fast Path vs Slow Path
      </h2>

      {/* Scenario Selector */}
      <div className="flex justify-center gap-2 mb-6">
        {Object.keys(SCENARIOS).map((s) => (
          <button
            key={s}
            onClick={() => setScenario(s)}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              scenario === s
                ? "bg-amber-500 text-white shadow-md"
                : "bg-white dark:bg-gray-800 text-slate-600 dark:text-gray-300 border border-slate-200 dark:border-gray-700 hover:bg-slate-50"
            }`}
          >
            {s}
          </button>
        ))}
      </div>

      {/* Path Indicator */}
      <div className="flex justify-center gap-6 mb-4">
        <div className={`flex items-center gap-2 px-4 py-2 rounded-lg ${step.path === "fast" ? "bg-green-100 dark:bg-green-900/30 border-2 border-green-400" : "bg-slate-100 dark:bg-gray-800 border-2 border-transparent"}`}>
          <Zap className={`w-4 h-4 ${step.path === "fast" ? "text-green-600" : "text-slate-400"}`} />
          <span className={`text-sm font-medium ${step.path === "fast" ? "text-green-700 dark:text-green-300" : "text-slate-400"}`}>
            Fast Path (~10ns)
          </span>
        </div>
        <div className={`flex items-center gap-2 px-4 py-2 rounded-lg ${step.path === "slow" ? "bg-amber-100 dark:bg-amber-900/30 border-2 border-amber-400" : "bg-slate-100 dark:bg-gray-800 border-2 border-transparent"}`}>
          <Clock className={`w-4 h-4 ${step.path === "slow" ? "text-amber-600" : "text-slate-400"}`} />
          <span className={`text-sm font-medium ${step.path === "slow" ? "text-amber-700 dark:text-amber-300" : "text-slate-400"}`}>
            Slow Path (~1us)
          </span>
        </div>
      </div>

      {/* Main Visualization */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        {/* User Space */}
        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4 border-2 border-blue-200 dark:border-blue-800">
          <h4 className="text-sm font-bold text-blue-700 dark:text-blue-300 mb-3 text-center">User Space</h4>
          <div className="space-y-3">
            {/* Thread A */}
            {step.threadAType !== "none" && (
              <motion.div
                layout
                className={`p-3 rounded-lg border-2 ${getThreadColor(step.threadAType)}`}
              >
                <div className="flex items-center gap-2">
                  <span className="text-xs font-bold">Thread A</span>
                  <span className="text-xs capitalize">{step.threadAType}</span>
                </div>
                {step.threadAType === "user" && (
                  <div className="mt-2 text-xs">Trying CAS(lock, 0, 1)...</div>
                )}
                {step.threadAType === "running" && (
                  <motion.div initial={{ scale: 0 }} animate={{ scale: 1 }} className="mt-2 text-xs font-bold text-green-600">
                    In Critical Section
                  </motion.div>
                )}
              </motion.div>
            )}
            {/* Thread B */}
            {step.threadBType !== "none" && step.threadBType !== "kernel" && (
              <motion.div layout className={`p-3 rounded-lg border-2 ${getThreadColor(step.threadBType)}`}>
                <div className="flex items-center gap-2">
                  <span className="text-xs font-bold">Thread B</span>
                  <span className="text-xs capitalize">{step.threadBType}</span>
                </div>
                {step.threadBType === "running" && (
                  <div className="mt-2 text-xs font-bold text-green-600">In Critical Section</div>
                )}
              </motion.div>
            )}
          </div>
        </div>

        {/* Futex / Lock */}
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border-2 border-slate-200 dark:border-gray-700 flex flex-col items-center justify-center">
          <h4 className="text-sm font-bold text-slate-700 dark:text-gray-200 mb-3">Futex Word (lock)</h4>
          <motion.div
            animate={{
              backgroundColor: step.lockValue === 1 ? "#ef4444" : "#22c55e",
              scale: step.lockValue === 1 ? [1, 1.05, 1] : 1,
            }}
            transition={{ repeat: step.lockValue === 1 ? Infinity : 0, duration: 1.5 }}
            className="w-20 h-20 rounded-xl flex flex-col items-center justify-center text-white font-bold shadow-lg mb-4"
          >
            <Lock className="w-6 h-6 mb-1" />
            <span className="text-2xl">{step.lockValue}</span>
            <span className="text-xs">{step.lockValue === 1 ? "LOCKED" : "FREE"}</span>
          </motion.div>

          {/* Wait Queue */}
          <div className="w-full">
            <div className="text-xs text-center text-slate-500 dark:text-gray-400 mb-1">Wait Queue</div>
            <div className="flex justify-center gap-1 min-h-[2rem]">
              <AnimatePresence>
                {Array.from({ length: step.waitQueueCount }).map((_, i) => (
                  <motion.div
                    key={i}
                    initial={{ scale: 0, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    exit={{ scale: 0, opacity: 0 }}
                    className="w-8 h-8 rounded-full bg-amber-200 dark:bg-amber-800 flex items-center justify-center"
                  >
                    <Moon className="w-4 h-4 text-amber-600 dark:text-amber-300" />
                  </motion.div>
                ))}
              </AnimatePresence>
              {step.waitQueueCount === 0 && (
                <span className="text-xs text-slate-400 dark:text-gray-500">Empty</span>
              )}
            </div>
          </div>
        </div>

        {/* Kernel Space */}
        <div className={`rounded-lg p-4 border-2 transition-all ${
          step.threadAType === "kernel" || step.threadBType === "kernel"
            ? "bg-amber-50 dark:bg-amber-900/20 border-amber-300 dark:border-amber-700"
            : "bg-slate-50 dark:bg-gray-800/50 border-slate-200 dark:border-gray-700"
        }`}>
          <h4 className="text-sm font-bold text-slate-700 dark:text-gray-200 mb-3 text-center">Kernel Space</h4>
          <div className="space-y-3">
            {step.threadAType === "kernel" && (
              <motion.div initial={{ x: -20, opacity: 0 }} animate={{ x: 0, opacity: 1 }} className="p-3 rounded-lg border-2 bg-amber-100 border-amber-300 dark:bg-amber-900/30 dark:border-amber-600">
                <div className="flex items-center gap-2">
                  <ArrowDown className="w-4 h-4 text-amber-600" />
                  <span className="text-xs font-bold text-amber-700 dark:text-amber-300">Thread A in kernel</span>
                </div>
                <div className="mt-1 text-xs text-amber-600 dark:text-amber-400">Processing FUTEX_WAIT...</div>
              </motion.div>
            )}
            {step.threadBType === "kernel" && (
              <motion.div initial={{ x: -20, opacity: 0 }} animate={{ x: 0, opacity: 1 }} className="p-3 rounded-lg border-2 bg-amber-100 border-amber-300 dark:bg-amber-900/30 dark:border-amber-600">
                <div className="flex items-center gap-2">
                  <ArrowUp className="w-4 h-4 text-amber-600" />
                  <span className="text-xs font-bold text-amber-700 dark:text-amber-300">Thread B in kernel</span>
                </div>
                <div className="mt-1 text-xs text-amber-600 dark:text-amber-400">Processing FUTEX_WAKE...</div>
              </motion.div>
            )}
            {step.threadAType !== "kernel" && step.threadBType !== "kernel" && (
              <div className="text-center text-xs text-slate-400 dark:text-gray-500 py-4">
                No kernel activity
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Step Description */}
      <AnimatePresence mode="wait">
        <motion.div
          key={currentStep}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
          className="text-center mb-4 p-4 bg-white dark:bg-gray-800 rounded-lg border border-slate-200 dark:border-gray-700"
        >
          <div className="text-sm font-bold text-slate-700 dark:text-gray-200 mb-1">{step.label}</div>
          <div className="text-sm text-slate-600 dark:text-gray-300">{step.description}</div>
          {step.timingNs > 0 && (
            <div className="mt-2 flex items-center justify-center gap-1">
              <Clock className="w-4 h-4 text-slate-400" />
              <span className="text-sm font-mono text-slate-500 dark:text-gray-400">
                Cumulative: {formatTiming(step.timingNs)}
              </span>
            </div>
          )}
        </motion.div>
      </AnimatePresence>

      {/* Progress */}
      <div className="mb-4">
        <div className="w-full h-2 bg-slate-200 dark:bg-gray-700 rounded-full overflow-hidden">
          <motion.div
            className={`h-full rounded-full ${step.path === "fast" ? "bg-green-500" : "bg-amber-500"}`}
            animate={{ width: `${((currentStep + 1) / steps.length) * 100}%` }}
          />
        </div>
      </div>

      {/* Controls */}
      <div className="flex items-center justify-center gap-3">
        <button onClick={reset} className="p-2 rounded-lg bg-slate-200 hover:bg-slate-300 dark:bg-gray-700 dark:hover:bg-gray-600 text-slate-600 dark:text-gray-300">
          <RotateCcw className="w-5 h-5" />
        </button>
        <button
          onClick={() => setCurrentStep((s) => Math.max(0, s - 1))}
          disabled={currentStep === 0}
          className="px-4 py-2 rounded-lg bg-slate-200 hover:bg-slate-300 dark:bg-gray-700 dark:hover:bg-gray-600 text-slate-600 dark:text-gray-300 disabled:opacity-40 text-sm"
        >
          Prev
        </button>
        <button
          onClick={() => setIsPlaying(!isPlaying)}
          className="px-5 py-2 rounded-lg bg-amber-500 hover:bg-amber-600 text-white font-medium flex items-center gap-2"
        >
          {isPlaying ? <><Clock className="w-4 h-4 animate-pulse" /> Pause</> : <><Play className="w-4 h-4" /> Play</>}
        </button>
        <button
          onClick={() => setCurrentStep((s) => Math.min(steps.length - 1, s + 1))}
          disabled={currentStep >= steps.length - 1}
          className="px-4 py-2 rounded-lg bg-slate-200 hover:bg-slate-300 dark:bg-gray-700 dark:hover:bg-gray-600 text-slate-600 dark:text-gray-300 disabled:opacity-40 text-sm"
        >
          Next
        </button>
      </div>

      {/* Timing Comparison */}
      <div className="mt-6 grid grid-cols-2 gap-4">
        <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4 border border-green-200 dark:border-green-800 text-center">
          <Zap className="w-6 h-6 text-green-500 mx-auto mb-1" />
          <div className="text-lg font-bold text-green-700 dark:text-green-300">~10ns</div>
          <div className="text-xs text-green-600 dark:text-green-400">Fast Path (user-space CAS)</div>
          <div className="text-xs text-slate-500 dark:text-gray-400 mt-1">No syscall, no context switch</div>
        </div>
        <div className="bg-amber-50 dark:bg-amber-900/20 rounded-lg p-4 border border-amber-200 dark:border-amber-800 text-center">
          <Clock className="w-6 h-6 text-amber-500 mx-auto mb-1" />
          <div className="text-lg font-bold text-amber-700 dark:text-amber-300">~1us</div>
          <div className="text-xs text-amber-600 dark:text-amber-400">Slow Path (kernel wait/wake)</div>
          <div className="text-xs text-slate-500 dark:text-gray-400 mt-1">Syscall + context switch + schedule</div>
        </div>
      </div>
    </div>
  );
}
