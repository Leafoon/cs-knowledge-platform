"use client";

import { useState, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Play, RotateCcw, Cpu, Zap, RefreshCw } from "lucide-react";

type InstructionType = "TSL" | "CAS" | "LLSC";

interface StepLog {
  cpu: number;
  action: string;
  result?: string;
}

const TSL_STEPS: StepLog[] = [
  { cpu: 0, action: "CPU0: TSL(old_val, lock)" },
  { cpu: 0, action: "CPU0: Reads lock=0, sets lock=1 atomically" },
  { cpu: 0, action: "CPU0: Got lock! old_val=0 -> enter CS", result: "成功获取锁" },
  { cpu: 1, action: "CPU1: TSL(old_val, lock)" },
  { cpu: 1, action: "CPU1: Reads lock=1, sets lock=1 atomically" },
  { cpu: 1, action: "CPU1: old_val=1 -> spin wait" },
  { cpu: 0, action: "CPU0: Leaving CS, lock = 0" },
  { cpu: 1, action: "CPU1: TSL(old_val, lock) -> reads 0, sets 1" },
  { cpu: 1, action: "CPU1: Got lock! Enter CS" },
];

const CAS_STEPS: StepLog[] = [
  { cpu: 0, action: "CPU0: Read lock -> val=0" },
  { cpu: 1, action: "CPU1: Read lock -> val=0" },
  { cpu: 0, action: "CPU0: CAS(lock, expected=0, new=1)" },
  { cpu: 0, action: "CPU0: Success! lock was 0, now 1", result: "成功获取锁" },
  { cpu: 1, action: "CPU1: CAS(lock, expected=0, new=1)" },
  { cpu: 1, action: "CPU1: FAIL! lock=1 != expected=0" },
  { cpu: 1, action: "CPU1: Retry: Read lock -> val=1" },
  { cpu: 1, action: "CPU1: CAS(lock, expected=1, new=1) -> FAIL, still 1" },
  { cpu: 0, action: "CPU0: Release lock, lock = 0" },
  { cpu: 1, action: "CPU1: Retry: CAS(lock, expected=0, new=1)" },
  { cpu: 1, action: "CPU1: Success! Enter CS", result: "成功获取锁" },
];

const LLSC_STEPS: StepLog[] = [
  { cpu: 0, action: "CPU0: LL(lock) -> loads lock=0, sets reservation" },
  { cpu: 1, action: "CPU1: LL(lock) -> loads lock=0, sets reservation" },
  { cpu: 0, action: "CPU0: SC(lock, 1) -> reservation valid, lock=1" },
  { cpu: 0, action: "CPU0: SC success! Enter CS", result: "成功获取锁" },
  { cpu: 1, action: "CPU1: SC(lock, 1) -> reservation invalidated by CPU0!" },
  { cpu: 1, action: "CPU1: SC FAIL! Returns 0" },
  { cpu: 1, action: "CPU1: Retry: LL(lock) -> loads lock=1" },
  { cpu: 1, action: "CPU1: SC(lock, 1) -> but lock already 1, spin" },
  { cpu: 0, action: "CPU0: Release lock, lock = 0" },
  { cpu: 1, action: "CPU1: LL(lock) -> loads lock=0" },
  { cpu: 1, action: "CPU1: SC(lock, 1) -> success!" },
  { cpu: 1, action: "CPU1: Enter CS", result: "成功获取锁" },
];

const INSTRUCTION_DATA = {
  TSL: {
    name: "Test and Set Lock (TSL)",
    description: "Atomic read-modify-write. Reads old value and sets to 1 in one bus transaction.",
    steps: TSL_STEPS,
    color: "blue",
    pros: ["Simple to implement", "Hardware guaranteed atomicity"],
    cons: ["Always writes (bus traffic)", "No ABA awareness"],
    syntax: "TSL(old, lock): {\n  old = lock;\n  lock = 1;\n  return old;\n}",
  },
  CAS: {
    name: "Compare-And-Swap (CAS)",
    description: "Atomically compares value with expected, swaps only if equal.",
    steps: CAS_STEPS,
    color: "emerald",
    pros: ["Only writes on success", "ABA detectable with version"],
    cons: ["ABA problem without counter", "Retry loop needed"],
    syntax: "CAS(lock, expected, new): {\n  if (lock == expected) {\n    lock = new;\n    return true;\n  }\n  return false;\n}",
  },
  LLSC: {
    name: "Load-Linked / Store-Conditional (LL/SC)",
    description: "LL loads and sets reservation; SC succeeds only if reservation not invalidated.",
    steps: LLSC_STEPS,
    color: "purple",
    pros: ["No ABA problem", "More flexible than CAS"],
    cons: ["Spurious SC failures", "Complex hardware"],
    syntax: "LL(lock):  load & set reservation\nSC(lock, val): \n  if reservation valid:\n    lock = val; return 1\n  else: return 0",
  },
};

export default function AtomicInstructionComparison() {
  const [activeTab, setActiveTab] = useState<InstructionType>("TSL");
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [lockState, setLockState] = useState<{ lock: number; cpu0Has: boolean; cpu1Has: boolean }>({
    lock: 0,
    cpu0Has: false,
    cpu1Has: false,
  });

  const data = INSTRUCTION_DATA[activeTab];
  const steps = data.steps;

  const reset = useCallback(() => {
    setCurrentStep(0);
    setIsPlaying(false);
    setLockState({ lock: 0, cpu0Has: false, cpu1Has: false });
  }, []);

  useEffect(() => {
    reset();
  }, [activeTab, reset]);

  useEffect(() => {
    if (!isPlaying) return;
    const timer = setInterval(() => {
      setCurrentStep((s) => {
        if (s >= steps.length - 1) {
          setIsPlaying(false);
          return s;
        }
        return s + 1;
      });
    }, 1500);
    return () => clearInterval(timer);
  }, [isPlaying, steps.length]);

  // Update lock state based on step
  useEffect(() => {
    const s = steps[currentStep];
    if (!s) return;
    const text = s.action + (s.result || "");
    const newLock = { ...lockState };
    if (text.includes("lock=1") || text.includes("sets lock=1") || text.includes("now 1")) newLock.lock = 1;
    if (text.includes("lock = 0") || text.includes("lock=0")) newLock.lock = 0;
    if (text.includes("Got lock") || text.includes("Success!") || text.includes("Enter CS")) {
      if (s.cpu === 0) newLock.cpu0Has = true;
      else newLock.cpu1Has = true;
    }
    if (text.includes("Release lock")) {
      if (s.cpu === 0) newLock.cpu0Has = false;
      else newLock.cpu1Has = false;
      newLock.lock = 0;
    }
    setLockState(newLock);
  }, [currentStep, activeTab]);

  const colorMap: Record<string, { bg: string; border: string; text: string; light: string }> = {
    blue: { bg: "bg-blue-500", border: "border-blue-400", text: "text-blue-700", light: "bg-blue-50 dark:bg-blue-900/20" },
    emerald: { bg: "bg-emerald-500", border: "border-emerald-400", text: "text-emerald-700", light: "bg-emerald-50 dark:bg-emerald-900/20" },
    purple: { bg: "bg-purple-500", border: "border-purple-400", text: "text-purple-700", light: "bg-purple-50 dark:bg-purple-900/20" },
  };
  const colors = colorMap[data.color];

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-indigo-50 rounded-xl shadow-lg dark:from-gray-900 dark:to-gray-800">
      <h2 className="text-2xl font-bold text-slate-800 dark:text-gray-100 mb-6 text-center">
        Atomic Instruction Comparison
      </h2>

      {/* Tab Selector */}
      <div className="flex justify-center gap-2 mb-6">
        {(["TSL", "CAS", "LLSC"] as InstructionType[]).map((type) => {
          const c = colorMap[INSTRUCTION_DATA[type].color];
          return (
            <button
              key={type}
              onClick={() => setActiveTab(type)}
              className={`px-4 py-2 rounded-lg font-medium text-sm transition-all ${
                activeTab === type
                  ? `${c.bg} text-white shadow-md`
                  : "bg-white dark:bg-gray-800 text-slate-600 dark:text-gray-300 border border-slate-200 dark:border-gray-700 hover:bg-slate-50 dark:hover:bg-gray-700"
              }`}
            >
              {type}
            </button>
          );
        })}
      </div>

      {/* Instruction Info */}
      <div className={`${colors.light} rounded-lg p-4 mb-6 border ${colors.border}`}>
        <h3 className={`font-bold text-lg ${colors.text} dark:text-white mb-1`}>{data.name}</h3>
        <p className="text-sm text-slate-600 dark:text-gray-300 mb-3">{data.description}</p>
        <pre className="text-xs font-mono bg-white dark:bg-gray-800 p-3 rounded border border-slate-200 dark:border-gray-700 text-slate-700 dark:text-gray-200 overflow-x-auto">
          {data.syntax}
        </pre>
      </div>

      {/* Race Visualization */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        {/* CPU 0 */}
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-slate-200 dark:border-gray-700">
          <div className="flex items-center gap-2 mb-3">
            <Cpu className="w-5 h-5 text-blue-500" />
            <span className="font-bold text-slate-700 dark:text-gray-200">CPU 0</span>
            {lockState.cpu0Has && (
              <motion.div initial={{ scale: 0 }} animate={{ scale: 1 }} className="ml-auto px-2 py-0.5 bg-green-100 text-green-700 dark:bg-green-800 dark:text-green-200 text-xs rounded-full">
                In CS
              </motion.div>
            )}
          </div>
          <div className="h-20 flex items-center justify-center">
            <motion.div
              animate={{
                scale: lockState.cpu0Has ? [1, 1.1, 1] : 1,
                backgroundColor: lockState.cpu0Has ? "#10b981" : "#e2e8f0",
              }}
              transition={{ repeat: lockState.cpu0Has ? Infinity : 0, duration: 1 }}
              className="w-16 h-16 rounded-full flex items-center justify-center"
            >
              <Cpu className={`w-8 h-8 ${lockState.cpu0Has ? "text-white" : "text-slate-400"}`} />
            </motion.div>
          </div>
        </div>

        {/* Lock */}
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-slate-200 dark:border-gray-700">
          <div className="flex items-center justify-center gap-2 mb-3">
            <Zap className="w-5 h-5 text-amber-500" />
            <span className="font-bold text-slate-700 dark:text-gray-200">Shared Lock</span>
          </div>
          <div className="h-20 flex items-center justify-center">
            <motion.div
              animate={{
                backgroundColor: lockState.lock === 1 ? "#ef4444" : "#22c55e",
                scale: lockState.lock === 1 ? 1.05 : 1,
              }}
              className="w-20 h-14 rounded-lg flex flex-col items-center justify-center text-white font-bold shadow-lg"
            >
              <span className="text-2xl">{lockState.lock}</span>
              <span className="text-xs">{lockState.lock === 1 ? "LOCKED" : "FREE"}</span>
            </motion.div>
          </div>
        </div>

        {/* CPU 1 */}
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-slate-200 dark:border-gray-700">
          <div className="flex items-center gap-2 mb-3">
            <Cpu className="w-5 h-5 text-purple-500" />
            <span className="font-bold text-slate-700 dark:text-gray-200">CPU 1</span>
            {lockState.cpu1Has && (
              <motion.div initial={{ scale: 0 }} animate={{ scale: 1 }} className="ml-auto px-2 py-0.5 bg-green-100 text-green-700 dark:bg-green-800 dark:text-green-200 text-xs rounded-full">
                In CS
              </motion.div>
            )}
          </div>
          <div className="h-20 flex items-center justify-center">
            <motion.div
              animate={{
                scale: lockState.cpu1Has ? [1, 1.1, 1] : 1,
                backgroundColor: lockState.cpu1Has ? "#10b981" : "#e2e8f0",
              }}
              transition={{ repeat: lockState.cpu1Has ? Infinity : 0, duration: 1 }}
              className="w-16 h-16 rounded-full flex items-center justify-center"
            >
              <Cpu className={`w-8 h-8 ${lockState.cpu1Has ? "text-white" : "text-slate-400"}`} />
            </motion.div>
          </div>
        </div>
      </div>

      {/* Step Timeline */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-slate-200 dark:border-gray-700 mb-4">
        <h4 className="font-semibold text-slate-700 dark:text-gray-200 mb-3">Execution Trace</h4>
        <div className="space-y-1 max-h-48 overflow-y-auto">
          {steps.map((s, i) => (
            <motion.div
              key={i}
              initial={false}
              animate={{
                backgroundColor: i === currentStep ? (s.result ? "#dcfce7" : "#fef9c3") : "transparent",
                opacity: i <= currentStep ? 1 : 0.4,
              }}
              className={`flex items-start gap-2 px-3 py-1.5 rounded text-sm font-mono ${
                i === currentStep ? "font-medium" : ""
              }`}
            >
              <span className={`flex-shrink-0 w-5 h-5 rounded-full flex items-center justify-center text-xs ${
                s.cpu === 0 ? "bg-blue-100 text-blue-700 dark:bg-blue-800 dark:text-blue-200"
                  : "bg-purple-100 text-purple-700 dark:bg-purple-800 dark:text-purple-200"
              }`}>
                {s.cpu}
              </span>
              <span className="text-slate-700 dark:text-gray-200">
                {s.result || s.action}
                {s.result && <span className="ml-2 text-green-600 dark:text-green-400">&#10003;</span>}
              </span>
            </motion.div>
          ))}
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
          className="p-2 rounded-lg bg-slate-200 hover:bg-slate-300 dark:bg-gray-700 dark:hover:bg-gray-600 text-slate-600 dark:text-gray-300 disabled:opacity-40"
        >
          <RefreshCw className="w-5 h-5 rotate-180" />
        </button>
        <button
          onClick={() => setIsPlaying(!isPlaying)}
          className="px-5 py-2 rounded-lg bg-indigo-500 hover:bg-indigo-600 text-white font-medium flex items-center gap-2"
        >
          {isPlaying ? <><RefreshCw className="w-4 h-4 animate-spin" /> Pause</> : <><Play className="w-4 h-4" /> Play</>}
        </button>
        <button
          onClick={() => setCurrentStep((s) => Math.min(steps.length - 1, s + 1))}
          disabled={currentStep >= steps.length - 1}
          className="p-2 rounded-lg bg-slate-200 hover:bg-slate-300 dark:bg-gray-700 dark:hover:bg-gray-600 text-slate-600 dark:text-gray-300 disabled:opacity-40"
        >
          <RefreshCw className="w-5 h-5" />
        </button>
      </div>

      {/* Pros/Cons */}
      <div className="mt-6 grid grid-cols-2 gap-4">
        <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-3 border border-green-200 dark:border-green-800">
          <h5 className="text-sm font-bold text-green-700 dark:text-green-300 mb-1">Advantages</h5>
          <ul className="text-xs text-green-600 dark:text-green-400 space-y-1">
            {data.pros.map((p, i) => <li key={i}>+ {p}</li>)}
          </ul>
        </div>
        <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-3 border border-red-200 dark:border-red-800">
          <h5 className="text-sm font-bold text-red-700 dark:text-red-300 mb-1">Disadvantages</h5>
          <ul className="text-xs text-red-600 dark:text-red-400 space-y-1">
            {data.cons.map((c, i) => <li key={i}>- {c}</li>)}
          </ul>
        </div>
      </div>
    </div>
  );
}
