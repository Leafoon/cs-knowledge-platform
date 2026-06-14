"use client";

import { useState, useCallback, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Play, Pause, RotateCcw, SkipForward, Lock, Unlock, AlertTriangle, CheckCircle, Activity } from "lucide-react";

type ProcessState = "running" | "waiting" | "blocked" | "deadlocked";
type LockState = "free" | "held";
type Scenario = "deadlock" | "safe";

interface ProcessInfo {
  id: string;
  name: string;
  state: ProcessState;
  holdsLock: string | null;
  waitingFor: string | null;
  order: ("A" | "B")[];
}

interface LockInfo {
  id: string;
  state: LockState;
  heldBy: string | null;
}

interface Step {
  label: string;
  description: string;
  action: string;
}

const PROCESS_COLORS: Record<ProcessState, string> = {
  running: "from-green-400 to-emerald-500",
  waiting: "from-yellow-400 to-amber-500",
  blocked: "from-red-400 to-rose-500",
  deadlocked: "from-red-600 to-red-800",
};

const PROCESS_TEXT: Record<ProcessState, string> = {
  running: "text-green-900",
  waiting: "text-yellow-900",
  blocked: "text-red-900",
  deadlocked: "text-white",
};

const LOCK_COLORS: Record<LockState, string> = {
  free: "from-blue-300 to-blue-400",
  held: "from-orange-400 to-orange-500",
};

const deadlockSteps: Step[] = [
  { label: "Step 1", description: "P1 requests Lock A", action: "P1 acquires Lock A" },
  { label: "Step 2", description: "P2 requests Lock B", action: "P2 acquires Lock B" },
  { label: "Step 3", description: "P1 requests Lock B", action: "P1 waits for Lock B (held by P2)" },
  { label: "Step 4", description: "P2 requests Lock A", action: "P2 waits for Lock A (held by P1)" },
  { label: "DEADLOCK", description: "Circular wait detected!", action: "Neither process can proceed" },
];

const safeSteps: Step[] = [
  { label: "Step 1", description: "P1 requests Lock A", action: "P1 acquires Lock A" },
  { label: "Step 2", description: "P1 requests Lock B", action: "P1 acquires Lock B (consistent ordering)" },
  { label: "Step 3", description: "P1 releases both locks", action: "P1 releases Lock A and Lock B" },
  { label: "Step 4", description: "P2 requests Lock A", action: "P2 acquires Lock A" },
  { label: "Step 5", description: "P2 requests Lock B", action: "P2 acquires Lock B" },
  { label: "SAFE", description: "No deadlock! Consistent lock ordering prevents it.", action: "Both processes complete successfully" },
];

export default function DeadlockScenarioDemo() {
  const [scenario, setScenario] = useState<Scenario>("deadlock");
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [log, setLog] = useState<string[]>([]);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  const steps = scenario === "deadlock" ? deadlockSteps : safeSteps;

  const getProcessStates = useCallback((step: number): ProcessInfo[] => {
    if (scenario === "deadlock") {
      switch (step) {
        case 0:
          return [
            { id: "P1", name: "Process 1", state: "running", holdsLock: null, waitingFor: null, order: ["A", "B"] },
            { id: "P2", name: "Process 2", state: "running", holdsLock: null, waitingFor: null, order: ["B", "A"] },
          ];
        case 1:
          return [
            { id: "P1", name: "Process 1", state: "running", holdsLock: "A", waitingFor: null, order: ["A", "B"] },
            { id: "P2", name: "Process 2", state: "running", holdsLock: null, waitingFor: null, order: ["B", "A"] },
          ];
        case 2:
          return [
            { id: "P1", name: "Process 1", state: "running", holdsLock: "A", waitingFor: null, order: ["A", "B"] },
            { id: "P2", name: "Process 2", state: "running", holdsLock: "B", waitingFor: null, order: ["B", "A"] },
          ];
        case 3:
          return [
            { id: "P1", name: "Process 1", state: "waiting", holdsLock: "A", waitingFor: "B", order: ["A", "B"] },
            { id: "P2", name: "Process 2", state: "running", holdsLock: "B", waitingFor: null, order: ["B", "A"] },
          ];
        case 4:
          return [
            { id: "P1", name: "Process 1", state: "deadlocked", holdsLock: "A", waitingFor: "B", order: ["A", "B"] },
            { id: "P2", name: "Process 2", state: "deadlocked", holdsLock: "B", waitingFor: "A", order: ["B", "A"] },
          ];
        default:
          return [];
      }
    } else {
      // safe scenario
      switch (step) {
        case 0:
          return [
            { id: "P1", name: "Process 1", state: "running", holdsLock: null, waitingFor: null, order: ["A", "B"] },
            { id: "P2", name: "Process 2", state: "running", holdsLock: null, waitingFor: null, order: ["A", "B"] },
          ];
        case 1:
          return [
            { id: "P1", name: "Process 1", state: "running", holdsLock: "A", waitingFor: null, order: ["A", "B"] },
            { id: "P2", name: "Process 2", state: "running", holdsLock: null, waitingFor: null, order: ["A", "B"] },
          ];
        case 2:
          return [
            { id: "P1", name: "Process 1", state: "running", holdsLock: "AB", waitingFor: null, order: ["A", "B"] },
            { id: "P2", name: "Process 2", state: "running", holdsLock: null, waitingFor: null, order: ["A", "B"] },
          ];
        case 3:
          return [
            { id: "P1", name: "Process 1", state: "running", holdsLock: null, waitingFor: null, order: ["A", "B"] },
            { id: "P2", name: "Process 2", state: "running", holdsLock: null, waitingFor: null, order: ["A", "B"] },
          ];
        case 4:
          return [
            { id: "P1", name: "Process 1", state: "running", holdsLock: null, waitingFor: null, order: ["A", "B"] },
            { id: "P2", name: "Process 2", state: "running", holdsLock: "A", waitingFor: null, order: ["A", "B"] },
          ];
        case 5:
          return [
            { id: "P1", name: "Process 1", state: "running", holdsLock: null, waitingFor: null, order: ["A", "B"] },
            { id: "P2", name: "Process 2", state: "running", holdsLock: "AB", waitingFor: null, order: ["A", "B"] },
          ];
        default:
          return [];
      }
    }
  }, [scenario]);

  const getLockStates = useCallback((step: number): LockInfo[] => {
    if (scenario === "deadlock") {
      switch (step) {
        case 0: return [
          { id: "A", state: "free", heldBy: null },
          { id: "B", state: "free", heldBy: null },
        ];
        case 1: return [
          { id: "A", state: "held", heldBy: "P1" },
          { id: "B", state: "free", heldBy: null },
        ];
        case 2: return [
          { id: "A", state: "held", heldBy: "P1" },
          { id: "B", state: "held", heldBy: "P2" },
        ];
        case 3: return [
          { id: "A", state: "held", heldBy: "P1" },
          { id: "B", state: "held", heldBy: "P2" },
        ];
        case 4: return [
          { id: "A", state: "held", heldBy: "P1" },
          { id: "B", state: "held", heldBy: "P2" },
        ];
        default: return [];
      }
    } else {
      switch (step) {
        case 0: return [
          { id: "A", state: "free", heldBy: null },
          { id: "B", state: "free", heldBy: null },
        ];
        case 1: return [
          { id: "A", state: "held", heldBy: "P1" },
          { id: "B", state: "free", heldBy: null },
        ];
        case 2: return [
          { id: "A", state: "held", heldBy: "P1" },
          { id: "B", state: "held", heldBy: "P1" },
        ];
        case 3: return [
          { id: "A", state: "free", heldBy: null },
          { id: "B", state: "free", heldBy: null },
        ];
        case 4: return [
          { id: "A", state: "held", heldBy: "P2" },
          { id: "B", state: "free", heldBy: null },
        ];
        case 5: return [
          { id: "A", state: "held", heldBy: "P2" },
          { id: "B", state: "held", heldBy: "P2" },
        ];
        default: return [];
      }
    }
  }, [scenario]);

  const processes = getProcessStates(currentStep);
  const locks = getLockStates(currentStep);
  const isFinished = currentStep >= steps.length - 1;

  const addLog = useCallback((msg: string) => {
    setLog((prev) => [msg, ...prev].slice(0, 12));
  }, []);

  const stepForward = useCallback(() => {
    if (currentStep < steps.length - 1) {
      const next = currentStep + 1;
      setCurrentStep(next);
      addLog(`${steps[next].label}: ${steps[next].action}`);
    } else {
      setIsPlaying(false);
      if (intervalRef.current) clearInterval(intervalRef.current);
    }
  }, [currentStep, steps, addLog]);

  const handlePlay = useCallback(() => {
    if (isPlaying) {
      setIsPlaying(false);
      if (intervalRef.current) clearInterval(intervalRef.current);
      return;
    }
    if (isFinished) {
      setCurrentStep(0);
      setLog([]);
    }
    setIsPlaying(true);
    intervalRef.current = setInterval(() => {
      setCurrentStep((prev) => {
        const maxSteps = scenario === "deadlock" ? deadlockSteps.length : safeSteps.length;
        if (prev >= maxSteps - 1) {
          setIsPlaying(false);
          if (intervalRef.current) clearInterval(intervalRef.current);
          return prev;
        }
        const next = prev + 1;
        const allSteps = scenario === "deadlock" ? deadlockSteps : safeSteps;
        addLog(`${allSteps[next].label}: ${allSteps[next].action}`);
        return next;
      });
    }, 1500);
  }, [isPlaying, isFinished, scenario, addLog]);

  const handleReset = useCallback(() => {
    setIsPlaying(false);
    if (intervalRef.current) clearInterval(intervalRef.current);
    setCurrentStep(0);
    setLog([]);
  }, []);

  const handleScenarioSwitch = useCallback((s: Scenario) => {
    setIsPlaying(false);
    if (intervalRef.current) clearInterval(intervalRef.current);
    setScenario(s);
    setCurrentStep(0);
    setLog([]);
  }, []);

  const isDeadlockStep = scenario === "deadlock" && currentStep === deadlockSteps.length - 1;
  const isSafeStep = scenario === "safe" && currentStep === safeSteps.length - 1;

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-red-50 rounded-xl shadow-lg dark:from-gray-900 dark:to-gray-800">
      <h2 className="text-2xl font-bold text-slate-800 dark:text-gray-100 mb-6 text-center">
        Deadlock Scenario Demonstration
      </h2>

      {/* Scenario Selector */}
      <div className="flex justify-center gap-3 mb-6">
        {(["deadlock", "safe"] as const).map((s) => (
          <button
            key={s}
            onClick={() => handleScenarioSwitch(s)}
            className={`px-5 py-2 rounded-lg font-semibold text-sm transition-all ${
              scenario === s
                ? s === "deadlock"
                  ? "bg-red-500 text-white shadow-md"
                  : "bg-green-500 text-white shadow-md"
                : "bg-white dark:bg-gray-700 text-slate-600 dark:text-gray-300 border border-slate-200 dark:border-gray-600 hover:bg-slate-100 dark:hover:bg-gray-600"
            }`}
          >
            {s === "deadlock" ? "Deadlock Scenario" : "Safe Ordering"}
          </button>
        ))}
      </div>

      {/* Explanation */}
      <div className={`mb-6 p-4 rounded-lg text-sm ${
        scenario === "deadlock"
          ? "bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-800 dark:text-red-200"
          : "bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 text-green-800 dark:text-green-200"
      }`}>
        {scenario === "deadlock" ? (
          <p><strong>Deadlock:</strong> P1 acquires Lock A then waits for Lock B. P2 acquires Lock B then waits for Lock A. Both processes are stuck forever.</p>
        ) : (
          <p><strong>Safe Ordering:</strong> Both processes acquire locks in the same order (A then B). No circular wait is possible, so deadlock is prevented.</p>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Main Visualization */}
        <div className="lg:col-span-2">
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-slate-200 dark:border-gray-700">
            {/* Current Step */}
            <div className="text-center mb-4">
              <motion.div
                key={currentStep}
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                className={`inline-block px-4 py-1 rounded-full text-sm font-semibold ${
                  isDeadlockStep
                    ? "bg-red-600 text-white animate-pulse"
                    : isSafeStep
                    ? "bg-green-500 text-white"
                    : "bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300"
                }`}
              >
                {steps[currentStep].label}
              </motion.div>
              <p className="text-sm text-slate-600 dark:text-gray-400 mt-2">{steps[currentStep].description}</p>
            </div>

            {/* Processes */}
            <div className="flex justify-center gap-8 mb-8">
              {processes.map((p) => (
                <motion.div
                  key={p.id}
                  layout
                  className="flex flex-col items-center gap-2"
                >
                  <motion.div
                    animate={{
                      scale: p.state === "deadlocked" ? [1, 1.05, 1] : 1,
                    }}
                    transition={{ repeat: p.state === "deadlocked" ? Infinity : 0, duration: 0.8 }}
                    className={`w-28 h-28 rounded-2xl bg-gradient-to-br ${PROCESS_COLORS[p.state]} flex flex-col items-center justify-center shadow-lg border-2 ${
                      p.state === "deadlocked" ? "border-red-900" : "border-white/30"
                    }`}
                  >
                    <span className="text-lg font-bold text-white">{p.id}</span>
                    <span className={`text-xs font-medium ${PROCESS_TEXT[p.state]} mt-1`}>
                      {p.state.toUpperCase()}
                    </span>
                  </motion.div>
                  <div className="text-xs text-center text-slate-500 dark:text-gray-400">
                    <div>Order: {p.order.join(" -> ")}</div>
                    {p.holdsLock && (
                      <div className="text-orange-600 dark:text-orange-400 font-medium mt-1">
                        Holds: {p.holdsLock === "AB" ? "A, B" : p.holdsLock}
                      </div>
                    )}
                    {p.waitingFor && (
                      <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="text-red-600 dark:text-red-400 font-medium mt-1"
                      >
                        Waiting: {p.waitingFor}
                      </motion.div>
                    )}
                  </div>
                </motion.div>
              ))}
            </div>

            {/* Locks */}
            <div className="flex justify-center gap-8 mb-6">
              {locks.map((l) => (
                <motion.div
                  key={l.id}
                  layout
                  className="flex flex-col items-center gap-2"
                >
                  <motion.div
                    animate={{
                      scale: l.state === "held" ? [1, 1.08, 1] : 1,
                    }}
                    transition={{ duration: 0.3 }}
                    className={`w-20 h-20 rounded-xl bg-gradient-to-br ${LOCK_COLORS[l.state]} flex flex-col items-center justify-center shadow-md border-2 border-white/30`}
                  >
                    {l.state === "free" ? (
                      <Unlock className="w-6 h-6 text-blue-700" />
                    ) : (
                      <Lock className="w-6 h-6 text-orange-800" />
                    )}
                    <span className="text-sm font-bold text-white mt-1">Lock {l.id}</span>
                  </motion.div>
                  <span className="text-xs text-slate-500 dark:text-gray-400">
                    {l.state === "held" ? `Held by ${l.heldBy}` : "Free"}
                  </span>
                </motion.div>
              ))}
            </div>

            {/* Deadlock / Safe indicator */}
            <AnimatePresence>
              {isDeadlockStep && (
                <motion.div
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.8 }}
                  className="flex items-center justify-center gap-3 p-4 bg-red-100 dark:bg-red-900/30 rounded-xl border-2 border-red-400 dark:border-red-600"
                >
                  <AlertTriangle className="w-8 h-8 text-red-600 dark:text-red-400" />
                  <div>
                    <p className="text-lg font-bold text-red-700 dark:text-red-300">DEADLOCK DETECTED</p>
                    <p className="text-sm text-red-600 dark:text-red-400">Circular wait: P1 {'->'} Lock B {'->'} P2 {'->'} Lock A {'->'} P1</p>
                  </div>
                </motion.div>
              )}
              {isSafeStep && (
                <motion.div
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.8 }}
                  className="flex items-center justify-center gap-3 p-4 bg-green-100 dark:bg-green-900/30 rounded-xl border-2 border-green-400 dark:border-green-600"
                >
                  <CheckCircle className="w-8 h-8 text-green-600 dark:text-green-400" />
                  <div>
                    <p className="text-lg font-bold text-green-700 dark:text-green-300">NO DEADLOCK</p>
                    <p className="text-sm text-green-600 dark:text-green-400">Consistent lock ordering prevents circular wait</p>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Controls */}
            <div className="flex items-center justify-center gap-3 mt-6">
              <button
                onClick={handleReset}
                className="p-2 rounded-lg bg-slate-200 dark:bg-gray-700 text-slate-600 dark:text-gray-300 hover:bg-slate-300 dark:hover:bg-gray-600 transition-colors"
                title="Reset"
              >
                <RotateCcw className="w-5 h-5" />
              </button>
              <button
                onClick={handlePlay}
                className={`p-3 rounded-xl shadow-md transition-all ${
                  isPlaying
                    ? "bg-amber-500 text-white hover:bg-amber-600"
                    : "bg-blue-500 text-white hover:bg-blue-600"
                }`}
                title={isPlaying ? "Pause" : "Play"}
              >
                {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
              </button>
              <button
                onClick={stepForward}
                disabled={isFinished}
                className="p-2 rounded-lg bg-slate-200 dark:bg-gray-700 text-slate-600 dark:text-gray-300 hover:bg-slate-300 dark:hover:bg-gray-600 transition-colors disabled:opacity-40"
                title="Step Forward"
              >
                <SkipForward className="w-5 h-5" />
              </button>
            </div>

            {/* Progress */}
            <div className="flex gap-1 mt-4 justify-center">
              {steps.map((s, i) => (
                <div
                  key={i}
                  className={`w-8 h-1.5 rounded-full transition-all ${
                    i <= currentStep
                      ? i === steps.length - 1 && scenario === "deadlock"
                        ? "bg-red-500"
                        : "bg-blue-500"
                      : "bg-slate-200 dark:bg-gray-700"
                  }`}
                />
              ))}
            </div>
          </div>
        </div>

        {/* Event Log */}
        <div className="bg-white dark:bg-gray-800 rounded-xl p-4 border border-slate-200 dark:border-gray-700">
          <h3 className="text-sm font-semibold text-slate-700 dark:text-gray-300 mb-3 flex items-center gap-2">
            <Activity className="w-4 h-4" /> Event Log
          </h3>
          <div className="space-y-1.5 max-h-80 overflow-y-auto">
            <AnimatePresence initial={false}>
              {log.length === 0 && (
                <p className="text-xs text-slate-400 dark:text-gray-500 italic">Press Play or Step to begin...</p>
              )}
              {log.map((entry, i) => (
                <motion.div
                  key={`${entry}-${i}`}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0 }}
                  className={`text-xs p-2 rounded ${
                    i === 0
                      ? "bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 font-medium"
                      : "text-slate-500 dark:text-gray-400"
                  }`}
                >
                  {entry}
                </motion.div>
              ))}
            </AnimatePresence>
          </div>

          {/* Lock ordering comparison */}
          <div className="mt-6 p-3 bg-slate-50 dark:bg-gray-900 rounded-lg">
            <h4 className="text-xs font-semibold text-slate-600 dark:text-gray-400 mb-2">Lock Acquisition Order</h4>
            <div className="space-y-2">
              {processes.map((p) => (
                <div key={p.id} className="flex items-center gap-2">
                  <span className="text-xs font-bold text-slate-700 dark:text-gray-300 w-6">{p.id}</span>
                  <div className="flex gap-1">
                    {p.order.map((lock, idx) => (
                      <span key={idx} className="flex items-center">
                        {idx > 0 && <span className="text-xs text-slate-400 mx-0.5">{'->'}</span>}
                        <span className={`text-xs px-1.5 py-0.5 rounded ${
                          p.holdsLock === lock || p.holdsLock === "AB"
                            ? "bg-orange-200 dark:bg-orange-800 text-orange-800 dark:text-orange-200"
                            : p.waitingFor === lock
                            ? "bg-red-200 dark:bg-red-800 text-red-800 dark:text-red-200"
                            : "bg-slate-200 dark:bg-gray-700 text-slate-600 dark:text-gray-400"
                        }`}>
                          {lock}
                        </span>
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
