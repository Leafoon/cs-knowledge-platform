"use client";

import { useState, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Play,
  Pause,
  RotateCcw,
  ChevronRight,
  ChevronLeft,
  Cpu,
  AlertTriangle,
  ArrowRight,
  ArrowLeft,
} from "lucide-react";

const REGISTERS_A = [
  { name: "ra", val: "0x00401234" },
  { name: "sp", val: "0x7fff_8000" },
  { name: "a0", val: "0x00000005" },
  { name: "a1", val: "0x00000003" },
  { name: "t0", val: "0xdeadbeef" },
  { name: "t1", val: "0xcafebabe" },
];

const REGISTERS_B = [
  { name: "ra", val: "0x00405678" },
  { name: "sp", val: "0x7fff_4000" },
  { name: "a0", val: "0x0000000a" },
  { name: "a1", val: "0x00000007" },
  { name: "t0", val: "0x12345678" },
  { name: "t1", val: "0x9abcdef0" },
];

const KERNEL_CTX_A = [
  { name: "ra", val: "0x80002340" },
  { name: "sp", val: "0x80010000" },
  { name: "s0", val: "0x00000001" },
  { name: "s1", val: "0x00000002" },
];

const KERNEL_CTX_B = [
  { name: "ra", val: "0x80004560" },
  { name: "sp", val: "0x80014000" },
  { name: "s0", val: "0x00000009" },
  { name: "s1", val: "0x0000000a" },
];

const STEPS = [
  {
    title: "Step 0: Process A Running (User Mode)",
    desc: "Process A executes in user mode (U-mode). CPU registers hold A's user context.",
    cpuMode: "U-mode",
    activeProc: "A",
    highlight: "userA",
  },
  {
    title: "Step 1: Timer Interrupt Fires",
    desc: "Hardware timer triggers interrupt. CPU saves PC to sepc, sets scause, switches to S-mode. stvec points to kernelvec.",
    cpuMode: "S-mode",
    activeProc: "A",
    highlight: "interrupt",
  },
  {
    title: "Step 2: uservec() Saves User Registers",
    desc: "uservec() saves all user registers to Process A's trapframe (in p->trapframe). Registers now in memory.",
    cpuMode: "S-mode",
    activeProc: "A",
    highlight: "trapframe",
  },
  {
    title: "Step 3: swtch() Saves A's Kernel Context",
    desc: "sched() calls swtch(&p->context, &cpu->context). Saves A's kernel ra, sp, s0-s11 to A's context struct.",
    cpuMode: "S-mode",
    activeProc: "A",
    highlight: "swtchSave",
  },
  {
    title: "Step 4: Scheduler Picks Process B",
    desc: "Scheduler selects Process B from ready queue. swtch(&cpu->context, &p->context) loads B's kernel context.",
    cpuMode: "S-mode",
    activeProc: "B",
    highlight: "swtchLoad",
  },
  {
    title: "Step 5: usertrapret() + userret()",
    desc: "usertrapret() restores trap registers. userret() restores B's user registers from B's trapframe back to CPU.",
    cpuMode: "S-mode→U-mode",
    activeProc: "B",
    highlight: "restore",
  },
  {
    title: "Step 6: Process B Running (User Mode)",
    desc: "Process B resumes execution in user mode with its saved register values restored.",
    cpuMode: "U-mode",
    activeProc: "B",
    highlight: "userB",
  },
];

const colorForStep = (i: number) => {
  if (i === 1) return "red";
  if (i >= 3 && i <= 4) return "purple";
  return "blue";
};

export default function ContextSwitchAnimation() {
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => {
      setStep((s) => {
        if (s >= STEPS.length - 1) {
          setPlaying(false);
          return s;
        }
        return s + 1;
      });
    }, 2200);
    return () => clearInterval(id);
  }, [playing]);

  const reset = useCallback(() => {
    setStep(0);
    setPlaying(false);
  }, []);

  const prev = useCallback(() => setStep((s) => Math.max(0, s - 1)), []);
  const next = useCallback(
    () => setStep((s) => Math.min(STEPS.length - 1, s + 1)),
    []
  );

  const s = STEPS[step];

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-blue-50 rounded-xl shadow-lg dark:from-gray-900 dark:to-gray-800">
      <h2 className="text-2xl font-bold text-slate-800 dark:text-gray-100 mb-6 text-center flex items-center justify-center gap-2">
        <Cpu className="w-7 h-7 text-blue-600 dark:text-blue-400" />
        Context Switch Animation (xv6 RISC-V)
      </h2>

      {/* Step description */}
      <AnimatePresence mode="wait">
        <motion.div
          key={step}
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: 10 }}
          transition={{ duration: 0.3 }}
          className={`mb-6 p-4 rounded-lg border-l-4 ${
            colorForStep(step) === "red"
              ? "bg-red-50 border-red-500 dark:bg-red-900/30 dark:border-red-400"
              : colorForStep(step) === "purple"
              ? "bg-purple-50 border-purple-500 dark:bg-purple-900/30 dark:border-purple-400"
              : "bg-blue-50 border-blue-500 dark:bg-blue-900/30 dark:border-blue-400"
          }`}
        >
          <h3 className="font-bold text-slate-800 dark:text-gray-100 text-lg">
            {s.title}
          </h3>
          <p className="text-sm text-slate-600 dark:text-gray-300 mt-1">
            {s.desc}
          </p>
        </motion.div>
      </AnimatePresence>

      {/* Main diagram area */}
      <div className="grid grid-cols-3 gap-4 mb-6 items-start">
        {/* Process A */}
        <motion.div
          animate={{
            opacity:
              s.highlight === "userA" || s.highlight === "trapframe" || s.highlight === "swtchSave"
                ? 1
                : s.activeProc === "A"
                ? 0.7
                : 0.4,
          }}
          className="flex flex-col items-center"
        >
          <div
            className={`w-full p-3 rounded-lg border-2 transition-colors duration-300 ${
              s.activeProc === "A"
                ? "border-blue-500 bg-blue-100 dark:bg-blue-900/40 dark:border-blue-400"
                : "border-slate-300 bg-white dark:bg-gray-800 dark:border-gray-600"
            }`}
          >
            <div className="flex items-center gap-2 mb-2">
              <div className="w-8 h-8 rounded-full bg-blue-500 flex items-center justify-center text-white font-bold text-sm">
                A
              </div>
              <span className="font-semibold text-slate-800 dark:text-gray-100 text-sm">
                Process A
              </span>
            </div>

            {/* User registers */}
            <div className="mb-2">
              <p className="text-xs font-semibold text-blue-700 dark:text-blue-300 mb-1">
                User Registers (user context)
              </p>
              <div className="space-y-0.5">
                {REGISTERS_A.map((r) => (
                  <motion.div
                    key={r.name}
                    animate={{
                      opacity:
                        s.highlight === "userA" || s.highlight === "restore" ? 1 : 0.5,
                      x:
                        s.highlight === "trapframe" && step === 2 ? 20 : 0,
                    }}
                    transition={{ duration: 0.5 }}
                    className="flex justify-between text-xs font-mono bg-blue-50 dark:bg-blue-900/30 px-2 py-0.5 rounded"
                  >
                    <span className="text-blue-800 dark:text-blue-300">{r.name}</span>
                    <span className="text-slate-600 dark:text-gray-400">{r.val}</span>
                  </motion.div>
                ))}
              </div>
            </div>

            {/* Kernel context */}
            <div>
              <p className="text-xs font-semibold text-indigo-700 dark:text-indigo-300 mb-1">
                Kernel Context (swtch)
              </p>
              <div className="space-y-0.5">
                {KERNEL_CTX_A.map((r) => (
                  <motion.div
                    key={r.name}
                    animate={{
                      opacity: s.highlight === "swtchSave" ? 1 : 0.5,
                    }}
                    className="flex justify-between text-xs font-mono bg-indigo-50 dark:bg-indigo-900/30 px-2 py-0.5 rounded"
                  >
                    <span className="text-indigo-800 dark:text-indigo-300">{r.name}</span>
                    <span className="text-slate-600 dark:text-gray-400">{r.val}</span>
                  </motion.div>
                ))}
              </div>
            </div>
          </div>
        </motion.div>

        {/* CPU Center */}
        <div className="flex flex-col items-center">
          <motion.div
            animate={{
              borderColor:
                s.cpuMode === "U-mode"
                  ? "#3b82f6"
                  : s.cpuMode === "S-mode"
                  ? "#ef4444"
                  : "#a855f7",
            }}
            className="w-full p-3 rounded-lg border-2 bg-white dark:bg-gray-800"
          >
            <div className="flex items-center justify-center gap-2 mb-2">
              <Cpu className="w-5 h-5 text-slate-700 dark:text-gray-300" />
              <span className="font-semibold text-slate-800 dark:text-gray-100 text-sm">
                CPU
              </span>
            </div>

            {/* Mode badge */}
            <motion.div
              layout
              className={`text-center text-xs font-bold px-3 py-1 rounded-full mb-2 ${
                s.cpuMode === "U-mode"
                  ? "bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-300"
                  : s.cpuMode === "S-mode"
                  ? "bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-300"
                  : "bg-purple-100 text-purple-700 dark:bg-purple-900/40 dark:text-purple-300"
              }`}
            >
              {s.cpuMode}
            </motion.div>

            {/* Registers in CPU */}
            <p className="text-xs font-semibold text-slate-600 dark:text-gray-400 mb-1">
              Register File
            </p>
            <div className="space-y-0.5">
              {(s.activeProc === "A" ? REGISTERS_A : REGISTERS_B).map((r) => (
                <motion.div
                  key={r.name}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="flex justify-between text-xs font-mono bg-amber-50 dark:bg-amber-900/20 px-2 py-0.5 rounded"
                >
                  <span className="text-amber-800 dark:text-amber-300">{r.name}</span>
                  <span className="text-slate-600 dark:text-gray-400">{r.val}</span>
                </motion.div>
              ))}
            </div>

            {/* Special registers */}
            <div className="mt-2 pt-2 border-t border-slate-200 dark:border-gray-600">
              <div className="flex justify-between text-xs font-mono px-2">
                <span className="text-slate-500 dark:text-gray-400">sepc</span>
                <span className="text-slate-600 dark:text-gray-300">
                  {step >= 1 ? "0x00401238" : "---"}
                </span>
              </div>
              <div className="flex justify-between text-xs font-mono px-2">
                <span className="text-slate-500 dark:text-gray-400">scause</span>
                <span className="text-slate-600 dark:text-gray-300">
                  {step >= 1 ? "0x8000000000000005" : "---"}
                </span>
              </div>
              <div className="flex justify-between text-xs font-mono px-2">
                <span className="text-slate-500 dark:text-gray-400">stvec</span>
                <span className="text-slate-600 dark:text-gray-300">0x80002000</span>
              </div>
            </div>
          </motion.div>

          {/* Interrupt indicator */}
          <AnimatePresence>
            {step === 1 && (
              <motion.div
                initial={{ opacity: 0, scale: 0.5 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.5 }}
                className="mt-2 flex items-center gap-1 text-red-600 dark:text-red-400"
              >
                <AlertTriangle className="w-4 h-4" />
                <span className="text-xs font-bold">TIMER INTERRUPT</span>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Data flow arrows */}
          <AnimatePresence>
            {(step === 2 || step === 3 || step === 4 || step === 5) && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="mt-3 flex items-center gap-2 text-xs font-semibold"
              >
                {step <= 3 ? (
                  <>
                    <ArrowRight className="w-4 h-4 text-purple-500" />
                    <span className="text-purple-600 dark:text-purple-400">
                      {step === 2
                        ? "Regs → Trapframe"
                        : step === 3
                        ? "CPU → context"
                        : ""}
                    </span>
                  </>
                ) : (
                  <>
                    <ArrowLeft className="w-4 h-4 text-green-500" />
                    <span className="text-green-600 dark:text-green-400">
                      {step === 4 ? "context → CPU" : "Trapframe → Regs"}
                    </span>
                  </>
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Process B */}
        <motion.div
          animate={{
            opacity:
              s.highlight === "userB" || s.highlight === "restore" || s.highlight === "swtchLoad"
                ? 1
                : s.activeProc === "B"
                ? 0.7
                : 0.4,
          }}
          className="flex flex-col items-center"
        >
          <div
            className={`w-full p-3 rounded-lg border-2 transition-colors duration-300 ${
              s.activeProc === "B"
                ? "border-green-500 bg-green-100 dark:bg-green-900/40 dark:border-green-400"
                : "border-slate-300 bg-white dark:bg-gray-800 dark:border-gray-600"
            }`}
          >
            <div className="flex items-center gap-2 mb-2">
              <div className="w-8 h-8 rounded-full bg-green-500 flex items-center justify-center text-white font-bold text-sm">
                B
              </div>
              <span className="font-semibold text-slate-800 dark:text-gray-100 text-sm">
                Process B
              </span>
            </div>

            {/* User registers */}
            <div className="mb-2">
              <p className="text-xs font-semibold text-green-700 dark:text-green-300 mb-1">
                User Registers (user context)
              </p>
              <div className="space-y-0.5">
                {REGISTERS_B.map((r) => (
                  <motion.div
                    key={r.name}
                    animate={{
                      opacity:
                        s.highlight === "userB" || s.highlight === "restore" ? 1 : 0.5,
                      x: s.highlight === "restore" && step === 5 ? -20 : 0,
                    }}
                    transition={{ duration: 0.5 }}
                    className="flex justify-between text-xs font-mono bg-green-50 dark:bg-green-900/30 px-2 py-0.5 rounded"
                  >
                    <span className="text-green-800 dark:text-green-300">{r.name}</span>
                    <span className="text-slate-600 dark:text-gray-400">{r.val}</span>
                  </motion.div>
                ))}
              </div>
            </div>

            {/* Kernel context */}
            <div>
              <p className="text-xs font-semibold text-teal-700 dark:text-teal-300 mb-1">
                Kernel Context (swtch)
              </p>
              <div className="space-y-0.5">
                {KERNEL_CTX_B.map((r) => (
                  <motion.div
                    key={r.name}
                    animate={{
                      opacity: s.highlight === "swtchLoad" ? 1 : 0.5,
                    }}
                    className="flex justify-between text-xs font-mono bg-teal-50 dark:bg-teal-900/30 px-2 py-0.5 rounded"
                  >
                    <span className="text-teal-800 dark:text-teal-300">{r.name}</span>
                    <span className="text-slate-600 dark:text-gray-400">{r.val}</span>
                  </motion.div>
                ))}
              </div>
            </div>
          </div>
        </motion.div>
      </div>

      {/* Step indicator dots */}
      <div className="flex justify-center gap-2 mb-4">
        {STEPS.map((_, i) => (
          <button
            key={i}
            onClick={() => setStep(i)}
            className={`w-3 h-3 rounded-full transition-all duration-200 ${
              i === step
                ? "bg-blue-600 scale-125 dark:bg-blue-400"
                : "bg-slate-300 dark:bg-gray-600 hover:bg-slate-400 dark:hover:bg-gray-500"
            }`}
            aria-label={`Go to step ${i}`}
          />
        ))}
      </div>

      {/* Controls */}
      <div className="flex justify-center gap-3">
        <button
          onClick={reset}
          className="px-4 py-2 bg-slate-600 text-white rounded-lg hover:bg-slate-700 disabled:opacity-50 flex items-center gap-1 text-sm"
        >
          <RotateCcw className="w-4 h-4" />
          Reset
        </button>
        <button
          onClick={prev}
          disabled={step === 0}
          className="px-4 py-2 bg-slate-600 text-white rounded-lg hover:bg-slate-700 disabled:opacity-50 flex items-center gap-1 text-sm"
        >
          <ChevronLeft className="w-4 h-4" />
          Prev
        </button>
        <button
          onClick={() => setPlaying(!playing)}
          className={`px-4 py-2 text-white rounded-lg flex items-center gap-1 text-sm ${
            playing
              ? "bg-amber-600 hover:bg-amber-700"
              : "bg-blue-600 hover:bg-blue-700"
          }`}
        >
          {playing ? (
            <>
              <Pause className="w-4 h-4" /> Pause
            </>
          ) : (
            <>
              <Play className="w-4 h-4" /> Play
            </>
          )}
        </button>
        <button
          onClick={next}
          disabled={step === STEPS.length - 1}
          className="px-4 py-2 bg-slate-600 text-white rounded-lg hover:bg-slate-700 disabled:opacity-50 flex items-center gap-1 text-sm"
        >
          Next
          <ChevronRight className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}
