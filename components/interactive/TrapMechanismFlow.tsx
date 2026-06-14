"use client";

import { useState, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Play,
  RotateCcw,
  Cpu,
  Terminal,
  ArrowDown,
  Zap,
  Server,
  RefreshCw,
  CheckCircle2,
  ChevronRight,
} from "lucide-react";

interface TrapStep {
  id: number;
  title: string;
  subtitle: string;
  mode: "User" | "Hardware" | "Kernel" | "Return";
  icon: React.ReactNode;
  description: string;
}

const steps: TrapStep[] = [
  {
    id: 1,
    title: "write() → ecall",
    subtitle: "User program invokes syscall",
    mode: "User",
    icon: <Terminal className="w-5 h-5" />,
    description: "User program calls write(). The C library wrapper places syscall number in a7 and executes the ecall instruction.",
  },
  {
    id: 2,
    title: "Hardware Trap",
    subtitle: "CPU switches to S-mode",
    mode: "Hardware",
    icon: <Cpu className="w-5 h-5" />,
    description: "Hardware saves PC to sepc, sets scause to cause code, switches privilege to S-mode, and jumps to stvec (kernel trap handler).",
  },
  {
    id: 3,
    title: "uservec (trampoline)",
    subtitle: "Save user registers to trapframe",
    mode: "Kernel",
    icon: <Server className="w-5 h-5" />,
    description: "Trampoline code runs at the same virtual address in user & kernel space. Saves all user registers to trapframe, loads kernel page table and stack.",
  },
  {
    id: 4,
    title: "usertrap()",
    subtitle: "Determine trap type & dispatch",
    mode: "Kernel",
    icon: <Zap className="w-5 h-5" />,
    description: "Kernel reads scause to determine trap type (ecall, interrupt, page fault). For ecall, it dispatches to syscall(). Saves stvec, switches to kernel trap handler.",
  },
  {
    id: 5,
    title: "syscall() → sys_write",
    subtitle: "Look up & execute syscall",
    mode: "Kernel",
    icon: <ChevronRight className="w-5 h-5" />,
    description: "syscall() reads a7 for the syscall number, looks up the function pointer in the syscall table, and calls sys_write with arguments from a0-a5.",
  },
  {
    id: 6,
    title: "usertrapret()",
    subtitle: "Prepare return to user mode",
    mode: "Return",
    icon: <RefreshCw className="w-5 h-5" />,
    description: "Sets up trapframe for next trap (kernel satp, kernel stack), sets stvec back to uservec, restores user page table, prepares sret registers.",
  },
  {
    id: 7,
    title: "userret (trampoline)",
    subtitle: "Restore user registers & sret",
    mode: "Return",
    icon: <RefreshCw className="w-5 h-5" />,
    description: "Trampoline restores all user registers from trapframe, switches to user page table, then executes sret to return to user mode at sepc.",
  },
  {
    id: 8,
    title: "Back in User Mode",
    subtitle: "write() returns to user program",
    mode: "User",
    icon: <CheckCircle2 className="w-5 h-5" />,
    description: "sret switches to U-mode and resumes at sepc. The write() library call returns the result. User program continues execution.",
  },
];

const modeConfig = {
  User: {
    bg: "bg-blue-100 dark:bg-blue-950",
    border: "border-blue-400 dark:border-blue-600",
    glow: "shadow-[0_0_20px_rgba(59,130,246,0.5)]",
    badge: "bg-blue-600 text-white",
    text: "text-blue-700 dark:text-blue-300",
    arrow: "text-blue-400 dark:text-blue-500",
  },
  Hardware: {
    bg: "bg-amber-100 dark:bg-amber-950",
    border: "border-amber-400 dark:border-amber-600",
    glow: "shadow-[0_0_20px_rgba(245,158,11,0.5)]",
    badge: "bg-amber-600 text-white",
    text: "text-amber-700 dark:text-amber-300",
    arrow: "text-amber-400 dark:text-amber-500",
  },
  Kernel: {
    bg: "bg-red-100 dark:bg-red-950",
    border: "border-red-400 dark:border-red-600",
    glow: "shadow-[0_0_20px_rgba(239,68,68,0.5)]",
    badge: "bg-red-600 text-white",
    text: "text-red-700 dark:text-red-300",
    arrow: "text-red-400 dark:text-red-500",
  },
  Return: {
    bg: "bg-green-100 dark:bg-green-950",
    border: "border-green-400 dark:border-green-600",
    glow: "shadow-[0_0_20px_rgba(34,197,94,0.5)]",
    badge: "bg-green-600 text-white",
    text: "text-green-700 dark:text-green-300",
    arrow: "text-green-400 dark:text-green-500",
  },
};

export default function TrapMechanismFlow() {
  const [currentStep, setCurrentStep] = useState(-1);
  const [isPlaying, setIsPlaying] = useState(false);
  const [autoPlay, setAutoPlay] = useState(false);

  const advance = useCallback(() => {
    setCurrentStep((prev) => {
      if (prev >= steps.length - 1) {
        setIsPlaying(false);
        setAutoPlay(false);
        return prev;
      }
      return prev + 1;
    });
  }, []);

  const startAnimation = useCallback(() => {
    setCurrentStep(0);
    setIsPlaying(true);
    setAutoPlay(true);
  }, []);

  const reset = useCallback(() => {
    setCurrentStep(-1);
    setIsPlaying(false);
    setAutoPlay(false);
  }, []);

  useEffect(() => {
    if (!autoPlay || currentStep >= steps.length - 1) return;
    const timer = setTimeout(advance, 2000);
    return () => clearTimeout(timer);
  }, [autoPlay, currentStep, advance]);

  const active = currentStep >= 0 ? steps[currentStep] : null;
  const cfg = active ? modeConfig[active.mode] : null;

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-gray-100 dark:from-slate-900 dark:to-slate-800 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2 text-center">
        xv6 Trap Flow
      </h3>
      <p className="text-sm text-slate-500 dark:text-slate-400 mb-6 text-center">
        Step-by-step animation of a trap from user mode through the kernel and back
      </p>

      <div className="flex justify-center gap-3 mb-6">
        <button
          onClick={startAnimation}
          disabled={isPlaying}
          className="flex items-center gap-2 px-5 py-2.5 rounded-lg font-semibold text-sm bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          <Play className="w-4 h-4" />
          {isPlaying ? "Playing..." : "Play"}
        </button>
        <button
          onClick={advance}
          disabled={isPlaying || currentStep >= steps.length - 1}
          className="flex items-center gap-2 px-5 py-2.5 rounded-lg font-semibold text-sm bg-slate-600 text-white hover:bg-slate-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          Step →
        </button>
        <button
          onClick={reset}
          className="flex items-center gap-2 px-5 py-2.5 rounded-lg font-semibold text-sm bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-200 hover:bg-slate-300 dark:hover:bg-slate-600 transition-colors"
        >
          <RotateCcw className="w-4 h-4" />
          Reset
        </button>
      </div>

      <div className="flex flex-col items-center gap-1">
        {steps.map((step, i) => {
          const isActive = i === currentStep;
          const isPast = i < currentStep;
          const sc = modeConfig[step.mode];
          const isFuture = i > currentStep;
          return (
            <div key={step.id} className="flex flex-col items-center">
              <motion.div
                onClick={() => { if (!isPlaying) setCurrentStep(i); }}
                initial={false}
                animate={{
                  scale: isActive ? 1.03 : 1,
                  opacity: isFuture && currentStep >= 0 ? 0.4 : 1,
                }}
                transition={{ type: "spring", stiffness: 300, damping: 25 }}
                className={`
                  relative w-full max-w-2xl rounded-xl border-2 p-4 cursor-pointer
                  transition-colors duration-300
                  ${isActive ? `${sc.bg} ${sc.border} ${sc.glow}` : isPast ? `${sc.bg} ${sc.border} opacity-70` : "bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-700"}
                `}
              >
                <div className="flex items-center gap-3">
                  <div className={`flex items-center justify-center w-9 h-9 rounded-full text-white text-sm font-bold shrink-0 ${isPast || isActive ? sc.badge : "bg-slate-300 dark:bg-slate-600"}`}>
                    {isPast ? <CheckCircle2 className="w-5 h-5" /> : step.id}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className={`font-bold text-sm ${isActive || isPast ? sc.text : "text-slate-700 dark:text-slate-300"}`}>
                        {step.icon}
                      </span>
                      <span className={`font-bold text-sm ${isActive || isPast ? sc.text : "text-slate-700 dark:text-slate-300"}`}>
                        {step.title}
                      </span>
                      <span className={`text-xs px-2 py-0.5 rounded-full font-semibold ${sc.badge}`}>
                        {step.mode}
                      </span>
                    </div>
                    <div className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
                      {step.subtitle}
                    </div>
                  </div>
                </div>

                <AnimatePresence>
                  {isActive && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: "auto", opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      transition={{ duration: 0.3 }}
                      className="overflow-hidden"
                    >
                      <p className="mt-3 pt-3 border-t border-slate-200 dark:border-slate-600 text-sm text-slate-700 dark:text-slate-300 leading-relaxed">
                        {step.description}
                      </p>
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.div>

              {i < steps.length - 1 && (
                <div className="flex flex-col items-center my-0.5">
                  <motion.div
                    animate={{
                      opacity: i < currentStep ? 1 : 0.3,
                      y: isActive ? [0, 3, 0] : 0,
                    }}
                    transition={
                      isActive
                        ? { repeat: Infinity, duration: 1 }
                        : { duration: 0.3 }
                    }
                  >
                    <ArrowDown
                      className={`w-5 h-5 ${
                        i < currentStep
                          ? modeConfig[step.mode].arrow
                          : "text-slate-300 dark:text-slate-600"
                      }`}
                    />
                  </motion.div>
                </div>
              )}
            </div>
          );
        })}
      </div>

      <AnimatePresence>
        {cfg && active && (
          <motion.div
            key={active.id}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className={`mt-6 rounded-xl border-2 p-5 ${cfg.bg} ${cfg.border}`}
          >
            <div className="flex items-center gap-2 mb-2">
              {active.icon}
              <span className={`font-bold ${cfg.text}`}>
                Step {active.id}: {active.title}
              </span>
            </div>
            <p className="text-sm text-slate-700 dark:text-slate-300 leading-relaxed">
              {active.description}
            </p>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="mt-6 flex flex-wrap justify-center gap-4 text-xs">
        {(Object.keys(modeConfig) as Array<keyof typeof modeConfig>).map((m) => {
          const c = modeConfig[m];
          return (
            <div key={m} className="flex items-center gap-1.5">
              <div className={`w-3 h-3 rounded-full ${c.badge}`} />
              <span className="text-slate-600 dark:text-slate-400 font-medium">{m} Mode</span>
            </div>
          );
        })}
      </div>

      <div className="mt-4 bg-slate-100 dark:bg-slate-800 rounded-lg p-4 text-xs text-slate-600 dark:text-slate-400">
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
          <div><strong>sepc:</strong> Saved exception PC (user&apos;s return address)</div>
          <div><strong>scause:</strong> Trap cause code (ecall = 8)</div>
          <div><strong>stvec:</strong> Kernel trap handler address</div>
          <div><strong>satp:</strong> Page table register (switches address space)</div>
          <div><strong>sret:</strong> Return from supervisor trap</div>
          <div><strong>Trapframe:</strong> Per-process saved register state</div>
        </div>
      </div>
    </div>
  );
}
