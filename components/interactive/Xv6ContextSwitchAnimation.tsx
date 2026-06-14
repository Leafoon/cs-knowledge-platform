"use client";
import React, { useState, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Play, Pause, RotateCcw, SkipForward, ArrowRight, Cpu, HardDrive } from "lucide-react";

interface Step {
  title: string;
  description: string;
  location: string;
  registers: { name: string; value: string; highlight: boolean }[];
  codeHighlight?: number[];
}

const steps: Step[] = [
  {
    title: "进程 A 在用户态运行",
    description: "进程 A 正在用户空间执行代码，使用自己的页表（satp 指向 A 的页表）。",
    location: "user-A",
    registers: [
      { name: "pc", value: "0x10048", highlight: false },
      { name: "sp", value: "0x7fff0000", highlight: false },
      { name: "ra", value: "0x100a0", highlight: false },
      { name: "s0", value: "42", highlight: false },
      { name: "s1", value: "100", highlight: false },
    ],
  },
  {
    title: "定时器中断触发",
    description: "硬件定时器触发中断：CPU 自动保存 pc → sepc，切换到 S 模式，跳转到 stvec（trampoline.S 的 uservec）。",
    location: "hardware",
    registers: [
      { name: "sepc", value: "0x10048", highlight: true },
      { name: "sstatus", value: "SPP=1", highlight: true },
      { name: "scause", value: "0x8000000000000005", highlight: true },
    ],
    codeHighlight: [0, 1],
  },
  {
    title: "uservec: 保存用户寄存器到 trapframe",
    description: "trampoline.S 的 uservec 将所有用户寄存器保存到进程 A 的 trapframe 页。该页在用户页表和内核页表中映射到同一物理地址。",
    location: "kernel-A",
    registers: [
      { name: "a0", value: "trapframe", highlight: true },
      { name: "sp (kernel)", value: "A->kstack+4096", highlight: true },
    ],
    codeHighlight: [2, 3, 4],
  },
  {
    title: "usertrap → yield → sched: 准备切换",
    description: "usertrap() 检测到是定时器中断，调用 yield()。yield() 将进程状态设为 RUNNABLE，获取 ptable.lock，然后调用 sched()。",
    location: "kernel-A",
    registers: [
      { name: "p->state", value: "RUNNABLE", highlight: true },
      { name: "ptable.lock", value: "held", highlight: true },
    ],
    codeHighlight: [5, 6, 7],
  },
  {
    title: "swtch(&p->context, &c->context): 保存 A 的内核寄存器",
    description: "swtch() 将进程 A 的 14 个 callee-saved 寄存器（ra, sp, s0-s11）保存到 A->context 中，然后加载 CPU 调度器的寄存器。",
    location: "swtch-1",
    registers: [
      { name: "A->context.ra", value: "sched+0x40", highlight: true },
      { name: "A->context.sp", value: "A->kstack+2048", highlight: true },
      { name: "A->context.s0-s11", value: "saved ✓", highlight: true },
    ],
    codeHighlight: [8, 9, 10],
  },
  {
    title: "scheduler(): 遍历进程表，选择进程 B",
    description: "swtch 返回到 scheduler()（在 CPU 的调度器栈上运行）。scheduler 遍历 proc[] 表，找到状态为 RUNNABLE 的进程 B。",
    location: "scheduler",
    registers: [
      { name: "c->proc", value: "B", highlight: true },
      { name: "B->state", value: "RUNNING", highlight: true },
    ],
    codeHighlight: [11, 12, 13],
  },
  {
    title: "swtch(&c->context, &B->context): 加载 B 的内核寄存器",
    description: "swtch() 保存调度器的寄存器到 c->context，从 B->context 加载进程 B 的内核寄存器。ret 跳转到 B 上次 sched() 的返回地址。",
    location: "swtch-2",
    registers: [
      { name: "ra", value: "B->context.ra", highlight: true },
      { name: "sp", value: "B->context.sp", highlight: true },
      { name: "s0-s11", value: "restored ✓", highlight: true },
    ],
    codeHighlight: [14, 15, 16],
  },
  {
    title: "usertrapret → userret: 返回用户态",
    description: "进程 B 从 sched() 返回，经过 yield() → usertrap() → usertrapret()。usertrapret 设置 trapframe、页表，最终跳转到 userret 恢复用户寄存器并执行 sret。",
    location: "kernel-B",
    registers: [
      { name: "satp", value: "B->pagetable", highlight: true },
      { name: "sepc", value: "B->trapframe->epc", highlight: true },
    ],
    codeHighlight: [17, 18, 19],
  },
  {
    title: "进程 B 在用户态继续运行",
    description: "sret 将 CPU 切换回用户模式，从 sepc 恢复执行。进程 B 从上次被中断的地方继续运行，完全无感知切换的发生。",
    location: "user-B",
    registers: [
      { name: "pc", value: "0x20048", highlight: false },
      { name: "sp", value: "0x7ffe0000", highlight: false },
      { name: "ra", value: "0x200a0", highlight: false },
      { name: "s0", value: "99", highlight: false },
      { name: "s1", value: "200", highlight: false },
    ],
  },
];

const locationColors: Record<string, { bg: string; border: string; label: string }> = {
  "user-A": { bg: "bg-blue-50 dark:bg-blue-900/30", border: "border-blue-400", label: "进程 A 用户态" },
  "hardware": { bg: "bg-red-50 dark:bg-red-900/30", border: "border-red-400", label: "硬件自动操作" },
  "kernel-A": { bg: "bg-amber-50 dark:bg-amber-900/30", border: "border-amber-400", label: "进程 A 内核态" },
  "swtch-1": { bg: "bg-purple-50 dark:bg-purple-900/30", border: "border-purple-400", label: "swtch() #1" },
  "scheduler": { bg: "bg-green-50 dark:bg-green-900/30", border: "border-green-400", label: "调度器" },
  "swtch-2": { bg: "bg-purple-50 dark:bg-purple-900/30", border: "border-purple-400", label: "swtch() #2" },
  "kernel-B": { bg: "bg-cyan-50 dark:bg-cyan-900/30", border: "border-cyan-400", label: "进程 B 内核态" },
  "user-B": { bg: "bg-green-50 dark:bg-green-900/30", border: "border-green-400", label: "进程 B 用户态" },
};

export default function Xv6ContextSwitchAnimation() {
  const [step, setStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  const current = steps[step];
  const locInfo = locationColors[current.location] || locationColors["user-A"];

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isPlaying && step < steps.length - 1) {
      interval = setInterval(() => {
        setStep((prev) => {
          if (prev >= steps.length - 1) {
            setIsPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, 2500);
    }
    return () => clearInterval(interval);
  }, [isPlaying, step]);

  const handleReset = useCallback(() => {
    setStep(0);
    setIsPlaying(false);
  }, []);

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-indigo-50 rounded-xl shadow-lg dark:from-gray-900 dark:to-gray-800">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-gray-100 mb-6 text-center flex items-center justify-center gap-2">
        <Cpu className="w-6 h-6 text-indigo-600" />
        xv6 上下文切换完整流程
      </h3>

      {/* Progress bar */}
      <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2 mb-6">
        <motion.div
          className="bg-indigo-600 h-2 rounded-full"
          animate={{ width: `${((step + 1) / steps.length) * 100}%` }}
          transition={{ duration: 0.3 }}
        />
      </div>

      {/* Control buttons */}
      <div className="flex justify-center gap-4 mb-6">
        <button
          onClick={handleReset}
          className="px-4 py-2 bg-slate-600 text-white rounded-lg hover:bg-slate-700 disabled:opacity-50 flex items-center gap-2"
        >
          <RotateCcw className="w-4 h-4" /> 重置
        </button>
        <button
          onClick={() => setStep(Math.max(0, step - 1))}
          disabled={step === 0}
          className="px-4 py-2 bg-slate-600 text-white rounded-lg hover:bg-slate-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          上一步
        </button>
        <button
          onClick={() => setIsPlaying(!isPlaying)}
          className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 flex items-center gap-2"
        >
          {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
          {isPlaying ? "暂停" : "播放"}
        </button>
        <button
          onClick={() => setStep(Math.min(steps.length - 1, step + 1))}
          disabled={step === steps.length - 1}
          className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
        >
          下一步 <SkipForward className="w-4 h-4" />
        </button>
      </div>

      {/* Step indicator */}
      <div className="text-center mb-4">
        <span className="text-sm text-slate-500 dark:text-gray-400">
          步骤 {step + 1} / {steps.length}
        </span>
      </div>

      {/* Main content */}
      <AnimatePresence mode="wait">
        <motion.div
          key={step}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -20 }}
          transition={{ duration: 0.3 }}
        >
          {/* Location badge */}
          <div className="flex justify-center mb-4">
            <span className={`px-3 py-1 rounded-full text-sm font-semibold text-white ${
              current.location.includes("user") ? "bg-blue-600" :
              current.location.includes("hardware") ? "bg-red-600" :
              current.location.includes("swtch") ? "bg-purple-600" :
              current.location.includes("scheduler") ? "bg-green-600" :
              "bg-amber-600"
            }`}>
              {locInfo.label}
            </span>
          </div>

          {/* Step title and description */}
          <div className={`${locInfo.bg} border-l-4 ${locInfo.border} p-4 rounded-lg mb-4`}>
            <h4 className="font-bold text-slate-800 dark:text-gray-100 mb-2">{current.title}</h4>
            <p className="text-sm text-slate-600 dark:text-gray-300">{current.description}</p>
          </div>

          {/* Register state */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4">
            <h5 className="font-semibold text-slate-700 dark:text-gray-200 mb-3 flex items-center gap-2">
              <HardDrive className="w-4 h-4" /> 寄存器状态
            </h5>
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-2">
              {current.registers.map((reg, i) => (
                <motion.div
                  key={`${step}-${i}`}
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: i * 0.05 }}
                  className={`flex items-center justify-between px-3 py-2 rounded ${
                    reg.highlight
                      ? "bg-yellow-100 dark:bg-yellow-900/40 border border-yellow-400"
                      : "bg-slate-50 dark:bg-gray-700 border border-slate-200 dark:border-gray-600"
                  }`}
                >
                  <span className="font-mono text-xs font-semibold text-slate-700 dark:text-gray-200">
                    {reg.name}
                  </span>
                  <span className={`font-mono text-xs ${
                    reg.highlight ? "text-yellow-800 dark:text-yellow-300 font-bold" : "text-slate-500 dark:text-gray-400"
                  }`}>
                    {reg.value}
                  </span>
                </motion.div>
              ))}
            </div>
          </div>

          {/* Flow diagram */}
          <div className="mt-4 flex items-center justify-center gap-2 text-xs text-slate-500 dark:text-gray-400">
            <span className={step >= 0 && step <= 3 ? "text-blue-600 font-bold" : ""}>用户A</span>
            <ArrowRight className="w-3 h-3" />
            <span className={step === 1 ? "text-red-600 font-bold" : ""}>中断</span>
            <ArrowRight className="w-3 h-3" />
            <span className={step >= 2 && step <= 3 ? "text-amber-600 font-bold" : ""}>内核A</span>
            <ArrowRight className="w-3 h-3" />
            <span className={step >= 4 && step <= 6 ? "text-purple-600 font-bold" : ""}>调度器</span>
            <ArrowRight className="w-3 h-3" />
            <span className={step >= 6 && step <= 7 ? "text-cyan-600 font-bold" : ""}>内核B</span>
            <ArrowRight className="w-3 h-3" />
            <span className={step === 8 ? "text-green-600 font-bold" : ""}>用户B</span>
          </div>
        </motion.div>
      </AnimatePresence>

      {/* Key insight */}
      <div className="mt-4 bg-indigo-50 dark:bg-indigo-900/30 border border-indigo-200 dark:border-indigo-700 rounded-lg p-3">
        <p className="text-xs text-indigo-700 dark:text-indigo-300">
          <strong>关键洞察：</strong>swtch() 被调用了两次——第一次从进程 A 切换到调度器，第二次从调度器切换到进程 B。每次 swtch() 执行"保存旧寄存器 + 加载新寄存器 + ret"，是一个对称操作。
        </p>
      </div>
    </div>
  );
}
