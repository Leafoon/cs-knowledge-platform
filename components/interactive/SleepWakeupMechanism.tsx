"use client";
import React, { useState, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Play, Pause, RotateCcw, SkipForward, Moon, Sun, AlertTriangle } from "lucide-react";

interface Process {
  id: string;
  name: string;
  state: "running" | "sleeping" | "wakeup" | "waiting";
  chan: string;
  lock: boolean;
}

const initialProcesses: Process[] = [
  { id: "producer", name: "生产者", state: "running", chan: "", lock: false },
  { id: "consumer", name: "消费者", state: "running", chan: "", lock: false },
];

interface Step {
  title: string;
  description: string;
  processes: Process[];
  highlight: string;
  isLostWakeup?: boolean;
}

const correctSteps: Step[] = [
  {
    title: "初始状态",
    description: "生产者和消费者都在运行。共享变量 count=0，MAX=5。",
    processes: [
      { id: "producer", name: "生产者", state: "running", chan: "", lock: true },
      { id: "consumer", name: "消费者", state: "running", chan: "", lock: false },
    ],
    highlight: "init",
  },
  {
    title: "生产者检查条件",
    description: "生产者持有锁，检查 count==MAX（缓冲区满？）。count=0，不满，继续生产。",
    processes: [
      { id: "producer", name: "生产者", state: "running", chan: "", lock: true },
      { id: "consumer", name: "消费者", state: "waiting", chan: "", lock: false },
    ],
    highlight: "check",
  },
  {
    title: "生产者生产数据",
    description: "生产者将数据放入缓冲区，count++。现在 count=1。",
    processes: [
      { id: "producer", name: "生产者", state: "running", chan: "", lock: true },
      { id: "consumer", name: "消费者", state: "waiting", chan: "", lock: false },
    ],
    highlight: "produce",
  },
  {
    title: "生产者唤醒消费者",
    description: "生产者调用 wakeup(&nread)，通知消费者有数据可读。然后释放锁。",
    processes: [
      { id: "producer", name: "生产者", state: "running", chan: "", lock: false },
      { id: "consumer", name: "消费者", state: "wakeup", chan: "nread", lock: false },
    ],
    highlight: "wakeup",
  },
  {
    title: "消费者获取锁并读取",
    description: "消费者被唤醒，获取锁，读取数据，count--。count=0。",
    processes: [
      { id: "producer", name: "生产者", state: "running", chan: "", lock: false },
      { id: "consumer", name: "消费者", state: "running", chan: "", lock: true },
    ],
    highlight: "consume",
  },
];

const lostWakeupSteps: Step[] = [
  {
    title: "初始状态（有问题的代码）",
    description: "生产者和消费者不使用锁保护条件检查。",
    processes: [
      { id: "producer", name: "生产者", state: "running", chan: "", lock: false },
      { id: "consumer", name: "消费者", state: "running", chan: "", lock: false },
    ],
    highlight: "init",
  },
  {
    title: "生产者检查 count==MAX",
    description: "生产者检查 count==MAX（是）。准备调用 sleep()。",
    processes: [
      { id: "producer", name: "生产者", state: "running", chan: "", lock: false },
      { id: "consumer", name: "消费者", state: "running", chan: "", lock: false },
    ],
    highlight: "check",
  },
  {
    title: "消费者修改 count 并调用 wakeup",
    description: "在生产者 sleep 之前，消费者执行 count--，然后调用 wakeup()。但此时生产者还没有 sleep！",
    processes: [
      { id: "producer", name: "生产者", state: "running", chan: "", lock: false },
      { id: "consumer", name: "消费者", state: "running", chan: "", lock: false },
    ],
    highlight: "wakeup-empty",
    isLostWakeup: true,
  },
  {
    title: "生产者调用 sleep()",
    description: "生产者现在调用 sleep()。但 wakeup 已经执行过了！没有人会再来唤醒生产者。",
    processes: [
      { id: "producer", name: "生产者", state: "sleeping", chan: "nwrite", lock: false },
      { id: "consumer", name: "消费者", state: "running", chan: "", lock: false },
    ],
    highlight: "sleep-lost",
    isLostWakeup: true,
  },
  {
    title: "Lost Wakeup！",
    description: "生产者永远睡眠。wakeup 在 sleep 之前执行，唤醒丢失。这就是 Lost Wakeup 问题。",
    processes: [
      { id: "producer", name: "生产者", state: "sleeping", chan: "nwrite", lock: false },
      { id: "consumer", name: "消费者", state: "running", chan: "", lock: false },
    ],
    highlight: "lost",
    isLostWakeup: true,
  },
];

export default function SleepWakeupMechanism() {
  const [mode, setMode] = useState<"correct" | "lost">("correct");
  const [step, setStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  const steps = mode === "correct" ? correctSteps : lostWakeupSteps;
  const current = steps[step];

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
  }, [isPlaying, step, steps.length]);

  const handleReset = useCallback(() => {
    setStep(0);
    setIsPlaying(false);
  }, []);

  const stateColors: Record<string, { bg: string; text: string; icon: string }> = {
    running: { bg: "bg-green-100 dark:bg-green-900/40 border-green-400", text: "text-green-700 dark:text-green-300", icon: "▶" },
    sleeping: { bg: "bg-blue-100 dark:bg-blue-900/40 border-blue-400", text: "text-blue-700 dark:text-blue-300", icon: "🌙" },
    wakeup: { bg: "bg-yellow-100 dark:bg-yellow-900/40 border-yellow-400", text: "text-yellow-700 dark:text-yellow-300", icon: "☀" },
    waiting: { bg: "bg-gray-100 dark:bg-gray-700 border-gray-400", text: "text-gray-600 dark:text-gray-300", icon: "⏳" },
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-indigo-50 rounded-xl shadow-lg dark:from-gray-900 dark:to-gray-800">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-gray-100 mb-6 text-center flex items-center justify-center gap-2">
        <Moon className="w-6 h-6 text-indigo-600" />
        xv6 sleep/wakeup 机制
      </h3>

      {/* Mode selector */}
      <div className="flex justify-center gap-3 mb-6">
        <button
          onClick={() => { setMode("correct"); setStep(0); setIsPlaying(false); }}
          className={`px-4 py-2 rounded-lg text-sm font-semibold ${mode === "correct" ? "bg-green-600 text-white" : "bg-white dark:bg-gray-700 text-slate-600 dark:text-gray-300"}`}
        >
          ✓ 正确实现
        </button>
        <button
          onClick={() => { setMode("lost"); setStep(0); setIsPlaying(false); }}
          className={`px-4 py-2 rounded-lg text-sm font-semibold ${mode === "lost" ? "bg-red-600 text-white" : "bg-white dark:bg-gray-700 text-slate-600 dark:text-gray-300"}`}
        >
          ✗ Lost Wakeup
        </button>
      </div>

      {/* Controls */}
      <div className="flex justify-center gap-4 mb-6">
        <button onClick={handleReset} className="px-4 py-2 bg-slate-600 text-white rounded-lg hover:bg-slate-700 flex items-center gap-2">
          <RotateCcw className="w-4 h-4" /> 重置
        </button>
        <button onClick={() => setStep(Math.max(0, step - 1))} disabled={step === 0} className="px-4 py-2 bg-slate-600 text-white rounded-lg hover:bg-slate-700 disabled:opacity-50">上一步</button>
        <button onClick={() => setIsPlaying(!isPlaying)} className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 flex items-center gap-2">
          {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
          {isPlaying ? "暂停" : "播放"}
        </button>
        <button onClick={() => setStep(Math.min(steps.length - 1, step + 1))} disabled={step === steps.length - 1} className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50 flex items-center gap-2">
          下一步 <SkipForward className="w-4 h-4" />
        </button>
      </div>

      {/* Progress */}
      <div className="text-center mb-4">
        <span className="text-sm text-slate-500 dark:text-gray-400">步骤 {step + 1} / {steps.length}</span>
      </div>

      <AnimatePresence mode="wait">
        <motion.div key={`${mode}-${step}`} initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -20 }} transition={{ duration: 0.3 }}>
          {/* Lost wakeup warning */}
          {current.isLostWakeup && (
            <div className="bg-red-50 dark:bg-red-900/30 border-l-4 border-red-400 p-3 rounded-lg mb-4 flex items-center gap-2">
              <AlertTriangle className="w-5 h-5 text-red-600" />
              <span className="text-sm text-red-700 dark:text-red-300 font-semibold">Lost Wakeup 问题！</span>
            </div>
          )}

          {/* Title */}
          <div className="bg-indigo-50 dark:bg-indigo-900/30 border-l-4 border-indigo-400 p-4 rounded-lg mb-4">
            <h4 className="font-bold text-slate-800 dark:text-gray-100 mb-2">{current.title}</h4>
            <p className="text-sm text-slate-600 dark:text-gray-300">{current.description}</p>
          </div>

          {/* Process states */}
          <div className="grid grid-cols-2 gap-4">
            {current.processes.map((p) => {
              const sc = stateColors[p.state];
              return (
                <motion.div key={p.id} className={`${sc.bg} border-2 rounded-lg p-4`} whileHover={{ scale: 1.02 }}>
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-lg">{sc.icon}</span>
                    <span className={`font-bold ${sc.text}`}>{p.name}</span>
                  </div>
                  <div className="text-xs space-y-1">
                    <div className={sc.text}>状态：{p.state === "running" ? "运行中" : p.state === "sleeping" ? "睡眠中" : p.state === "wakeup" ? "被唤醒" : "等待中"}</div>
                    {p.chan && <div className={sc.text}>通道：{p.chan}</div>}
                    <div className={sc.text}>锁：{p.lock ? "已持有" : "未持有"}</div>
                  </div>
                </motion.div>
              );
            })}
          </div>
        </motion.div>
      </AnimatePresence>

      {/* Key insight */}
      <div className="mt-4 bg-indigo-50 dark:bg-indigo-900/30 border border-indigo-200 dark:border-indigo-700 rounded-lg p-3">
        <p className="text-xs text-indigo-700 dark:text-indigo-300">
          <strong>解决方案：</strong>在持有锁的情况下检查条件和调用 sleep()。sleep(chan, lk) 在持有 ptable.lock 的情况下释放 lk 并设置 SLEEPING 状态，确保 wakeup 不会在检查和睡眠之间执行。
        </p>
      </div>
    </div>
  );
}
