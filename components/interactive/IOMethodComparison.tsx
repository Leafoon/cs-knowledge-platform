"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Cpu, HardDrive, MemoryStick, Zap, Clock, ArrowRight } from "lucide-react";

interface IOMethod {
  id: string;
  name: string;
  nameZh: string;
  icon: React.ReactNode;
  color: string;
  bgColor: string;
  borderColor: string;
  cpuUtil: number;
  complexity: string;
  latency: string;
  bestFor: string;
  steps: { label: string; active: "cpu" | "device" | "bus" | "idle" }[];
}

const methods: IOMethod[] = [
  {
    id: "polling",
    name: "Polling",
    nameZh: "轮询",
    icon: <Clock className="w-5 h-5" />,
    color: "text-red-700 dark:text-red-300",
    bgColor: "bg-red-50 dark:bg-red-950/40",
    borderColor: "border-red-300 dark:border-red-700",
    cpuUtil: 95,
    complexity: "最简单",
    latency: "最低",
    bestFor: "快速设备、小数据量",
    steps: [
      { label: "CPU 写入命令到设备", active: "cpu" },
      { label: "CPU 反复读取状态寄存器", active: "cpu" },
      { label: "CPU 检测到 READY", active: "cpu" },
      { label: "CPU 逐字节读取数据", active: "cpu" },
      { label: "CPU 返回数据给用户", active: "cpu" },
    ],
  },
  {
    id: "interrupt",
    name: "Interrupt-Driven",
    nameZh: "中断驱动",
    icon: <Zap className="w-5 h-5" />,
    color: "text-amber-700 dark:text-amber-300",
    bgColor: "bg-amber-50 dark:bg-amber-950/40",
    borderColor: "border-amber-300 dark:border-amber-700",
    cpuUtil: 30,
    complexity: "中等",
    latency: "较低",
    bestFor: "中等数据量、通用设备",
    steps: [
      { label: "CPU 写入命令到设备", active: "cpu" },
      { label: "CPU 执行其他进程", active: "idle" },
      { label: "设备完成，发送中断", active: "device" },
      { label: "CPU 响应中断，逐字节读取", active: "cpu" },
      { label: "CPU 唤醒等待进程", active: "cpu" },
    ],
  },
  {
    id: "dma",
    name: "DMA",
    nameZh: "直接内存访问",
    icon: <MemoryStick className="w-5 h-5" />,
    color: "text-emerald-700 dark:text-emerald-300",
    bgColor: "bg-emerald-50 dark:bg-emerald-950/40",
    borderColor: "border-emerald-300 dark:border-emerald-700",
    cpuUtil: 5,
    complexity: "最复杂",
    latency: "略高",
    bestFor: "大数据量、磁盘/网络",
    steps: [
      { label: "CPU 设置 DMA 参数", active: "cpu" },
      { label: "CPU 执行其他进程", active: "idle" },
      { label: "DMA 控制器传输数据到内存", active: "bus" },
      { label: "DMA 完成，发送中断", active: "device" },
      { label: "CPU 处理完成中断", active: "cpu" },
    ],
  },
];

export default function IOMethodComparison() {
  const [selected, setSelected] = useState<string | null>(null);
  const [step, setStep] = useState(-1);
  const [isAnimating, setIsAnimating] = useState(false);

  const selectedMethod = methods.find((m) => m.id === selected);

  const startAnimation = () => {
    if (!selectedMethod || isAnimating) return;
    setIsAnimating(true);
    setStep(0);
    let i = 0;
    const interval = setInterval(() => {
      i++;
      if (i >= selectedMethod.steps.length) {
        clearInterval(interval);
        setIsAnimating(false);
      }
      setStep(i);
    }, 1200);
  };

  const resetAnimation = () => {
    setStep(-1);
    setIsAnimating(false);
  };

  const getStepColor = (active: string) => {
    switch (active) {
      case "cpu": return "bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 border-blue-300 dark:border-blue-700";
      case "device": return "bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300 border-purple-300 dark:border-purple-700";
      case "bus": return "bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-300 border-emerald-300 dark:border-emerald-700";
      case "idle": return "bg-slate-100 dark:bg-gray-700 text-slate-500 dark:text-gray-400 border-slate-300 dark:border-gray-600";
      default: return "bg-slate-100 dark:bg-gray-700 text-slate-500 dark:text-gray-400 border-slate-300 dark:border-gray-600";
    }
  };

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-cyan-50 rounded-xl shadow-lg dark:from-gray-900 dark:to-gray-800">
      <h2 className="text-2xl font-bold text-slate-800 dark:text-gray-100 mb-2 text-center">
        I/O 方式对比
      </h2>
      <p className="text-sm text-slate-500 dark:text-gray-400 text-center mb-6">
        轮询 vs 中断驱动 vs DMA — 点击查看每种方式的工作流程
      </p>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-6">
        {methods.map((m) => (
          <button
            key={m.id}
            onClick={() => { setSelected(m.id); resetAnimation(); }}
            className={`p-4 rounded-lg border-2 text-left transition-all ${m.bgColor} ${
              selected === m.id ? `${m.borderColor} shadow-md` : "border-transparent hover:border-slate-300 dark:hover:border-gray-600"
            }`}
          >
            <div className="flex items-center gap-2 mb-2">
              <span className={m.color}>{m.icon}</span>
              <span className={`font-bold text-sm ${m.color}`}>{m.nameZh}</span>
            </div>
            <div className="flex items-center gap-2 mb-1">
              <span className="text-xs text-slate-500 dark:text-gray-400">CPU 占用</span>
              <div className="flex-1 h-2 bg-slate-200 dark:bg-gray-700 rounded-full overflow-hidden">
                <motion.div
                  className={`h-full rounded-full ${m.cpuUtil > 60 ? "bg-red-400" : m.cpuUtil > 20 ? "bg-amber-400" : "bg-emerald-400"}`}
                  initial={{ width: 0 }}
                  animate={{ width: `${m.cpuUtil}%` }}
                  transition={{ duration: 0.8, delay: 0.2 }}
                />
              </div>
              <span className="text-xs font-mono text-slate-600 dark:text-gray-300 w-8">{m.cpuUtil}%</span>
            </div>
            <div className="text-xs text-slate-500 dark:text-gray-400">{m.bestFor}</div>
          </button>
        ))}
      </div>

      {selectedMethod && (
        <motion.div
          key={selectedMethod.id}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white dark:bg-gray-800 rounded-lg p-5 border border-slate-200 dark:border-gray-700"
        >
          <div className="flex items-center gap-3 mb-4">
            <span className={selectedMethod.color}>{selectedMethod.icon}</span>
            <h3 className={`text-lg font-bold ${selectedMethod.color}`}>{selectedMethod.nameZh} I/O</h3>
          </div>

          <div className="flex gap-3 mb-4">
            <button
              onClick={startAnimation}
              disabled={isAnimating}
              className="flex items-center gap-2 px-4 py-2 bg-cyan-500 text-white rounded-lg hover:bg-cyan-600 disabled:opacity-50 text-sm"
            >
              播放动画
            </button>
            <button onClick={resetAnimation} className="px-4 py-2 bg-slate-500 text-white rounded-lg hover:bg-slate-600 text-sm">
              重置
            </button>
          </div>

          <div className="space-y-2">
            {selectedMethod.steps.map((s, i) => {
              const isVisible = i <= step;
              const isCurrent = i === step;
              return (
                <AnimatePresence key={i}>
                  {isVisible && (
                    <motion.div
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      className={`flex items-center gap-3 p-3 rounded-lg border ${getStepColor(s.active)} ${
                        isCurrent ? "shadow-md" : ""
                      }`}
                    >
                      <span className="w-6 h-6 rounded-full bg-white dark:bg-gray-800 flex items-center justify-center text-xs font-bold border">
                        {i + 1}
                      </span>
                      <span className="text-sm">{s.label}</span>
                      {isCurrent && (
                        <motion.span
                          className="ml-auto w-2 h-2 rounded-full bg-cyan-500"
                          animate={{ scale: [1, 1.3, 1] }}
                          transition={{ repeat: Infinity, duration: 1 }}
                        />
                      )}
                    </motion.div>
                  )}
                </AnimatePresence>
              );
            })}
          </div>
        </motion.div>
      )}
    </div>
  );
}
