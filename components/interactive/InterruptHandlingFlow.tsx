"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Play, RotateCcw, AlertTriangle, Clock, Zap } from "lucide-react";

interface InterruptStep {
  id: number;
  title: string;
  description: string;
  detail: string;
  duration: number;
}

const interruptSteps: InterruptStep[] = [
  {
    id: 1,
    title: "中断请求",
    description: "外部设备发送中断信号",
    detail: "时钟芯片向 CPU 发送 IRQ 信号",
    duration: 1000,
  },
  {
    id: 2,
    title: "保存现场",
    description: "CPU 自动保存当前状态",
    detail: "保存 CS, RIP, RFLAGS, RSP, SS 到内核栈",
    duration: 1500,
  },
  {
    id: 3,
    title: "查找 IDT",
    description: "根据中断向量号查找处理程序",
    detail: "IDT[32] → 时钟中断处理程序地址",
    duration: 1200,
  },
  {
    id: 4,
    title: "切换到内核态",
    description: "特权级别切换",
    detail: "CPL: 3 (用户态) → 0 (内核态)",
    duration: 800,
  },
  {
    id: 5,
    title: "执行处理程序",
    description: "调用中断服务例程 (ISR)",
    detail: "更新系统时钟、检查进程时间片",
    duration: 2000,
  },
  {
    id: 6,
    title: "发送 EOI",
    description: "通知中断控制器",
    detail: "向 PIC/APIC 发送 End Of Interrupt 信号",
    duration: 800,
  },
  {
    id: 7,
    title: "恢复现场",
    description: "恢复保存的寄存器",
    detail: "从内核栈弹出 RIP, RFLAGS, RSP 等",
    duration: 1500,
  },
  {
    id: 8,
    title: "返回用户态",
    description: "执行 iretq 指令",
    detail: "CPL: 0 → 3, 继续执行被中断的代码",
    duration: 1000,
  },
];

const interruptTypes = [
  { name: "时钟中断", vector: 32, color: "#3b82f6", icon: <Clock className="w-5 h-5" /> },
  { name: "键盘中断", vector: 33, color: "#10b981", icon: <Zap className="w-5 h-5" /> },
  { name: "缺页异常", vector: 14, color: "#f59e0b", icon: <AlertTriangle className="w-5 h-5" /> },
];

export function InterruptHandlingFlow() {
  const [currentStep, setCurrentStep] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [selectedType, setSelectedType] = useState(0);
  const [cpuState, setCpuState] = useState({
    mode: "用户态",
    privilege: 3,
    executing: "应用程序代码",
  });

  const handlePlay = () => {
    setIsPlaying(true);
    setCurrentStep(0);
    setCpuState({ mode: "用户态", privilege: 3, executing: "应用程序代码" });
  };

  const handleReset = () => {
    setIsPlaying(false);
    setCurrentStep(0);
    setCpuState({ mode: "用户态", privilege: 3, executing: "应用程序代码" });
  };

  const handleStepForward = () => {
    if (currentStep < interruptSteps.length - 1) {
      setCurrentStep((prev) => prev + 1);
    } else {
      setIsPlaying(false);
    }
  };

  // Auto-advance when playing
  useState(() => {
    if (isPlaying && currentStep < interruptSteps.length) {
      // Update CPU state based on current step
      if (currentStep === 3) {
        setCpuState({ mode: "内核态", privilege: 0, executing: "切换中..." });
      } else if (currentStep === 4) {
        setCpuState({ mode: "内核态", privilege: 0, executing: "中断处理程序" });
      } else if (currentStep === 7) {
        setCpuState({ mode: "用户态", privilege: 3, executing: "应用程序代码" });
      }

      const timer = setTimeout(() => {
        handleStepForward();
      }, interruptSteps[currentStep]?.duration || 1000);

      return () => clearTimeout(timer);
    }
  });

  const progress = ((currentStep + 1) / interruptSteps.length) * 100;

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-6 text-text-primary">
        中断处理流程动画演示
      </h3>

      {/* Interrupt Type Selection */}
      <div className="mb-6">
        <h4 className="font-semibold mb-3 text-text-primary">选择中断类型</h4>
        <div className="flex gap-3">
          {interruptTypes.map((type, index) => (
            <button
              key={type.name}
              onClick={() => setSelectedType(index)}
              disabled={isPlaying}
              className={`flex-1 p-3 rounded-lg border-2 transition flex items-center gap-2 justify-center ${
                selectedType === index
                  ? `border-[${type.color}] bg-opacity-10`
                  : "border-gray-300 dark:border-gray-700"
              }`}
              style={{
                borderColor: selectedType === index ? type.color : undefined,
                backgroundColor:
                  selectedType === index ? `${type.color}20` : undefined,
              }}
            >
              <div style={{ color: type.color }}>{type.icon}</div>
              <div className="text-left">
                <div className="font-semibold text-sm text-text-primary">
                  {type.name}
                </div>
                <div className="text-xs text-text-secondary">
                  向量 {type.vector}
                </div>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* CPU State Display */}
      <div className="mb-6 p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
        <h4 className="font-semibold mb-3 text-text-primary">CPU 当前状态</h4>
        <div className="grid grid-cols-3 gap-4">
          <div>
            <div className="text-sm text-text-secondary">运行模式</div>
            <div
              className={`text-lg font-semibold ${
                cpuState.mode === "内核态"
                  ? "text-red-600 dark:text-red-400"
                  : "text-blue-600 dark:text-blue-400"
              }`}
            >
              {cpuState.mode}
            </div>
          </div>
          <div>
            <div className="text-sm text-text-secondary">特权级别 (CPL)</div>
            <div className="text-lg font-semibold text-text-primary">
              Ring {cpuState.privilege}
            </div>
          </div>
          <div>
            <div className="text-sm text-text-secondary">正在执行</div>
            <div className="text-lg font-semibold text-text-primary">
              {cpuState.executing}
            </div>
          </div>
        </div>
      </div>

      {/* Controls */}
      <div className="flex gap-3 mb-6">
        <button
          onClick={handlePlay}
          disabled={isPlaying}
          className="flex items-center gap-2 px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed transition"
        >
          <Play className="w-5 h-5" />
          开始演示
        </button>
        <button
          onClick={handleReset}
          className="flex items-center gap-2 px-6 py-3 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition"
        >
          <RotateCcw className="w-5 h-5" />
          重置
        </button>
        <div className="flex-1" />
        <div className="text-sm text-text-secondary self-center">
          步骤 {currentStep + 1} / {interruptSteps.length}
        </div>
      </div>

      {/* Progress Bar */}
      <div className="mb-6">
        <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
          <motion.div
            className="h-full rounded-full"
            style={{ backgroundColor: interruptTypes[selectedType].color }}
            initial={{ width: 0 }}
            animate={{ width: `${progress}%` }}
            transition={{ duration: 0.3 }}
          />
        </div>
      </div>

      {/* Steps Visualization */}
      <div className="space-y-3">
        <AnimatePresence>
          {interruptSteps.map((step, index) => {
            const isActive = index === currentStep;
            const isCompleted = index < currentStep;
            const isFuture = index > currentStep;

            return (
              <motion.div
                key={step.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
                className={`p-4 rounded-lg border-2 transition-all ${
                  isActive
                    ? "border-blue-500 bg-blue-50 dark:bg-blue-900/20 shadow-lg scale-105"
                    : isCompleted
                    ? "border-green-500 bg-green-50 dark:bg-green-900/20"
                    : "border-gray-300 dark:border-gray-700 opacity-50"
                }`}
              >
                <div className="flex items-start gap-4">
                  <div
                    className={`flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center font-bold ${
                      isActive
                        ? "bg-blue-600 text-white"
                        : isCompleted
                        ? "bg-green-600 text-white"
                        : "bg-gray-300 dark:bg-gray-700 text-gray-600 dark:text-gray-400"
                    }`}
                  >
                    {isCompleted ? "✓" : step.id}
                  </div>
                  <div className="flex-1">
                    <h4 className="font-semibold text-text-primary">
                      {step.title}
                    </h4>
                    <p className="text-sm text-text-secondary mt-1">
                      {step.description}
                    </p>
                    {isActive && (
                      <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: "auto" }}
                        className="mt-2 p-2 bg-blue-100 dark:bg-blue-800/30 rounded text-sm text-text-primary"
                      >
                        {step.detail}
                      </motion.div>
                    )}
                  </div>
                  {isActive && (
                    <motion.div
                      animate={{ rotate: 360 }}
                      transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                      className="flex-shrink-0"
                    >
                      <div className="w-6 h-6 border-3 border-blue-600 border-t-transparent rounded-full" />
                    </motion.div>
                  )}
                </div>
              </motion.div>
            );
          })}
        </AnimatePresence>
      </div>

      {/* Summary */}
      {currentStep === interruptSteps.length - 1 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-6 p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800"
        >
          <h4 className="font-semibold text-green-700 dark:text-green-300 mb-2">
            中断处理完成！
          </h4>
          <p className="text-sm text-text-secondary">
            总耗时约 {interruptSteps.reduce((sum, s) => sum + s.duration, 0) / 1000}{" "}
            毫秒。程序从用户态陷入内核态，处理中断后返回用户态继续执行。
          </p>
        </motion.div>
      )}
    </div>
  );
}
