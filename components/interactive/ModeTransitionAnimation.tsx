"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ArrowDown, Shield, User, Cpu, AlertTriangle } from "lucide-react";

type ProcessorMode = "user" | "kernel";

interface TransitionStep {
  id: number;
  description: string;
  detail: string;
  mode: ProcessorMode;
}

const syscallSteps: TransitionStep[] = [
  { id: 1, description: "应用程序执行 syscall", detail: "用户态代码调用库函数", mode: "user" },
  { id: 2, description: "触发软中断 (INT 0x80/SYSCALL)", detail: "陷入内核态", mode: "user" },
  { id: 3, description: "保存用户态上下文", detail: "保存寄存器到内核栈", mode: "kernel" },
  { id: 4, description: "执行系统调用处理函数", detail: "sys_read, sys_write 等", mode: "kernel" },
  { id: 5, description: "恢复用户态上下文", detail: "从内核栈恢复寄存器", mode: "kernel" },
  { id: 6, description: "返回用户态 (SYSRET/IRET)", detail: "切换回 Ring 3", mode: "user" }
];

export default function ModeTransitionAnimation() {
  const [currentStep, setCurrentStep] = useState<number>(0);
  const [isAnimating, setIsAnimating] = useState<boolean>(false);

  const startAnimation = () => {
    setIsAnimating(true);
    setCurrentStep(0);
    animateSteps(0);
  };

  const animateSteps = (index: number) => {
    if (index >= syscallSteps.length) {
      setIsAnimating(false);
      return;
    }
    setTimeout(() => {
      setCurrentStep(index + 1);
      animateSteps(index + 1);
    }, 1200);
  };

  const reset = () => {
    setCurrentStep(0);
    setIsAnimating(false);
  };

  const getCurrentMode = (): ProcessorMode => {
    if (currentStep === 0) return "user";
    return syscallSteps[currentStep - 1].mode;
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-violet-50 to-violet-100 dark:from-violet-950 dark:to-slate-900 rounded-xl shadow-2xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-6">
        处理器模式转换动画 (系统调用)
      </h3>

      <div className="grid lg:grid-cols-2 gap-6">
        {/* 可视化区域 */}
        <div className="space-y-6">
          {/* 模式指示器 */}
          <div className="grid grid-cols-2 gap-4">
            <motion.div
              animate={{
                scale: getCurrentMode() === "user" ? 1.05 : 1,
                borderWidth: getCurrentMode() === "user" ? 3 : 1
              }}
              className="p-6 bg-white dark:bg-slate-800 rounded-lg border border-blue-300 dark:border-blue-700"
            >
              <div className="flex items-center gap-2 mb-2">
                <User className="w-6 h-6 text-blue-600" />
                <h4 className="font-bold text-blue-700 dark:text-blue-300">用户态 (Ring 3)</h4>
              </div>
              <div className="text-xs text-slate-600 dark:text-slate-400">
                权限受限，不能访问硬件
              </div>
              {getCurrentMode() === "user" && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="mt-2 px-2 py-1 bg-blue-100 dark:bg-blue-900/30 rounded text-xs text-blue-700 dark:text-blue-300"
                >
                  ● 当前模式
                </motion.div>
              )}
            </motion.div>

            <motion.div
              animate={{
                scale: getCurrentMode() === "kernel" ? 1.05 : 1,
                borderWidth: getCurrentMode() === "kernel" ? 3 : 1
              }}
              className="p-6 bg-white dark:bg-slate-800 rounded-lg border border-red-300 dark:border-red-700"
            >
              <div className="flex items-center gap-2 mb-2">
                <Shield className="w-6 h-6 text-red-600" />
                <h4 className="font-bold text-red-700 dark:text-red-300">内核态 (Ring 0)</h4>
              </div>
              <div className="text-xs text-slate-600 dark:text-slate-400">
                完全权限，可访问所有资源
              </div>
              {getCurrentMode() === "kernel" && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="mt-2 px-2 py-1 bg-red-100 dark:bg-red-900/30 rounded text-xs text-red-700 dark:text-red-300"
                >
                  ● 当前模式
                </motion.div>
              )}
            </motion.div>
          </div>

          {/* 转换箭头动画 */}
          <div className="relative p-6 bg-white dark:bg-slate-800 rounded-lg shadow-lg h-64 flex items-center justify-center">
            <AnimatePresence mode="wait">
              {currentStep > 0 && currentStep <= 3 && (
                <motion.div
                  key="to-kernel"
                  initial={{ opacity: 0, y: -50 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  className="flex flex-col items-center"
                >
                  <ArrowDown className="w-16 h-16 text-red-600 animate-bounce" />
                  <div className="mt-4 px-4 py-2 bg-red-100 dark:bg-red-900/30 rounded-lg text-red-700 dark:text-red-300 font-semibold">
                    陷入内核态
                  </div>
                </motion.div>
              )}
              {currentStep > 3 && currentStep <= 6 && (
                <motion.div
                  key="to-user"
                  initial={{ opacity: 0, y: 50 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  className="flex flex-col items-center"
                >
                  <motion.div
                    animate={{ rotate: 180 }}
                    className="w-16 h-16"
                  >
                    <ArrowDown className="w-full h-full text-blue-600 animate-bounce" />
                  </motion.div>
                  <div className="mt-4 px-4 py-2 bg-blue-100 dark:bg-blue-900/30 rounded-lg text-blue-700 dark:text-blue-300 font-semibold">
                    返回用户态
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* 控制按钮 */}
          <div className="flex gap-2">
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={startAnimation}
              disabled={isAnimating}
              className="flex-1 px-4 py-3 bg-violet-600 text-white rounded-lg disabled:opacity-50 disabled:cursor-not-allowed font-semibold"
            >
              {isAnimating ? "动画播放中..." : "开始动画"}
            </motion.button>
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={reset}
              className="px-4 py-3 bg-slate-600 text-white rounded-lg font-semibold"
            >
              重置
            </motion.button>
          </div>
        </div>

        {/* 步骤列表 */}
        <div className="space-y-3">
          <h4 className="font-semibold text-slate-700 dark:text-slate-300 mb-3">系统调用流程</h4>
          {syscallSteps.map((step) => (
            <motion.div
              key={step.id}
              initial={{ opacity: 0.5, x: -20 }}
              animate={{
                opacity: currentStep >= step.id ? 1 : 0.5,
                x: currentStep >= step.id ? 0 : -20,
                scale: currentStep === step.id ? 1.05 : 1
              }}
              className={`
                p-4 rounded-lg border-l-4 transition-all
                ${currentStep === step.id
                  ? step.mode === "kernel"
                    ? "bg-red-50 dark:bg-red-900/20 border-red-500"
                    : "bg-blue-50 dark:bg-blue-900/20 border-blue-500"
                  : "bg-white dark:bg-slate-800 border-slate-300 dark:border-slate-700"
                }
              `}
            >
              <div className="flex items-center gap-2 mb-1">
                <div className={`
                  w-8 h-8 rounded-full flex items-center justify-center font-bold text-white
                  ${step.mode === "kernel" ? "bg-red-600" : "bg-blue-600"}
                `}>
                  {step.id}
                </div>
                <h5 className="font-semibold text-slate-800 dark:text-slate-200">
                  {step.description}
                </h5>
              </div>
              <p className="text-sm text-slate-600 dark:text-slate-400 ml-10">
                {step.detail}
              </p>
              {currentStep === step.id && (
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: "100%" }}
                  className={`mt-2 h-1 rounded ${step.mode === "kernel" ? "bg-red-500" : "bg-blue-500"}`}
                />
              )}
            </motion.div>
          ))}
        </div>
      </div>

      {/* 代码示例 */}
      <div className="mt-6 p-4 bg-white dark:bg-slate-800 rounded-lg shadow">
        <h4 className="font-semibold text-slate-700 dark:text-slate-300 mb-2">系统调用示例</h4>
        <div className="p-3 bg-slate-100 dark:bg-slate-900 rounded font-mono text-xs text-slate-800 dark:text-slate-200">
          <pre>{`// 用户态代码
int fd = open("/path/file", O_RDONLY);  // Ring 3
// ↓ 触发 syscall (陷入 Ring 0)
// 内核态执行 sys_open()
// ↓ 返回用户态 (回到 Ring 3)
// fd 现在包含文件描述符`}</pre>
        </div>
      </div>

      {/* 说明 */}
      <div className="mt-6 p-4 bg-violet-50 dark:bg-violet-900/20 rounded-lg border border-violet-200 dark:border-violet-800">
        <p className="text-sm text-violet-900 dark:text-violet-100">
          <strong>模式转换：</strong> 处理器在用户态和内核态之间切换时会经历上下文保存、特权级切换、指令执行、上下文恢复等步骤。
          系统调用是用户程序访问内核功能的唯一受控入口，通过软中断或专用指令（SYSCALL/SYSENTER）触发。
        </p>
      </div>
    </div>
  );
}
