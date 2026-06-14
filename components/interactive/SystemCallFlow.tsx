"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ArrowRight, Clock, Cpu, User } from "lucide-react";

interface SystemCallStep {
  id: number;
  phase: string;
  location: "用户态" | "内核态" | "转换中";
  description: string;
  details: string;
  color: string;
}

const syscallSteps: SystemCallStep[] = [
  {
    id: 1,
    phase: "1. 用户程序调用",
    location: "用户态",
    description: "应用程序调用库函数",
    details: "例如：write(fd, buf, len) 在 glibc 中",
    color: "#3b82f6",
  },
  {
    id: 2,
    phase: "2. 准备参数",
    location: "用户态",
    description: "将参数放入寄存器",
    details: "x86-64: rdi=fd, rsi=buf, rdx=len, rax=syscall_number(1)",
    color: "#3b82f6",
  },
  {
    id: 3,
    phase: "3. 执行 syscall 指令",
    location: "转换中",
    description: "触发模式切换",
    details: "syscall/sysenter 指令，切换到内核态",
    color: "#f59e0b",
  },
  {
    id: 4,
    phase: "4. 保存用户态上下文",
    location: "内核态",
    description: "保存寄存器到内核栈",
    details: "保存 RIP, RSP, RFLAGS, CS, SS 等",
    color: "#dc2626",
  },
  {
    id: 5,
    phase: "5. 查找系统调用表",
    location: "内核态",
    description: "根据系统调用号查找",
    details: "sys_call_table[rax] → sys_write 地址",
    color: "#dc2626",
  },
  {
    id: 6,
    phase: "6. 执行内核函数",
    location: "内核态",
    description: "执行实际的系统调用处理",
    details: "sys_write() → vfs_write() → 文件系统写入",
    color: "#dc2626",
  },
  {
    id: 7,
    phase: "7. 恢复用户态上下文",
    location: "转换中",
    description: "从内核栈恢复寄存器",
    details: "恢复 RIP, RSP, RFLAGS 等",
    color: "#f59e0b",
  },
  {
    id: 8,
    phase: "8. 返回用户态",
    location: "用户态",
    description: "sysret/sysexit 指令返回",
    details: "CPL: 0 → 3, 返回值在 rax 中",
    color: "#3b82f6",
  },
];

export function SystemCallFlow() {
  const [currentStep, setCurrentStep] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [selectedSyscall, setSelectedSyscall] = useState("write");

  const syscallExamples = [
    { name: "write", number: 1, description: "写文件", args: "(fd, buf, count)" },
    { name: "read", number: 0, description: "读文件", args: "(fd, buf, count)" },
    { name: "open", number: 2, description: "打开文件", args: "(pathname, flags, mode)" },
    { name: "fork", number: 57, description: "创建进程", args: "()" },
    { name: "execve", number: 59, description: "执行程序", args: "(filename, argv, envp)" },
  ];

  const handlePlay = () => {
    setIsPlaying(true);
    setCurrentStep(0);
  };

  const handleReset = () => {
    setIsPlaying(false);
    setCurrentStep(0);
  };

  const handleStepForward = () => {
    if (currentStep < syscallSteps.length - 1) {
      setCurrentStep((prev) => prev + 1);
    } else {
      setIsPlaying(false);
    }
  };

  const handleStepBack = () => {
    if (currentStep > 0) {
      setCurrentStep((prev) => prev - 1);
    }
  };

  // Auto-advance when playing
  useState(() => {
    if (isPlaying && currentStep < syscallSteps.length) {
      const timer = setTimeout(() => {
        handleStepForward();
      }, 1500);
      return () => clearTimeout(timer);
    }
  });

  const currentLocation = syscallSteps[currentStep]?.location || "用户态";

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-6 text-text-primary">
        系统调用执行流程
      </h3>

      {/* System Call Selection */}
      <div className="mb-6">
        <h4 className="font-semibold mb-3 text-text-primary">
          选择系统调用示例
        </h4>
        <div className="grid grid-cols-5 gap-3">
          {syscallExamples.map((syscall) => (
            <button
              key={syscall.name}
              onClick={() => setSelectedSyscall(syscall.name)}
              disabled={isPlaying}
              className={`p-3 rounded-lg border-2 transition ${
                selectedSyscall === syscall.name
                  ? "border-blue-500 bg-blue-50 dark:bg-blue-900/20"
                  : "border-gray-300 dark:border-gray-700 hover:border-gray-400"
              }`}
            >
              <div className="font-mono font-semibold text-sm text-text-primary">
                {syscall.name}()
              </div>
              <div className="text-xs text-text-secondary mt-1">
                #{syscall.number}
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Current Mode Indicator */}
      <div className="mb-6 p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
        <h4 className="font-semibold mb-3 text-text-primary">当前执行状态</h4>
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2">
            <Cpu className="w-5 h-5 text-blue-600 dark:text-blue-400" />
            <div>
              <div className="text-sm text-text-secondary">运行模式</div>
              <div
                className={`text-lg font-semibold ${
                  currentLocation === "内核态"
                    ? "text-red-600 dark:text-red-400"
                    : currentLocation === "转换中"
                    ? "text-yellow-600 dark:text-yellow-400"
                    : "text-blue-600 dark:text-blue-400"
                }`}
              >
                {currentLocation}
              </div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <User className="w-5 h-5 text-purple-600 dark:text-purple-400" />
            <div>
              <div className="text-sm text-text-secondary">特权级别</div>
              <div className="text-lg font-semibold text-text-primary">
                {currentLocation === "内核态" ? "Ring 0" : "Ring 3"}
              </div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Clock className="w-5 h-5 text-green-600 dark:text-green-400" />
            <div>
              <div className="text-sm text-text-secondary">当前步骤</div>
              <div className="text-lg font-semibold text-text-primary">
                {currentStep + 1} / {syscallSteps.length}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Controls */}
      <div className="flex gap-3 mb-6">
        <button
          onClick={handlePlay}
          disabled={isPlaying}
          className="px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed transition"
        >
          自动播放
        </button>
        <button
          onClick={handleStepBack}
          disabled={isPlaying || currentStep === 0}
          className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition"
        >
          上一步
        </button>
        <button
          onClick={handleStepForward}
          disabled={isPlaying || currentStep === syscallSteps.length - 1}
          className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition"
        >
          下一步
        </button>
        <button
          onClick={handleReset}
          className="px-6 py-3 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition"
        >
          重置
        </button>
      </div>

      {/* Progress Indicator */}
      <div className="mb-6">
        <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
          <motion.div
            className="h-full bg-gradient-to-r from-blue-600 via-yellow-600 to-red-600"
            initial={{ width: 0 }}
            animate={{ width: `${((currentStep + 1) / syscallSteps.length) * 100}%` }}
            transition={{ duration: 0.3 }}
          />
        </div>
      </div>

      {/* Flow Diagram */}
      <div className="space-y-3">
        <AnimatePresence>
          {syscallSteps.map((step, index) => {
            const isActive = index === currentStep;
            const isCompleted = index < currentStep;
            const isFuture = index > currentStep;

            return (
              <motion.div
                key={step.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.05 }}
                className={`border-2 rounded-lg overflow-hidden ${
                  isActive
                    ? "border-blue-500 shadow-lg"
                    : isCompleted
                    ? "border-green-500"
                    : "border-gray-300 dark:border-gray-700"
                } ${isFuture ? "opacity-50" : "opacity-100"}`}
              >
                <div
                  className={`p-4 ${
                    isActive
                      ? "bg-blue-50 dark:bg-blue-900/20"
                      : isCompleted
                      ? "bg-green-50 dark:bg-green-900/20"
                      : "bg-gray-50 dark:bg-gray-900"
                  }`}
                >
                  <div className="flex items-center gap-4">
                    <div
                      className={`flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center font-bold text-white`}
                      style={{
                        backgroundColor: isCompleted
                          ? "#10b981"
                          : isActive
                          ? step.color
                          : "#6b7280",
                      }}
                    >
                      {isCompleted ? "✓" : step.id}
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center gap-3">
                        <h4 className="font-semibold text-text-primary">
                          {step.phase}
                        </h4>
                        <span
                          className={`px-2 py-1 rounded text-xs font-semibold text-white`}
                          style={{ backgroundColor: step.color }}
                        >
                          {step.location}
                        </span>
                      </div>
                      <p className="text-sm text-text-secondary mt-1">
                        {step.description}
                      </p>
                    </div>
                    {index < syscallSteps.length - 1 && (
                      <ArrowRight className="w-6 h-6 text-text-secondary" />
                    )}
                  </div>

                  {isActive && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: "auto", opacity: 1 }}
                      className="mt-3 p-3 bg-white dark:bg-gray-950 rounded border border-border-subtle"
                    >
                      <p className="text-sm text-text-primary font-mono">
                        {step.details}
                      </p>
                    </motion.div>
                  )}
                </div>
              </motion.div>
            );
          })}
        </AnimatePresence>
      </div>

      {/* Summary */}
      {currentStep === syscallSteps.length - 1 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-6 p-6 bg-green-50 dark:bg-green-900/20 rounded-lg border-2 border-green-500"
        >
          <h4 className="font-semibold text-lg text-green-700 dark:text-green-300 mb-3">
            系统调用完成！
          </h4>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="text-text-secondary">总耗时：</span>
              <span className="font-semibold text-text-primary ml-2">
                ~100-300 ns (快速路径)
              </span>
            </div>
            <div>
              <span className="text-text-secondary">模式切换：</span>
              <span className="font-semibold text-text-primary ml-2">
                2 次 (用户→内核→用户)
              </span>
            </div>
            <div className="col-span-2">
              <span className="text-text-secondary">开销来源：</span>
              <span className="font-semibold text-text-primary ml-2">
                上下文保存/恢复、TLB 刷新、特权级切换
              </span>
            </div>
          </div>
        </motion.div>
      )}

      {/* Info Box */}
      <div className="mt-6 p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg border border-purple-200 dark:border-purple-800">
        <h4 className="font-semibold mb-2 text-purple-700 dark:text-purple-300">
          x86-64 系统调用机制
        </h4>
        <div className="text-sm text-text-secondary space-y-1">
          <p>
            <strong className="text-text-primary">syscall/sysret</strong>：
            快速系统调用指令，直接从 MSR (Model-Specific Registers) 读取内核入口地址
          </p>
          <p>
            <strong className="text-text-primary">参数传递</strong>：
            rdi, rsi, rdx, r10, r8, r9 (最多 6 个参数)
          </p>
          <p>
            <strong className="text-text-primary">返回值</strong>：
            rax (错误码为负数)
          </p>
        </div>
      </div>
    </div>
  );
}
