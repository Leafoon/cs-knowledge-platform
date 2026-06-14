"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Play, RotateCcw, Cpu, HardDrive, Settings } from "lucide-react";

interface BootStage {
  id: number;
  name: string;
  description: string;
  details: string[];
  color: string;
  duration: number;
}

const bootStages: BootStage[] = [
  {
    id: 1,
    name: "BIOS/UEFI POST",
    description: "上电自检",
    details: [
      "CPU 从 0xFFFFFFF0 开始执行",
      "检测内存、显卡、键盘等硬件",
      "初始化中断向量表 (IVT)",
      "查找可引导设备",
    ],
    color: "#8b5cf6",
    duration: 2000,
  },
  {
    id: 2,
    name: "MBR/GPT 加载",
    description: "主引导记录",
    details: [
      "从磁盘第一个扇区读取 512 字节",
      "验证魔数 0x55AA",
      "加载引导代码到 0x7C00",
      "跳转到引导加载器",
    ],
    color: "#3b82f6",
    duration: 1500,
  },
  {
    id: 3,
    name: "Boot Loader (GRUB Stage 1.5)",
    description: "第二阶段加载器",
    details: [
      "加载文件系统驱动",
      "读取 /boot/grub/grub.cfg",
      "显示启动菜单",
      "用户选择内核版本",
    ],
    color: "#10b981",
    duration: 2000,
  },
  {
    id: 4,
    name: "内核加载",
    description: "加载 Linux 内核",
    details: [
      "从 /boot/vmlinuz-* 加载内核映像",
      "加载 initramfs (临时根文件系统)",
      "解压内核到内存",
      "跳转到内核入口点",
    ],
    color: "#f59e0b",
    duration: 2500,
  },
  {
    id: 5,
    name: "内核初始化",
    description: "start_kernel() 执行",
    details: [
      "初始化内存管理 (mm_init)",
      "初始化调度器 (sched_init)",
      "初始化中断系统 (trap_init)",
      "挂载 initramfs",
    ],
    color: "#ef4444",
    duration: 3000,
  },
  {
    id: 6,
    name: "init 进程启动",
    description: "用户空间初始化",
    details: [
      "内核启动 /sbin/init (PID 1)",
      "systemd 读取 /etc/systemd/system/",
      "启动基础服务 (udev, dbus, network)",
      "挂载真实根文件系统",
    ],
    color: "#ec4899",
    duration: 2500,
  },
  {
    id: 7,
    name: "启动服务",
    description: "系统服务初始化",
    details: [
      "启动 default.target 依赖的服务",
      "启动网络管理器",
      "启动登录管理器 (getty/gdm)",
      "系统准备就绪",
    ],
    color: "#06b6d4",
    duration: 2000,
  },
];

export function SystemBootVisualizer() {
  const [currentStage, setCurrentStage] = useState<number>(-1);
  const [isBooting, setIsBooting] = useState(false);
  const [bootTime, setBootTime] = useState(0);
  const [cpuMode, setCpuMode] = useState("实模式 (16-bit)");
  const [memoryUsage, setMemoryUsage] = useState(0);

  const handleBoot = () => {
    setIsBooting(true);
    setCurrentStage(0);
    setBootTime(0);
    setMemoryUsage(0);
    setCpuMode("实模式 (16-bit)");
  };

  const handleReset = () => {
    setIsBooting(false);
    setCurrentStage(-1);
    setBootTime(0);
    setMemoryUsage(0);
    setCpuMode("实模式 (16-bit)");
  };

  // Auto-advance stages
  useState(() => {
    if (isBooting && currentStage >= 0 && currentStage < bootStages.length) {
      const stage = bootStages[currentStage];

      // Update CPU mode based on stage
      if (currentStage === 2) {
        setCpuMode("保护模式 (32-bit)");
      } else if (currentStage === 4) {
        setCpuMode("长模式 (64-bit)");
      }

      // Update memory usage
      setMemoryUsage((prev) => Math.min(prev + 15, 95));

      const timer = setTimeout(() => {
        setBootTime((prev) => prev + stage.duration);
        if (currentStage < bootStages.length - 1) {
          setCurrentStage((prev) => prev + 1);
        } else {
          setIsBooting(false);
        }
      }, stage.duration);

      return () => clearTimeout(timer);
    }
  });

  const progress =
    currentStage >= 0 ? ((currentStage + 1) / bootStages.length) * 100 : 0;

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-6 text-text-primary">
        系统启动流程可视化
      </h3>

      {/* System Status */}
      <div className="mb-6 grid grid-cols-3 gap-4">
        <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg border border-purple-200 dark:border-purple-800">
          <div className="flex items-center gap-2 mb-2">
            <Cpu className="w-5 h-5 text-purple-600 dark:text-purple-400" />
            <span className="text-sm text-text-secondary">CPU 模式</span>
          </div>
          <div className="text-lg font-semibold text-text-primary">
            {cpuMode}
          </div>
        </div>
        <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
          <div className="flex items-center gap-2 mb-2">
            <HardDrive className="w-5 h-5 text-blue-600 dark:text-blue-400" />
            <span className="text-sm text-text-secondary">内存使用</span>
          </div>
          <div className="text-lg font-semibold text-text-primary">
            {memoryUsage.toFixed(1)}%
          </div>
        </div>
        <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800">
          <div className="flex items-center gap-2 mb-2">
            <Settings className="w-5 h-5 text-green-600 dark:text-green-400" />
            <span className="text-sm text-text-secondary">启动时间</span>
          </div>
          <div className="text-lg font-semibold text-text-primary">
            {(bootTime / 1000).toFixed(1)}s
          </div>
        </div>
      </div>

      {/* Controls */}
      <div className="flex gap-3 mb-6">
        <button
          onClick={handleBoot}
          disabled={isBooting}
          className="flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition"
        >
          <Play className="w-5 h-5" />
          开始启动
        </button>
        <button
          onClick={handleReset}
          className="flex items-center gap-2 px-6 py-3 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition"
        >
          <RotateCcw className="w-5 h-5" />
          重置
        </button>
        <div className="flex-1" />
        {currentStage >= 0 && (
          <div className="text-sm text-text-secondary self-center">
            阶段 {currentStage + 1} / {bootStages.length}
          </div>
        )}
      </div>

      {/* Progress Bar */}
      <div className="mb-6">
        <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
          <motion.div
            className="h-full bg-gradient-to-r from-blue-600 to-purple-600 rounded-full"
            initial={{ width: 0 }}
            animate={{ width: `${progress}%` }}
            transition={{ duration: 0.5 }}
          />
        </div>
      </div>

      {/* Boot Stages */}
      <div className="space-y-4">
        <AnimatePresence>
          {bootStages.map((stage, index) => {
            const isActive = index === currentStage;
            const isCompleted = index < currentStage;
            const isFuture = index > currentStage;

            return (
              <motion.div
                key={stage.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                className={`border-2 rounded-lg overflow-hidden transition-all ${
                  isActive
                    ? "border-blue-500 shadow-lg shadow-blue-500/50"
                    : isCompleted
                    ? "border-green-500"
                    : "border-gray-300 dark:border-gray-700"
                } ${isFuture ? "opacity-50" : "opacity-100"}`}
              >
                {/* Stage Header */}
                <div
                  className={`p-4 flex items-center gap-4 ${
                    isActive
                      ? "bg-blue-50 dark:bg-blue-900/20"
                      : isCompleted
                      ? "bg-green-50 dark:bg-green-900/20"
                      : "bg-gray-50 dark:bg-gray-900"
                  }`}
                >
                  <div
                    className={`flex-shrink-0 w-12 h-12 rounded-full flex items-center justify-center font-bold text-white`}
                    style={{
                      backgroundColor: isCompleted
                        ? "#10b981"
                        : isActive
                        ? stage.color
                        : "#6b7280",
                    }}
                  >
                    {isCompleted ? "✓" : stage.id}
                  </div>
                  <div className="flex-1">
                    <h4 className="font-semibold text-lg text-text-primary">
                      {stage.name}
                    </h4>
                    <p className="text-sm text-text-secondary">
                      {stage.description}
                    </p>
                  </div>
                  {isActive && (
                    <motion.div
                      animate={{ rotate: 360 }}
                      transition={{
                        duration: 1,
                        repeat: Infinity,
                        ease: "linear",
                      }}
                      className="flex-shrink-0"
                    >
                      <div
                        className="w-8 h-8 border-4 border-t-transparent rounded-full"
                        style={{ borderColor: stage.color, borderTopColor: "transparent" }}
                      />
                    </motion.div>
                  )}
                </div>

                {/* Stage Details */}
                {isActive && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: "auto", opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    className="border-t border-border-subtle"
                  >
                    <div className="p-4 bg-white dark:bg-gray-950">
                      <ul className="space-y-2">
                        {stage.details.map((detail, idx) => (
                          <motion.li
                            key={idx}
                            initial={{ opacity: 0, x: -10 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: idx * 0.2 }}
                            className="flex items-start gap-2 text-sm text-text-secondary"
                          >
                            <span
                              className="flex-shrink-0 w-1.5 h-1.5 rounded-full mt-1.5"
                              style={{ backgroundColor: stage.color }}
                            />
                            {detail}
                          </motion.li>
                        ))}
                      </ul>
                    </div>
                  </motion.div>
                )}
              </motion.div>
            );
          })}
        </AnimatePresence>
      </div>

      {/* Completion Message */}
      {currentStage === bootStages.length - 1 && !isBooting && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-6 p-6 bg-green-50 dark:bg-green-900/20 rounded-lg border-2 border-green-500"
        >
          <h4 className="font-semibold text-xl text-green-700 dark:text-green-300 mb-2">
            🎉 系统启动完成！
          </h4>
          <p className="text-text-secondary">
            总启动时间：<strong>{(bootTime / 1000).toFixed(2)} 秒</strong>
          </p>
          <p className="text-text-secondary mt-1">
            系统已进入运行状态，所有核心服务已就绪。
          </p>
        </motion.div>
      )}

      {/* Info Box */}
      <div className="mt-6 p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg border border-yellow-200 dark:border-yellow-800">
        <p className="text-sm text-text-secondary">
          <strong className="text-text-primary">现代 Linux 启动</strong>：
          整个过程涉及从实模式 (16-bit) → 保护模式 (32-bit) → 长模式 (64-bit)
          的 CPU 模式切换，以及从 BIOS/UEFI 固件 → Boot Loader → 内核 → init
          进程的多级引导链。
        </p>
      </div>
    </div>
  );
}
