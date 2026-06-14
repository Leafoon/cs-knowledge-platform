"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Play, Pause, RotateCcw, Cpu, HardDrive, Database } from "lucide-react";

interface Stage {
  id: number;
  name: string;
  description: string;
  subsystems: string[];
  duration: number;
  color: string;
}

const stages: Stage[] = [
  {
    id: 1,
    name: "引导加载",
    description: "GRUB/UEFI 加载内核镜像到内存",
    subsystems: ["加载 vmlinuz", "加载 initramfs", "解压缩内核"],
    duration: 500,
    color: "from-blue-500 to-blue-600"
  },
  {
    id: 2,
    name: "内核自解压",
    description: "内核自解压并初始化早期环境",
    subsystems: ["解压内核", "设置页表", "启用 MMU"],
    duration: 300,
    color: "from-purple-500 to-purple-600"
  },
  {
    id: 3,
    name: "start_kernel",
    description: "内核主初始化函数",
    subsystems: [
      "中断初始化",
      "内存管理初始化",
      "进程调度初始化",
      "设备驱动初始化"
    ],
    duration: 800,
    color: "from-green-500 to-green-600"
  },
  {
    id: 4,
    name: "rest_init",
    description: "创建 init 进程并启动用户空间",
    subsystems: ["创建 kernel_init", "创建 kthreadd", "启动 init 进程"],
    duration: 400,
    color: "from-orange-500 to-orange-600"
  }
];

export default function KernelInitializationTimeline() {
  const [currentStage, setCurrentStage] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);

  const startAnimation = () => {
    setIsPlaying(true);
    setCurrentStage(0);
    animateStages(0);
  };

  const animateStages = (index: number) => {
    if (index >= stages.length) {
      setIsPlaying(false);
      return;
    }
    setTimeout(() => {
      setCurrentStage(index + 1);
      animateStages(index + 1);
    }, stages[index].duration);
  };

  const reset = () => {
    setCurrentStage(0);
    setIsPlaying(false);
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-indigo-100 dark:from-indigo-950 dark:to-slate-900 rounded-xl shadow-2xl">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <Cpu className="w-8 h-8 text-indigo-600 dark:text-indigo-400" />
          <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100">
            内核初始化时间线
          </h3>
        </div>
        <div className="flex gap-2">
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={startAnimation}
            disabled={isPlaying}
            className="px-4 py-2 bg-indigo-600 text-white rounded-lg disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
          >
            <Play className="w-4 h-4" />
            播放
          </motion.button>
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={reset}
            className="px-4 py-2 bg-slate-600 text-white rounded-lg flex items-center gap-2"
          >
            <RotateCcw className="w-4 h-4" />
            重置
          </motion.button>
        </div>
      </div>

      {/* 时间轴 */}
      <div className="mb-8 relative">
        <div className="flex items-center justify-between mb-2">
          {stages.map((stage, index) => (
            <div key={stage.id} className="flex-1 flex flex-col items-center">
              <motion.div
                className={`
                  w-12 h-12 rounded-full flex items-center justify-center font-bold text-white z-10
                  ${currentStage > index
                    ? `bg-gradient-to-r ${stage.color}`
                    : "bg-slate-300 dark:bg-slate-700"
                  }
                `}
                animate={currentStage === index ? {
                  scale: [1, 1.2, 1],
                  transition: { repeat: Infinity, duration: 1 }
                } : {}}
              >
                {stage.id}
              </motion.div>
              <div className="text-xs text-center mt-2 text-slate-600 dark:text-slate-400">
                {stage.name}
              </div>
            </div>
          ))}
        </div>
        <div className="absolute top-6 left-0 right-0 h-1 bg-slate-300 dark:bg-slate-700 -z-0">
          <motion.div
            className="h-full bg-gradient-to-r from-blue-500 to-orange-500"
            initial={{ width: "0%" }}
            animate={{ width: `${(currentStage / stages.length) * 100}%` }}
            transition={{ duration: 0.5 }}
          />
        </div>
      </div>

      {/* 当前阶段详情 */}
      <AnimatePresence mode="wait">
        {currentStage > 0 && currentStage <= stages.length && (
          <motion.div
            key={currentStage}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="p-6 bg-white dark:bg-slate-800 rounded-lg shadow-lg mb-6"
          >
            <div className={`inline-block px-4 py-2 rounded-lg bg-gradient-to-r ${stages[currentStage - 1].color} text-white font-bold mb-4`}>
              阶段 {currentStage}: {stages[currentStage - 1].name}
            </div>
            <p className="text-slate-700 dark:text-slate-300 mb-4">
              {stages[currentStage - 1].description}
            </p>
            <div className="space-y-2">
              {stages[currentStage - 1].subsystems.map((subsystem, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.1 }}
                  className="flex items-center gap-2 text-sm text-slate-600 dark:text-slate-400"
                >
                  <div className={`w-2 h-2 rounded-full bg-gradient-to-r ${stages[currentStage - 1].color}`} />
                  {subsystem}
                </motion.div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* 详细阶段说明 */}
      <div className="grid md:grid-cols-2 gap-4">
        {stages.map((stage) => (
          <div
            key={stage.id}
            className={`p-4 rounded-lg transition-all ${
              currentStage === stage.id
                ? "bg-gradient-to-r " + stage.color + " text-white shadow-lg"
                : "bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300"
            }`}
          >
            <h4 className="font-bold mb-2">阶段 {stage.id}: {stage.name}</h4>
            <p className={`text-sm mb-2 ${
              currentStage === stage.id ? "text-white/90" : "text-slate-600 dark:text-slate-400"
            }`}>
              {stage.description}
            </p>
            <div className="text-xs">
              {stage.subsystems.map((sub, i) => (
                <div key={i} className="flex items-center gap-1 mt-1">
                  <span className={currentStage === stage.id ? "text-white/70" : "text-slate-500"}>•</span>
                  {sub}
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* 代码参考 */}
      <div className="mt-6 p-4 bg-white dark:bg-slate-800 rounded-lg shadow">
        <h4 className="font-semibold text-slate-700 dark:text-slate-300 mb-2">init/main.c 代码片段</h4>
        <div className="p-3 bg-slate-100 dark:bg-slate-900 rounded font-mono text-xs text-slate-800 dark:text-slate-200 overflow-x-auto">
          <pre>{`asmlinkage __visible void __init start_kernel(void)
{
    set_task_stack_end_magic(&init_task);
    trap_init();           // 中断初始化
    mm_init();             // 内存管理初始化
    sched_init();          // 调度器初始化
    console_init();        // 控制台初始化
    
    rest_init();           // 创建 init 进程
}`}</pre>
        </div>
      </div>

      {/* 说明 */}
      <div className="mt-6 p-4 bg-indigo-50 dark:bg-indigo-900/20 rounded-lg border border-indigo-200 dark:border-indigo-800">
        <p className="text-sm text-indigo-900 dark:text-indigo-100">
          <strong>内核初始化流程：</strong> Linux 内核初始化经历多个阶段，从引导加载器加载内核镜像开始，
          经过自解压、start_kernel 主初始化、rest_init 创建 init 进程，最终启动用户空间。
          整个过程涉及中断、内存、调度、设备等多个子系统的初始化。
        </p>
      </div>
    </div>
  );
}
