"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Play, CheckCircle, ArrowRight } from "lucide-react";

interface Stage {
  id: number;
  name: string;
  description: string;
  tasks: string[];
  color: string;
}

const stages: Stage[] = [
  {
    id: 1,
    name: "BIOS/UEFI",
    description: "固件初始化硬件并执行 POST",
    tasks: ["硬件自检 (POST)", "初始化芯片组", "查找启动设备", "加载 Boot Loader"],
    color: "from-blue-500 to-blue-600"
  },
  {
    id: 2,
    name: "Boot Loader (GRUB)",
    description: "加载操作系统内核",
    tasks: ["读取配置文件", "显示启动菜单", "加载内核到内存", "传递启动参数"],
    color: "from-green-500 to-green-600"
  },
  {
    id: 3,
    name: "Kernel 初始化",
    description: "内核启动并初始化子系统",
    tasks: ["解压内核", "初始化内存管理", "初始化调度器", "挂载根文件系统"],
    color: "from-purple-500 to-purple-600"
  },
  {
    id: 4,
    name: "Init 进程",
    description: "启动用户空间第一个进程",
    tasks: ["执行 /sbin/init", "启动系统服务", "初始化网络", "准备用户登录"],
    color: "from-orange-500 to-orange-600"
  }
];

export default function BootLoaderWorkflow() {
  const [currentStage, setCurrentStage] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  const startAnimation = () => {
    setIsPlaying(true);
    setCurrentStage(0);
    
    const interval = setInterval(() => {
      setCurrentStage(prev => {
        if (prev >= stages.length - 1) {
          clearInterval(interval);
          setIsPlaying(false);
          return prev;
        }
        return prev + 1;
      });
    }, 2000);
  };

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800 rounded-xl shadow-2xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-6">
        系统启动工作流程
      </h3>

      {/* 阶段流程图 */}
      <div className="flex items-center justify-between mb-8 relative">
        {stages.map((stage, index) => (
          <React.Fragment key={stage.id}>
            <motion.div
              className={`relative flex-1 ${index < stages.length - 1 ? "mr-4" : ""}`}
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ delay: index * 0.1 }}
            >
              <motion.div
                className={`
                  p-4 rounded-lg bg-gradient-to-r ${stage.color} text-white
                  ${currentStage === index ? "ring-4 ring-white shadow-2xl" : "shadow-lg"}
                  transition-all cursor-pointer
                `}
                whileHover={{ scale: 1.05 }}
                onClick={() => !isPlaying && setCurrentStage(index)}
                animate={{
                  scale: currentStage === index ? 1.05 : 1,
                  y: currentStage === index ? -5 : 0
                }}
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs opacity-80">阶段 {stage.id}</span>
                  {currentStage >= index && !isPlaying && (
                    <CheckCircle className="w-5 h-5" />
                  )}
                </div>
                <h4 className="font-bold text-sm mb-1">{stage.name}</h4>
                <p className="text-xs opacity-90">{stage.description}</p>
              </motion.div>
            </motion.div>

            {index < stages.length - 1 && (
              <motion.div
                className="flex items-center"
                initial={{ opacity: 0 }}
                animate={{ opacity: currentStage > index ? 1 : 0.3 }}
                transition={{ duration: 0.5 }}
              >
                <ArrowRight className={`w-6 h-6 ${currentStage > index ? "text-green-600" : "text-slate-400"}`} />
              </motion.div>
            )}
          </React.Fragment>
        ))}
      </div>

      {/* 详细任务列表 */}
      <AnimatePresence mode="wait">
        <motion.div
          key={currentStage}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -20 }}
          className="p-6 bg-white dark:bg-slate-800 rounded-lg shadow-lg"
        >
          <h4 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-4">
            {stages[currentStage].name} - 详细任务
          </h4>
          <ul className="space-y-3">
            {stages[currentStage].tasks.map((task, i) => (
              <motion.li
                key={i}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.1 }}
                className="flex items-center gap-3 text-slate-700 dark:text-slate-300"
              >
                <CheckCircle className="w-5 h-5 text-green-600 flex-shrink-0" />
                <span>{task}</span>
              </motion.li>
            ))}
          </ul>
        </motion.div>
      </AnimatePresence>

      {/* 控制按钮 */}
      <div className="mt-6 flex gap-3 justify-center">
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={startAnimation}
          disabled={isPlaying}
          className="px-6 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-400 text-white rounded-lg font-semibold flex items-center gap-2"
        >
          <Play className="w-5 h-5" />
          {isPlaying ? "播放中..." : "开始演示"}
        </motion.button>
      </div>

      {/* 说明 */}
      <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
        <p className="text-sm text-blue-900 dark:text-blue-100">
          <strong>启动流程：</strong> 系统从 BIOS/UEFI 固件开始，经过 Boot Loader 加载内核，
          再由内核初始化系统，最后启动用户空间的 init 进程。整个过程通常在几秒到几十秒内完成。
        </p>
      </div>
    </div>
  );
}
