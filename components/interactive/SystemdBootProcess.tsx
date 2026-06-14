"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Play, Pause, RotateCcw, Server, Cpu, Settings } from "lucide-react";

interface InitStage {
  id: number;
  name: string;
  description: string;
  services: string[];
  duration: number;
  color: string;
}

const stages: InitStage[] = [
  {
    id: 1,
    name: "内核移交控制",
    description: "内核启动 systemd (PID 1)",
    services: ["执行 /sbin/init → /lib/systemd/systemd"],
    duration: 400,
    color: "from-blue-500 to-blue-600"
  },
  {
    id: 2,
    name: "default.target",
    description: "加载默认目标单元",
    services: [
      "multi-user.target (文本模式)",
      "graphical.target (图形模式)"
    ],
    duration: 600,
    color: "from-purple-500 to-purple-600"
  },
  {
    id: 3,
    name: "sysinit.target",
    description: "系统初始化",
    services: [
      "挂载文件系统",
      "启用交换分区",
      "加载内核模块",
      "设置主机名和时区"
    ],
    duration: 800,
    color: "from-green-500 to-green-600"
  },
  {
    id: 4,
    name: "basic.target",
    description: "基础系统服务",
    services: [
      "udev (设备管理)",
      "journald (日志服务)",
      "dbus (消息总线)"
    ],
    duration: 700,
    color: "from-orange-500 to-orange-600"
  },
  {
    id: 5,
    name: "multi-user.target",
    description: "多用户环境",
    services: [
      "网络服务 (NetworkManager)",
      "SSH 服务 (sshd)",
      "计划任务 (crond)"
    ],
    duration: 900,
    color: "from-red-500 to-red-600"
  },
  {
    id: 6,
    name: "graphical.target",
    description: "图形界面 (可选)",
    services: [
      "显示管理器 (gdm/lightdm)",
      "桌面环境 (GNOME/KDE)"
    ],
    duration: 500,
    color: "from-pink-500 to-pink-600"
  }
];

export default function SystemdBootProcess() {
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
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-rose-50 to-rose-100 dark:from-rose-950 dark:to-slate-900 rounded-xl shadow-2xl">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <Server className="w-8 h-8 text-rose-600 dark:text-rose-400" />
          <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100">
            Systemd 启动流程
          </h3>
        </div>
        <div className="flex gap-2">
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={startAnimation}
            disabled={isPlaying}
            className="px-4 py-2 bg-rose-600 text-white rounded-lg disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
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

      {/* 进度时间轴 */}
      <div className="mb-8 relative">
        <div className="flex justify-between mb-2">
          {stages.map((stage, index) => (
            <div key={stage.id} className="flex flex-col items-center flex-1">
              <motion.div
                className={`
                  w-10 h-10 rounded-full flex items-center justify-center font-bold text-white z-10 text-sm
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
              <div className="text-xs text-center mt-1 text-slate-600 dark:text-slate-400 max-w-[100px]">
                {stage.name}
              </div>
            </div>
          ))}
        </div>
        <div className="absolute top-5 left-0 right-0 h-1 bg-slate-300 dark:bg-slate-700 -z-0">
          <motion.div
            className="h-full bg-gradient-to-r from-blue-500 to-pink-500"
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
              {stages[currentStage - 1].services.map((service, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.15 }}
                  className="flex items-center gap-2 text-sm text-slate-600 dark:text-slate-400 p-2 bg-slate-50 dark:bg-slate-900 rounded"
                >
                  <div className={`w-2 h-2 rounded-full bg-gradient-to-r ${stages[currentStage - 1].color}`} />
                  {service}
                </motion.div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* 依赖关系图 */}
      <div className="grid md:grid-cols-2 gap-4 mb-6">
        <div className="p-6 bg-white dark:bg-slate-800 rounded-lg shadow-lg">
          <h4 className="font-semibold text-slate-700 dark:text-slate-300 mb-4 flex items-center gap-2">
            <Settings className="w-5 h-5 text-rose-600" />
            Target 依赖关系
          </h4>
          <div className="space-y-3 text-sm">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-blue-500" />
              <span className="text-slate-700 dark:text-slate-300">default.target</span>
            </div>
            <div className="ml-6 flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-green-500" />
              <span className="text-slate-700 dark:text-slate-300">sysinit.target</span>
            </div>
            <div className="ml-6 flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-orange-500" />
              <span className="text-slate-700 dark:text-slate-300">basic.target</span>
            </div>
            <div className="ml-12 flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-red-500" />
              <span className="text-slate-700 dark:text-slate-300">multi-user.target</span>
            </div>
            <div className="ml-16 flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-pink-500" />
              <span className="text-slate-700 dark:text-slate-300">graphical.target</span>
            </div>
          </div>
        </div>

        <div className="p-6 bg-white dark:bg-slate-800 rounded-lg shadow-lg">
          <h4 className="font-semibold text-slate-700 dark:text-slate-300 mb-4 flex items-center gap-2">
            <Cpu className="w-5 h-5 text-rose-600" />
            关键特性
          </h4>
          <ul className="space-y-2 text-sm text-slate-600 dark:text-slate-400">
            <li className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-rose-500" />
              并行启动服务，提升启动速度
            </li>
            <li className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-rose-500" />
              基于依赖关系自动排序
            </li>
            <li className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-rose-500" />
              按需激活 (socket/path/timer)
            </li>
            <li className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-rose-500" />
              统一日志管理 (journalctl)
            </li>
            <li className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-rose-500" />
              服务监控与自动重启
            </li>
          </ul>
        </div>
      </div>

      {/* 常用命令 */}
      <div className="p-6 bg-white dark:bg-slate-800 rounded-lg shadow-lg mb-6">
        <h4 className="font-semibold text-slate-700 dark:text-slate-300 mb-3">常用 systemctl 命令</h4>
        <div className="grid md:grid-cols-2 gap-3">
          <div className="p-3 bg-slate-100 dark:bg-slate-900 rounded">
            <div className="font-mono text-xs text-rose-600 mb-1">systemctl status service</div>
            <div className="text-xs text-slate-600 dark:text-slate-400">查看服务状态</div>
          </div>
          <div className="p-3 bg-slate-100 dark:bg-slate-900 rounded">
            <div className="font-mono text-xs text-rose-600 mb-1">systemctl start service</div>
            <div className="text-xs text-slate-600 dark:text-slate-400">启动服务</div>
          </div>
          <div className="p-3 bg-slate-100 dark:bg-slate-900 rounded">
            <div className="font-mono text-xs text-rose-600 mb-1">systemctl enable service</div>
            <div className="text-xs text-slate-600 dark:text-slate-400">开机自启</div>
          </div>
          <div className="p-3 bg-slate-100 dark:bg-slate-900 rounded">
            <div className="font-mono text-xs text-rose-600 mb-1">systemctl list-units</div>
            <div className="text-xs text-slate-600 dark:text-slate-400">列出所有单元</div>
          </div>
        </div>
      </div>

      {/* 说明 */}
      <div className="p-4 bg-rose-50 dark:bg-rose-900/20 rounded-lg border border-rose-200 dark:border-rose-800">
        <p className="text-sm text-rose-900 dark:text-rose-100">
          <strong>Systemd 启动流程：</strong> Systemd 是现代 Linux 发行版的标准初始化系统 (PID 1)，
          采用 target 单元组织启动流程。default.target 定义启动目标 (通常为 multi-user 或 graphical)，
          依赖关系决定启动顺序。Systemd 支持并行启动、按需激活、服务监控等高级特性，相比传统 SysV init 大幅提升启动速度和管理能力。
        </p>
      </div>
    </div>
  );
}
