"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Layers, ChevronDown, ChevronUp, Info } from "lucide-react";

interface Layer {
  id: string;
  name: string;
  description: string;
  color: string;
  examples: string[];
  responsibilities: string[];
}

const layers: Layer[] = [
  {
    id: "application",
    name: "应用程序层",
    description: "用户直接交互的应用软件",
    color: "from-purple-500 to-purple-600",
    examples: ["浏览器", "文本编辑器", "视频播放器", "游戏"],
    responsibilities: ["用户界面", "业务逻辑", "数据处理", "用户体验"]
  },
  {
    id: "api",
    name: "系统 API 层",
    description: "操作系统提供的编程接口",
    color: "from-blue-500 to-blue-600",
    examples: ["POSIX API", "Win32 API", "系统调用接口", "库函数"],
    responsibilities: ["接口封装", "参数验证", "权限检查", "错误处理"]
  },
  {
    id: "kernel",
    name: "内核层",
    description: "操作系统核心功能实现",
    color: "from-green-500 to-green-600",
    examples: ["进程调度", "内存管理", "文件系统", "设备驱动"],
    responsibilities: ["资源分配", "进程管理", "内存管理", "I/O 控制"]
  },
  {
    id: "hal",
    name: "硬件抽象层 (HAL)",
    description: "隔离硬件差异的抽象层",
    color: "from-yellow-500 to-yellow-600",
    examples: ["中断控制器抽象", "定时器抽象", "DMA 抽象", "总线抽象"],
    responsibilities: ["硬件隔离", "接口统一", "平台适配", "驱动框架"]
  },
  {
    id: "hardware",
    name: "硬件层",
    description: "物理计算机硬件",
    color: "from-red-500 to-red-600",
    examples: ["CPU", "内存", "磁盘", "网卡"],
    responsibilities: ["指令执行", "数据存储", "信号传输", "电源管理"]
  }
];

export default function AbstractionLayersVisualization() {
  const [selectedLayer, setSelectedLayer] = useState<string | null>(null);
  const [showInfo, setShowInfo] = useState<{ [key: string]: boolean }>({});

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800 rounded-xl shadow-2xl">
      <div className="flex items-center gap-3 mb-6">
        <Layers className="w-8 h-8 text-blue-600 dark:text-blue-400" />
        <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100">
          操作系统抽象层次结构
        </h3>
      </div>

      <div className="space-y-3">
        {layers.map((layer, index) => (
          <motion.div
            key={layer.id}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.1 }}
            className="relative"
          >
            {/* 层级卡片 */}
            <motion.div
              whileHover={{ scale: 1.02 }}
              className={`
                cursor-pointer rounded-lg p-4 
                bg-gradient-to-r ${layer.color} text-white
                shadow-lg hover:shadow-xl transition-shadow
                ${selectedLayer === layer.id ? "ring-4 ring-white/50" : ""}
              `}
              onClick={() => setSelectedLayer(selectedLayer === layer.id ? null : layer.id)}
            >
              <div className="flex items-center justify-between">
                <div className="flex-1">
                  <h4 className="text-xl font-bold mb-1">{layer.name}</h4>
                  <p className="text-white/90 text-sm">{layer.description}</p>
                </div>
                <motion.div
                  animate={{ rotate: selectedLayer === layer.id ? 180 : 0 }}
                  transition={{ duration: 0.3 }}
                >
                  {selectedLayer === layer.id ? (
                    <ChevronUp className="w-6 h-6" />
                  ) : (
                    <ChevronDown className="w-6 h-6" />
                  )}
                </motion.div>
              </div>
            </motion.div>

            {/* 展开的详细信息 */}
            <AnimatePresence>
              {selectedLayer === layer.id && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: "auto", opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  transition={{ duration: 0.3 }}
                  className="overflow-hidden"
                >
                  <div className="mt-2 p-4 bg-white dark:bg-slate-800 rounded-lg shadow-inner">
                    <div className="grid md:grid-cols-2 gap-4">
                      {/* 典型示例 */}
                      <div>
                        <h5 className="font-semibold text-slate-700 dark:text-slate-300 mb-2 flex items-center gap-2">
                          <Info className="w-4 h-4" />
                          典型示例
                        </h5>
                        <ul className="space-y-1">
                          {layer.examples.map((example, i) => (
                            <li key={i} className="text-sm text-slate-600 dark:text-slate-400 flex items-center gap-2">
                              <span className="w-1.5 h-1.5 rounded-full bg-blue-500" />
                              {example}
                            </li>
                          ))}
                        </ul>
                      </div>

                      {/* 主要职责 */}
                      <div>
                        <h5 className="font-semibold text-slate-700 dark:text-slate-300 mb-2 flex items-center gap-2">
                          <Info className="w-4 h-4" />
                          主要职责
                        </h5>
                        <ul className="space-y-1">
                          {layer.responsibilities.map((resp, i) => (
                            <li key={i} className="text-sm text-slate-600 dark:text-slate-400 flex items-center gap-2">
                              <span className="w-1.5 h-1.5 rounded-full bg-green-500" />
                              {resp}
                            </li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* 层级间的连接线 */}
            {index < layers.length - 1 && (
              <div className="flex justify-center py-2">
                <motion.div
                  initial={{ scaleY: 0 }}
                  animate={{ scaleY: 1 }}
                  transition={{ delay: index * 0.1 + 0.2 }}
                  className="w-1 h-8 bg-gradient-to-b from-slate-400 to-slate-300 dark:from-slate-600 dark:to-slate-500 rounded-full"
                />
              </div>
            )}
          </motion.div>
        ))}
      </div>

      {/* 说明文字 */}
      <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
        <p className="text-sm text-blue-900 dark:text-blue-100">
          <strong>抽象层次设计原则：</strong> 每一层只能调用其下一层的接口，上层不需要了解底层实现细节。
          这种分层设计提高了系统的可维护性、可移植性和可扩展性。
        </p>
      </div>
    </div>
  );
}
