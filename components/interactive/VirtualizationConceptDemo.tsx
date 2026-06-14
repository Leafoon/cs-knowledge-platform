"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Layers, Box, Cpu, HardDrive, Monitor } from "lucide-react";

interface VirtualizationType {
  id: string;
  name: string;
  description: string;
  layers: string[];
  advantages: string[];
  examples: string[];
}

const types: VirtualizationType[] = [
  {
    id: "cpu",
    name: "CPU 虚拟化",
    description: "时间片轮转，多个进程共享 CPU",
    layers: ["进程A", "进程B", "进程C", "操作系统调度器", "物理 CPU"],
    advantages: ["并发执行", "CPU 利用率高", "进程隔离"],
    examples: ["进程调度", "线程切换", "时间片轮转"]
  },
  {
    id: "memory",
    name: "内存虚拟化",
    description: "虚拟地址空间，独立内存视图",
    layers: ["进程虚拟地址空间", "页表映射", "物理内存", "磁盘 Swap"],
    advantages: ["内存隔离", "地址空间独立", "超额分配"],
    examples: ["虚拟内存", "分页机制", "内存保护"]
  },
  {
    id: "storage",
    name: "存储虚拟化",
    description: "文件系统抽象底层存储设备",
    layers: ["文件", "文件系统层", "块设备层", "物理磁盘"],
    advantages: ["统一接口", "位置透明", "容量管理"],
    examples: ["文件系统", "RAID", "逻辑卷管理"]
  },
  {
    id: "vm",
    name: "虚拟机",
    description: "完整操作系统虚拟化",
    layers: ["Guest OS", "Hypervisor", "Host OS", "物理硬件"],
    advantages: ["完全隔离", "多系统共存", "资源灵活分配"],
    examples: ["VMware", "KVM", "Docker"]
  }
];

export default function VirtualizationConceptDemo() {
  const [selectedType, setSelectedType] = useState<string>("cpu");

  const selected = types.find(t => t.id === selectedType)!;

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-teal-50 to-teal-100 dark:from-teal-950 dark:to-slate-900 rounded-xl shadow-2xl">
      <div className="flex items-center gap-3 mb-6">
        <Layers className="w-8 h-8 text-teal-600 dark:text-teal-400" />
        <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100">
          虚拟化概念演示
        </h3>
      </div>

      <div className="grid lg:grid-cols-3 gap-6">
        {/* 虚拟化类型选择 */}
        <div className="lg:col-span-1 space-y-2">
          {types.map((type) => {
            const Icon = type.id === "cpu" ? Cpu : type.id === "memory" ? HardDrive : type.id === "storage" ? Box : Monitor;
            return (
              <motion.button
                key={type.id}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => setSelectedType(type.id)}
                className={`
                  w-full text-left p-4 rounded-lg transition-all flex items-center gap-3
                  ${selectedType === type.id
                    ? "bg-teal-600 text-white shadow-lg"
                    : "bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-teal-50 dark:hover:bg-slate-700"
                  }
                `}
              >
                <Icon className="w-6 h-6" />
                <div className="flex-1">
                  <div className="font-semibold">{type.name}</div>
                  <div className={`text-xs mt-1 ${selectedType === type.id ? "text-teal-100" : "text-slate-500"}`}>
                    {type.description}
                  </div>
                </div>
              </motion.button>
            );
          })}
        </div>

        {/* 详细信息 */}
        <div className="lg:col-span-2 space-y-4">
          {/* 层次结构可视化 */}
          <div className="p-5 bg-white dark:bg-slate-800 rounded-lg shadow">
            <h4 className="font-semibold text-slate-700 dark:text-slate-300 mb-4">
              层次结构
            </h4>
            <div className="space-y-3">
              {selected.layers.map((layer, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.1 }}
                  className={`
                    p-3 rounded-lg text-center font-medium
                    ${i === 0
                      ? "bg-gradient-to-r from-purple-500 to-purple-600 text-white"
                      : i === selected.layers.length - 1
                      ? "bg-gradient-to-r from-slate-600 to-slate-700 text-white"
                      : "bg-gradient-to-r from-teal-400 to-teal-500 text-white"
                    }
                  `}
                >
                  {layer}
                </motion.div>
              ))}
            </div>

            {/* 向下箭头指示 */}
            <div className="flex justify-center my-2">
              <motion.div
                animate={{ y: [0, 5, 0] }}
                transition={{ duration: 1.5, repeat: Infinity }}
                className="text-slate-400"
              >
                ↓ 抽象层次递减 ↓
              </motion.div>
            </div>
          </div>

          {/* 优势 */}
          <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800">
            <h5 className="font-semibold text-green-800 dark:text-green-300 mb-3">
              主要优势
            </h5>
            <ul className="space-y-2">
              {selected.advantages.map((adv, i) => (
                <li key={i} className="text-sm text-green-700 dark:text-green-200 flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-green-500" />
                  {adv}
                </li>
              ))}
            </ul>
          </div>

          {/* 实际应用 */}
          <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
            <h5 className="font-semibold text-blue-800 dark:text-blue-300 mb-3">
              典型应用
            </h5>
            <div className="flex flex-wrap gap-2">
              {selected.examples.map((example, i) => (
                <span
                  key={i}
                  className="px-3 py-1 bg-blue-600 text-white text-sm rounded-full"
                >
                  {example}
                </span>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* 总结 */}
      <div className="mt-6 p-4 bg-teal-50 dark:bg-teal-900/20 rounded-lg border border-teal-200 dark:border-teal-800">
        <p className="text-sm text-teal-900 dark:text-teal-100">
          <strong>虚拟化核心思想：</strong> 在有限物理资源之上构建抽象层，为每个程序提供"独占全部资源"的假象。
          这是操作系统最核心的设计理念之一，实现了资源共享与隔离的平衡。
        </p>
      </div>
    </div>
  );
}
