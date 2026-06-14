"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Shield, User, Check, X } from "lucide-react";

interface Mode {
  id: string;
  name: string;
  description: string;
  privileges: string[];
  restrictions: string[];
  color: string;
  icon: React.ReactNode;
}

const modes: Mode[] = [
  {
    id: "kernel",
    name: "内核模式 (Kernel Mode)",
    description: "Ring 0 - 最高权限，可执行所有指令",
    privileges: [
      "执行特权指令 (CLI/STI/HLT)",
      "访问所有内存地址",
      "直接访问硬件",
      "修改控制寄存器 (CR0/CR3)",
      "处理中断和异常",
      "切换进程上下文"
    ],
    restrictions: [],
    color: "from-red-500 to-red-600",
    icon: <Shield className="w-6 h-6" />
  },
  {
    id: "user",
    name: "用户模式 (User Mode)",
    description: "Ring 3 - 受限权限，保护系统安全",
    privileges: [
      "执行普通算术/逻辑指令",
      "访问分配的虚拟内存",
      "调用系统调用 (syscall)",
      "使用用户级库函数"
    ],
    restrictions: [
      "不能执行特权指令",
      "不能直接访问硬件",
      "不能访问内核内存",
      "不能修改页表",
      "不能禁用中断"
    ],
    color: "from-blue-500 to-blue-600",
    icon: <User className="w-6 h-6" />
  }
];

export default function ProcessorModeComparison() {
  const [selectedMode, setSelectedMode] = useState<string>("kernel");
  const selected = modes.find(m => m.id === selectedMode)!;

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800 rounded-xl shadow-2xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-6">
        处理器模式对比
      </h3>

      <div className="grid md:grid-cols-2 gap-4 mb-6">
        {modes.map((mode) => (
          <motion.button
            key={mode.id}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={() => setSelectedMode(mode.id)}
            className={`
              p-6 rounded-lg text-left transition-all
              ${selectedMode === mode.id
                ? `bg-gradient-to-r ${mode.color} text-white shadow-2xl`
                : "bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300 shadow-lg"
              }
            `}
          >
            <div className="flex items-center gap-3 mb-3">
              {mode.icon}
              <h4 className="text-xl font-bold">{mode.name}</h4>
            </div>
            <p className={`text-sm ${selectedMode === mode.id ? "text-white/90" : "text-slate-600 dark:text-slate-400"}`}>
              {mode.description}
            </p>
          </motion.button>
        ))}
      </div>

      {/* 详细权限对比 */}
      <motion.div
        key={selectedMode}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="grid md:grid-cols-2 gap-6"
      >
        {/* 允许的操作 */}
        <div className="p-6 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800">
          <h5 className="font-bold text-green-800 dark:text-green-300 mb-4 flex items-center gap-2">
            <Check className="w-5 h-5" />
            允许的操作
          </h5>
          <ul className="space-y-3">
            {selected.privileges.map((priv, i) => (
              <motion.li
                key={i}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.05 }}
                className="flex items-start gap-3 text-sm text-green-700 dark:text-green-200"
              >
                <Check className="w-4 h-4 flex-shrink-0 mt-0.5" />
                <span>{priv}</span>
              </motion.li>
            ))}
          </ul>
        </div>

        {/* 限制的操作 */}
        <div className="p-6 bg-red-50 dark:bg-red-900/20 rounded-lg border border-red-200 dark:border-red-800">
          <h5 className="font-bold text-red-800 dark:text-red-300 mb-4 flex items-center gap-2">
            <X className="w-5 h-5" />
            {selected.restrictions.length > 0 ? "禁止的操作" : "无限制"}
          </h5>
          {selected.restrictions.length > 0 ? (
            <ul className="space-y-3">
              {selected.restrictions.map((rest, i) => (
                <motion.li
                  key={i}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.05 }}
                  className="flex items-start gap-3 text-sm text-red-700 dark:text-red-200"
                >
                  <X className="w-4 h-4 flex-shrink-0 mt-0.5" />
                  <span>{rest}</span>
                </motion.li>
              ))}
            </ul>
          ) : (
            <p className="text-sm text-red-700 dark:text-red-200">
              内核模式拥有完全权限，可以执行所有指令和访问所有资源。
            </p>
          )}
        </div>
      </motion.div>

      {/* 模式切换示意 */}
      <div className="mt-6 p-6 bg-white dark:bg-slate-800 rounded-lg shadow-lg">
        <h4 className="font-semibold text-slate-700 dark:text-slate-300 mb-4">模式切换</h4>
        <div className="flex items-center justify-center gap-6">
          <div className="text-center">
            <div className="w-24 h-24 bg-blue-100 dark:bg-blue-900/30 rounded-full flex items-center justify-center mb-2">
              <User className="w-12 h-12 text-blue-600" />
            </div>
            <div className="font-semibold text-sm text-slate-700 dark:text-slate-300">用户模式</div>
          </div>

          <div className="flex flex-col items-center gap-2">
            <motion.div
              animate={{ x: [0, 10, 0] }}
              transition={{ duration: 1.5, repeat: Infinity }}
              className="text-green-600 dark:text-green-400"
            >
              <div className="text-xs font-mono">syscall →</div>
            </motion.div>
            <motion.div
              animate={{ x: [0, -10, 0] }}
              transition={{ duration: 1.5, repeat: Infinity }}
              className="text-purple-600 dark:text-purple-400"
            >
              <div className="text-xs font-mono">← sysret</div>
            </motion.div>
          </div>

          <div className="text-center">
            <div className="w-24 h-24 bg-red-100 dark:bg-red-900/30 rounded-full flex items-center justify-center mb-2">
              <Shield className="w-12 h-12 text-red-600" />
            </div>
            <div className="font-semibold text-sm text-slate-700 dark:text-slate-300">内核模式</div>
          </div>
        </div>
      </div>

      {/* 说明 */}
      <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
        <p className="text-sm text-blue-900 dark:text-blue-100">
          <strong>双模式设计：</strong> 处理器通过硬件支持的双模式机制保护系统。
          用户程序在受限的用户模式运行，需要系统服务时通过系统调用切换到内核模式。
          这种设计防止恶意程序直接破坏系统。
        </p>
      </div>
    </div>
  );
}
