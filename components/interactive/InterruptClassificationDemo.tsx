"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Zap, AlertTriangle, Clock, Cpu } from "lucide-react";

interface InterruptType {
  id: string;
  name: string;
  description: string;
  vector: string;
  priority: number;
  examples: string[];
  color: string;
}

const interrupts: InterruptType[] = [
  {
    id: "hardware",
    name: "硬件中断 (IRQ)",
    description: "外部设备触发的异步中断",
    vector: "32-255",
    priority: 3,
    examples: ["键盘输入", "网卡收发包", "磁盘I/O完成", "定时器滴答"],
    color: "from-blue-500 to-blue-600"
  },
  {
    id: "exception",
    name: "异常 (Exception)",
    description: "CPU 执行指令时检测到的错误",
    vector: "0-31",
    priority: 1,
    examples: ["除零错误", "页故障", "非法指令", "断点"],
    color: "from-red-500 to-red-600"
  },
  {
    id: "software",
    name: "软件中断 (INT n)",
    description: "程序主动触发的同步中断",
    vector: "0-255",
    priority: 2,
    examples: ["系统调用 (INT 0x80)", "调试断点 (INT 3)", "溢出检查 (INT 4)"],
    color: "from-green-500 to-green-600"
  },
  {
    id: "nmi",
    name: "不可屏蔽中断 (NMI)",
    description: "最高优先级，不能被禁用",
    vector: "2",
    priority: 0,
    examples: ["硬件故障", "内存错误", "看门狗超时"],
    color: "from-purple-500 to-purple-600"
  }
];

export default function InterruptClassificationDemo() {
  const [selectedType, setSelectedType] = useState<string>("hardware");
  const selected = interrupts.find(i => i.id === selectedType)!;

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-950 dark:to-slate-900 rounded-xl shadow-2xl">
      <div className="flex items-center gap-3 mb-6">
        <Zap className="w-8 h-8 text-purple-600 dark:text-purple-400" />
        <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100">
          中断分类与特性
        </h3>
      </div>

      <div className="grid lg:grid-cols-3 gap-6">
        {/* 中断类型选择 */}
        <div className="lg:col-span-1 space-y-3">
          {interrupts.map((interrupt, index) => (
            <motion.button
              key={interrupt.id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1 }}
              whileHover={{ scale: 1.02 }}
              onClick={() => setSelectedType(interrupt.id)}
              className={`
                w-full text-left p-4 rounded-lg transition-all
                ${selectedType === interrupt.id
                  ? `bg-gradient-to-r ${interrupt.color} text-white shadow-lg`
                  : "bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300"
                }
              `}
            >
              <div className="flex items-center justify-between mb-2">
                <span className="font-bold">{interrupt.name}</span>
                <span className={`text-xs px-2 py-1 rounded ${
                  selectedType === interrupt.id 
                    ? "bg-white/20" 
                    : "bg-slate-100 dark:bg-slate-700"
                }`}>
                  优先级 {interrupt.priority}
                </span>
              </div>
              <p className={`text-xs ${
                selectedType === interrupt.id 
                  ? "text-white/90" 
                  : "text-slate-600 dark:text-slate-400"
              }`}>
                {interrupt.description}
              </p>
            </motion.button>
          ))}
        </div>

        {/* 详细信息 */}
        <div className="lg:col-span-2 space-y-4">
          <motion.div
            key={selectedType}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="p-6 bg-white dark:bg-slate-800 rounded-lg shadow-lg"
          >
            <h4 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-4">
              {selected.name}
            </h4>

            {/* 中断向量 */}
            <div className="mb-6 p-4 bg-slate-100 dark:bg-slate-900 rounded-lg">
              <div className="text-sm text-slate-600 dark:text-slate-400 mb-1">中断向量号</div>
              <div className="text-2xl font-bold text-purple-600">{selected.vector}</div>
            </div>

            {/* 典型示例 */}
            <div className="mb-4">
              <h5 className="font-semibold text-slate-700 dark:text-slate-300 mb-3 flex items-center gap-2">
                <AlertTriangle className="w-5 h-5" />
                典型示例
              </h5>
              <div className="grid gap-2">
                {selected.examples.map((example, i) => (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.1 }}
                    className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded border border-purple-200 dark:border-purple-800 text-sm text-purple-700 dark:text-purple-300"
                  >
                    {example}
                  </motion.div>
                ))}
              </div>
            </div>
          </motion.div>

          {/* 对比表格 */}
          <div className="p-6 bg-white dark:bg-slate-800 rounded-lg shadow-lg overflow-x-auto">
            <h4 className="font-semibold text-slate-700 dark:text-slate-300 mb-4">特性对比</h4>
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-200 dark:border-slate-700">
                  <th className="text-left p-2 text-slate-600 dark:text-slate-400">特性</th>
                  <th className="text-left p-2 text-slate-600 dark:text-slate-400">异常</th>
                  <th className="text-left p-2 text-slate-600 dark:text-slate-400">硬件中断</th>
                  <th className="text-left p-2 text-slate-600 dark:text-slate-400">软件中断</th>
                </tr>
              </thead>
              <tbody className="text-slate-700 dark:text-slate-300">
                <tr className="border-b border-slate-200 dark:border-slate-700">
                  <td className="p-2 font-medium">触发方式</td>
                  <td className="p-2">CPU 检测</td>
                  <td className="p-2">设备信号</td>
                  <td className="p-2">指令触发</td>
                </tr>
                <tr className="border-b border-slate-200 dark:border-slate-700">
                  <td className="p-2 font-medium">同步/异步</td>
                  <td className="p-2">同步</td>
                  <td className="p-2">异步</td>
                  <td className="p-2">同步</td>
                </tr>
                <tr className="border-b border-slate-200 dark:border-slate-700">
                  <td className="p-2 font-medium">可屏蔽</td>
                  <td className="p-2">否</td>
                  <td className="p-2">是</td>
                  <td className="p-2">否</td>
                </tr>
                <tr>
                  <td className="p-2 font-medium">处理紧急性</td>
                  <td className="p-2">立即</td>
                  <td className="p-2">可延迟</td>
                  <td className="p-2">立即</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* 说明 */}
      <div className="mt-6 p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg border border-purple-200 dark:border-purple-800">
        <p className="text-sm text-purple-900 dark:text-purple-100">
          <strong>中断分类：</strong> 中断根据触发源和特性分为多种类型。
          异常是 CPU 内部产生的同步事件，硬件中断来自外部设备的异步信号，
          软件中断由程序主动触发。不可屏蔽中断 (NMI) 具有最高优先级，用于处理严重硬件故障。
        </p>
      </div>
    </div>
  );
}
