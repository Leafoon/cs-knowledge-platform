"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { List, Zap, TrendingUp, Server } from "lucide-react";

interface Technique {
  id: string;
  name: string;
  description: string;
  latency: number;
  throughput: number;
  cpuUsage: number;
  complexity: string;
  advantages: string[];
  disadvantages: string[];
}

const techniques: Technique[] = [
  {
    id: "polling",
    name: "轮询 (Polling)",
    description: "CPU 定期检查设备状态",
    latency: 85,
    throughput: 40,
    cpuUsage: 90,
    complexity: "低",
    advantages: ["实现简单", "适合高速设备"],
    disadvantages: ["CPU 占用高", "响应延迟大"]
  },
  {
    id: "interrupt",
    name: "中断驱动 (Interrupt)",
    description: "设备通过中断通知 CPU",
    latency: 30,
    throughput: 75,
    cpuUsage: 40,
    complexity: "中",
    advantages: ["CPU 利用率高", "响应及时"],
    disadvantages: ["中断开销", "上下文切换成本"]
  },
  {
    id: "top-bottom",
    name: "上下半部 (Top/Bottom Half)",
    description: "分离紧急和延迟处理",
    latency: 25,
    throughput: 85,
    cpuUsage: 35,
    complexity: "中",
    advantages: ["降低中断延迟", "提高吞吐量"],
    disadvantages: ["实现复杂", "需要同步"]
  },
  {
    id: "napi",
    name: "NAPI",
    description: "混合中断和轮询",
    latency: 20,
    throughput: 95,
    cpuUsage: 25,
    complexity: "高",
    advantages: ["高性能", "自适应负载"],
    disadvantages: ["实现复杂", "调参困难"]
  }
];

export default function InterruptOptimizationComparison() {
  const [selectedTechnique, setSelectedTechnique] = useState<string>("interrupt");
  const technique = techniques.find(t => t.id === selectedTechnique)!;

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-cyan-50 to-cyan-100 dark:from-cyan-950 dark:to-slate-900 rounded-xl shadow-2xl">
      <div className="flex items-center gap-3 mb-6">
        <Zap className="w-8 h-8 text-cyan-600 dark:text-cyan-400" />
        <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100">
          中断处理优化技术对比
        </h3>
      </div>

      {/* 技术选择 */}
      <div className="grid md:grid-cols-4 gap-3 mb-6">
        {techniques.map((tech) => (
          <motion.button
            key={tech.id}
            whileHover={{ scale: 1.02 }}
            onClick={() => setSelectedTechnique(tech.id)}
            className={`
              p-4 rounded-lg transition-all
              ${selectedTechnique === tech.id
                ? "bg-cyan-600 text-white shadow-lg"
                : "bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300"
              }
            `}
          >
            <h4 className="font-bold mb-1">{tech.name}</h4>
            <p className={`text-xs ${selectedTechnique === tech.id ? "text-cyan-100" : "text-slate-500"}`}>
              复杂度: {tech.complexity}
            </p>
          </motion.button>
        ))}
      </div>

      {/* 详细信息 */}
      <div className="grid lg:grid-cols-2 gap-6 mb-6">
        {/* 性能指标 */}
        <div className="p-6 bg-white dark:bg-slate-800 rounded-lg shadow-lg">
          <h4 className="font-semibold text-slate-700 dark:text-slate-300 mb-4 flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-cyan-600" />
            性能指标
          </h4>
          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-slate-600 dark:text-slate-400">响应延迟</span>
                <span className="font-semibold text-cyan-600">{100 - technique.latency}%</span>
              </div>
              <div className="h-4 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
                <motion.div
                  className="h-full bg-gradient-to-r from-green-400 to-green-600"
                  initial={{ width: 0 }}
                  animate={{ width: `${100 - technique.latency}%` }}
                  transition={{ duration: 0.8 }}
                />
              </div>
            </div>

            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-slate-600 dark:text-slate-400">吞吐量</span>
                <span className="font-semibold text-cyan-600">{technique.throughput}%</span>
              </div>
              <div className="h-4 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
                <motion.div
                  className="h-full bg-gradient-to-r from-blue-400 to-blue-600"
                  initial={{ width: 0 }}
                  animate={{ width: `${technique.throughput}%` }}
                  transition={{ duration: 0.8, delay: 0.2 }}
                />
              </div>
            </div>

            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-slate-600 dark:text-slate-400">CPU 利用率</span>
                <span className="font-semibold text-cyan-600">{100 - technique.cpuUsage}%</span>
              </div>
              <div className="h-4 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
                <motion.div
                  className="h-full bg-gradient-to-r from-purple-400 to-purple-600"
                  initial={{ width: 0 }}
                  animate={{ width: `${100 - technique.cpuUsage}%` }}
                  transition={{ duration: 0.8, delay: 0.4 }}
                />
              </div>
            </div>
          </div>
        </div>

        {/* 优劣势 */}
        <div className="p-6 bg-white dark:bg-slate-800 rounded-lg shadow-lg">
          <h4 className="font-semibold text-slate-700 dark:text-slate-300 mb-4 flex items-center gap-2">
            <List className="w-5 h-5 text-cyan-600" />
            优劣势分析
          </h4>
          <div className="space-y-4">
            <div>
              <div className="text-sm font-semibold text-green-600 mb-2">✓ 优势</div>
              <div className="space-y-1">
                {technique.advantages.map((adv, i) => (
                  <div key={i} className="text-sm text-slate-600 dark:text-slate-400 flex items-center gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-green-500" />
                    {adv}
                  </div>
                ))}
              </div>
            </div>
            <div>
              <div className="text-sm font-semibold text-red-600 mb-2">✗ 劣势</div>
              <div className="space-y-1">
                {technique.disadvantages.map((dis, i) => (
                  <div key={i} className="text-sm text-slate-600 dark:text-slate-400 flex items-center gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-red-500" />
                    {dis}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* NAPI 工作原理 */}
      {selectedTechnique === "napi" && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="p-6 bg-gradient-to-r from-cyan-500 to-cyan-600 rounded-lg shadow-lg text-white mb-6"
        >
          <h4 className="font-bold mb-4 flex items-center gap-2">
            <Server className="w-6 h-6" />
            NAPI (New API) 自适应工作原理
          </h4>
          <div className="grid md:grid-cols-3 gap-4">
            <div className="p-3 bg-white/10 rounded">
              <div className="font-semibold mb-1">低负载</div>
              <div className="text-sm text-cyan-100">使用中断模式，及时响应</div>
            </div>
            <div className="p-3 bg-white/10 rounded">
              <div className="font-semibold mb-1">高负载</div>
              <div className="text-sm text-cyan-100">切换到轮询模式，降低开销</div>
            </div>
            <div className="p-3 bg-white/10 rounded">
              <div className="font-semibold mb-1">自适应</div>
              <div className="text-sm text-cyan-100">动态调整策略，平衡性能</div>
            </div>
          </div>
        </motion.div>
      )}

      {/* 对比表格 */}
      <div className="p-6 bg-white dark:bg-slate-800 rounded-lg shadow-lg overflow-x-auto">
        <h4 className="font-semibold text-slate-700 dark:text-slate-300 mb-4">技术对比</h4>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-200 dark:border-slate-700">
              <th className="text-left p-2 text-slate-600 dark:text-slate-400">技术</th>
              <th className="text-left p-2 text-slate-600 dark:text-slate-400">延迟</th>
              <th className="text-left p-2 text-slate-600 dark:text-slate-400">吞吐</th>
              <th className="text-left p-2 text-slate-600 dark:text-slate-400">CPU</th>
              <th className="text-left p-2 text-slate-600 dark:text-slate-400">适用场景</th>
            </tr>
          </thead>
          <tbody className="text-slate-700 dark:text-slate-300">
            {techniques.map((tech) => (
              <tr key={tech.id} className="border-b border-slate-200 dark:border-slate-700">
                <td className="p-2 font-medium">{tech.name}</td>
                <td className="p-2">{100 - tech.latency}%</td>
                <td className="p-2">{tech.throughput}%</td>
                <td className="p-2">{100 - tech.cpuUsage}%</td>
                <td className="p-2 text-xs">
                  {tech.id === "polling" && "简单设备"}
                  {tech.id === "interrupt" && "通用设备"}
                  {tech.id === "top-bottom" && "复杂设备"}
                  {tech.id === "napi" && "高速网络"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* 说明 */}
      <div className="mt-6 p-4 bg-cyan-50 dark:bg-cyan-900/20 rounded-lg border border-cyan-200 dark:border-cyan-800">
        <p className="text-sm text-cyan-900 dark:text-cyan-100">
          <strong>中断优化：</strong> 不同的中断处理技术适用于不同场景。
          轮询适合简单设备，中断驱动是通用方案，上下半部用于复杂设备，NAPI 专为高速网络设计。
          Linux 网络栈使用 NAPI 在低负载时用中断，高负载时切换为轮询，实现自适应优化。
        </p>
      </div>
    </div>
  );
}
