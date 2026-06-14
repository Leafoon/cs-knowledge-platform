"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { TrendingUp, TrendingDown, Zap, HardDrive, Cpu, Clock } from "lucide-react";

interface Strategy {
  id: string;
  name: string;
  description: string;
  metrics: {
    speed: number;
    memory: number;
    complexity: number;
    security: number;
  };
  tradeoffs: {
    pros: string[];
    cons: string[];
  };
}

const strategies: Strategy[] = [
  {
    id: "cache",
    name: "CPU 缓存",
    description: "使用高速缓存减少内存访问延迟",
    metrics: { speed: 95, memory: 30, complexity: 60, security: 80 },
    tradeoffs: {
      pros: ["极快的访问速度", "减少内存总线压力", "降低功耗"],
      cons: ["容量有限", "一致性维护复杂", "成本高"]
    }
  },
  {
    id: "buffer",
    name: "I/O 缓冲",
    description: "批量处理 I/O 请求减少系统调用",
    metrics: { speed: 70, memory: 50, complexity: 40, security: 90 },
    tradeoffs: {
      pros: ["减少系统调用开销", "提高吞吐量", "实现简单"],
      cons: ["数据可能丢失", "延迟增加", "占用内存"]
    }
  },
  {
    id: "preemptive",
    name: "抢占式调度",
    description: "强制切换进程保证响应性",
    metrics: { speed: 60, memory: 70, complexity: 75, security: 70 },
    tradeoffs: {
      pros: ["响应时间短", "公平性好", "支持实时任务"],
      cons: ["上下文切换开销", "实现复杂", "可能优先级反转"]
    }
  },
  {
    id: "lazy",
    name: "延迟分配",
    description: "推迟资源分配直到实际使用",
    metrics: { speed: 50, memory: 90, complexity: 55, security: 60 },
    tradeoffs: {
      pros: ["节省内存", "启动快", "减少浪费"],
      cons: ["首次访问慢", "可能分配失败", "难以预测性能"]
    }
  }
];

export default function PerformanceTradeoffSimulator() {
  const [selectedStrategy, setSelectedStrategy] = useState<string>("cache");

  const selected = strategies.find(s => s.id === selectedStrategy)!;

  const MetricBar = ({ label, value, icon, color }: { label: string; value: number; icon: React.ReactNode; color: string }) => (
    <div className="mb-4">
      <div className="flex items-center justify-between mb-1">
        <div className="flex items-center gap-2 text-sm font-medium text-slate-700 dark:text-slate-300">
          {icon}
          {label}
        </div>
        <span className="text-sm font-bold text-slate-600 dark:text-slate-400">
          {value}%
        </span>
      </div>
      <div className="h-3 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${value}%` }}
          transition={{ duration: 0.8, ease: "easeOut" }}
          className={`h-full ${color} rounded-full`}
        />
      </div>
    </div>
  );

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-violet-50 to-violet-100 dark:from-violet-950 dark:to-slate-900 rounded-xl shadow-2xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-6">
        性能权衡分析器
      </h3>

      <div className="grid lg:grid-cols-3 gap-6">
        {/* 策略选择 */}
        <div className="lg:col-span-1 space-y-2">
          <h4 className="font-semibold text-slate-700 dark:text-slate-300 mb-3">
            优化策略
          </h4>
          {strategies.map((strategy) => (
            <motion.button
              key={strategy.id}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => setSelectedStrategy(strategy.id)}
              className={`
                w-full text-left p-3 rounded-lg transition-all
                ${selectedStrategy === strategy.id
                  ? "bg-violet-600 text-white shadow-lg"
                  : "bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-violet-50 dark:hover:bg-slate-700"
                }
              `}
            >
              <div className="font-semibold">{strategy.name}</div>
              <div className={`text-xs mt-1 ${selectedStrategy === strategy.id ? "text-violet-100" : "text-slate-500"}`}>
                {strategy.description}
              </div>
            </motion.button>
          ))}
        </div>

        {/* 性能指标 */}
        <div className="lg:col-span-2 space-y-4">
          <div className="p-5 bg-white dark:bg-slate-800 rounded-lg shadow">
            <h4 className="font-semibold text-slate-700 dark:text-slate-300 mb-4">
              性能指标
            </h4>
            <MetricBar
              label="执行速度"
              value={selected.metrics.speed}
              icon={<Zap className="w-4 h-4" />}
              color="bg-gradient-to-r from-green-500 to-green-600"
            />
            <MetricBar
              label="内存效率"
              value={selected.metrics.memory}
              icon={<HardDrive className="w-4 h-4" />}
              color="bg-gradient-to-r from-blue-500 to-blue-600"
            />
            <MetricBar
              label="实现复杂度"
              value={selected.metrics.complexity}
              icon={<Cpu className="w-4 h-4" />}
              color="bg-gradient-to-r from-orange-500 to-orange-600"
            />
            <MetricBar
              label="安全性"
              value={selected.metrics.security}
              icon={<Clock className="w-4 h-4" />}
              color="bg-gradient-to-r from-purple-500 to-purple-600"
            />
          </div>

          {/* 优缺点 */}
          <div className="grid md:grid-cols-2 gap-4">
            <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800">
              <h5 className="font-semibold text-green-800 dark:text-green-300 mb-3 flex items-center gap-2">
                <TrendingUp className="w-5 h-5" />
                优势
              </h5>
              <ul className="space-y-2">
                {selected.tradeoffs.pros.map((pro, i) => (
                  <li key={i} className="text-sm text-green-700 dark:text-green-200 flex items-start gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-green-500 mt-1.5" />
                    {pro}
                  </li>
                ))}
              </ul>
            </div>

            <div className="p-4 bg-red-50 dark:bg-red-900/20 rounded-lg border border-red-200 dark:border-red-800">
              <h5 className="font-semibold text-red-800 dark:text-red-300 mb-3 flex items-center gap-2">
                <TrendingDown className="w-5 h-5" />
                劣势
              </h5>
              <ul className="space-y-2">
                {selected.tradeoffs.cons.map((con, i) => (
                  <li key={i} className="text-sm text-red-700 dark:text-red-200 flex items-start gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-red-500 mt-1.5" />
                    {con}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      </div>

      {/* 总结 */}
      <div className="mt-6 p-4 bg-violet-50 dark:bg-violet-900/20 rounded-lg border border-violet-200 dark:border-violet-800">
        <p className="text-sm text-violet-900 dark:text-violet-100">
          <strong>权衡原则：</strong> 操作系统设计中没有"银弹"。每种优化策略都有其适用场景，
          需要根据实际需求（实时性、吞吐量、内存限制、功耗等）选择合适的方案。
        </p>
      </div>
    </div>
  );
}
