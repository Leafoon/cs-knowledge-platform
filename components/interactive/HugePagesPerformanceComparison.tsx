"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { ArrowUpDown, Zap, TrendingUp } from "lucide-react";

interface Scenario {
  id: string;
  name: string;
  description: string;
  normalPages: number;
  hugePages: number;
  improvement: number;
}

const scenarios: Scenario[] = [
  {
    id: "database",
    name: "数据库服务器",
    description: "大内存数据库 (100GB)",
    normalPages: 25600000,
    hugePages: 51200,
    improvement: 15
  },
  {
    id: "vm",
    name: "虚拟化平台",
    description: "运行 20 个 VM",
    normalPages: 20480000,
    hugePages: 40960,
    improvement: 12
  },
  {
    id: "bigdata",
    name: "大数据处理",
    description: "Spark/Hadoop 集群",
    normalPages: 30720000,
    hugePages: 61440,
    improvement: 18
  }
];

export default function HugePagesPerformanceComparison() {
  const [selectedScenario, setSelectedScenario] = useState<string>("database");
  const scenario = scenarios.find(s => s.id === selectedScenario)!;

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-emerald-50 to-emerald-100 dark:from-emerald-950 dark:to-slate-900 rounded-xl shadow-2xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-6">
        大页 (Huge Pages) 性能对比
      </h3>

      {/* 场景选择 */}
      <div className="grid md:grid-cols-3 gap-4 mb-6">
        {scenarios.map((s) => (
          <motion.button
            key={s.id}
            whileHover={{ scale: 1.02 }}
            onClick={() => setSelectedScenario(s.id)}
            className={`
              p-4 rounded-lg transition-all
              ${selectedScenario === s.id
                ? "bg-emerald-600 text-white shadow-lg"
                : "bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300"
              }
            `}
          >
            <h4 className="font-bold mb-1">{s.name}</h4>
            <p className={`text-xs ${selectedScenario === s.id ? "text-emerald-100" : "text-slate-500"}`}>
              {s.description}
            </p>
          </motion.button>
        ))}
      </div>

      {/* 对比可视化 */}
      <div className="grid md:grid-cols-2 gap-6 mb-6">
        {/* 4KB 普通页 */}
        <div className="p-6 bg-white dark:bg-slate-800 rounded-lg shadow-lg">
          <h4 className="font-semibold text-slate-700 dark:text-slate-300 mb-4 flex items-center gap-2">
            <ArrowUpDown className="w-5 h-5 text-blue-600" />
            4KB 普通页
          </h4>
          <div className="space-y-4">
            <div>
              <div className="text-sm text-slate-600 dark:text-slate-400 mb-1">页表项数量</div>
              <div className="text-3xl font-bold text-blue-600">
                {scenario.normalPages.toLocaleString()}
              </div>
            </div>
            <div>
              <div className="text-sm text-slate-600 dark:text-slate-400 mb-1">TLB 缺失率</div>
              <div className="h-6 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
                <motion.div
                  className="h-full bg-red-500"
                  initial={{ width: 0 }}
                  animate={{ width: "85%" }}
                  transition={{ duration: 1, delay: 0.3 }}
                />
              </div>
              <div className="text-sm text-red-600 mt-1">高 (85%)</div>
            </div>
            <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded text-sm text-red-700 dark:text-red-300">
              ⚠️ 大量 TLB 缺失导致性能下降
            </div>
          </div>
        </div>

        {/* 2MB 大页 */}
        <div className="p-6 bg-white dark:bg-slate-800 rounded-lg shadow-lg border-2 border-emerald-500">
          <h4 className="font-semibold text-slate-700 dark:text-slate-300 mb-4 flex items-center gap-2">
            <Zap className="w-5 h-5 text-emerald-600" />
            2MB 大页 (Huge Pages)
          </h4>
          <div className="space-y-4">
            <div>
              <div className="text-sm text-slate-600 dark:text-slate-400 mb-1">页表项数量</div>
              <div className="text-3xl font-bold text-emerald-600">
                {scenario.hugePages.toLocaleString()}
              </div>
              <div className="text-xs text-emerald-600 mt-1">
                ↓ 减少 {((1 - scenario.hugePages / scenario.normalPages) * 100).toFixed(1)}%
              </div>
            </div>
            <div>
              <div className="text-sm text-slate-600 dark:text-slate-400 mb-1">TLB 缺失率</div>
              <div className="h-6 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
                <motion.div
                  className="h-full bg-green-500"
                  initial={{ width: 0 }}
                  animate={{ width: "15%" }}
                  transition={{ duration: 1, delay: 0.3 }}
                />
              </div>
              <div className="text-sm text-green-600 mt-1">低 (15%)</div>
            </div>
            <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded text-sm text-green-700 dark:text-green-300">
              ✓ TLB 命中率大幅提升
            </div>
          </div>
        </div>
      </div>

      {/* 性能提升 */}
      <div className="p-6 bg-gradient-to-r from-emerald-500 to-emerald-600 rounded-lg shadow-lg text-white">
        <div className="flex items-center justify-between">
          <div>
            <div className="text-sm opacity-90 mb-1">性能提升</div>
            <div className="text-5xl font-bold flex items-center gap-2">
              <TrendingUp className="w-12 h-12" />
              {scenario.improvement}%
            </div>
          </div>
          <div className="text-right">
            <div className="text-sm opacity-90">内存访问延迟降低</div>
            <div className="text-2xl font-bold mt-2">~{scenario.improvement * 2}%</div>
          </div>
        </div>
      </div>

      {/* 配置说明 */}
      <div className="mt-6 grid md:grid-cols-2 gap-4">
        <div className="p-4 bg-white dark:bg-slate-800 rounded-lg shadow">
          <h5 className="font-semibold text-slate-700 dark:text-slate-300 mb-2">Linux 配置</h5>
          <div className="p-3 bg-slate-100 dark:bg-slate-900 rounded font-mono text-xs text-slate-800 dark:text-slate-200">
            # 分配 1024 个 2MB 大页<br/>
            echo 1024 &gt; /proc/sys/vm/nr_hugepages<br/><br/>
            # 挂载 hugetlbfs<br/>
            mount -t hugetlbfs none /mnt/huge
          </div>
        </div>

        <div className="p-4 bg-white dark:bg-slate-800 rounded-lg shadow">
          <h5 className="font-semibold text-slate-700 dark:text-slate-300 mb-2">使用场景</h5>
          <ul className="space-y-2 text-sm text-slate-600 dark:text-slate-400">
            <li className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
              大内存数据库 (Oracle, PostgreSQL)
            </li>
            <li className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
              虚拟化平台 (KVM, QEMU)
            </li>
            <li className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
              高性能计算 (HPC)
            </li>
          </ul>
        </div>
      </div>

      {/* 说明 */}
      <div className="mt-6 p-4 bg-emerald-50 dark:bg-emerald-900/20 rounded-lg border border-emerald-200 dark:border-emerald-800">
        <p className="text-sm text-emerald-900 dark:text-emerald-100">
          <strong>大页优势：</strong> 使用 2MB 或 1GB 大页可以大幅减少页表项数量，提高 TLB 命中率，
          降低地址转换开销。特别适合内存密集型应用，可获得 10-20% 的性能提升。
        </p>
      </div>
    </div>
  );
}
