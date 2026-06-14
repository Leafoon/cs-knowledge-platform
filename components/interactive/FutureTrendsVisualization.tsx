"use client";

import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Cpu, Wifi, Atom, BrainCircuit, Zap, TrendingUp, Check } from "lucide-react";

interface Trend {
  id: string;
  title: string;
  icon: React.ReactNode;
  maturity: number;
  timeframe: string;
  description: string;
  technologies: string[];
  challenges: string[];
}

const TRENDS: Trend[] = [
  { id: "hw-co", title: "软硬件协同设计", icon: <Cpu className="w-5 h-5" />, maturity: 65, timeframe: "当前", description: "NVM、CXL、SmartNIC 等新硬件要求 OS 全新抽象层", technologies: ["PMFS", "DPDK", "io_uring", "CXL"], challenges: ["NVM 崩溃一致性", "异构调度"] },
  { id: "edge", title: "边缘计算 OS", icon: <Wifi className="w-5 h-5" />, maturity: 40, timeframe: "5 年内", description: "在资源受限的边缘设备上运行精简 OS", technologies: ["Unikernel", "eBPF", "Zephyr RTOS"], challenges: ["安全性", "异构硬件"] },
  { id: "quantum", title: "量子计算 OS", icon: <Atom className="w-5 h-5" />, maturity: 10, timeframe: "10+ 年", description: "管理量子比特、量子-经典混合计算", technologies: ["Qiskit Runtime", "Cirq"], challenges: ["量子纠错", "编程模型"] },
  { id: "ai-native", title: "AI 原生 OS", icon: <BrainCircuit className="w-5 h-5" />, maturity: 30, timeframe: "5-10 年", description: "AI 驱动的自适应资源管理和调度", technologies: ["Reinforcement Learning", "Auto-tuning"], challenges: ["可解释性", "安全性"] },
  { id: "energy", title: "能效优先设计", icon: <Zap className="w-5 h-5" />, maturity: 55, timeframe: "当前", description: "碳中和目标下的绿色计算", technologies: ["DVFS", "Heterogeneous", "Power Gating"], challenges: ["性能-能效权衡"] },
];

export default function FutureTrendsVisualization() {
  const [selected, setSelected] = useState<string | null>(null);
  const [animatedMaturity, setAnimatedMaturity] = useState<Record<string, number>>({});

  useEffect(() => {
    TRENDS.forEach((t) => {
      setTimeout(() => {
        setAnimatedMaturity((prev) => ({ ...prev, [t.id]: t.maturity }));
      }, 300 + TRENDS.indexOf(t) * 200);
    });
  }, []);

  const active = TRENDS.find((t) => t.id === selected);

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-emerald-50 rounded-xl shadow-lg dark:from-gray-900 dark:to-gray-800">
      <h2 className="text-2xl font-bold text-slate-800 dark:text-gray-100 mb-2 text-center">
        操作系统未来趋势
      </h2>
      <p className="text-sm text-slate-500 dark:text-gray-400 text-center mb-6">
        五大技术方向的成熟度与发展时间线
      </p>

      {/* Animated maturity bars */}
      <div className="bg-white dark:bg-gray-800 rounded-xl p-5 border border-slate-200 dark:border-gray-700 mb-6">
        <h3 className="text-sm font-bold text-slate-600 dark:text-gray-300 mb-4">技术成熟度</h3>
        <div className="space-y-3">
          {TRENDS.map((t, i) => (
            <div key={t.id} className="flex items-center gap-3">
              <span className="w-28 text-xs font-bold text-slate-600 dark:text-gray-300 text-right">{t.title}</span>
              <div className="flex-1 h-6 bg-slate-100 dark:bg-gray-700 rounded-full overflow-hidden relative">
                <motion.div
                  className="h-full rounded-full bg-gradient-to-r from-emerald-400 to-emerald-600"
                  initial={{ width: 0 }}
                  animate={{ width: `${animatedMaturity[t.id] || 0}%` }}
                  transition={{ duration: 1.2, delay: i * 0.15, ease: "easeOut" }}
                />
                <span className="absolute right-2 top-1/2 -translate-y-1/2 text-[10px] font-bold text-slate-600 dark:text-gray-300">
                  {animatedMaturity[t.id] || 0}%
                </span>
              </div>
              <span className="text-[10px] text-slate-400 w-16">{t.timeframe}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Trend cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 mb-6">
        {TRENDS.map((t) => (
          <motion.button key={t.id} onClick={() => setSelected(selected === t.id ? null : t.id)}
            className={`text-left p-4 rounded-xl border-2 transition-all ${
              selected === t.id ? "border-emerald-400 bg-emerald-50 dark:bg-emerald-950/30 shadow-md" : "border-transparent bg-white dark:bg-gray-800 hover:border-slate-300"
            }`} whileHover={{ scale: 1.01 }}>
            <div className="flex items-center gap-2 mb-2">
              <span className="text-emerald-500">{t.icon}</span>
              <span className="text-sm font-bold text-slate-700 dark:text-gray-200">{t.title}</span>
            </div>
            <p className="text-xs text-slate-500 dark:text-gray-400 mb-2">{t.description}</p>
            <div className="flex flex-wrap gap-1">
              {t.technologies.map((tech) => (
                <span key={tech} className="px-1.5 py-0.5 bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-300 rounded text-[10px] font-mono">
                  {tech}
                </span>
              ))}
            </div>
          </motion.button>
        ))}
      </div>

      {/* Detail */}
      <AnimatePresence>
        {active && (
          <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
            className="bg-white dark:bg-gray-800 rounded-xl p-5 border border-slate-200 dark:border-gray-700">
            <h3 className="text-base font-bold text-slate-800 dark:text-gray-100 mb-3">{active.title}</h3>
            <div className="mb-3">
              <span className="text-xs font-bold text-slate-500">关键挑战</span>
              <ul className="mt-1 space-y-1">
                {active.challenges.map((c) => (
                  <li key={c} className="flex items-center gap-2 text-xs text-slate-600 dark:text-gray-300">
                    <span className="text-red-400">!</span> {c}
                  </li>
                ))}
              </ul>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
