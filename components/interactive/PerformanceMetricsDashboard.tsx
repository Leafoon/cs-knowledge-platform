"use client";

import React, { useState, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Gauge, Cpu, HardDrive, MemoryStick, Network, RefreshCw } from "lucide-react";

interface Metric {
  id: string;
  name: string;
  icon: React.ReactNode;
  value: number;
  target: number;
  unit: string;
  max: number;
  color: string;
  colorLight: string;
  history: number[];
}

const INITIAL_METRICS: Omit<Metric, "history">[] = [
  { id: "cpu", name: "CPU 使用率", icon: <Cpu className="w-4 h-4" />, value: 0, target: 73.2, unit: "%", max: 100, color: "#3b82f6", colorLight: "bg-blue-500" },
  { id: "mem", name: "内存使用", icon: <MemoryStick className="w-4 h-4" />, value: 0, target: 62.8, unit: "GB", max: 128, color: "#10b981", colorLight: "bg-emerald-500" },
  { id: "disk", name: "磁盘 IOPS", icon: <HardDrive className="w-4 h-4" />, value: 0, target: 45200, unit: "", max: 60000, color: "#8b5cf6", colorLight: "bg-violet-500" },
  { id: "net", name: "网络吞吐", icon: <Network className="w-4 h-4" />, value: 0, target: 8.7, unit: "Gbps", max: 10, color: "#f59e0b", colorLight: "bg-amber-500" },
];

export default function PerformanceMetricsDashboard() {
  const [metrics, setMetrics] = useState<Metric[]>(
    INITIAL_METRICS.map((m) => ({ ...m, history: [] }))
  );
  const [isAnimating, setIsAnimating] = useState(false);
  const [tick, setTick] = useState(0);

  const simulateValues = useCallback(() => {
    return INITIAL_METRICS.map((m) => {
      const jitter = m.target * 0.15 * (Math.random() - 0.5);
      return Math.max(0, Math.min(m.max, m.target + jitter));
    });
  }, []);

  useEffect(() => {
    if (!isAnimating) return;
    const interval = setInterval(() => {
      setTick((t) => t + 1);
      const targets = simulateValues();
      setMetrics((prev) =>
        prev.map((m, i) => {
          const newVal = targets[i];
          return {
            ...m,
            value: m.value + (newVal - m.value) * 0.3,
            history: [...m.history.slice(-29), newVal],
          };
        })
      );
    }, 200);
    return () => clearInterval(interval);
  }, [isAnimating, simulateValues]);

  const handleStart = () => {
    setMetrics(INITIAL_METRICS.map((m) => ({ ...m, history: [] })));
    setTick(0);
    setIsAnimating(true);
  };

  const handleStop = () => setIsAnimating(false);

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-sky-50 rounded-xl shadow-lg dark:from-gray-900 dark:to-gray-800">
      <h2 className="text-2xl font-bold text-slate-800 dark:text-gray-100 mb-2 text-center">
        性能指标实时仪表盘
      </h2>
      <p className="text-sm text-slate-500 dark:text-gray-400 text-center mb-6">
        模拟真实系统性能监控 — 观察指标波动与趋势
      </p>

      <div className="flex gap-3 mb-6 justify-center">
        <button onClick={handleStart} disabled={isAnimating} className="flex items-center gap-2 px-4 py-2 bg-sky-500 text-white rounded-lg hover:bg-sky-600 disabled:opacity-50 text-sm">
          <RefreshCw className={`w-4 h-4 ${isAnimating ? "animate-spin" : ""}`} />
          {isAnimating ? "监控中..." : "开始监控"}
        </button>
        <button onClick={handleStop} disabled={!isAnimating} className="px-4 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600 disabled:opacity-50 text-sm">
          停止
        </button>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        {metrics.map((m) => (
          <motion.div key={m.id} className="bg-white dark:bg-gray-800 rounded-xl p-5 border border-slate-200 dark:border-gray-700" layout>
            <div className="flex items-center gap-2 mb-3">
              <span style={{ color: m.color }}>{m.icon}</span>
              <span className="text-sm font-bold text-slate-700 dark:text-gray-200">{m.name}</span>
            </div>

            <div className="flex items-baseline gap-1 mb-2">
              <span className="text-3xl font-bold tabular-nums" style={{ color: m.color }}>
                {m.id === "disk" ? Math.round(m.value).toLocaleString() : m.value.toFixed(1)}
              </span>
              <span className="text-sm text-slate-400">{m.unit}</span>
            </div>

            {/* SVG ring gauge */}
            <div className="flex justify-center mb-2">
              <svg width="100" height="50" viewBox="0 0 100 50">
                <path d="M 10 45 A 40 40 0 0 1 90 45" fill="none" stroke="#e2e8f0" strokeWidth="8" strokeLinecap="round" />
                <motion.path
                  d="M 10 45 A 40 40 0 0 1 90 45"
                  fill="none"
                  stroke={m.color}
                  strokeWidth="8"
                  strokeLinecap="round"
                  strokeDasharray={`${(m.value / m.max) * 126} 126`}
                />
              </svg>
            </div>

            {/* Sparkline */}
            {m.history.length > 1 && (
              <svg width="100%" height="32" viewBox="0 0 300 32" preserveAspectRatio="none">
                <polyline
                  points={m.history.map((v, i) => `${(i / 29) * 300},${32 - (v / m.max) * 28}`).join(" ")}
                  fill="none"
                  stroke={m.color}
                  strokeWidth="1.5"
                  opacity="0.6"
                />
                {m.history.length > 0 && (
                  <circle
                    cx={((m.history.length - 1) / 29) * 300}
                    cy={32 - (m.history[m.history.length - 1] / m.max) * 28}
                    r="3"
                    fill={m.color}
                  />
                )}
              </svg>
            )}

            <div className="flex justify-between text-[10px] text-slate-400 mt-1">
              <span>Min: {m.history.length > 0 ? Math.min(...m.history).toFixed(1) : "—"}</span>
              <span>Max: {m.history.length > 0 ? Math.max(...m.history).toFixed(1) : "—"}</span>
            </div>
          </motion.div>
        ))}
      </div>

      <div className="text-center text-xs text-slate-400">
        采样间隔 200ms | Tick: {tick} | 模拟数据
      </div>
    </div>
  );
}

export { PerformanceMetricsDashboard };
