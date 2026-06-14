"use client";

import React, { useState, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Shuffle, RotateCcw, Shield, Eye, EyeOff } from "lucide-react";

interface Segment {
  name: string;
  color: string;
  darkColor: string;
  textColor: string;
  darkTextColor: string;
  offset: number;
  size: number;
  randomOffset?: number;
}

const baseSegments: Segment[] = [
  {
    name: "内核空间",
    color: "bg-red-200",
    darkColor: "dark:bg-red-900/50",
    textColor: "text-red-700",
    darkTextColor: "dark:text-red-300",
    offset: 0,
    size: 15,
  },
  {
    name: "栈 (Stack)",
    color: "bg-blue-200",
    darkColor: "dark:bg-blue-900/50",
    textColor: "text-blue-700",
    darkTextColor: "dark:text-blue-300",
    offset: 15,
    size: 20,
  },
  {
    name: "mmap (共享库)",
    color: "bg-purple-200",
    darkColor: "dark:bg-purple-900/50",
    textColor: "text-purple-700",
    darkTextColor: "dark:text-purple-300",
    offset: 35,
    size: 20,
  },
  {
    name: "堆 (Heap)",
    color: "bg-emerald-200",
    darkColor: "dark:bg-emerald-900/50",
    textColor: "text-emerald-700",
    darkTextColor: "dark:text-emerald-300",
    offset: 55,
    size: 15,
  },
  {
    name: "BSS / 数据段",
    color: "bg-amber-200",
    darkColor: "dark:bg-amber-900/50",
    textColor: "text-amber-700",
    darkTextColor: "dark:text-amber-300",
    offset: 70,
    size: 12,
  },
  {
    name: "代码段 (.text)",
    color: "bg-cyan-200",
    darkColor: "dark:bg-cyan-900/50",
    textColor: "text-cyan-700",
    darkTextColor: "dark:text-cyan-300",
    offset: 82,
    size: 10,
  },
];

function generateRandomOffset(): number {
  return Math.floor(Math.random() * 12) - 6; // -6 to +5
}

function formatAddr(base: number, offset: number): string {
  const addr = base + offset * 0x1000;
  return `0x${addr.toString(16).padStart(12, "0")}`;
}

export default function ASLRVisualization() {
  const [aslEnabled, setAslrEnabled] = useState(true);
  const [iterations, setIterations] = useState<{ segments: Segment[] }[]>([]);
  const [currentIter, setCurrentIter] = useState(0);
  const [showAddrs, setShowAddrs] = useState(true);

  const generateLayout = useCallback(() => {
    if (!aslEnabled) {
      return baseSegments.map((s) => ({ ...s, randomOffset: 0 }));
    }
    return baseSegments.map((s) => ({
      ...s,
      randomOffset:
        s.name === "内核空间" ? 0 : generateRandomOffset() * (s.size / 10),
    }));
  }, [aslEnabled]);

  const addIteration = useCallback(() => {
    const layout = generateLayout();
    setIterations((prev) => [...prev.slice(-4), { segments: layout }]);
    setCurrentIter((prev) => Math.min(prev + 1, 4));
  }, [generateLayout]);

  const reset = useCallback(() => {
    setIterations([]);
    setCurrentIter(0);
  }, []);

  useEffect(() => {
    reset();
  }, [aslEnabled, reset]);

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-blue-50 dark:from-indigo-950 dark:to-slate-900 rounded-xl shadow-2xl">
      <div className="flex items-center gap-3 mb-4">
        <Shuffle className="w-8 h-8 text-indigo-600 dark:text-indigo-400" />
        <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100">
          ASLR 地址空间布局随机化
        </h3>
      </div>

      {/* Controls */}
      <div className="flex flex-wrap gap-3 mb-6">
        <button
          onClick={() => setAslrEnabled(!aslEnabled)}
          className={`flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
            aslEnabled
              ? "bg-indigo-100 dark:bg-indigo-900/40 text-indigo-700 dark:text-indigo-300 border border-indigo-300 dark:border-indigo-700"
              : "bg-white dark:bg-slate-700 text-slate-600 dark:text-slate-300 border border-slate-200 dark:border-slate-600"
          }`}
        >
          <Shield className="w-4 h-4" />
          ASLR: {aslEnabled ? "ON" : "OFF"}
        </button>

        <button
          onClick={addIteration}
          className="flex items-center gap-1.5 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm font-medium transition-all"
        >
          <Shuffle className="w-4 h-4" />
          重新加载程序
        </button>

        <button
          onClick={reset}
          className="flex items-center gap-1.5 px-4 py-2 bg-slate-600 hover:bg-slate-700 text-white rounded-lg text-sm font-medium transition-all"
        >
          <RotateCcw className="w-4 h-4" />
          清空
        </button>

        <button
          onClick={() => setShowAddrs(!showAddrs)}
          className={`flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm font-medium transition-all ${
            showAddrs
              ? "bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-200"
              : "bg-white dark:bg-slate-800 text-slate-500 dark:text-slate-400 border border-slate-200 dark:border-slate-600"
          }`}
        >
          {showAddrs ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />}
          地址
        </button>
      </div>

      {/* Visualizations */}
      <div className="space-y-3">
        {iterations.length === 0 && (
          <div className="bg-white dark:bg-slate-800 rounded-lg p-8 border border-slate-200 dark:border-slate-700 text-center text-slate-400 dark:text-slate-500">
            点击 &ldquo;重新加载程序&rdquo; 查看内存布局
          </div>
        )}

        <AnimatePresence>
          {iterations.map((iter, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.3 }}
              className="bg-white dark:bg-slate-800 rounded-lg p-4 border border-slate-200 dark:border-slate-700"
            >
              <div className="flex items-center gap-2 mb-2">
                <span className="text-xs font-medium text-slate-500 dark:text-slate-400">
                  加载 #{idx + 1}
                </span>
                {aslEnabled && idx > 0 && (
                  <span className="text-xs px-1.5 py-0.5 bg-indigo-100 dark:bg-indigo-900/30 text-indigo-600 dark:text-indigo-400 rounded">
                    布局不同!
                  </span>
                )}
              </div>

              <div className="relative h-12 rounded-lg overflow-hidden bg-slate-100 dark:bg-slate-700">
                {iter.segments.map((seg, segIdx) => {
                  const randomOff = seg.randomOffset || 0;
                  const left = Math.max(0, Math.min(95, seg.offset + randomOff));
                  const width = seg.size;

                  return (
                    <motion.div
                      key={segIdx}
                      initial={{ scaleX: 0 }}
                      animate={{ scaleX: 1 }}
                      transition={{ delay: segIdx * 0.08, duration: 0.3 }}
                      className={`absolute top-0 h-full ${seg.color} ${seg.darkColor} border-r border-white dark:border-slate-600 flex items-center justify-center`}
                      style={{
                        left: `${left}%`,
                        width: `${width}%`,
                        transformOrigin: "left",
                      }}
                    >
                      <span
                        className={`text-xs font-medium ${seg.textColor} ${seg.darkTextColor} truncate px-1`}
                      >
                        {seg.name}
                      </span>
                    </motion.div>
                  );
                })}
              </div>

              {showAddrs && aslEnabled && (
                <div className="mt-2 flex flex-wrap gap-3 text-xs font-mono text-slate-500 dark:text-slate-400">
                  {iter.segments.map((seg, segIdx) => {
                    const base = 0x7f0000000000;
                    const randomOff = seg.randomOffset || 0;
                    return (
                      <span key={segIdx}>
                        {seg.name}: {formatAddr(base + seg.offset * 0x10000, randomOff)}
                      </span>
                    );
                  })}
                </div>
              )}
            </motion.div>
          ))}
        </AnimatePresence>
      </div>

      {/* Legend */}
      <div className="mt-4 grid grid-cols-3 sm:grid-cols-6 gap-2 text-xs text-slate-500 dark:text-slate-400">
        {baseSegments.map((seg, i) => (
          <div key={i} className="flex items-center gap-1.5">
            <div
              className={`w-3 h-3 rounded ${seg.color} ${seg.darkColor}`}
            />
            <span>{seg.name}</span>
          </div>
        ))}
      </div>

      {/* Info */}
      <div className="mt-4 p-3 bg-indigo-50 dark:bg-indigo-900/20 rounded-lg border border-indigo-200 dark:border-indigo-800">
        <p className="text-xs text-indigo-700 dark:text-indigo-300">
          <strong>ASLR 原理：</strong>
          每次程序加载时，栈、堆、共享库、代码段的基址都会随机偏移。攻击者无法预测目标地址，大大增加了利用漏洞的难度。
          {aslEnabled
            ? " 当前 ASLR 已启用，注意每次加载后各段地址的变化。"
            : " 当前 ASLR 已禁用，每次加载地址相同，攻击者可以预先计算目标地址。"}
        </p>
      </div>
    </div>
  );
}
