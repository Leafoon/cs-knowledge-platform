"use client";

import React, { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Cpu, Play, Pause, RotateCcw, Zap, Shield } from "lucide-react";

interface CacheLine {
  bytes: { label: string; owner: number | null; color: string }[];
}

const CACHE_LINE_SIZE = 64;
const VAR_SIZE = 8;

export default function FalseSharingDemo() {
  const [mode, setMode] = useState<"problem" | "fix">("problem");
  const [isRunning, setIsRunning] = useState(false);
  const [iteration, setIteration] = useState(0);
  const [cpu0Count, setCpu0Count] = useState(0);
  const [cpu1Count, setCpu1Count] = useState(0);
  const [bounces, setBounces] = useState(0);
  const [activeCpu, setActiveCpu] = useState<0 | 1>(0);
  const [flashLine, setFlashLine] = useState<number | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  const isProblem = mode === "problem";

  useEffect(() => {
    if (isRunning) {
      intervalRef.current = setInterval(() => {
        setIteration(prev => prev + 1);
        setActiveCpu(prev => {
          const next = prev === 0 ? 1 : 0;
          return next;
        });
        setBounces(prev => prev + 1);
        setFlashLine(0);
        setTimeout(() => setFlashLine(null), 300);

        if (Math.random() > 0.5) {
          setCpu0Count(prev => prev + 1);
        } else {
          setCpu1Count(prev => prev + 1);
        }
      }, 600);
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [isRunning]);

  const reset = () => {
    setIsRunning(false);
    setIteration(0);
    setCpu0Count(0);
    setCpu1Count(0);
    setBounces(0);
    setActiveCpu(0);
    setFlashLine(null);
  };

  const cacheLines: CacheLine[] = isProblem
    ? [
        {
          bytes: [
            { label: "counter_a (8B)", owner: 0, color: "bg-blue-400" },
            { label: "counter_b (8B)", owner: 1, color: "bg-green-400" },
            ...Array.from({ length: 48 }, (_, i) => ({
              label: `padding (${i < 48 ? 48 - i : 0}B)`,
              owner: null,
              color: "bg-slate-200 dark:bg-slate-700",
            })),
          ],
        },
      ]
    : [
        {
          bytes: [
            { label: "counter_a (8B)", owner: 0, color: "bg-blue-400" },
            ...Array.from({ length: 56 }, (_, i) => ({
              label: `padding (${56 - i}B)`,
              owner: null,
              color: "bg-slate-200 dark:bg-slate-700",
            })),
          ],
        },
        {
          bytes: [
            { label: "counter_b (8B)", owner: 1, color: "bg-green-400" },
            ...Array.from({ length: 56 }, (_, i) => ({
              label: `padding (${56 - i}B)`,
              owner: null,
              color: "bg-slate-200 dark:bg-slate-700",
            })),
          ],
        },
      ];

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-rose-50 dark:from-slate-900 dark:to-rose-950 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2 text-center">
        False Sharing 演示
      </h3>
      <p className="text-sm text-slate-500 dark:text-slate-400 text-center mb-6">
        同一缓存行中的不同变量会导致 CPU 间不必要的缓存行传输
      </p>

      <div className="flex justify-center mb-6 gap-4">
        <button
          onClick={() => { setMode("problem"); reset(); }}
          className={`px-5 py-2.5 rounded-lg font-semibold transition-all flex items-center gap-2 ${
            isProblem
              ? "bg-red-600 text-white shadow-lg scale-105"
              : "bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700"
          }`}
        >
          <Zap className="w-4 h-4" /> 有问题版本
        </button>
        <button
          onClick={() => { setMode("fix"); reset(); }}
          className={`px-5 py-2.5 rounded-lg font-semibold transition-all flex items-center gap-2 ${
            !isProblem
              ? "bg-green-600 text-white shadow-lg scale-105"
              : "bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700"
          }`}
        >
          <Shield className="w-4 h-4" /> 修复版本 (Padding)
        </button>
      </div>

      <div className="bg-white dark:bg-slate-800 rounded-lg p-4 border border-slate-200 dark:border-slate-700 mb-6">
        <h4 className="font-semibold text-slate-700 dark:text-slate-200 mb-3">
          {isProblem ? "❌ 问题：两个变量在同一缓存行" : "✅ 修复：变量分配到不同缓存行"}
        </h4>

        <div className="space-y-3">
          {cacheLines.map((line, lineIdx) => (
            <div key={lineIdx}>
              <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">
                缓存行 {lineIdx} (64 字节) {lineIdx === 0 && flashLine === lineIdx && (
                  <motion.span
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="ml-2 text-orange-500 font-bold"
                  >
                    ← 缓存行弹跳!
                  </motion.span>
                )}
              </div>
              <div className="flex gap-0.5">
                {line.bytes.slice(0, 8).map((byte, byteIdx) => (
                  <motion.div
                    key={byteIdx}
                    animate={flashLine === lineIdx && byteIdx < 2 ? {
                      scale: [1, 1.2, 1],
                      backgroundColor: ["#fb923c", byte.color.includes("blue") ? "#60a5fa" : "#4ade80"],
                    } : {}}
                    transition={{ duration: 0.3 }}
                    className={`h-10 flex-1 rounded-sm ${byte.color} flex items-center justify-center relative group`}
                  >
                    <span className="text-[8px] text-white font-mono truncate px-0.5">
                      {byteIdx === 0 ? (isProblem ? "a" : "a") : byteIdx === 1 ? (isProblem ? "b" : "pad") : ""}
                    </span>
                    <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 hidden group-hover:block z-10">
                      <div className="bg-slate-800 text-white text-xs rounded px-2 py-1 whitespace-nowrap">
                        {byte.label}
                      </div>
                    </div>
                  </motion.div>
                ))}
                <div className="h-10 flex-1 rounded-sm bg-slate-100 dark:bg-slate-700 flex items-center justify-center">
                  <span className="text-[8px] text-slate-400">...</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4 mb-6">
        <motion.div
          animate={activeCpu === 0 && isRunning ? { borderColor: ["#60a5fa", "#f97316", "#60a5fa"] } : {}}
          transition={{ duration: 0.5, repeat: Infinity }}
          className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4 border-2 border-blue-300 dark:border-blue-700"
        >
          <div className="flex items-center gap-2 mb-2">
            <Cpu className="w-5 h-5 text-blue-600" />
            <span className="font-bold text-blue-700 dark:text-blue-300">CPU 0</span>
          </div>
          <div className="text-sm text-slate-600 dark:text-slate-300">
            <p>修改 <code className="bg-blue-100 dark:bg-blue-800 px-1 rounded">counter_a</code></p>
            <p className="font-mono mt-1">counter_a = {cpu0Count}</p>
          </div>
        </motion.div>

        <motion.div
          animate={activeCpu === 1 && isRunning ? { borderColor: ["#4ade80", "#f97316", "#4ade80"] } : {}}
          transition={{ duration: 0.5, repeat: Infinity }}
          className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4 border-2 border-green-300 dark:border-green-700"
        >
          <div className="flex items-center gap-2 mb-2">
            <Cpu className="w-5 h-5 text-green-600" />
            <span className="font-bold text-green-700 dark:text-green-300">CPU 1</span>
          </div>
          <div className="text-sm text-slate-600 dark:text-slate-300">
            <p>修改 <code className="bg-green-100 dark:bg-green-800 px-1 rounded">counter_b</code></p>
            <p className="font-mono mt-1">counter_b = {cpu1Count}</p>
          </div>
        </motion.div>
      </div>

      <div className="flex justify-center gap-3 mb-6">
        <button
          onClick={() => setIsRunning(!isRunning)}
          className={`flex items-center gap-2 px-5 py-2.5 rounded-lg font-semibold text-white transition-all ${
            isRunning ? "bg-orange-600 hover:bg-orange-700" : "bg-blue-600 hover:bg-blue-700"
          }`}
        >
          {isRunning ? <><Pause className="w-4 h-4" /> 暂停</> : <><Play className="w-4 h-4" /> 开始</>}
        </button>
        <button onClick={reset}
          className="flex items-center gap-2 px-5 py-2.5 bg-slate-600 hover:bg-slate-700 text-white rounded-lg font-semibold transition-all">
          <RotateCcw className="w-4 h-4" /> 重置
        </button>
      </div>

      <div className="grid grid-cols-3 gap-4 mb-6">
        <div className="bg-white dark:bg-slate-800 rounded-lg p-3 border border-slate-200 dark:border-slate-700 text-center">
          <div className="text-xs text-slate-500 mb-1">迭代次数</div>
          <div className="text-xl font-bold font-mono text-slate-800 dark:text-slate-100">{iteration}</div>
        </div>
        <div className="bg-white dark:bg-slate-800 rounded-lg p-3 border border-slate-200 dark:border-slate-700 text-center">
          <div className="text-xs text-slate-500 mb-1">缓存行弹跳</div>
          <motion.div
            key={bounces}
            initial={{ scale: 1.3 }}
            animate={{ scale: 1 }}
            className="text-xl font-bold font-mono text-orange-500"
          >
            {bounces}
          </motion.div>
        </div>
        <div className="bg-white dark:bg-slate-800 rounded-lg p-3 border border-slate-200 dark:border-slate-700 text-center">
          <div className="text-xs text-slate-500 mb-1">性能影响</div>
          <div className={`text-xl font-bold ${isProblem ? "text-red-500" : "text-green-500"}`}>
            {isProblem ? "严重" : "无"}
          </div>
        </div>
      </div>

      <div className="bg-slate-100 dark:bg-slate-800 rounded-lg p-4 font-mono text-xs">
        <div className="text-slate-500 dark:text-slate-400 mb-2">{"// " + (isProblem ? "有问题的代码" : "修复后的代码")}</div>
        {isProblem ? (
          <pre className="text-slate-700 dark:text-slate-300 whitespace-pre-wrap">{`struct {
    long counter_a;  // CPU 0 修改
    long counter_b;  // CPU 1 修改 ← 同一缓存行！
} data;`}</pre>
        ) : (
          <pre className="text-slate-700 dark:text-slate-300 whitespace-pre-wrap">{`struct {
    long counter_a;
    char padding[56];  // 填充到 64 字节
    long counter_b;    // 不同缓存行 ✓
    char padding2[56];
} data;`}</pre>
        )}
      </div>
    </div>
  );
}
