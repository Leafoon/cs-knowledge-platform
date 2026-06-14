"use client";

import React, { useState, useCallback, useEffect } from "react";
import { motion } from "framer-motion";
import { Play, RotateCcw } from "lucide-react";

type BufferMode = "single" | "double" | "circular";

export default function IOBufferingDemo() {
  const [mode, setMode] = useState<BufferMode>("single");
  const [isRunning, setIsRunning] = useState(false);
  const [step, setStep] = useState(0);
  const [singleBuf, setSingleBuf] = useState<number[]>([]);
  const [doubleBufA, setDoubleBufA] = useState<number[]>([]);
  const [doubleBufB, setDoubleBufB] = useState<number[]>([]);
  const [circBuf, setcircBuf] = useState<number[]>([]);
  const [circHead, setCircHead] = useState(0);
  const [circTail, setCircTail] = useState(0);

  const reset = useCallback(() => {
    setIsRunning(false);
    setStep(0);
    setSingleBuf([]);
    setDoubleBufA([]);
    setDoubleBufB([]);
    setcircBuf(new Array(8).fill(0));
    setCircHead(0);
    setCircTail(0);
  }, []);

  useEffect(() => { reset(); }, [mode, reset]);

  const startDemo = useCallback(() => {
    reset();
    setIsRunning(true);
    let s = 0;
    const interval = setInterval(() => {
      s++;
      setStep(s);
      if (mode === "single") {
        if (s % 2 === 1) setSingleBuf([1, 2, 3, 4].slice(0, Math.min(4, Math.ceil(s / 2))));
        else setSingleBuf([]);
      } else if (mode === "double") {
        if (s % 4 < 2) {
          setDoubleBufA([1, 2, 3].slice(0, Math.min(3, s)));
          setDoubleBufB(doubleBufB.length > 0 ? doubleBufB : []);
        } else {
          setDoubleBufB([4, 5, 6].slice(0, Math.min(3, s - 2)));
        }
      } else {
        setcircBuf((prev) => {
          const next = [...prev];
          if (s % 2 === 1) {
            next[circTail % 8] = s;
            setCircTail((t) => t + 1);
          } else {
            next[circHead % 8] = 0;
            setCircHead((h) => h + 1);
          }
          return next;
        });
      }
      if (s >= 12) { clearInterval(interval); setIsRunning(false); }
    }, 800);
  }, [mode, reset]);

  const bufSize = 32;
  const renderBuffer = (data: number[], label: string, color: string) => (
    <div className="text-center">
      <div className="text-xs font-bold text-slate-500 dark:text-gray-400 mb-1">{label}</div>
      <div className="flex gap-1 justify-center">
        {Array.from({ length: 8 }).map((_, i) => (
          <motion.div
            key={i}
            className={`w-${bufSize / 8} h-10 rounded flex items-center justify-center text-xs font-mono border-2 transition-all ${
              data[i] ? `${color} border-current` : "bg-slate-100 dark:bg-gray-700 border-slate-200 dark:border-gray-600 text-slate-400"
            }`}
            animate={data[i] ? { scale: [1, 1.1, 1] } : {}}
          >
            {data[i] || ""}
          </motion.div>
        ))}
      </div>
    </div>
  );

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-violet-50 rounded-xl shadow-lg dark:from-gray-900 dark:to-gray-800">
      <h2 className="text-2xl font-bold text-slate-800 dark:text-gray-100 mb-2 text-center">
        I/O 缓冲策略演示
      </h2>
      <p className="text-sm text-slate-500 dark:text-gray-400 text-center mb-6">
        对比单缓冲、双缓冲和循环缓冲的工作方式
      </p>

      <div className="flex gap-3 mb-6 justify-center">
        {(["single", "double", "circular"] as BufferMode[]).map((m) => (
          <button
            key={m}
            onClick={() => setMode(m)}
            className={`px-4 py-2 rounded-lg text-sm font-bold transition-colors ${
              mode === m ? "bg-violet-500 text-white" : "bg-slate-200 text-slate-600 dark:bg-gray-700 dark:text-gray-300"
            }`}
          >
            {m === "single" ? "单缓冲" : m === "double" ? "双缓冲" : "循环缓冲"}
          </button>
        ))}
      </div>

      <div className="flex gap-3 mb-6 justify-center">
        <button onClick={startDemo} disabled={isRunning} className="flex items-center gap-2 px-4 py-2 bg-violet-500 text-white rounded-lg hover:bg-violet-600 disabled:opacity-50 text-sm">
          <Play className="w-4 h-4" /> 演示
        </button>
        <button onClick={reset} className="flex items-center gap-2 px-4 py-2 bg-slate-500 text-white rounded-lg hover:bg-slate-600 text-sm">
          <RotateCcw className="w-4 h-4" /> 重置
        </button>
      </div>

      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-slate-200 dark:border-gray-700">
        {mode === "single" && (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-sm font-bold text-blue-600 dark:text-blue-400">设备 → 缓冲区 → CPU</span>
              <span className="text-xs text-slate-500 dark:text-gray-400">步骤 {step}</span>
            </div>
            <div className="text-center text-sm text-slate-600 dark:text-gray-300 mb-2">
              设备填充缓冲区时，CPU 必须等待；CPU 处理缓冲区时，设备必须等待。
            </div>
            {renderBuffer(singleBuf, "缓冲区", "bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300")}
            <div className="flex justify-between text-xs text-slate-500 dark:text-gray-400">
              <span>{step % 2 === 1 ? "设备正在填充..." : "CPU 正在处理..."}</span>
              <span>{step % 2 === 1 ? "CPU 等待" : "设备等待"}</span>
            </div>
          </div>
        )}

        {mode === "double" && (
          <div className="space-y-4">
            <div className="text-center text-sm text-slate-600 dark:text-gray-300 mb-2">
              两个缓冲区交替使用：一个被 CPU 处理时，另一个被设备填充。
            </div>
            {renderBuffer(doubleBufA, "缓冲区 A", "bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300")}
            {renderBuffer(doubleBufB, "缓冲区 B", "bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-300")}
            <div className="text-xs text-center text-slate-500 dark:text-gray-400">
              步骤 {step}：{step % 4 < 2 ? "设备填充 A，CPU 处理 B" : "设备填充 B，CPU 处理 A"}
            </div>
          </div>
        )}

        {mode === "circular" && (
          <div className="space-y-4">
            <div className="text-center text-sm text-slate-600 dark:text-gray-300 mb-2">
              环形队列：生产者写入 tail，消费者读取 head。
            </div>
            <div className="flex justify-center gap-1">
              {circBuf.map((v, i) => (
                <motion.div
                  key={i}
                  className={`w-10 h-10 rounded-full flex items-center justify-center text-xs font-mono border-2 ${
                    v ? "bg-violet-100 dark:bg-violet-900/30 border-violet-400 text-violet-700 dark:text-violet-300"
                      : "bg-slate-100 dark:bg-gray-700 border-slate-200 dark:border-gray-600 text-slate-400"
                  } ${i === circHead % 8 ? "ring-2 ring-blue-400" : ""} ${i === circTail % 8 ? "ring-2 ring-emerald-400" : ""}`}
                >
                  {v || i}
                </motion.div>
              ))}
            </div>
            <div className="flex justify-center gap-4 text-xs text-slate-500 dark:text-gray-400">
              <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full ring-2 ring-blue-400" /> Head (读)</span>
              <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full ring-2 ring-emerald-400" /> Tail (写)</span>
            </div>
          </div>
        )}
      </div>

      <div className="mt-4 text-xs text-slate-500 dark:text-gray-400 text-center">
        {mode === "single" && "单缓冲：设备和 CPU 无法并行，性能最差"}
        {mode === "double" && "双缓冲：设备和 CPU 可以并行，性能较好"}
        {mode === "circular" && "循环缓冲：适用于持续数据流，如网络包处理"}
      </div>
    </div>
  );
}
