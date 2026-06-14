"use client";

import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Cpu, HardDrive } from "lucide-react";

interface Process {
  id: number;
  name: string;
  color: string;
  progress: number;
}

export default function ResourceMultiplexingDemo() {
  const [mode, setMode] = useState<"time" | "space">("time");
  const [running, setRunning] = useState(false);

  const [processes, setProcesses] = useState<Process[]>([
    { id: 1, name: "浏览器", color: "bg-blue-500", progress: 0 },
    { id: 2, name: "音乐播放器", color: "bg-green-500", progress: 0 },
    { id: 3, name: "文本编辑器", color: "bg-purple-500", progress: 0 }
  ]);

  const [activeProcess, setActiveProcess] = useState(0);

  useEffect(() => {
    if (!running) return;

    const interval = setInterval(() => {
      if (mode === "time") {
        setActiveProcess(prev => {
          const next = (prev + 1) % processes.length;
          setProcesses(procs =>
            procs.map((p, i) =>
              i === prev && p.progress < 100
                ? { ...p, progress: Math.min(100, p.progress + 10) }
                : p
            )
          );
          return next;
        });
      } else {
        setProcesses(prev =>
          prev.map(p =>
            p.progress < 100 ? { ...p, progress: Math.min(100, p.progress + 5) } : p
          )
        );
      }
    }, 500);

    return () => clearInterval(interval);
  }, [running, mode, processes.length]);

  useEffect(() => {
    if (processes.every(p => p.progress >= 100)) {
      setRunning(false);
    }
  }, [processes]);

  const reset = () => {
    setProcesses(prev => prev.map(p => ({ ...p, progress: 0 })));
    setActiveProcess(0);
    setRunning(false);
  };

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-cyan-50 to-cyan-100 dark:from-cyan-950 dark:to-slate-900 rounded-xl shadow-2xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-6">
        资源复用演示
      </h3>

      <div className="flex gap-3 mb-6">
        <button
          onClick={() => { setMode("time"); reset(); }}
          className={`flex-1 p-4 rounded-lg transition-all ${
            mode === "time"
              ? "bg-cyan-600 text-white shadow-lg"
              : "bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300"
          }`}
        >
          <div className="flex items-center justify-center gap-2 mb-2">
            <Cpu className="w-6 h-6" />
            <span className="font-semibold">时分复用 (Time Multiplexing)</span>
          </div>
          <p className="text-sm opacity-90">
            单个 CPU 在多个进程间快速切换
          </p>
        </button>

        <button
          onClick={() => { setMode("space"); reset(); }}
          className={`flex-1 p-4 rounded-lg transition-all ${
            mode === "space"
              ? "bg-cyan-600 text-white shadow-lg"
              : "bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300"
          }`}
        >
          <div className="flex items-center justify-center gap-2 mb-2">
            <HardDrive className="w-6 h-6" />
            <span className="font-semibold">空分复用 (Space Multiplexing)</span>
          </div>
          <p className="text-sm opacity-90">
            多个进程同时使用内存不同区域
          </p>
        </button>
      </div>

      {mode === "time" && (
        <div className="mb-6 p-4 bg-white dark:bg-slate-800 rounded-lg shadow">
          <h4 className="font-semibold text-slate-700 dark:text-slate-300 mb-3 flex items-center gap-2">
            <Cpu className="w-5 h-5" />
            CPU 状态
          </h4>
          <div className="h-20 bg-slate-100 dark:bg-slate-900 rounded-lg flex items-center justify-center">
            <AnimatePresence mode="wait">
              {running && (
                <motion.div
                  key={activeProcess}
                  initial={{ scale: 0, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  exit={{ scale: 0, opacity: 0 }}
                  className={`px-6 py-3 ${processes[activeProcess].color} text-white rounded-lg font-semibold`}
                >
                  正在执行: {processes[activeProcess].name}
                </motion.div>
              )}
            </AnimatePresence>
            {!running && (
              <span className="text-slate-500">CPU 空闲</span>
            )}
          </div>
        </div>
      )}

      {mode === "space" && (
        <div className="mb-6 p-4 bg-white dark:bg-slate-800 rounded-lg shadow">
          <h4 className="font-semibold text-slate-700 dark:text-slate-300 mb-3 flex items-center gap-2">
            <HardDrive className="w-5 h-5" />
            内存分配
          </h4>
          <div className="flex gap-2">
            {processes.map((proc) => (
              <div
                key={proc.id}
                className={`flex-1 h-20 ${proc.color} rounded-lg flex items-center justify-center text-white font-semibold`}
              >
                {proc.name}
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="space-y-3 mb-6">
        {processes.map((proc) => (
          <div key={proc.id}>
            <div className="flex items-center justify-between mb-1">
              <span className="text-sm font-medium text-slate-700 dark:text-slate-300">
                {proc.name}
              </span>
              <span className="text-sm text-slate-600 dark:text-slate-400">
                {proc.progress}%
              </span>
            </div>
            <div className="h-4 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
              <motion.div
                className={`h-full ${proc.color}`}
                initial={{ width: 0 }}
                animate={{ width: `${proc.progress}%` }}
                transition={{ duration: 0.3 }}
              />
            </div>
          </div>
        ))}
      </div>

      <div className="flex gap-3 justify-center">
        <button
          onClick={() => setRunning(!running)}
          className="px-6 py-3 bg-cyan-600 hover:bg-cyan-700 text-white rounded-lg font-semibold"
        >
          {running ? "暂停" : "开始"}
        </button>
        <button
          onClick={reset}
          className="px-6 py-3 bg-slate-600 hover:bg-slate-700 text-white rounded-lg font-semibold"
        >
          重置
        </button>
      </div>

      <div className="mt-6 p-4 bg-cyan-50 dark:bg-cyan-900/20 rounded-lg border border-cyan-200 dark:border-cyan-800">
        <p className="text-sm text-cyan-900 dark:text-cyan-100">
          <strong>资源复用：</strong>
          {mode === "time"
            ? " CPU 通过快速切换在多个进程间分配时间片，每个进程轮流执行一小段时间，给用户造成\"同时运行\"的假象。"
            : " 内存被划分为多个区域，每个进程独占一块空间，可以真正并行访问内存而不会冲突。"}
        </p>
      </div>
    </div>
  );
}
