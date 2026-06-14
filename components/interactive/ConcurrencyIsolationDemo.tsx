"use client";

import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Cpu, Play, Pause, RotateCcw, Shield } from "lucide-react";

export default function ConcurrencyIsolationDemo() {
  const [isRunning, setIsRunning] = useState(false);
  const [timeSlice, setTimeSlice] = useState(0);
  const [activeProcess, setActiveProcess] = useState(0);
  const [mode, setMode] = useState<"concurrency" | "isolation">("concurrency");

  const processes = [
    { id: 0, name: "Process A", color: "bg-blue-500", task: "音乐播放", address: "0x1000" },
    { id: 1, name: "Process B", color: "bg-green-500", task: "文件下载", address: "0x1000" },
    { id: 2, name: "Process C", color: "bg-purple-500", task: "文档编辑", address: "0x1000" }
  ];

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isRunning) {
      interval = setInterval(() => {
        setTimeSlice(prev => (prev + 1) % 30);
        setActiveProcess(prev => (prev + 1) % processes.length);
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [isRunning]);

  const handleReset = () => {
    setIsRunning(false);
    setTimeSlice(0);
    setActiveProcess(0);
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-purple-50 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold text-slate-800 mb-6 text-center">
        并发与隔离演示
      </h3>

      {/* Mode Toggle */}
      <div className="flex justify-center mb-6 gap-4">
        <button
          onClick={() => setMode("concurrency")}
          className={`px-6 py-3 rounded-lg font-semibold transition-all flex items-center gap-2 ${
            mode === "concurrency"
              ? "bg-blue-600 text-white shadow-lg scale-105"
              : "bg-white text-slate-600 hover:bg-slate-100"
          }`}
        >
          <Cpu className="w-5 h-5" />
          并发（Concurrency）
        </button>
        <button
          onClick={() => setMode("isolation")}
          className={`px-6 py-3 rounded-lg font-semibold transition-all flex items-center gap-2 ${
            mode === "isolation"
              ? "bg-green-600 text-white shadow-lg scale-105"
              : "bg-white text-slate-600 hover:bg-slate-100"
          }`}
        >
          <Shield className="w-5 h-5" />
          隔离（Isolation）
        </button>
      </div>

      {/* Concurrency Mode */}
      {mode === "concurrency" && (
        <div className="space-y-6">
          {/* Control Panel */}
          <div className="flex justify-center gap-4">
            <button
              onClick={() => setIsRunning(!isRunning)}
              className={`px-6 py-3 rounded-lg font-semibold text-white transition-all ${
                isRunning ? "bg-orange-600 hover:bg-orange-700" : "bg-blue-600 hover:bg-blue-700"
              }`}
            >
              {isRunning ? (
                <>
                  <Pause className="inline w-5 h-5 mr-2" />
                  暂停
                </>
              ) : (
                <>
                  <Play className="inline w-5 h-5 mr-2" />
                  开始
                </>
              )}
            </button>
            <button
              onClick={handleReset}
              className="px-6 py-3 rounded-lg font-semibold bg-slate-600 text-white hover:bg-slate-700 transition-all"
            >
              <RotateCcw className="inline w-5 h-5 mr-2" />
              重置
            </button>
          </div>

          {/* Time Slice Visualization */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex items-center justify-between mb-4">
              <h4 className="font-bold text-slate-800">时间片轮转（Time Slicing）</h4>
              <span className="text-sm text-slate-600">时间: {timeSlice * 10}ms</span>
            </div>
            <div className="grid grid-cols-3 gap-4 mb-4">
              {processes.map((proc, idx) => (
                <motion.div
                  key={proc.id}
                  animate={{
                    scale: activeProcess === idx ? 1.1 : 1,
                    borderWidth: activeProcess === idx ? 3 : 2
                  }}
                  className={`p-4 rounded-lg border-2 transition-all ${
                    activeProcess === idx
                      ? `${proc.color} text-white border-white shadow-lg`
                      : "bg-slate-100 border-slate-300"
                  }`}
                >
                  <div className="text-center">
                    <div className="font-bold mb-1">{proc.name}</div>
                    <div className="text-xs opacity-80">{proc.task}</div>
                    {activeProcess === idx && (
                      <div className="mt-2 text-xs font-semibold animate-pulse">
                        正在 CPU 上运行
                      </div>
                    )}
                  </div>
                </motion.div>
              ))}
            </div>
            <div className="bg-slate-50 p-3 rounded-lg">
              <p className="text-sm text-slate-700">
                <strong>并发原理：</strong>CPU 快速在多个进程之间切换（每 10ms 一次），宏观上看像是"同时"运行。
                用户感知不到切换延迟，体验流畅。
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Isolation Mode */}
      {mode === "isolation" && (
        <div className="space-y-6">
          <div className="bg-white rounded-lg shadow-md p-6">
            <h4 className="font-bold text-slate-800 mb-4">虚拟地址空间隔离</h4>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {processes.map(proc => (
                <div key={proc.id} className="border-2 border-slate-300 rounded-lg p-4">
                  <div className={`${proc.color} text-white p-2 rounded-lg mb-3 text-center font-bold`}>
                    {proc.name}
                  </div>
                  <div className="space-y-2">
                    <div className="bg-slate-100 p-2 rounded">
                      <div className="text-xs text-slate-600">虚拟地址</div>
                      <div className="font-mono text-sm">{proc.address}</div>
                    </div>
                    <div className="bg-slate-100 p-2 rounded">
                      <div className="text-xs text-slate-600">物理地址</div>
                      <div className="font-mono text-sm">
                        0x{(Math.random() * 0xffffff | 0).toString(16).toUpperCase()}
                      </div>
                    </div>
                    <div className="text-xs text-slate-600 mt-2">
                      通过 MMU 页表映射到不同的物理内存
                    </div>
                  </div>
                </div>
              ))}
            </div>
            <div className="mt-4 bg-green-50 border-l-4 border-green-400 p-4 rounded">
              <p className="text-sm text-slate-700">
                <strong>隔离原理：</strong>每个进程拥有独立的虚拟地址空间。相同的虚拟地址 0x1000 在不同进程中
                映射到不同的物理内存，进程之间互不干扰、无法访问彼此的数据。
              </p>
            </div>
          </div>

          {/* Security Demo */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h4 className="font-bold text-slate-800 mb-4">隔离的安全性</h4>
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-red-50 border-2 border-red-300 rounded-lg p-4">
                <h5 className="font-semibold text-red-700 mb-2">没有隔离（危险）</h5>
                <ul className="text-sm text-slate-700 space-y-1">
                  <li>✗ 进程可读取其他进程内存</li>
                  <li>✗ 一个进程崩溃导致系统崩溃</li>
                  <li>✗ 恶意程序窃取密码</li>
                </ul>
              </div>
              <div className="bg-green-50 border-2 border-green-300 rounded-lg p-4">
                <h5 className="font-semibold text-green-700 mb-2">有隔离（安全）</h5>
                <ul className="text-sm text-slate-700 space-y-1">
                  <li>✓ 进程无法访问其他进程内存</li>
                  <li>✓ 进程崩溃不影响其他进程</li>
                  <li>✓ 数据受保护</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
