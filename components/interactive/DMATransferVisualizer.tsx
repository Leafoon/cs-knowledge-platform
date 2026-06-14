"use client";

import React, { useState, useCallback } from "react";
import { motion } from "framer-motion";
import { Play, RotateCcw, Cpu, HardDrive, MemoryStick } from "lucide-react";

export default function DMATransferVisualizer() {
  const [step, setStep] = useState(-1);
  const [isRunning, setIsRunning] = useState(false);

  const steps = [
    { label: "CPU 设置 DMA 参数", sub: "源地址=设备, 目标地址=内存0x8000, 长度=4096字节", who: "cpu" as const },
    { label: "CPU 启动 DMA 传输", sub: "写入 DMA 控制寄存器，启动传输", who: "cpu" as const },
    { label: "DMA 控制器读取设备数据", sub: "DMA 控制器从设备逐块读取数据", who: "dma" as const },
    { label: "DMA 写入内存", sub: "DMA 控制器将数据直接写入内存地址 0x8000", who: "dma" as const },
    { label: "传输完成，DMA 发送中断", sub: "DMA 控制器向 CPU 发送完成中断", who: "dma" as const },
    { label: "CPU 处理中断", sub: "CPU 响应中断，唤醒等待 I/O 的进程", who: "cpu" as const },
  ];

  const reset = useCallback(() => { setStep(-1); setIsRunning(false); }, []);

  const autoPlay = useCallback(() => {
    reset();
    setIsRunning(true);
    let i = 0;
    const interval = setInterval(() => {
      setStep(i);
      i++;
      if (i >= steps.length) { clearInterval(interval); setIsRunning(false); }
    }, 1500);
  }, [reset, steps.length]);

  const stepForward = () => { if (step < steps.length - 1) setStep((p) => p + 1); };

  const getWhoColor = (who: string, active: boolean) => {
    if (!active) return "bg-slate-100 dark:bg-gray-750 border-slate-200 dark:border-gray-700";
    switch (who) {
      case "cpu": return "bg-blue-50 dark:bg-blue-950/30 border-blue-400 dark:border-blue-600";
      case "dma": return "bg-emerald-50 dark:bg-emerald-950/30 border-emerald-400 dark:border-emerald-600";
      default: return "bg-slate-100 dark:bg-gray-750 border-slate-200 dark:border-gray-700";
    }
  };

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-emerald-50 rounded-xl shadow-lg dark:from-gray-900 dark:to-gray-800">
      <h2 className="text-2xl font-bold text-slate-800 dark:text-gray-100 mb-2 text-center">
        DMA 传输过程可视化
      </h2>
      <p className="text-sm text-slate-500 dark:text-gray-400 text-center mb-6">
        观察 DMA 如何让 CPU 从数据传输中解放出来
      </p>

      <div className="flex gap-3 mb-6 justify-center">
        <button onClick={autoPlay} disabled={isRunning} className="flex items-center gap-2 px-4 py-2 bg-emerald-500 text-white rounded-lg hover:bg-emerald-600 disabled:opacity-50 text-sm">
          <Play className="w-4 h-4" /> 自动播放
        </button>
        <button onClick={stepForward} disabled={isRunning || step >= steps.length - 1} className="px-4 py-2 bg-teal-500 text-white rounded-lg hover:bg-teal-600 disabled:opacity-50 text-sm">
          单步
        </button>
        <button onClick={reset} className="flex items-center gap-2 px-4 py-2 bg-slate-500 text-white rounded-lg hover:bg-slate-600 text-sm">
          <RotateCcw className="w-4 h-4" /> 重置
        </button>
      </div>

      {/* Hardware visualization */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        <div className={`rounded-lg p-4 border-2 text-center transition-all ${
          step >= 0 && step <= 1 ? "bg-blue-50 dark:bg-blue-950/30 border-blue-400" : step >= 5 ? "bg-blue-50 dark:bg-blue-950/30 border-blue-400" : "bg-white dark:bg-gray-800 border-slate-200 dark:border-gray-700"
        }`}>
          <Cpu className="w-8 h-8 text-blue-500 mx-auto mb-2" />
          <div className="text-sm font-bold text-slate-700 dark:text-gray-200">CPU</div>
          <div className="text-xs text-slate-500 dark:text-gray-400 mt-1">
            {step >= 0 && step <= 1 ? "正在设置 DMA" : step >= 2 && step <= 4 ? "执行其他任务" : step >= 5 ? "处理中断" : "空闲"}
          </div>
        </div>

        <div className={`rounded-lg p-4 border-2 text-center transition-all ${
          step >= 2 && step <= 4 ? "bg-emerald-50 dark:bg-emerald-950/30 border-emerald-400" : "bg-white dark:bg-gray-800 border-slate-200 dark:border-gray-700"
        }`}>
          <HardDrive className="w-8 h-8 text-emerald-500 mx-auto mb-2" />
          <div className="text-sm font-bold text-slate-700 dark:text-gray-200">DMA 控制器</div>
          <div className="text-xs text-slate-500 dark:text-gray-400 mt-1">
            {step >= 2 && step <= 3 ? "正在传输数据" : step === 4 ? "发送中断" : "空闲"}
          </div>
        </div>

        <div className={`rounded-lg p-4 border-2 text-center transition-all ${
          step >= 3 ? "bg-purple-50 dark:bg-purple-950/30 border-purple-400" : "bg-white dark:bg-gray-800 border-slate-200 dark:border-gray-700"
        }`}>
          <MemoryStick className="w-8 h-8 text-purple-500 mx-auto mb-2" />
          <div className="text-sm font-bold text-slate-700 dark:text-gray-200">内存</div>
          <div className="text-xs text-slate-500 dark:text-gray-400 mt-1">
            {step >= 3 ? "接收数据" : "等待"}
          </div>
        </div>
      </div>

      {/* Steps */}
      <div className="space-y-2">
        {steps.map((s, i) => {
          const isVisible = i <= step;
          const isCurrent = i === step;
          return isVisible ? (
            <motion.div
              key={i}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              className={`p-3 rounded-lg border-2 ${getWhoColor(s.who, isCurrent)}`}
            >
              <div className="flex items-center gap-3">
                <span className="w-6 h-6 rounded-full bg-white dark:bg-gray-800 flex items-center justify-center text-xs font-bold border">{i + 1}</span>
                <div>
                  <div className="text-sm font-bold text-slate-800 dark:text-gray-100">{s.label}</div>
                  <div className="text-xs text-slate-500 dark:text-gray-400">{s.sub}</div>
                </div>
                <span className={`ml-auto px-2 py-0.5 rounded text-xs font-bold ${
                  s.who === "cpu" ? "bg-blue-100 text-blue-600 dark:bg-blue-900/30 dark:text-blue-400" : "bg-emerald-100 text-emerald-600 dark:bg-emerald-900/30 dark:text-emerald-400"
                }`}>
                  {s.who === "cpu" ? "CPU" : "DMA"}
                </span>
              </div>
            </motion.div>
          ) : null;
        })}
      </div>
    </div>
  );
}
