"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Zap, Play, RotateCcw, CheckCircle } from "lucide-react";

const steps = [
  { label: "设备准备数据", src: "device", desc: "外设独立完成数据准备工作，CPU执行其他任务" },
  { label: "设备发送中断请求", src: "device", desc: "设备就绪后，通过INTR线向CPU发送中断请求信号" },
  { label: "CPU完成当前指令", src: "cpu", desc: "CPU在当前指令执行完毕后，检测中断请求" },
  { label: "CPU响应中断", src: "cpu", desc: "CPU发出中断响应信号INTA，关中断（IF=0）" },
  { label: "保存现场", src: "cpu", desc: "将PSW、PC、寄存器等压入堆栈保存" },
  { label: "获取中断向量", src: "cpu", desc: "通过中断向量表获取中断服务程序入口地址" },
  { label: "执行中断服务程序", src: "isr", desc: "CPU执行ISR，完成数据传输（读/写一个字）" },
  { label: "恢复现场", src: "cpu", desc: "从堆栈恢复寄存器、PC、PSW" },
  { label: "开中断返回", src: "cpu", desc: "开中断（IF=1），返回断点继续执行主程序" },
];

export function InterruptDrivenIO() {
  const [step, setStep] = useState(-1);
  const [autoPlay, setAutoPlay] = useState(false);

  const handleAutoPlay = () => {
    if (autoPlay) { setAutoPlay(false); return; }
    setAutoPlay(true);
    setStep(-1);
    let s = -1;
    const timer = setInterval(() => {
      s++;
      if (s >= steps.length) { setAutoPlay(false); clearInterval(timer); return; }
      setStep(s);
    }, 1200);
  };

  const srcColors: Record<string, string> = {
    device: "border-green-500 bg-green-500/10",
    cpu: "border-blue-500 bg-blue-500/10",
    isr: "border-purple-500 bg-purple-500/10",
  };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <div className="flex items-center gap-2 mb-4">
        <Zap className="w-5 h-5 text-yellow-400" />
        <h3 className="text-lg font-semibold">中断驱动 I/O</h3>
      </div>

      <div className="flex gap-2 mb-4">
        <button onClick={handleAutoPlay} className="px-4 py-1.5 bg-yellow-600 rounded text-sm text-white hover:bg-yellow-500 flex items-center gap-1">
          <Play className="w-3 h-3" /> {autoPlay ? "暂停" : "自动演示"}
        </button>
        <button onClick={() => { setStep(-1); setAutoPlay(false); }}
          className="px-3 py-1.5 bg-gray-700 rounded text-sm text-gray-300 hover:bg-gray-600">
          <RotateCcw className="w-3 h-3 inline mr-1" />重置
        </button>
        <button onClick={() => setStep((s) => Math.min(s + 1, steps.length - 1))}
          className="px-3 py-1.5 bg-gray-700 rounded text-sm text-gray-300 hover:bg-gray-600">
          单步
        </button>
      </div>

      <div className="relative">
        <div className="absolute left-4 top-0 bottom-0 w-0.5 bg-gray-700" />
        <div className="space-y-2">
          {steps.map((s, i) => (
            <motion.div
              key={i}
              className={`relative pl-10 p-3 rounded-lg border-l-4 transition-all ${
                step === i ? srcColors[s.src] : step > i ? "border-gray-600 bg-gray-800/20 opacity-50" : "border-gray-700 bg-gray-800/10"
              }`}
              animate={{ x: step === i ? 4 : 0 }}
            >
              <div className={`absolute left-2.5 w-3 h-3 rounded-full border-2 ${
                step === i ? "bg-yellow-400 border-yellow-400" : step > i ? "bg-gray-600 border-gray-600" : "bg-gray-800 border-gray-600"
              }`} style={{ top: 16 }} />
              <div className="flex items-center justify-between">
                <span className={`text-sm ${step === i ? "text-white font-medium" : "text-gray-400"}`}>
                  {s.label}
                </span>
                <span className={`text-[10px] px-1.5 py-0.5 rounded ${
                  s.src === "device" ? "bg-green-600/30 text-green-300" : s.src === "isr" ? "bg-purple-600/30 text-purple-300" : "bg-blue-600/30 text-blue-300"
                }`}>
                  {s.src === "device" ? "设备" : s.src === "isr" ? "ISR" : "CPU"}
                </span>
              </div>
              {step === i && (
                <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="text-xs text-gray-400 mt-1">
                  {s.desc}
                </motion.p>
              )}
            </motion.div>
          ))}
        </div>
      </div>

      {step === steps.length - 1 && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}
          className="mt-4 flex items-center gap-2 text-sm text-green-400 p-3 bg-green-500/10 rounded-lg">
          <CheckCircle className="w-4 h-4" /> 中断处理完成，CPU返回主程序继续执行
        </motion.div>
      )}
    </div>
  );
}
