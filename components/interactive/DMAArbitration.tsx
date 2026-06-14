"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Scale, ArrowRight } from "lucide-react";

const arbSteps = [
  { label: "DMA发送总线请求", desc: "DMA控制器需要传送数据，置BR=1", signals: { BR: 1, BG: 0, BS: 0 } },
  { label: "仲裁器响应", desc: "总线仲裁器检测到BR，检查总线是否空闲", signals: { BR: 1, BG: 0, BS: 0 } },
  { label: "CPU完成当前周期", desc: "CPU在当前总线周期结束时释放总线", signals: { BR: 1, BG: 0, BS: 1 } },
  { label: "仲裁器授权", desc: "仲裁器向DMA发送总线授权信号BG=1", signals: { BR: 1, BG: 1, BS: 0 } },
  { label: "DMA接管总线", desc: "DMA控制器接管总线，开始数据传送", signals: { BR: 1, BG: 1, BS: 1 } },
  { label: "数据传送中", desc: "DMA使用总线进行数据传送", signals: { BR: 1, BG: 1, BS: 1 } },
  { label: "传送完成", desc: "DMA释放总线，BG和BR复位", signals: { BR: 0, BG: 0, BS: 0 } },
  { label: "CPU恢复总线", desc: "CPU重新获得总线使用权，继续执行", signals: { BR: 0, BG: 0, BS: 0 } },
];

export function DMAArbitration() {
  const [step, setStep] = useState(-1);
  const [autoPlay, setAutoPlay] = useState(false);
  const current = step >= 0 ? arbSteps[step] : null;

  useEffect(() => {
    if (!autoPlay) return;
    const timer = setInterval(() => {
      setStep((s) => {
        if (s >= arbSteps.length - 1) { setAutoPlay(false); return s; }
        return s + 1;
      });
    }, 1200);
    return () => clearInterval(timer);
  }, [autoPlay]);

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <div className="flex items-center gap-2 mb-4">
        <Scale className="w-5 h-5 text-rose-400" />
        <h3 className="text-lg font-semibold">DMA 总线仲裁</h3>
      </div>

      <div className="flex gap-2 mb-4">
        <button onClick={() => setAutoPlay(!autoPlay)}
          className="px-4 py-1.5 bg-rose-600 rounded text-sm text-white hover:bg-rose-500">
          {autoPlay ? "暂停" : "自动演示"}
        </button>
        <button onClick={() => setStep((s) => Math.min(s + 1, arbSteps.length - 1))}
          className="px-3 py-1.5 bg-gray-700 rounded text-sm text-gray-300 hover:bg-gray-600">
          单步
        </button>
        <button onClick={() => { setStep(-1); setAutoPlay(false); }}
          className="px-3 py-1.5 bg-gray-700 rounded text-sm text-gray-300 hover:bg-gray-600">
          重置
        </button>
      </div>

      <div className="flex items-center justify-center gap-6 mb-6">
        <div className={`p-4 rounded-lg border-2 text-center transition-all ${
          current && current.signals.BS ? "border-gray-600 bg-gray-800/20 opacity-50" : "border-blue-500 bg-blue-500/10"
        }`}>
          <div className="text-sm font-medium text-blue-300">CPU</div>
          <div className="text-xs text-gray-400 mt-1">{current && current.signals.BS ? "等待" : "使用总线"}</div>
        </div>

        <div className="flex flex-col gap-1">
          <div className="flex items-center gap-2 text-xs">
            <span className="w-8 text-right text-gray-400">BR</span>
            <div className={`w-20 h-4 rounded ${current?.signals.BR ? "bg-rose-500" : "bg-gray-700"}`} />
          </div>
          <div className="flex items-center gap-2 text-xs">
            <span className="w-8 text-right text-gray-400">BG</span>
            <div className={`w-20 h-4 rounded ${current?.signals.BG ? "bg-green-500" : "bg-gray-700"}`} />
          </div>
          <div className="flex items-center gap-2 text-xs">
            <span className="w-8 text-right text-gray-400">BS</span>
            <div className={`w-20 h-4 rounded ${current?.signals.BS ? "bg-yellow-500" : "bg-gray-700"}`} />
          </div>
        </div>

        <div className={`p-4 rounded-lg border-2 text-center transition-all ${
          current?.signals.BG ? "border-teal-500 bg-teal-500/10" : "border-gray-600 bg-gray-800/20 opacity-50"
        }`}>
          <div className="text-sm font-medium text-teal-300">DMA 控制器</div>
          <div className="text-xs text-gray-400 mt-1">{current?.signals.BG ? "使用总线" : "请求等待"}</div>
        </div>
      </div>

      <AnimatePresence mode="wait">
        {current && (
          <motion.div key={step}
            initial={{ opacity: 0, y: 5 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -5 }}
            className="p-3 bg-gray-800/30 rounded-lg"
          >
            <div className="flex items-center gap-2 mb-1">
              <span className="text-sm font-medium text-rose-300">步骤 {step + 1}: {current.label}</span>
            </div>
            <p className="text-xs text-gray-400">{current.desc}</p>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="mt-4 flex gap-1">
        {arbSteps.map((_, i) => (
          <div key={i} className={`flex-1 h-1 rounded ${i <= step ? "bg-rose-400" : "bg-gray-700"}`} />
        ))}
      </div>
    </div>
  );
}
