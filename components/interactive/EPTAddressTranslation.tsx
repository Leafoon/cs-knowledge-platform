"use client";

import React, { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Cpu, ArrowRight, RotateCcw, Play, ChevronRight, Layers } from "lucide-react";

interface TranslationStep {
  id: number;
  level: string;
  table: string;
  index: string;
  indexValue: string;
  entryContent: string;
  result: string;
  color: string;
  phase: "guest" | "ept";
}

const gvaInput = "0x0040_1234";

const steps: TranslationStep[] = [
  {
    id: 0,
    level: "Guest PML4",
    table: "Guest CR3 → PML4",
    index: "PML4[0]",
    indexValue: "GVA[47:39] = 0",
    entryContent: "→ Guest PDPT @ GPA 0x1000",
    result: "定位 Guest PDPT",
    color: "bg-blue-500",
    phase: "guest",
  },
  {
    id: 1,
    level: "EPT: GPA 0x1000",
    table: "EPT PML4 → PDPT → PD → PT",
    index: "EPT 翻译 GPA→HPA",
    indexValue: "GPA 0x1000 → 4 级 EPT 查找",
    entryContent: "→ HPA 0xA000",
    result: "Guest PDPT 真实地址 HPA 0xA000",
    color: "bg-purple-500",
    phase: "ept",
  },
  {
    id: 2,
    level: "Guest PDPT",
    table: "HPA 0xA000 → Guest PDPT",
    index: "PDPT[1]",
    indexValue: "GVA[38:30] = 1",
    entryContent: "→ Guest PD @ GPA 0x2000",
    result: "定位 Guest PD",
    color: "bg-blue-500",
    phase: "guest",
  },
  {
    id: 3,
    level: "EPT: GPA 0x2000",
    table: "EPT PML4 → PDPT → PD → PT",
    index: "EPT 翻译 GPA→HPA",
    indexValue: "GPA 0x2000 → 4 级 EPT 查找",
    entryContent: "→ HPA 0xB000",
    result: "Guest PD 真实地址 HPA 0xB000",
    color: "bg-purple-500",
    phase: "ept",
  },
  {
    id: 4,
    level: "Guest PD",
    table: "HPA 0xB000 → Guest PD",
    index: "PD[0]",
    indexValue: "GVA[29:21] = 0",
    entryContent: "→ Guest PT @ GPA 0x3000",
    result: "定位 Guest PT",
    color: "bg-blue-500",
    phase: "guest",
  },
  {
    id: 5,
    level: "EPT: GPA 0x3000",
    table: "EPT PML4 → PDPT → PD → PT",
    index: "EPT 翻译 GPA→HPA",
    indexValue: "GPA 0x3000 → 4 级 EPT 查找",
    entryContent: "→ HPA 0xC000",
    result: "Guest PT 真实地址 HPA 0xC000",
    color: "bg-purple-500",
    phase: "ept",
  },
  {
    id: 6,
    level: "Guest PT",
    table: "HPA 0xC000 → Guest PT",
    index: "PT[18]",
    indexValue: "GVA[20:12] = 18",
    entryContent: "→ GPA 0x5000_1000",
    result: "数据页 GPA 0x5000_1000",
    color: "bg-blue-500",
    phase: "guest",
  },
  {
    id: 7,
    level: "EPT: GPA 0x5000_1000",
    table: "EPT PML4 → PDPT → PD → PT",
    index: "EPT 翻译 GPA→HPA",
    indexValue: "GPA 0x5000_1000 → 4 级 EPT 查找",
    entryContent: "→ HPA 0xF000_1000",
    result: "最终物理地址 HPA 0xF000_1000",
    color: "bg-purple-500",
    phase: "ept",
  },
];

export default function EPTAddressTranslation() {
  const [currentStep, setCurrentStep] = useState(-1);
  const [isAnimating, setIsAnimating] = useState(false);

  const runAnimation = useCallback(async () => {
    if (isAnimating) return;
    setIsAnimating(true);
    setCurrentStep(-1);
    for (let i = 0; i < steps.length; i++) {
      await new Promise((r) => setTimeout(r, 800));
      setCurrentStep(i);
    }
    setIsAnimating(false);
  }, [isAnimating]);

  const reset = useCallback(() => {
    setCurrentStep(-1);
    setIsAnimating(false);
  }, []);

  return (
    <div className="w-full space-y-5">
      <div className="bg-gray-800/60 rounded-xl p-4 border border-gray-700">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <Layers className="w-5 h-5 text-purple-400" />
            <h3 className="text-sm font-bold text-white">EPT 二级地址翻译</h3>
          </div>
          <div className="flex gap-2">
            <button
              onClick={runAnimation}
              disabled={isAnimating}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-purple-600 hover:bg-purple-500 disabled:opacity-50 text-white text-xs font-medium transition-colors"
            >
              <Play className="w-3 h-3" /> 开始翻译
            </button>
            <button
              onClick={reset}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-gray-700 hover:bg-gray-600 text-gray-300 text-xs font-medium transition-colors"
            >
              <RotateCcw className="w-3 h-3" /> 重置
            </button>
          </div>
        </div>
        <div className="flex items-center gap-3 text-xs">
          <span className="text-gray-400">GVA:</span>
          <span className="font-mono text-yellow-400 bg-gray-900 px-2 py-1 rounded">{gvaInput}</span>
          <span className="text-gray-400">→</span>
          <span className="font-mono text-gray-500">
            {currentStep >= 7 ? "HPA 0xF000_1000" : "翻译中..."}
          </span>
        </div>
      </div>

      <div className="flex gap-2 text-xs">
        <span className="flex items-center gap-1.5 px-2 py-1 rounded bg-blue-500/20 text-blue-400 border border-blue-500/30">
          <span className="w-2 h-2 rounded-full bg-blue-500" /> Guest 页表遍历
        </span>
        <span className="flex items-center gap-1.5 px-2 py-1 rounded bg-purple-500/20 text-purple-400 border border-purple-500/30">
          <span className="w-2 h-2 rounded-full bg-purple-500" /> EPT 页表翻译 (GPA→HPA)
        </span>
      </div>

      <div className="space-y-2">
        {steps.map((step, i) => {
          const isActive = currentStep === i;
          const isDone = currentStep > i;
          const isPending = currentStep < i;

          return (
            <motion.div
              key={step.id}
              initial={false}
              animate={{
                opacity: isPending ? 0.35 : 1,
                scale: isActive ? 1.01 : 1,
              }}
              transition={{ duration: 0.25 }}
              className={`flex items-start gap-3 p-3 rounded-lg border transition-colors ${
                isActive
                  ? step.phase === "ept"
                    ? "bg-purple-500/10 border-purple-500/50 shadow-lg shadow-purple-500/10"
                    : "bg-blue-500/10 border-blue-500/50 shadow-lg shadow-blue-500/10"
                  : isDone
                  ? "bg-gray-800/40 border-gray-700/50"
                  : "bg-gray-800/20 border-gray-800/50"
              }`}
            >
              <div
                className={`w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0 ${
                  isDone
                    ? "bg-green-600 text-white"
                    : isActive
                    ? `${step.color} text-white`
                    : "bg-gray-700 text-gray-500"
                }`}
              >
                {isDone ? "✓" : i + 1}
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-0.5">
                  <span className={`text-xs font-semibold ${isActive ? "text-white" : "text-gray-400"}`}>
                    {step.level}
                  </span>
                  <span
                    className={`text-[10px] px-1.5 py-0.5 rounded ${
                      step.phase === "ept"
                        ? "bg-purple-500/20 text-purple-400"
                        : "bg-blue-500/20 text-blue-400"
                    }`}
                  >
                    {step.phase === "ept" ? "EPT" : "Guest"}
                  </span>
                </div>
                <p className="text-[11px] text-gray-500">{step.indexValue}</p>
                <AnimatePresence>
                  {isActive && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: "auto" }}
                      exit={{ opacity: 0, height: 0 }}
                      className="overflow-hidden"
                    >
                      <p className="text-xs text-gray-300 mt-1 flex items-center gap-1">
                        <ArrowRight className="w-3 h-3 text-yellow-500" />
                        {step.entryContent}
                      </p>
                      <p className="text-xs font-medium text-yellow-400 mt-0.5">
                        → {step.result}
                      </p>
                    </motion.div>
                  )}
                </AnimatePresence>
                {isDone && (
                  <p className="text-[11px] text-green-400/70 mt-0.5">{step.result}</p>
                )}
              </div>
            </motion.div>
          );
        })}
      </div>

      <AnimatePresence>
        {currentStep >= 7 && (
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-green-500/10 rounded-xl p-4 border border-green-500/30 text-center"
          >
            <p className="text-sm font-semibold text-green-400">翻译完成</p>
            <p className="text-xs text-gray-400 mt-1">
              GVA <span className="text-yellow-400 font-mono">0x0040_1234</span>
              {" → "}
              GPA <span className="text-blue-400 font-mono">0x5000_1000</span>
              {" → "}
              HPA <span className="text-green-400 font-mono">0xF000_1000</span>
            </p>
            <p className="text-[11px] text-gray-500 mt-1">
              共经历 4 次 Guest 页表查找 + 4 次 EPT 翻译 = 8 步硬件翻译
            </p>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
