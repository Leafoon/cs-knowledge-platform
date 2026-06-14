"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { PlayCircle, RotateCcw } from "lucide-react";

const stages = [
  {
    name: "IF",
    full: "取指 (Instruction Fetch)",
    steps: [
      "PC → MAR: 将程序计数器的值送到地址总线",
      "M[MAR] → MDR: 从存储器读取指令到数据总线",
      "MDR → IR: 指令送入指令寄存器",
      "PC + 1 → PC: 程序计数器自增",
    ],
    dataFlow: "PC → 地址总线 → 存储器 → 数据总线 → IR",
    color: "#3b82f6",
  },
  {
    name: "ID",
    full: "译码 (Instruction Decode)",
    steps: [
      "IR[opcode] → 控制单元: 操作码送入CU译码",
      "IR[rs] → 寄存器读端口1: 读源操作数1",
      "IR[rt] → 寄存器读端口2: 读源操作数2",
      "IR[imm] → 符号扩展: 立即数符号扩展",
    ],
    dataFlow: "IR → CU(译码) + 寄存器堆(读数据)",
    color: "#8b5cf6",
  },
  {
    name: "EX",
    full: "执行 (Execute)",
    steps: [
      "ALU A端 ← 源操作数1 (rs)",
      "ALU B端 ← 源操作数2 (rt 或 立即数)",
      "ALU执行运算 (ADD/SUB/AND/OR等)",
      "ALU结果 → ALU输出寄存器",
    ],
    dataFlow: "rs/rt → ALU → 结果",
    color: "#10b981",
  },
  {
    name: "MEM",
    full: "访存 (Memory Access)",
    steps: [
      "LW: ALU结果 → MAR, M[MAR] → MDR",
      "SW: rt → MDR, ALU结果 → MAR, MDR → M[MAR]",
      "R-type: 无访存操作（跳过）",
      "BEQ: 若Z=1, 目标地址 → PC",
    ],
    dataFlow: "ALU结果 → MAR → 存储器 ↔ MDR",
    color: "#f59e0b",
  },
  {
    name: "WB",
    full: "写回 (Write Back)",
    steps: [
      "R-type: ALU结果 → 寄存器堆[rd]",
      "LW: MDR → 寄存器堆[rt]",
      "写入信号 → 寄存器堆写使能",
      "目标寄存器地址 ← IR[rd] 或 IR[rt]",
    ],
    dataFlow: "ALU结果/MDR → 寄存器堆",
    color: "#ef4444",
  },
];

export function InstructionExecutionFlow() {
  const [activeStage, setActiveStage] = useState(-1);
  const [step, setStep] = useState(0);

  const start = () => { setActiveStage(0); setStep(0); };
  const nextStep = () => {
    const currentStage = stages[activeStage];
    if (step < currentStage.steps.length - 1) {
      setStep(s => s + 1);
    } else if (activeStage < stages.length - 1) {
      setActiveStage(s => s + 1);
      setStep(0);
    }
  };
  const reset = () => { setActiveStage(-1); setStep(0); };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <PlayCircle className="w-5 h-5 text-blue-500" />
        指令执行流程
      </h3>

      <div className="flex items-center gap-1 mb-6">
        {stages.map((s, i) => (
          <div key={i} className="flex items-center">
            <div className={`px-3 py-2 rounded text-sm font-medium transition-colors ${
              i === activeStage ? "text-white" : i < activeStage ? "bg-bg-surface border border-border-subtle opacity-60" : "bg-bg-surface border border-border-subtle"
            }`}
              style={i === activeStage ? { backgroundColor: s.color } : undefined}>
              {s.name}
            </div>
            {i < stages.length - 1 && <div className="w-4 h-0.5 bg-border-subtle" />}
          </div>
        ))}
      </div>

      <AnimatePresence mode="wait">
        {activeStage >= 0 ? (
          <motion.div key={activeStage} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
            <div className="p-4 rounded border mb-4" style={{ borderColor: stages[activeStage].color + "60", backgroundColor: stages[activeStage].color + "08" }}>
              <div className="text-sm font-bold mb-1" style={{ color: stages[activeStage].color }}>{stages[activeStage].full}</div>
              <div className="text-xs text-text-muted mb-3">数据流: {stages[activeStage].dataFlow}</div>
              <div className="space-y-1">
                {stages[activeStage].steps.map((st, i) => (
                  <motion.div key={i}
                    className={`flex items-center gap-2 p-1.5 rounded text-xs font-mono ${
                      i <= step ? "text-text-secondary" : "text-text-muted opacity-40"
                    }`}
                    initial={{ opacity: 0 }} animate={{ opacity: i <= step ? 1 : 0.4 }}>
                    <span className={`w-5 h-5 rounded-full flex items-center justify-center text-xs ${
                      i <= step ? "text-white" : "bg-bg-surface text-text-muted"
                    }`} style={i <= step ? { backgroundColor: stages[activeStage].color } : undefined}>
                      {i + 1}
                    </span>
                    {st}
                  </motion.div>
                ))}
              </div>
            </div>

            <div className="flex gap-2">
              <button onClick={nextStep}
                className="px-4 py-1.5 rounded bg-blue-500 text-white text-sm hover:bg-blue-600">
                {step < stages[activeStage].steps.length - 1 ? "下一步" : activeStage < stages.length - 1 ? "下一阶段" : "完成"}
              </button>
              <button onClick={reset} className="px-3 py-1.5 rounded bg-bg-surface border border-border-subtle text-sm flex items-center gap-1 hover:border-blue-400">
                <RotateCcw className="w-4 h-4" /> 重置
              </button>
            </div>
          </motion.div>
        ) : (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="text-center py-8">
            <p className="text-text-muted mb-4">点击开始，逐步观察指令从取指到写回的完整执行流程</p>
            <button onClick={start} className="px-6 py-2 rounded bg-blue-500 text-white text-sm flex items-center gap-2 mx-auto hover:bg-blue-600">
              <PlayCircle className="w-5 h-5" /> 开始执行
            </button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
