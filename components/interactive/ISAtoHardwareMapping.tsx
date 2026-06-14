"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Cpu, ArrowRight } from "lucide-react";

interface Stage {
  name: string;
  desc: string;
  detail: string;
  color: string;
}

const instructions = [
  {
    name: "ADD R1, R2, R3",
    stages: [
      { name: "IF", desc: "取指", detail: "PC → 指令存储器，取出指令 ADD R1,R2,R3", color: "#3b82f6" },
      { name: "ID", desc: "译码", detail: "译码得到 ADD 操作，读寄存器 R2、R3", color: "#8b5cf6" },
      { name: "EX", desc: "执行", detail: "ALU 计算 R2 + R3", color: "#10b981" },
      { name: "MEM", desc: "访存", detail: "无访存操作（ALU指令跳过）", color: "#f59e0b" },
      { name: "WB", desc: "写回", detail: "结果写回寄存器 R1", color: "#ef4444" },
    ],
  },
  {
    name: "LW R1, 100(R2)",
    stages: [
      { name: "IF", desc: "取指", detail: "取出指令 LW R1,100(R2)", color: "#3b82f6" },
      { name: "ID", desc: "译码", detail: "译码得到 LW 操作，读寄存器 R2", color: "#8b5cf6" },
      { name: "EX", desc: "执行", detail: "ALU 计算 R2 + 100 = 有效地址", color: "#10b981" },
      { name: "MEM", desc: "访存", detail: "访问数据存储器，读取 M[R2+100]", color: "#f59e0b" },
      { name: "WB", desc: "写回", detail: "将读出的数据写回 R1", color: "#ef4444" },
    ],
  },
  {
    name: "BEQ R1, R2, offset",
    stages: [
      { name: "IF", desc: "取指", detail: "取出指令 BEQ R1,R2,offset", color: "#3b82f6" },
      { name: "ID", desc: "译码", detail: "译码得到 BEQ，读 R1、R2", color: "#8b5cf6" },
      { name: "EX", desc: "执行", detail: "ALU 比较 R1==R2，计算目标地址", color: "#10b981" },
      { name: "MEM", desc: "访存", detail: "若相等，更新 PC = 目标地址", color: "#f59e0b" },
      { name: "WB", desc: "写回", detail: "无写回操作", color: "#ef4444" },
    ],
  },
];

export function ISAtoHardwareMapping() {
  const [instrIdx, setInstrIdx] = useState(0);
  const [activeStage, setActiveStage] = useState(0);

  const instr = instructions[instrIdx];

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <Cpu className="w-5 h-5 text-indigo-500" />
        ISA到硬件映射
      </h3>

      <div className="flex gap-2 mb-4">
        {instructions.map((inst, i) => (
          <button key={i} onClick={() => { setInstrIdx(i); setActiveStage(0); }}
            className={`px-3 py-1 rounded text-xs font-mono ${instrIdx === i ? "bg-blue-500 text-white" : "bg-bg-surface border border-border-subtle"}`}>
            {inst.name}
          </button>
        ))}
      </div>

      <div className="flex items-center gap-1 mb-4">
        {instr.stages.map((s, i) => (
          <div key={i} className="flex items-center">
            <button onClick={() => setActiveStage(i)}
              className={`px-3 py-2 rounded text-xs font-medium transition-colors ${
                activeStage === i ? "text-white" : "bg-bg-surface border border-border-subtle hover:border-blue-400"
              }`}
              style={activeStage === i ? { backgroundColor: s.color } : undefined}>
              {s.name}
            </button>
            {i < instr.stages.length - 1 && <ArrowRight className="w-4 h-4 text-text-muted mx-0.5" />}
          </div>
        ))}
      </div>

      <AnimatePresence mode="wait">
        <motion.div key={`${instrIdx}-${activeStage}`}
          initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
          <div className="p-4 rounded border border-border-subtle" style={{ borderColor: instr.stages[activeStage].color + "60" }}>
            <div className="flex items-center gap-2 mb-2">
              <div className="px-2 py-0.5 rounded text-sm font-bold text-white"
                style={{ backgroundColor: instr.stages[activeStage].color }}>
                {instr.stages[activeStage].name}
              </div>
              <span className="font-medium">{instr.stages[activeStage].desc}</span>
            </div>
            <p className="text-sm text-text-secondary">{instr.stages[activeStage].detail}</p>
          </div>

          <div className="mt-4 grid grid-cols-5 gap-1">
            {instr.stages.map((s, i) => (
              <motion.div key={i} className="h-2 rounded-full" style={{ backgroundColor: i <= activeStage ? s.color : "#333" }}
                initial={{ scaleX: 0 }} animate={{ scaleX: 1 }} transition={{ delay: i * 0.1 }} />
            ))}
          </div>
        </motion.div>
      </AnimatePresence>

      <div className="flex gap-2 mt-4">
        <button onClick={() => setActiveStage(Math.max(0, activeStage - 1))} disabled={activeStage === 0}
          className="px-3 py-1 rounded bg-bg-surface border border-border-subtle text-sm disabled:opacity-30">上一步</button>
        <button onClick={() => setActiveStage(Math.min(4, activeStage + 1))} disabled={activeStage === 4}
          className="px-3 py-1 rounded bg-blue-500 text-white text-sm disabled:opacity-30">下一步</button>
      </div>
    </div>
  );
}
