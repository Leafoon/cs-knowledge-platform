"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Workflow } from "lucide-react";

export function PipelineFriendlyISA() {
  const [showVariable, setShowVariable] = useState(false);

  const fixedStages = [
    { stage: "IF", time: "1ns", desc: "取指（固定长度，一次取完）" },
    { stage: "ID", time: "1ns", desc: "译码（格式统一，快速译码）" },
    { stage: "EX", time: "1ns", desc: "执行" },
    { stage: "MEM", time: "1ns", desc: "访存" },
    { stage: "WB", time: "1ns", desc: "写回" },
  ];

  const variableStages = [
    { stage: "IF", time: "1~3ns", desc: "取指（变长，需多次取）" },
    { stage: "ID", time: "1~4ns", desc: "译码（格式复杂，耗时不同）" },
    { stage: "EX", time: "1~2ns", desc: "执行（复杂指令多周期）" },
    { stage: "MEM", time: "0~1ns", desc: "访存" },
    { stage: "WB", time: "0~1ns", desc: "写回" },
  ];

  const stages = showVariable ? variableStages : fixedStages;

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <Workflow className="w-5 h-5 text-indigo-500" />
        流水线友好ISA设计
      </h3>

      <div className="flex gap-2 mb-6">
        <button onClick={() => setShowVariable(false)}
          className={`px-4 py-1.5 rounded text-sm ${!showVariable ? "bg-green-500 text-white" : "bg-bg-surface border border-border-subtle"}`}>
          定长指令 (RISC)
        </button>
        <button onClick={() => setShowVariable(true)}
          className={`px-4 py-1.5 rounded text-sm ${showVariable ? "bg-orange-500 text-white" : "bg-bg-surface border border-border-subtle"}`}>
          变长指令 (CISC)
        </button>
      </div>

      <div className="space-y-2 mb-6">
        {stages.map((s, i) => (
          <motion.div key={`${showVariable}-${i}`} className="flex items-center gap-3"
            initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.1 }}>
            <div className="w-12 text-center font-mono text-sm font-bold text-blue-400">{s.stage}</div>
            <motion.div
              className="h-10 rounded flex items-center px-3 text-sm text-white"
              style={{ backgroundColor: showVariable ? "#f59e0b" : "#10b981" }}
              initial={{ width: 0 }}
              animate={{ width: showVariable ? `${60 + i * 15}px` : "100%" }}
              transition={{ duration: 0.5, delay: i * 0.1 }}
            >
              <span className="font-mono">{s.time}</span>
            </motion.div>
            <span className="text-xs text-text-muted">{s.desc}</span>
          </motion.div>
        ))}
      </div>

      <motion.div
        className={`p-4 rounded border ${showVariable ? "border-orange-500/30 bg-orange-500/5" : "border-green-500/30 bg-green-500/5"}`}
        initial={{ opacity: 0 }} animate={{ opacity: 1 }}
      >
        <div className="text-sm font-medium mb-2">{showVariable ? "变长指令的流水线问题" : "定长指令的流水线优势"}</div>
        <ul className="text-xs text-text-secondary space-y-1">
          {showVariable ? (
            <>
              <li>• 取指阶段耗时不一致，需要预取缓冲</li>
              <li>• 译码阶段需先确定指令长度，增加复杂度</li>
              <li>• 各阶段时间不均等，流水线吞吐率降低</li>
              <li>• 控制逻辑复杂，难以实现超标量</li>
            </>
          ) : (
            <>
              <li>• 每周期取固定长度指令，IF阶段简单高效</li>
              <li>• 译码逻辑统一，无需预判指令边界</li>
              <li>• 各阶段时间均等，流水线无气泡</li>
              <li>• 易于实现超标量和乱序执行</li>
            </>
          )}
        </ul>
      </motion.div>
    </div>
  );
}
