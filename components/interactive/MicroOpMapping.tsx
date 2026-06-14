"use client"

import { useState } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { ArrowRightLeft } from "lucide-react"

interface InstructionProgram {
  name: string
  entry: number
  steps: { microAddr: number; microOps: string; controlSignals: string[] }[]
}

const instructions: InstructionProgram[] = [
  {
    name: "ADD",
    entry: 4,
    steps: [
      { microAddr: 0, microOps: "PC → MAR", controlSignals: ["PCout", "MARin"] },
      { microAddr: 1, microOps: "M[MAR] → MDR, PC+1", controlSignals: ["Read", "MDRout", "PCin"] },
      { microAddr: 2, microOps: "MDR → IR", controlSignals: ["MDRout", "IRin"] },
      { microAddr: 3, microOps: "MAP (decode)", controlSignals: ["MAP"] },
      { microAddr: 4, microOps: "IR[addr] → MAR", controlSignals: ["IRout", "MARin"] },
      { microAddr: 5, microOps: "M[MAR] → MDR", controlSignals: ["Read", "MDRout"] },
      { microAddr: 6, microOps: "AC + MDR → AC", controlSignals: ["ALUadd", "ACin", "End"] },
    ],
  },
  {
    name: "LOAD",
    entry: 7,
    steps: [
      { microAddr: 0, microOps: "PC → MAR", controlSignals: ["PCout", "MARin"] },
      { microAddr: 1, microOps: "M[MAR] → MDR, PC+1", controlSignals: ["Read", "MDRout", "PCin"] },
      { microAddr: 2, microOps: "MDR → IR", controlSignals: ["MDRout", "IRin"] },
      { microAddr: 3, microOps: "MAP (decode)", controlSignals: ["MAP"] },
      { microAddr: 7, microOps: "IR[addr] → MAR", controlSignals: ["IRout", "MARin"] },
      { microAddr: 8, microOps: "M[MAR] → MDR", controlSignals: ["Read", "MDRout"] },
      { microAddr: 9, microOps: "MDR → AC", controlSignals: ["MDRout", "ACin", "End"] },
    ],
  },
  {
    name: "STORE",
    entry: 10,
    steps: [
      { microAddr: 0, microOps: "PC → MAR", controlSignals: ["PCout", "MARin"] },
      { microAddr: 1, microOps: "M[MAR] → MDR, PC+1", controlSignals: ["Read", "MDRout", "PCin"] },
      { microAddr: 2, microOps: "MDR → IR", controlSignals: ["MDRout", "IRin"] },
      { microAddr: 3, microOps: "MAP (decode)", controlSignals: ["MAP"] },
      { microAddr: 10, microOps: "IR[addr] → MAR", controlSignals: ["IRout", "MARin"] },
      { microAddr: 11, microOps: "AC → MDR, Write", controlSignals: ["ACout", "MDRin", "Write", "End"] },
    ],
  },
  {
    name: "JMP",
    entry: 12,
    steps: [
      { microAddr: 0, microOps: "PC → MAR", controlSignals: ["PCout", "MARin"] },
      { microAddr: 1, microOps: "M[MAR] → MDR, PC+1", controlSignals: ["Read", "MDRout", "PCin"] },
      { microAddr: 2, microOps: "MDR → IR", controlSignals: ["MDRout", "IRin"] },
      { microAddr: 3, microOps: "MAP (decode)", controlSignals: ["MAP"] },
      { microAddr: 12, microOps: "IR[addr] → PC", controlSignals: ["IRout", "PCin", "End"] },
    ],
  },
]

export function MicroOpMapping() {
  const [selectedInst, setSelectedInst] = useState(0)
  const [expandedStep, setExpandedStep] = useState<number | null>(null)
  const program = instructions[selectedInst]

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <div className="flex items-center gap-2 mb-4">
        <ArrowRightLeft className="w-5 h-5 text-accent" />
        <h3 className="text-lg font-semibold">微操作映射表</h3>
      </div>

      <div className="flex gap-2 mb-4 flex-wrap">
        {instructions.map((inst, i) => (
          <button
            key={inst.name}
            className={`px-4 py-1.5 text-xs rounded-md border font-medium ${
              selectedInst === i
                ? "bg-blue-600 text-white border-blue-600"
                : "border-border-subtle hover:border-blue-400"
            }`}
            onClick={() => { setSelectedInst(i); setExpandedStep(null) }}
          >
            {inst.name}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-3 mb-4">
        <div className="p-3 rounded bg-bg-surface border border-border-subtle">
          <div className="text-xs text-text-secondary mb-1">机器指令</div>
          <div className="text-lg font-bold">{program.name}</div>
        </div>
        <div className="p-3 rounded bg-bg-surface border border-border-subtle">
          <div className="text-xs text-text-secondary mb-1">微程序入口</div>
          <div className="text-lg font-mono font-bold">μPC = {program.entry}</div>
        </div>
        <div className="p-3 rounded bg-bg-surface border border-border-subtle">
          <div className="text-xs text-text-secondary mb-1">微指令数</div>
          <div className="text-lg font-bold">{program.steps.length}</div>
        </div>
        <div className="p-3 rounded bg-bg-surface border border-border-subtle">
          <div className="text-xs text-text-secondary mb-1">层次</div>
          <div className="text-xs">指令 → 微程序 → 微命令</div>
        </div>
      </div>

      <div className="relative">
        <div className="absolute left-[22px] top-0 bottom-0 w-px bg-border-subtle" />
        <div className="space-y-2">
          {program.steps.map((ms, i) => {
            const isShared = ms.microAddr < 4
            const isExpanded = expandedStep === i
            return (
              <motion.div
                key={i}
                className="relative"
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.05 }}
              >
                <div
                  className={`flex items-start gap-3 p-3 rounded-lg border cursor-pointer transition-colors ${
                    isExpanded ? "border-blue-500 bg-blue-900/20" : "border-border-subtle hover:bg-bg-surface"
                  }`}
                  onClick={() => setExpandedStep(isExpanded ? null : i)}
                >
                  <div className={`w-10 h-6 rounded flex items-center justify-center text-[10px] font-mono shrink-0 ${
                    isShared ? "bg-yellow-800/40 text-yellow-300" : "bg-blue-800/40 text-blue-300"
                  }`}>
                    μ{ms.microAddr}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-mono">{ms.microOps}</span>
                      {isShared && (
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-yellow-800/30 text-yellow-400">
                          共享
                        </span>
                      )}
                    </div>
                  </div>
                  <div className="flex flex-wrap gap-1 shrink-0 max-w-[200px]">
                    {ms.controlSignals.slice(0, 3).map(sig => (
                      <span key={sig} className="text-[10px] px-1.5 py-0.5 rounded bg-bg-surface border border-border-subtle">
                        {sig}
                      </span>
                    ))}
                    {ms.controlSignals.length > 3 && (
                      <span className="text-[10px] text-text-secondary">+{ms.controlSignals.length - 3}</span>
                    )}
                  </div>
                </div>

                <AnimatePresence>
                  {isExpanded && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: "auto", opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      className="overflow-hidden"
                    >
                      <div className="ml-14 mt-1 p-3 rounded bg-bg-surface border border-border-subtle text-xs">
                        <div className="mb-2">
                          <span className="text-text-secondary">微地址：</span>
                          <span className="font-mono ml-1">{ms.microAddr}</span>
                        </div>
                        <div className="mb-2">
                          <span className="text-text-secondary">微操作：</span>
                          <span className="ml-1">{ms.microOps}</span>
                        </div>
                        <div>
                          <span className="text-text-secondary">控制信号：</span>
                          <div className="flex flex-wrap gap-1 mt-1">
                            {ms.controlSignals.map(sig => (
                              <span key={sig} className="px-2 py-1 rounded bg-blue-900/30 text-blue-300 border border-blue-500/50 font-mono">
                                {sig}
                              </span>
                            ))}
                          </div>
                        </div>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
