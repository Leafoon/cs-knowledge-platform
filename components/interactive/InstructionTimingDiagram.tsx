"use client"

import { useState } from "react"
import { motion } from "framer-motion"
import { Clock, BarChart3 } from "lucide-react"

interface MachineCycle {
  type: string
  label: string
  color: string
  width: number
}

interface InstructionType {
  name: string
  desc: string
  cycles: MachineCycle[]
  cpi: number
}

const instructions: InstructionType[] = [
  {
    name: "寄存器寻址",
    desc: "ADD R1, R2",
    cpi: 1,
    cycles: [
      { type: "FE", label: "取指", color: "bg-blue-500", width: 1 },
      { type: "EX", label: "执行", color: "bg-green-500", width: 1 },
    ],
  },
  {
    name: "直接寻址",
    desc: "LOAD [addr]",
    cpi: 2,
    cycles: [
      { type: "FE", label: "取指", color: "bg-blue-500", width: 1 },
      { type: "EX", label: "执行(访存)", color: "bg-green-500", width: 1 },
    ],
  },
  {
    name: "间接寻址",
    desc: "LOAD [[addr]]",
    cpi: 3,
    cycles: [
      { type: "FE", label: "取指", color: "bg-blue-500", width: 1 },
      { type: "IND", label: "间址", color: "bg-amber-500", width: 1 },
      { type: "EX", label: "执行(访存)", color: "bg-green-500", width: 1 },
    ],
  },
  {
    name: "变址寻址",
    desc: "LOAD [X+addr]",
    cpi: 3,
    cycles: [
      { type: "FE", label: "取指", color: "bg-blue-500", width: 1 },
      { type: "EX", label: "计算地址", color: "bg-green-500", width: 1 },
      { type: "EX", label: "执行(访存)", color: "bg-green-500", width: 1 },
    ],
  },
  {
    name: "带中断检查",
    desc: "任意指令+中断",
    cpi: 4,
    cycles: [
      { type: "FE", label: "取指", color: "bg-blue-500", width: 1 },
      { type: "DE", label: "译码", color: "bg-purple-500", width: 1 },
      { type: "EX", label: "执行", color: "bg-green-500", width: 1 },
      { type: "INTR", label: "中断", color: "bg-red-500", width: 1 },
    ],
  },
]

const phaseLegend = [
  { label: "FE 取指", color: "bg-blue-500" },
  { label: "DE 译码", color: "bg-purple-500" },
  { label: "IND 间址", color: "bg-amber-500" },
  { label: "EX 执行", color: "bg-green-500" },
  { label: "INTR 中断", color: "bg-red-500" },
]

export function InstructionTimingDiagram() {
  const [selected, setSelected] = useState<number[]>([0, 2, 4])

  const toggle = (idx: number) => {
    setSelected((s) => s.includes(idx) ? s.filter((i) => i !== idx) : [...s, idx])
  }

  const maxCycles = Math.max(...selected.map((i) => instructions[i].cycles.length))
  const avgCpi = selected.length > 0 ? (selected.reduce((s, i) => s + instructions[i].cpi, 0) / selected.length).toFixed(1) : "—"

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <div className="flex items-center gap-2 mb-4">
        <BarChart3 className="w-5 h-5 text-accent" />
        <h3 className="text-lg font-semibold">指令时序图</h3>
      </div>

      <div className="mb-4">
        <h4 className="text-sm font-medium text-text-secondary mb-2">选择指令类型</h4>
        <div className="flex flex-wrap gap-2">
          {instructions.map((instr, i) => (
            <button key={i} onClick={() => toggle(i)} className={`px-3 py-1.5 rounded text-sm border transition-colors ${selected.includes(i) ? "border-accent bg-accent/10 text-accent" : "border-border-subtle bg-bg-card text-text-secondary"}`}>
              {instr.name}
            </button>
          ))}
        </div>
      </div>

      <div className="flex flex-wrap gap-3 mb-4">
        {phaseLegend.map((p) => (
          <div key={p.label} className="flex items-center gap-1.5 text-xs text-text-secondary">
            <div className={`w-3 h-3 rounded-sm ${p.color}`} />
            {p.label}
          </div>
        ))}
      </div>

      <div className="space-y-3 mb-4">
        {selected.sort((a, b) => a - b).map((idx) => {
          const instr = instructions[idx]
          return (
            <motion.div key={idx} initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} className="bg-bg-card border border-border-subtle rounded p-3">
              <div className="flex items-center gap-2 mb-2">
                <span className="text-sm font-medium">{instr.name}</span>
                <span className="text-xs text-text-secondary font-mono">({instr.desc})</span>
                <span className="ml-auto text-xs font-mono px-2 py-0.5 rounded bg-accent/10 text-accent">CPI = {instr.cpi}</span>
              </div>
              <div className="flex items-center gap-1">
                <Clock className="w-3 h-3 text-text-secondary shrink-0" />
                <div className="flex-1 flex gap-0.5">
                  {instr.cycles.map((c, ci) => (
                    <motion.div
                      key={ci}
                      initial={{ scaleX: 0 }}
                      animate={{ scaleX: 1 }}
                      transition={{ delay: ci * 0.15 }}
                      className={`h-8 rounded flex items-center justify-center text-xs text-white font-medium ${c.color}`}
                      style={{ flex: c.width }}
                    >
                      {c.label}
                    </motion.div>
                  ))}
                  {Array.from({ length: maxCycles - instr.cycles.length }, (_, i) => (
                    <div key={`pad-${i}`} className="flex-1 h-8" />
                  ))}
                </div>
              </div>
              <div className="flex mt-1">
                {instr.cycles.map((_, ci) => (
                  <div key={ci} className="flex-1 text-center text-xs text-text-secondary">T{ci + 1}</div>
                ))}
                {Array.from({ length: maxCycles - instr.cycles.length }, (_, i) => (
                  <div key={`t-${i}`} className="flex-1" />
                ))}
              </div>
            </motion.div>
          )
        })}
      </div>

      {selected.length > 0 && (
        <div className="grid grid-cols-3 gap-3">
          <div className="bg-bg-card border border-border-subtle rounded p-3 text-center">
            <div className="text-xs text-text-secondary mb-1">对比指令数</div>
            <div className="text-xl font-bold text-accent">{selected.length}</div>
          </div>
          <div className="bg-bg-card border border-border-subtle rounded p-3 text-center">
            <div className="text-xs text-text-secondary mb-1">平均 CPI</div>
            <div className="text-xl font-bold text-accent">{avgCpi}</div>
          </div>
          <div className="bg-bg-card border border-border-subtle rounded p-3 text-center">
            <div className="text-xs text-text-secondary mb-1">最大机器周期</div>
            <div className="text-xl font-bold text-accent">{maxCycles}</div>
          </div>
        </div>
      )}
    </div>
  )
}
