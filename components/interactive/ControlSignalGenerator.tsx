"use client"

import { useState } from "react"
import { motion } from "framer-motion"
import { Zap } from "lucide-react"

const opcodes = ["ADD", "LOAD", "STORE", "JMP", "SUB", "AND"]
const timingSignals = ["T1", "T2", "T3", "T4"]
const controlSignals = [
  "PCout", "MARin", "Read", "MDRout", "IRin",
  "PCin", "ALUadd", "ALUsub", "MARin", "WMFC", "End",
]

const signalTable: Record<string, Record<string, string[]>> = {
  ADD: {
    T1: ["PCout", "MARin"],
    T2: ["Read", "MDRout", "PCin"],
    T3: ["MDRout", "IRin"],
    T4: ["ALUadd", "End"],
  },
  LOAD: {
    T1: ["PCout", "MARin"],
    T2: ["Read", "MDRout", "PCin"],
    T3: ["MDRout", "IRin"],
    T4: ["MARin", "Read", "WMFC"],
  },
  STORE: {
    T1: ["PCout", "MARin"],
    T2: ["Read", "MDRout", "PCin"],
    T3: ["MDRout", "IRin"],
    T4: ["MARin"],
  },
  JMP: {
    T1: ["PCout", "MARin"],
    T2: ["Read", "MDRout", "PCin"],
    T3: ["MDRout", "IRin"],
    T4: ["PCin", "End"],
  },
  SUB: {
    T1: ["PCout", "MARin"],
    T2: ["Read", "MDRout", "PCin"],
    T3: ["MDRout", "IRin"],
    T4: ["ALUsub", "End"],
  },
  AND: {
    T1: ["PCout", "MARin"],
    T2: ["Read", "MDRout", "PCin"],
    T3: ["MDRout", "IRin"],
    T4: ["End"],
  },
}

const booleanExprs: Record<string, string> = {
  PCout: "PCout = T1·(ADD+LOAD+STORE+JMP+SUB+AND)",
  MARin: "MARin = T1·(ADD+LOAD+STORE+JMP+SUB+AND) + T4·LOAD + T4·STORE",
  Read: "Read = T2·(ADD+LOAD+STORE+JMP+SUB+AND) + T4·LOAD",
  MDRout: "MDRout = T2·(ADD+LOAD+STORE+JMP+SUB+AND) + T3·(ADD+LOAD+STORE+JMP+SUB+AND)",
  IRin: "IRin = T3·(ADD+LOAD+STORE+JMP+SUB+AND)",
  PCin: "PCin = T2·(ADD+LOAD+STORE+JMP+SUB+AND) + T4·JMP",
  ALUadd: "ALUadd = T4·ADD",
  ALUsub: "ALUsub = T4·SUB",
  WMFC: "WMFC = T4·LOAD",
  End: "End = T4·(ADD+STORE+JMP+SUB+AND)",
}

export function ControlSignalGenerator() {
  const [opcode, setOpcode] = useState("ADD")
  const [timing, setTiming] = useState("T1")
  const [animatingSignals, setAnimatingSignals] = useState<string[]>([])

  const activeSignals = signalTable[opcode]?.[timing] || []

  const handleCellClick = (op: string, t: string) => {
    setOpcode(op)
    setTiming(t)
    const signals = signalTable[op]?.[t] || []
    setAnimatingSignals([])
    signals.forEach((sig, i) => {
      setTimeout(() => {
        setAnimatingSignals(prev => [...prev, sig])
      }, i * 150)
    })
  }

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <div className="flex items-center gap-2 mb-4">
        <Zap className="w-5 h-5 text-accent" />
        <h3 className="text-lg font-semibold">控制信号生成器</h3>
      </div>

      <div className="flex gap-3 mb-4">
        <div>
          <label className="text-xs text-text-secondary mb-1 block">操作码</label>
          <select
            value={opcode}
            onChange={e => { setOpcode(e.target.value); setAnimatingSignals([]) }}
            className="px-3 py-1.5 bg-bg-surface border border-border-subtle rounded text-sm"
          >
            {opcodes.map(op => <option key={op} value={op}>{op}</option>)}
          </select>
        </div>
        <div>
          <label className="text-xs text-text-secondary mb-1 block">时序信号</label>
          <select
            value={timing}
            onChange={e => { setTiming(e.target.value); setAnimatingSignals([]) }}
            className="px-3 py-1.5 bg-bg-surface border border-border-subtle rounded text-sm"
          >
            {timingSignals.map(t => <option key={t} value={t}>{t}</option>)}
          </select>
        </div>
      </div>

      <div className="overflow-x-auto mb-4">
        <table className="w-full text-xs border-collapse">
          <thead>
            <tr>
              <th className="p-2 border border-border-subtle bg-bg-surface"></th>
              {timingSignals.map(t => (
                <th key={t} className="p-2 border border-border-subtle bg-bg-surface font-medium">{t}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {opcodes.map(op => (
              <tr key={op}>
                <td className="p-2 border border-border-subtle font-medium bg-bg-surface">{op}</td>
                {timingSignals.map(t => {
                  const signals = signalTable[op]?.[t] || []
                  const isSelected = op === opcode && t === timing
                  return (
                    <td
                      key={t}
                      className={`p-2 border border-border-subtle cursor-pointer transition-colors ${
                        isSelected ? "bg-blue-900/30" : "hover:bg-bg-surface"
                      }`}
                      onClick={() => handleCellClick(op, t)}
                    >
                      {signals.length > 0 ? signals.join(", ") : "—"}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="mb-4">
        <h4 className="text-sm font-medium mb-2">活跃控制信号</h4>
        <div className="flex flex-wrap gap-2">
          {controlSignals.map(sig => {
            const isActive = activeSignals.includes(sig)
            return (
              <motion.span
                key={sig}
                className={`px-3 py-1.5 rounded text-xs font-mono border ${
                  isActive
                    ? "border-blue-500 text-blue-400"
                    : "border-border-subtle text-text-secondary opacity-40"
                }`}
                animate={{
                  scale: isActive && animatingSignals.includes(sig) ? [1, 1.15, 1] : 1,
                  backgroundColor: isActive ? "rgba(59,130,246,0.15)" : "transparent",
                }}
                transition={{ duration: 0.3 }}
              >
                {sig}
              </motion.span>
            )
          })}
        </div>
      </div>

      <div>
        <h4 className="text-sm font-medium mb-2">布尔表达式</h4>
        <div className="space-y-1 max-h-40 overflow-y-auto">
          {Object.entries(booleanExprs).map(([sig, expr]) => (
            <div
              key={sig}
              className={`text-xs font-mono px-3 py-1 rounded ${
                activeSignals.includes(sig)
                  ? "bg-blue-900/20 text-blue-300"
                  : "text-text-secondary"
              }`}
            >
              {expr}
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
