"use client"

import { useState, useEffect, useCallback } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Play, Pause, SkipForward, RotateCcw, Cpu } from "lucide-react"

interface MicroOp {
  label: string
  signals: string[]
}

const phases: { name: string; color: string; ops: MicroOp[] }[] = [
  {
    name: "取指 Fetch",
    color: "bg-blue-500",
    ops: [
      { label: "PC → MAR", signals: ["PCout", "MARin"] },
      { label: "M[MAR] → MDR", signals: ["MemRead", "MARin", "MDRin"] },
      { label: "MDR → IR", signals: ["MDRout", "IRin"] },
      { label: "PC + 1 → PC", signals: ["PCout", "ALU+1", "PCin"] },
    ],
  },
  {
    name: "间址 Indirect",
    color: "bg-amber-500",
    ops: [
      { label: "IR[addr] → MAR", signals: ["IRaddr_out", "MARin"] },
      { label: "M[MAR] → MDR", signals: ["MemRead", "MARin", "MDRin"] },
      { label: "MDR → IR[addr]", signals: ["MDRout", "IRaddr_in"] },
    ],
  },
  {
    name: "执行 Execute",
    color: "bg-green-500",
    ops: [
      { label: "IR[addr] → MAR", signals: ["IRaddr_out", "MARin"] },
      { label: "M[MAR] → MDR", signals: ["MemRead", "MARin", "MDRin"] },
      { label: "MDR → Y", signals: ["MDRout", "Yin"] },
      { label: "AC + Y → Z", signals: ["ACout", "ALU+", "Zin"] },
      { label: "Z → AC", signals: ["Zout", "ACin"] },
    ],
  },
  {
    name: "中断 Interrupt",
    color: "bg-red-500",
    ops: [
      { label: "SP - 1 → SP", signals: ["SPout", "ALU-1", "SPin"] },
      { label: "PC → M[SP]", signals: ["PCout", "MARin", "MemWrite"] },
      { label: "ISR → PC", signals: ["IVRout", "PCin"] },
    ],
  },
]

const allSignals = [
  "PCout", "PCin", "MARin", "MemRead", "MemWrite", "MDRin", "MDRout",
  "IRin", "IRaddr_out", "IRaddr_in", "ACout", "ACin", "Yin", "Zin", "Zout",
  "SPout", "SPin", "ALU+1", "ALU-1", "ALU+", "IVRout",
]

export function MicrooperationSequencer() {
  const [phaseIdx, setPhaseIdx] = useState(0)
  const [opIdx, setOpIdx] = useState(0)
  const [playing, setPlaying] = useState(false)

  const currentPhase = phases[phaseIdx]
  const currentOp = currentPhase.ops[opIdx]
  const activeSignals = currentOp.signals

  const step = useCallback(() => {
    if (opIdx < currentPhase.ops.length - 1) {
      setOpIdx((i) => i + 1)
    } else if (phaseIdx < phases.length - 1) {
      setPhaseIdx((p) => p + 1)
      setOpIdx(0)
    } else {
      setPhaseIdx(0)
      setOpIdx(0)
      setPlaying(false)
    }
  }, [phaseIdx, opIdx, currentPhase.ops.length])

  useEffect(() => {
    if (!playing) return
    const timer = setInterval(step, 1200)
    return () => clearInterval(timer)
  }, [playing, step])

  const reset = () => {
    setPhaseIdx(0)
    setOpIdx(0)
    setPlaying(false)
  }

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <div className="flex items-center gap-2 mb-4">
        <Cpu className="w-5 h-5 text-accent" />
        <h3 className="text-lg font-semibold">微操作序列器</h3>
      </div>

      <div className="flex gap-2 mb-4">
        <button onClick={() => setPlaying(!playing)} className="px-3 py-1.5 rounded bg-accent text-white text-sm flex items-center gap-1">
          {playing ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
          {playing ? "暂停" : "自动播放"}
        </button>
        <button onClick={step} className="px-3 py-1.5 rounded bg-bg-card border border-border-subtle text-sm flex items-center gap-1">
          <SkipForward className="w-4 h-4" /> 单步
        </button>
        <button onClick={reset} className="px-3 py-1.5 rounded bg-bg-card border border-border-subtle text-sm flex items-center gap-1">
          <RotateCcw className="w-4 h-4" /> 重置
        </button>
      </div>

      <div className="flex gap-1 mb-4">
        {phases.map((p, i) => (
          <div key={p.name} className={`flex-1 h-2 rounded-full transition-colors ${i === phaseIdx ? p.color : i < phaseIdx ? "bg-text-secondary opacity-40" : "bg-bg-card"}`} />
        ))}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
        <div>
          <h4 className="text-sm font-medium text-text-secondary mb-2">阶段: {currentPhase.name}</h4>
          <div className="space-y-1.5">
            {currentPhase.ops.map((op, i) => (
              <motion.div
                key={`${phaseIdx}-${i}`}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                className={`flex items-center gap-2 px-3 py-2 rounded text-sm font-mono transition-colors ${i === opIdx ? "bg-accent/15 border border-accent text-accent" : i < opIdx ? "text-text-secondary opacity-50" : "bg-bg-card border border-border-subtle"}`}
              >
                <span className={`w-5 h-5 rounded-full flex items-center justify-center text-xs ${i === opIdx ? "bg-accent text-white" : "bg-bg-card"}`}>
                  {i < opIdx ? "✓" : i + 1}
                </span>
                {op.label}
              </motion.div>
            ))}
          </div>
        </div>

        <div>
          <h4 className="text-sm font-medium text-text-secondary mb-2">控制信号</h4>
          <div className="grid grid-cols-3 gap-1.5">
            {allSignals.map((sig) => {
              const active = activeSignals.includes(sig)
              return (
                <AnimatePresence key={sig}>
                  {active && (
                    <motion.div
                      initial={{ scale: 0.8, opacity: 0 }}
                      animate={{ scale: 1, opacity: 1 }}
                      exit={{ scale: 0.8, opacity: 0 }}
                      className="px-2 py-1 rounded text-xs font-mono bg-green-500/20 text-green-400 border border-green-500/30 text-center"
                    >
                      {sig}
                    </motion.div>
                  )}
                </AnimatePresence>
              )
            })}
          </div>
        </div>
      </div>

      <div className="flex items-center justify-between text-xs text-text-secondary">
        <span>步骤 {phaseIdx * 10 + opIdx + 1} / {phases.reduce((s, p) => s + p.ops.length, 0)}</span>
        <span>总微操作数: {phases.reduce((s, p) => s + p.ops.length, 0)}</span>
      </div>
    </div>
  )
}
