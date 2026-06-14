"use client"

import { useState, useEffect } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { GitBranch } from "lucide-react"

type BusType = "single" | "dual" | "triple"

const operations: Record<BusType, { name: string; steps: { label: string; from: string; to: string }[] }> = {
  single: {
    name: "ADD R1, R2, R3",
    steps: [
      { label: "R2 → Bus → ALU(A)", from: "R2", to: "ALU" },
      { label: "R3 → Bus → ALU(B)", from: "R3", to: "ALU" },
      { label: "ALU → Bus → R1", from: "ALU", to: "R1" },
    ],
  },
  dual: {
    name: "ADD R1, R2, R3",
    steps: [
      { label: "R2→Bus1 ALU(A), R3→Bus2 ALU(B)", from: "R2+R3", to: "ALU" },
      { label: "ALU → Bus1 → R1", from: "ALU", to: "R1" },
    ],
  },
  triple: {
    name: "ADD R1, R2, R3",
    steps: [{ label: "R2→Bus1, R3→Bus2, ALU→Bus3→R1", from: "R2+R3", to: "R1" }],
  },
}

const layout = [
  { id: "R1", x: 40, y: 40, w: 60, h: 40, color: "#3b82f6" },
  { id: "R2", x: 40, y: 110, w: 60, h: 40, color: "#3b82f6" },
  { id: "R3", x: 40, y: 180, w: 60, h: 40, color: "#3b82f6" },
  { id: "ALU", x: 280, y: 90, w: 80, h: 60, color: "#f59e0b" },
  { id: "bus1", x: 160, y: 60, w: 60, h: 20, color: "#10b981" },
  { id: "bus2", x: 160, y: 130, w: 60, h: 20, color: "#8b5cf6" },
  { id: "bus3", x: 160, y: 200, w: 60, h: 20, color: "#ec4899" },
]

export function DatapathBuilder() {
  const [busType, setBusType] = useState<BusType>("single")
  const [step, setStep] = useState(-1)
  const [playing, setPlaying] = useState(false)

  const op = operations[busType]
  const visibleBuses = busType === "single" ? 1 : busType === "dual" ? 2 : 3

  useEffect(() => {
    if (!playing) return
    if (step >= op.steps.length - 1) { setPlaying(false); return }
    const timer = setTimeout(() => setStep((s) => s + 1), 1500)
    return () => clearTimeout(timer)
  }, [playing, step, op.steps.length])

  const highlights = step >= 0 ? getHighlights(op.steps[step]) : []

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <div className="flex items-center gap-2 mb-4">
        <GitBranch className="w-5 h-5 text-accent" />
        <h3 className="text-lg font-semibold">数据通路构建器</h3>
      </div>
      <div className="flex gap-2 mb-4">
        {(["single", "dual", "triple"] as const).map((t) => (
          <button key={t} onClick={() => { setBusType(t); setStep(-1); setPlaying(false) }}
            className={`px-4 py-1.5 rounded text-sm font-medium transition-colors ${busType === t ? "bg-accent text-white" : "bg-bg-surface border border-border-subtle text-text-secondary hover:border-accent"}`}>
            {t === "single" ? "单总线" : t === "dual" ? "双总线" : "三总线"}
          </button>
        ))}
        <button onClick={() => { setStep(0); setPlaying(true) }} disabled={playing}
          className="ml-auto px-4 py-1.5 rounded text-sm font-medium bg-green-600 text-white hover:bg-green-700 disabled:opacity-50">
          {playing ? "运行中..." : "执行动画"}
        </button>
      </div>
      <div className="flex gap-6">
        <svg viewBox="0 0 420 260" className="flex-1 min-h-[220px]">
          {Array.from({ length: visibleBuses }).map((_, i) => {
            const bus = layout[4 + i]
            return (
              <g key={bus.id}>
                <motion.rect x={bus.x} y={bus.y} width={bus.w} height={bus.h} rx={4}
                  fill={step >= 0 ? bus.color : "#374151"} fillOpacity={step >= 0 ? 0.3 : 0.15}
                  stroke={bus.color} strokeWidth={step >= 0 ? 2 : 1}
                  animate={{ fillOpacity: highlights.includes(bus.id) ? [0.2, 0.6, 0.2] : step >= 0 ? 0.3 : 0.15 }}
                  transition={{ duration: 0.8, repeat: highlights.includes(bus.id) ? Infinity : 0 }} />
                <text x={bus.x + bus.w / 2} y={bus.y + 14} textAnchor="middle" fill="#9ca3af" fontSize={10}>{bus.id}</text>
              </g>
            )
          })}
          {layout.slice(0, 4).map((comp) => (
            <g key={comp.id}>
              <motion.rect x={comp.x} y={comp.y} width={comp.w} height={comp.h} rx={6}
                fill={highlights.includes(comp.id) ? comp.color : "transparent"}
                fillOpacity={highlights.includes(comp.id) ? 0.25 : 0.05}
                stroke={comp.color} strokeWidth={highlights.includes(comp.id) ? 2 : 1}
                animate={{ scale: highlights.includes(comp.id) ? [1, 1.05, 1] : 1 }}
                transition={{ duration: 0.6, repeat: highlights.includes(comp.id) ? Infinity : 0 }} />
              <text x={comp.x + comp.w / 2} y={comp.y + comp.h / 2 + 1} textAnchor="middle" dominantBaseline="middle" fill={comp.color} fontSize={13} fontWeight="600">{comp.id}</text>
            </g>
          ))}
          {step >= 0 && <AnimatePresence>
            <motion.line key={`flow-${step}`} x1={100} y1={100} x2={280} y2={120} stroke="#f59e0b" strokeWidth={2} strokeDasharray="8 4" initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: 0.8 }} />
          </AnimatePresence>}
        </svg>
        <div className="w-52 flex flex-col gap-2">
          <div className="text-sm font-medium text-text-secondary mb-1">
            {op.name} — {busType === "single" ? "3" : busType === "dual" ? "2" : "1"} 个周期
          </div>
          {op.steps.map((s, i) => (
            <motion.div key={i} className={`p-2 rounded text-xs border transition-all ${step === i ? "border-accent bg-accent/10 text-accent" : step > i ? "border-green-600/30 bg-green-600/5 text-green-400" : "border-border-subtle text-text-secondary"}`}
              animate={step === i ? { scale: [1, 1.02, 1] } : {}}
              transition={{ duration: 0.5, repeat: step === i ? Infinity : 0 }}>
              <span className="font-mono">周期 {i + 1}:</span> {s.label}
            </motion.div>
          ))}
          <div className="mt-2 p-3 rounded-lg border border-border-subtle bg-bg-surface text-xs text-text-secondary">
            <div className="font-medium text-text-primary mb-1">性能比较</div>
            <div>单总线: 3 周期 (串行)</div>
            <div>双总线: 2 周期 (部分并行)</div>
            <div>三总线: 1 周期 (完全并行)</div>
          </div>
        </div>
      </div>
    </div>
  )
}

function getHighlights(step: { from: string; to: string }): string[] {
  const h: string[] = []
  for (const p of (step.from + "+" + step.to).split("+")) {
    const t = p.trim()
    if (["R1", "R2", "R3", "ALU"].includes(t)) h.push(t)
  }
  h.push("bus1")
  return h
}
