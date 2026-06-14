"use client"

import { useState, useEffect } from "react"
import { motion } from "framer-motion"
import { Play, RotateCcw } from "lucide-react"

type Phase = "fetch" | "decode" | "execute"

const microSteps: { phase: Phase; label: string; description: string; active: string[]; from: string; to: string }[] = [
  { phase: "fetch", label: "Fetch 1", description: "PC 的内容送入 MAR", active: ["PC", "MAR"], from: "PC", to: "MAR" },
  { phase: "fetch", label: "Fetch 2", description: "MAR 指向的内存单元内容送入 MDR，PC+1", active: ["MAR", "Memory", "MDR", "PC"], from: "Memory", to: "MDR" },
  { phase: "fetch", label: "Fetch 3", description: "MDR 的内容(指令)送入 IR", active: ["MDR", "IR"], from: "MDR", to: "IR" },
  { phase: "decode", label: "Decode 1", description: "IR 的操作码部分送入 CU 进行译码", active: ["IR", "CU"], from: "IR", to: "CU" },
  { phase: "decode", label: "Decode 2", description: "CU 产生控制信号，确定操作数位置", active: ["CU", "RegisterFile"], from: "CU", to: "RegisterFile" },
  { phase: "execute", label: "Execute 1", description: "寄存器操作数送入 ALU", active: ["RegisterFile", "ALU"], from: "RegisterFile", to: "ALU" },
  { phase: "execute", label: "Execute 2", description: "ALU 运算结果写回寄存器", active: ["ALU", "RegisterFile"], from: "ALU", to: "RegisterFile" },
]

const components = [
  { id: "PC", label: "PC", x: 20, y: 30, w: 55, h: 35, color: "#8b5cf6" },
  { id: "MAR", label: "MAR", x: 100, y: 30, w: 55, h: 35, color: "#ec4899" },
  { id: "Memory", label: "Memory", x: 180, y: 20, w: 65, h: 55, color: "#6366f1" },
  { id: "MDR", label: "MDR", x: 270, y: 30, w: 55, h: 35, color: "#06b6d4" },
  { id: "IR", label: "IR", x: 350, y: 30, w: 55, h: 35, color: "#f97316" },
  { id: "CU", label: "CU", x: 350, y: 110, w: 65, h: 40, color: "#f59e0b" },
  { id: "RegisterFile", label: "Reg File", x: 150, y: 110, w: 110, h: 40, color: "#10b981" },
  { id: "ALU", label: "ALU", x: 150, y: 190, w: 110, h: 45, color: "#3b82f6" },
]

const phaseColors: Record<Phase, string> = { fetch: "#3b82f6", decode: "#f59e0b", execute: "#10b981" }
const phaseLabels: Record<Phase, string> = { fetch: "取指 (Fetch)", decode: "译码 (Decode)", execute: "执行 (Execute)" }

export function CPUComponentsInteraction() {
  const [step, setStep] = useState(0)
  const [autoPlay, setAutoPlay] = useState(false)

  const current = microSteps[step]

  useEffect(() => {
    if (!autoPlay) return
    if (step >= microSteps.length - 1) { setAutoPlay(false); return }
    const timer = setTimeout(() => setStep((s) => s + 1), 2000)
    return () => clearTimeout(timer)
  }, [autoPlay, step])

  const flowFrom = components.find((c) => c.id === current.from)
  const flowTo = components.find((c) => c.id === current.to)

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <div className="flex items-center gap-2 mb-4">
        <Play className="w-5 h-5 text-accent" />
        <h3 className="text-lg font-semibold">CPU 组件交互演示</h3>
      </div>
      <div className="flex items-center gap-3 mb-4">
        <button onClick={() => { setStep(0); setAutoPlay(true) }} disabled={autoPlay}
          className="px-4 py-1.5 rounded text-sm font-medium bg-green-600 text-white hover:bg-green-700 disabled:opacity-50 flex items-center gap-1">
          <Play className="w-3.5 h-3.5" />{autoPlay ? "运行中..." : "自动运行"}
        </button>
        <button onClick={() => { setStep(0); setAutoPlay(false) }}
          className="px-4 py-1.5 rounded text-sm border border-border-subtle text-text-secondary hover:border-accent flex items-center gap-1">
          <RotateCcw className="w-3.5 h-3.5" />重置
        </button>
        <div className="ml-auto flex gap-2">
          {(["fetch", "decode", "execute"] as const).map((ph) => (
            <span key={ph} className={`px-3 py-1 rounded text-xs font-medium ${current.phase === ph ? "text-white" : "text-text-secondary bg-bg-surface"}`}
              style={current.phase === ph ? { backgroundColor: phaseColors[ph] } : {}}>
              {phaseLabels[ph]}
            </span>
          ))}
        </div>
      </div>
      <div className="flex gap-6">
        <svg viewBox="0 0 440 260" className="flex-1 min-h-[230px]">
          {flowFrom && flowTo && (
            <>
              <motion.line x1={flowFrom.x + flowFrom.w / 2} y1={flowFrom.y + flowFrom.h / 2}
                x2={flowTo.x + flowTo.w / 2} y2={flowTo.y + flowTo.h / 2}
                stroke={phaseColors[current.phase]} strokeWidth={2.5} strokeDasharray="8 4"
                initial={{ pathLength: 0, opacity: 0 }} animate={{ pathLength: 1, opacity: 1 }} transition={{ duration: 0.8 }} />
              <motion.circle r={4} fill={phaseColors[current.phase]}
                initial={{ cx: flowFrom.x + flowFrom.w / 2, cy: flowFrom.y + flowFrom.h / 2 }}
                animate={{ cx: [flowFrom.x + flowFrom.w / 2, flowTo.x + flowTo.w / 2], cy: [flowFrom.y + flowFrom.h / 2, flowTo.y + flowTo.h / 2] }}
                transition={{ duration: 1.2, repeat: Infinity, ease: "linear" }} />
            </>
          )}
          {components.map((comp) => {
            const isActive = current.active.includes(comp.id)
            return (
              <g key={comp.id}>
                <motion.rect x={comp.x} y={comp.y} width={comp.w} height={comp.h} rx={6}
                  fill={isActive ? comp.color : "transparent"} fillOpacity={isActive ? 0.2 : 0.03}
                  stroke={comp.color} strokeWidth={isActive ? 2.5 : 1}
                  animate={isActive ? { scale: [1, 1.04, 1] } : {}}
                  transition={{ duration: 0.8, repeat: isActive ? Infinity : 0 }} />
                <text x={comp.x + comp.w / 2} y={comp.y + comp.h / 2 + 1} textAnchor="middle" dominantBaseline="middle"
                  fill={isActive ? comp.color : "#6b7280"} fontSize={11} fontWeight={isActive ? "700" : "500"}>
                  {comp.label}
                </text>
              </g>
            )
          })}
          <line x1={210} y1={75} x2={210} y2={110} stroke="#374151" strokeWidth={1} />
          <line x1={375} y1={65} x2={375} y2={110} stroke="#374151" strokeWidth={1} />
          <line x1={210} y1={150} x2={210} y2={190} stroke="#374151" strokeWidth={1} />
        </svg>
        <div className="w-48 flex flex-col gap-2">
          {microSteps.map((ms, i) => (
            <motion.button key={i} onClick={() => { setStep(i); setAutoPlay(false) }}
              className={`text-left p-2 rounded text-xs border transition-all ${step === i ? "border-accent bg-accent/10" : step > i ? "border-green-600/20 bg-green-600/5" : "border-border-subtle"}`}
              animate={step === i ? { scale: [1, 1.02, 1] } : {}}
              transition={{ duration: 0.6, repeat: step === i ? Infinity : 0 }}>
              <div className="flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full" style={{ backgroundColor: step >= i ? phaseColors[ms.phase] : "#4b5563" }} />
                <span className="font-mono font-medium" style={{ color: step === i ? phaseColors[ms.phase] : step > i ? "#22c55e" : "#6b7280" }}>{ms.label}</span>
              </div>
              {step === i && <motion.p initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} className="mt-1 text-text-secondary leading-snug">{ms.description}</motion.p>}
            </motion.button>
          ))}
        </div>
      </div>
    </div>
  )
}
