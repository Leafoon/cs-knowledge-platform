"use client"

import { useState, useEffect, useCallback } from "react"
import { motion } from "framer-motion"
import { Play, Pause, SkipForward, RotateCcw, Zap } from "lucide-react"

const phases = [
  { name: "取指 FE", short: "FE", color: "bg-blue-500", desc: "从内存取出指令" },
  { name: "译码 DE", short: "DE", color: "bg-amber-500", desc: "解析指令操作码和地址" },
  { name: "执行 EX", short: "EX", color: "bg-green-500", desc: "执行指令操作" },
  { name: "中断 INTR", short: "INTR", color: "bg-red-500", desc: "检查并处理中断" },
]

const registerNames = ["PC", "IR", "MAR", "MDR", "AC", "SP"]

const steps = [
  { phase: 0, action: "PC → MAR", detail: "程序计数器送MAR", regs: { PC: "0x1000", MAR: "0x1000" } },
  { phase: 0, action: "M[MAR] → MDR", detail: "读取内存单元到MDR", regs: { MDR: "LOAD 0x5000" } },
  { phase: 0, action: "MDR → IR", detail: "指令送指令寄存器", regs: { IR: "LOAD 0x5000" } },
  { phase: 0, action: "PC + 1 → PC", detail: "程序计数器加1", regs: { PC: "0x1001" } },
  { phase: 1, action: "IR[opcode] → CU", detail: "操作码送控制单元", regs: {} },
  { phase: 1, action: "IR[addr] → MAR", detail: "地址码送MAR", regs: { MAR: "0x5000" } },
  { phase: 2, action: "M[MAR] → MDR", detail: "读取操作数", regs: { MDR: "0x0042" } },
  { phase: 2, action: "MDR → AC", detail: "操作数送累加器", regs: { AC: "0x0042" } },
  { phase: 3, action: "检查中断请求", detail: "无中断则继续", regs: {} },
]

export function FetchDecodeExecute() {
  const [stepIdx, setStepIdx] = useState(0)
  const [cycle, setCycle] = useState(1)
  const [playing, setPlaying] = useState(false)
  const [regs, setRegs] = useState<Record<string, string>>({ PC: "0x1000" })

  const current = steps[stepIdx]
  const currentPhase = phases[current.phase]

  const advance = useCallback(() => {
    if (stepIdx < steps.length - 1) {
      setStepIdx((s) => s + 1)
      setRegs((r) => ({ ...r, ...(steps[stepIdx + 1].regs as Record<string, string>) }))
    } else {
      setStepIdx(0)
      setRegs({ PC: "0x1000" })
      setCycle((c) => c + 1)
    }
  }, [stepIdx])

  useEffect(() => {
    if (!playing) return
    const t = setInterval(advance, 1000)
    return () => clearInterval(t)
  }, [playing, advance])

  const reset = () => {
    setStepIdx(0)
    setCycle(1)
    setRegs({ PC: "0x1000" })
    setPlaying(false)
  }

  const globalIdx = (cycle - 1) * steps.length + stepIdx

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <div className="flex items-center gap-2 mb-4">
        <Zap className="w-5 h-5 text-accent" />
        <h3 className="text-lg font-semibold">取指-译码-执行周期</h3>
      </div>

      <div className="flex gap-2 mb-4">
        <button onClick={() => setPlaying(!playing)} className="px-3 py-1.5 rounded bg-accent text-white text-sm flex items-center gap-1">
          {playing ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
          {playing ? "暂停" : "自动播放"}
        </button>
        <button onClick={advance} className="px-3 py-1.5 rounded bg-bg-card border border-border-subtle text-sm flex items-center gap-1">
          <SkipForward className="w-4 h-4" /> 单步
        </button>
        <button onClick={reset} className="px-3 py-1.5 rounded bg-bg-card border border-border-subtle text-sm flex items-center gap-1">
          <RotateCcw className="w-4 h-4" /> 重置
        </button>
      </div>

      <div className="grid grid-cols-4 gap-1 mb-4">
        {phases.map((p, i) => (
          <div key={p.name} className={`text-center py-2 rounded text-xs font-medium transition-all ${i === current.phase ? `${p.color} text-white` : "bg-bg-card border border-border-subtle text-text-secondary"}`}>
            {p.name}
          </div>
        ))}
      </div>

      <motion.div key={stepIdx} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="bg-bg-card border border-border-subtle rounded p-4 mb-4">
        <div className="flex items-center gap-2 mb-1">
          <span className={`px-2 py-0.5 rounded text-xs text-white ${currentPhase.color}`}>{currentPhase.short}</span>
          <span className="font-mono text-sm font-semibold">{current.action}</span>
        </div>
        <p className="text-sm text-text-secondary">{current.detail}</p>
      </motion.div>

      <div className="mb-4">
        <h4 className="text-sm font-medium text-text-secondary mb-2">寄存器状态</h4>
        <div className="grid grid-cols-3 gap-2">
          {registerNames.map((name) => (
            <motion.div key={name} layout className="flex items-center justify-between bg-bg-card border border-border-subtle rounded px-3 py-1.5 text-sm">
              <span className="font-mono text-text-secondary">{name}</span>
              <motion.span key={regs[name] || "empty"} initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="font-mono text-accent">
                {regs[name] || "—"}
              </motion.span>
            </motion.div>
          ))}
        </div>
      </div>

      <div className="mb-3">
        <h4 className="text-sm font-medium text-text-secondary mb-2">流水线时间线</h4>
        <div className="space-y-1">
          {Array.from({ length: cycle }, (_, ci) => (
            <div key={ci} className="flex items-center gap-1">
              <span className="text-xs text-text-secondary w-12">C{ci + 1}</span>
              <div className="flex-1 flex gap-0.5">
                {phases.map((p, pi) => {
                  const isPast = ci < cycle - 1
                  const isCurrent = ci === cycle - 1 && pi <= current.phase
                  return (
                    <div key={pi} className={`flex-1 h-4 rounded-sm transition-colors ${isPast || isCurrent ? `${p.color} opacity-${isPast ? "40" : "100"}` : "bg-bg-card"}`} style={{ opacity: isPast ? 0.4 : isCurrent ? 1 : 0.2 }} />
                  )
                })}
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="flex items-center justify-between text-xs text-text-secondary">
        <span>周期 {cycle} · 步骤 {stepIdx + 1}/{steps.length}</span>
        <span>全局步骤 {globalIdx + 1}</span>
      </div>
    </div>
  )
}
