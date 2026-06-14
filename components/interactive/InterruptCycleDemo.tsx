"use client"

import { useState, useEffect, useCallback } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Play, Pause, SkipForward, RotateCcw, AlertTriangle } from "lucide-react"

interface InterruptSource {
  name: string
  vector: number
  priority: number
  color: string
}

const interrupts: InterruptSource[] = [
  { name: "打印机完成", vector: 0x04, priority: 1, color: "text-blue-400" },
  { name: "磁盘I/O", vector: 0x08, priority: 2, color: "text-green-400" },
  { name: "定时器溢出", vector: 0x0C, priority: 3, color: "text-amber-400" },
  { name: "键盘输入", vector: 0x10, priority: 4, color: "text-purple-400" },
  { name: "硬件故障", vector: 0x14, priority: 5, color: "text-red-400" },
]

const steps = [
  { label: "关中断", desc: "将中断允许标志 IF 置 0", reg: "IF = 0" },
  { label: "保存断点 (SP-1→SP)", desc: "栈指针减1，准备压栈", reg: "SP = 0xFF" },
  { label: "保存PC到栈", desc: "PC → M[SP]，保存返回地址", reg: "M[0xFF] = 0x1000" },
  { label: "保存PSW", desc: "程序状态字压栈保护", reg: "M[0xFE] = PSW" },
  { label: "查中断向量表", desc: "根据中断号查ISR入口地址", reg: "IVR → PC" },
  { label: "跳转到ISR", desc: "PC指向中断服务程序入口", reg: "PC = ISR入口" },
]

export function InterruptCycleDemo() {
  const [stepIdx, setStepIdx] = useState(-1)
  const [playing, setPlaying] = useState(false)
  const [triggered, setTriggered] = useState<InterruptSource | null>(null)
  const [pc, setPc] = useState("0x1000")
  const [sp, setSp] = useState(0x100)
  const [ifFlag, setIfFlag] = useState(true)
  const [stack, setStack] = useState<{ addr: number; val: string }[]>([])

  const advance = useCallback(() => {
    if (stepIdx >= steps.length - 1) return
    const next = stepIdx + 1
    setStepIdx(next)
    switch (next) {
      case 0: setIfFlag(false); break
      case 1: setSp((s) => s - 1); break
      case 2: setStack((s) => [{ addr: sp - 1, val: pc }, ...s]); break
      case 3: setStack((s) => [{ addr: sp - 2, val: "PSW" }, ...s]); setSp((s) => s - 1); break
      case 4: break
      case 5: if (triggered) setPc(`0x${triggered.vector.toString(16).padStart(4, "0")}`); break
    }
  }, [stepIdx, sp, pc, triggered])

  useEffect(() => {
    if (!playing) return
    const t = setInterval(advance, 1200)
    return () => clearInterval(t)
  }, [playing, advance])

  const trigger = (intr: InterruptSource) => {
    setTriggered(intr)
    setStepIdx(-1)
    setPc("0x1000")
    setSp(0x100)
    setIfFlag(true)
    setStack([])
    setPlaying(false)
  }

  const reset = () => {
    setStepIdx(-1)
    setTriggered(null)
    setPc("0x1000")
    setSp(0x100)
    setIfFlag(true)
    setStack([])
    setPlaying(false)
  }

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <div className="flex items-center gap-2 mb-4">
        <AlertTriangle className="w-5 h-5 text-accent" />
        <h3 className="text-lg font-semibold">中断周期演示</h3>
      </div>

      <div className="mb-4">
        <h4 className="text-sm font-medium text-text-secondary mb-2">触发中断源</h4>
        <div className="flex flex-wrap gap-2">
          {interrupts.map((intr) => (
            <button key={intr.name} onClick={() => trigger(intr)} className={`px-3 py-1.5 rounded text-sm border border-border-subtle bg-bg-card hover:border-accent transition-colors ${intr.color}`}>
              {intr.name} (vec:{intr.vector})
            </button>
          ))}
        </div>
      </div>

      {triggered && (
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
          <div className="flex items-center gap-2 mb-3 px-3 py-2 rounded bg-red-500/10 border border-red-500/30">
            <AlertTriangle className="w-4 h-4 text-red-400" />
            <span className="text-sm text-red-400">中断请求: {triggered.name} (向量: 0x{triggered.vector.toString(16).padStart(2, "0").toUpperCase()})</span>
          </div>

          <div className="flex gap-2 mb-4">
            <button onClick={() => setPlaying(!playing)} className="px-3 py-1.5 rounded bg-accent text-white text-sm flex items-center gap-1">
              {playing ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
              {playing ? "暂停" : "自动执行"}
            </button>
            <button onClick={advance} className="px-3 py-1.5 rounded bg-bg-card border border-border-subtle text-sm flex items-center gap-1">
              <SkipForward className="w-4 h-4" /> 单步
            </button>
            <button onClick={reset} className="px-3 py-1.5 rounded bg-bg-card border border-border-subtle text-sm flex items-center gap-1">
              <RotateCcw className="w-4 h-4" /> 重置
            </button>
          </div>

          <div className="space-y-2 mb-4">
            {steps.map((step, i) => (
              <motion.div key={i} initial={{ opacity: 0, x: -10 }} animate={{ opacity: i <= stepIdx ? 1 : 0.3, x: 0 }} className={`flex items-center gap-3 px-3 py-2 rounded text-sm ${i === stepIdx ? "bg-accent/15 border border-accent" : i < stepIdx ? "bg-bg-card/50" : "bg-bg-card border border-border-subtle"}`}>
                <span className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${i === stepIdx ? "bg-accent text-white" : i < stepIdx ? "bg-green-500 text-white" : "bg-bg-card border border-border-subtle"}`}>
                  {i < stepIdx ? "✓" : i + 1}
                </span>
                <div className="flex-1">
                  <div className="font-medium">{step.label}</div>
                  <div className="text-xs text-text-secondary">{step.desc}</div>
                </div>
                <AnimatePresence>
                  {i <= stepIdx && (
                    <motion.span initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="text-xs font-mono text-accent">{step.reg}</motion.span>
                  )}
                </AnimatePresence>
              </motion.div>
            ))}
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <h4 className="text-sm font-medium text-text-secondary mb-2">寄存器</h4>
              <div className="space-y-1 text-sm font-mono">
                <div className="flex justify-between bg-bg-card border border-border-subtle rounded px-3 py-1.5">
                  <span className="text-text-secondary">PC</span>
                  <motion.span key={pc} initial={{ color: "#22c55e" }} animate={{ color: "var(--accent)" }} className="text-accent">{pc}</motion.span>
                </div>
                <div className="flex justify-between bg-bg-card border border-border-subtle rounded px-3 py-1.5">
                  <span className="text-text-secondary">SP</span>
                  <span className="text-accent">0x{sp.toString(16).padStart(3, "0").toUpperCase()}</span>
                </div>
                <div className="flex justify-between bg-bg-card border border-border-subtle rounded px-3 py-1.5">
                  <span className="text-text-secondary">IF</span>
                  <span className={ifFlag ? "text-green-400" : "text-red-400"}>{ifFlag ? "1 (开)" : "0 (关)"}</span>
                </div>
              </div>
            </div>

            <div>
              <h4 className="text-sm font-medium text-text-secondary mb-2">栈内容</h4>
              <div className="bg-bg-card border border-border-subtle rounded p-2 min-h-[80px]">
                {stack.length === 0 ? (
                  <div className="text-xs text-text-secondary text-center py-4">栈为空</div>
                ) : (
                  stack.map((s, i) => (
                    <motion.div key={i} initial={{ opacity: 0, y: -5 }} animate={{ opacity: 1, y: 0 }} className="flex justify-between text-xs font-mono px-2 py-1 border-b border-border-subtle last:border-0">
                      <span className="text-text-secondary">0x{s.addr.toString(16).padStart(3, "0").toUpperCase()}</span>
                      <span className="text-accent">{s.val}</span>
                    </motion.div>
                  ))
                )}
              </div>
            </div>
          </div>
        </motion.div>
      )}

      {!triggered && (
        <div className="text-center py-8 text-text-secondary text-sm">点击上方按钮触发中断</div>
      )}
    </div>
  )
}
