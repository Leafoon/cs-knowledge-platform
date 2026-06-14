"use client"

import { useState, useEffect, useCallback } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Play, Pause, RotateCcw, HardDrive } from "lucide-react"

type Mode = "read" | "write"

interface TimingSignal {
  name: string
  ticks: (string | null)[]
}

const readSignals: TimingSignal[] = [
  { name: "CLK", ticks: ["↑", "↓", "↑", "↓", "↑", "↓"] },
  { name: "MAR", ticks: [null, "有效", "有效", null, null, null] },
  { name: "CS̄", ticks: [null, null, "低", "低", null, null] },
  { name: "OĒ", ticks: [null, null, "低", "低", null, null] },
  { name: "MDR", ticks: [null, null, null, null, "有效", "有效"] },
  { name: "Data Bus", ticks: [null, null, null, "数据", "数据", null] },
]

const writeSignals: TimingSignal[] = [
  { name: "CLK", ticks: ["↑", "↓", "↑", "↓", "↑", "↓"] },
  { name: "MAR", ticks: [null, "有效", "有效", null, null, null] },
  { name: "CS̄", ticks: [null, null, "低", "低", null, null] },
  { name: "WĒ", ticks: [null, null, null, "低", "低", null] },
  { name: "MDR", ticks: [null, null, "数据", "数据", null, null] },
  { name: "Data Bus", ticks: [null, null, null, "数据", "数据", null] },
]

const memoryCells = Array.from({ length: 16 }, (_, i) => ({
  addr: i.toString(16).padStart(4, "0").toUpperCase(),
  val: Math.floor(Math.random() * 256).toString(16).padStart(2, "0").toUpperCase(),
}))

export function MemoryAccessCycle() {
  const [mode, setMode] = useState<Mode>("read")
  const [tick, setTick] = useState(-1)
  const [playing, setPlaying] = useState(false)
  const [selectedAddr, setSelectedAddr] = useState(0)
  const totalTicks = 6

  const signals = mode === "read" ? readSignals : writeSignals

  const advance = useCallback(() => {
    setTick((t) => (t < totalTicks - 1 ? t + 1 : -1))
  }, [])

  useEffect(() => {
    if (!playing) return
    const t = setInterval(advance, 900)
    return () => clearInterval(t)
  }, [playing, advance])

  const reset = () => {
    setTick(-1)
    setPlaying(false)
  }

  const getPhaseLabel = () => {
    if (tick < 0) return "就绪"
    if (tick <= 1) return "地址建立"
    if (tick <= 3) return mode === "read" ? "存储体读取" : "存储体写入"
    return "数据输出"
  }

  const memoryActive = tick >= 1 && tick <= 3
  const dataOnBus = tick >= 3 && tick <= 4

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <div className="flex items-center gap-2 mb-4">
        <HardDrive className="w-5 h-5 text-accent" />
        <h3 className="text-lg font-semibold">访存周期演示</h3>
      </div>

      <div className="flex items-center gap-2 mb-4">
        <div className="flex rounded overflow-hidden border border-border-subtle">
          {(["read", "write"] as Mode[]).map((m) => (
            <button key={m} onClick={() => { setMode(m); reset() }} className={`px-3 py-1.5 text-sm ${mode === m ? "bg-accent text-white" : "bg-bg-card text-text-secondary"}`}>
              {m === "read" ? "读周期" : "写周期"}
            </button>
          ))}
        </div>
        <button onClick={() => setPlaying(!playing)} className="px-3 py-1.5 rounded bg-accent text-white text-sm flex items-center gap-1">
          {playing ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
        </button>
        <button onClick={advance} className="px-3 py-1.5 rounded bg-bg-card border border-border-subtle text-sm">单步</button>
        <button onClick={reset} className="px-3 py-1.5 rounded bg-bg-card border border-border-subtle text-sm flex items-center gap-1">
          <RotateCcw className="w-4 h-4" />
        </button>
      </div>

      <div className="flex items-center gap-4 mb-4">
        <motion.div layout className={`px-4 py-3 rounded border text-center font-mono text-sm ${tick >= 0 ? "border-accent bg-accent/10" : "border-border-subtle bg-bg-card"}`}>
          <div className="text-xs text-text-secondary mb-1">MAR</div>
          <div>{tick >= 0 ? memoryCells[selectedAddr].addr : "—"}</div>
        </motion.div>

        <motion.div animate={{ opacity: memoryActive ? 1 : 0.3 }} className="flex-1 flex items-center justify-center">
          <div className={`relative w-28 h-20 rounded border-2 flex items-center justify-center font-mono text-sm ${memoryActive ? "border-amber-500 bg-amber-500/10" : "border-border-subtle bg-bg-card"}`}>
            <div className="text-xs absolute top-1 text-text-secondary">存储体</div>
            {memoryActive && (
              <motion.div initial={{ scale: 0 }} animate={{ scale: 1 }} className="font-bold text-amber-500">
                {memoryCells[selectedAddr].val}
              </motion.div>
            )}
          </div>
        </motion.div>

        <motion.div layout className={`px-4 py-3 rounded border text-center font-mono text-sm ${dataOnBus ? "border-green-500 bg-green-500/10" : "border-border-subtle bg-bg-card"}`}>
          <div className="text-xs text-text-secondary mb-1">MDR</div>
          <div>{dataOnBus ? memoryCells[selectedAddr].val : "—"}</div>
        </motion.div>
      </div>

      <AnimatePresence>
        {dataOnBus && (
          <motion.div initial={{ scaleX: 0 }} animate={{ scaleX: 1 }} exit={{ scaleX: 0 }} className="h-1 bg-green-500 rounded mb-4 mx-8" />
        )}
      </AnimatePresence>

      <div className="mb-4">
        <h4 className="text-sm font-medium text-text-secondary mb-2">时序图 - {getPhaseLabel()}</h4>
        <div className="bg-bg-card border border-border-subtle rounded overflow-hidden">
          <table className="w-full text-xs font-mono">
            <thead>
              <tr className="border-b border-border-subtle">
                <th className="px-2 py-1.5 text-left text-text-secondary">信号</th>
                {Array.from({ length: totalTicks }, (_, i) => (
                  <th key={i} className={`px-2 py-1.5 text-center ${i === tick ? "bg-accent/20 text-accent" : "text-text-secondary"}`}>T{i + 1}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {signals.map((sig) => (
                <tr key={sig.name} className="border-b border-border-subtle last:border-0">
                  <td className="px-2 py-1.5 text-text-secondary">{sig.name}</td>
                  {sig.ticks.map((v, ti) => (
                    <td key={ti} className={`px-2 py-1.5 text-center ${ti === tick ? "bg-accent/10" : ""}`}>
                      {v && <motion.span initial={{ opacity: 0 }} animate={{ opacity: 1 }} className={v === "低" || v === "有效" || v === "数据" ? "text-green-500 font-bold" : ""}>{v}</motion.span>}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div>
        <h4 className="text-sm font-medium text-text-secondary mb-2">内存单元</h4>
        <div className="grid grid-cols-4 md:grid-cols-8 gap-1">
          {memoryCells.map((cell, i) => (
            <button key={i} onClick={() => setSelectedAddr(i)} className={`px-2 py-1 rounded text-xs font-mono border transition-colors ${i === selectedAddr ? "border-accent bg-accent/10 text-accent" : "border-border-subtle bg-bg-card text-text-secondary hover:border-accent/50"}`}>
              {cell.addr}:{cell.val}
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}
