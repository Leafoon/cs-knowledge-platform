"use client"

import { useState, useEffect, useRef, useCallback } from "react"
import { motion } from "framer-motion"
import { Play, Pause, RotateCcw, Zap, Cpu, Timer } from "lucide-react"

const N_STAGES = ["IF", "ID", "EX", "MEM", "WB"]
const S_STAGES = ["IF1", "IF2", "ID1", "ID2", "EX1", "EX2", "MEM1", "MEM2", "WB1", "WB2"]
const SC: Record<string, string> = {
  IF: "bg-sky-500", IF1: "bg-sky-400", IF2: "bg-sky-600",
  ID: "bg-emerald-500", ID1: "bg-emerald-400", ID2: "bg-emerald-600",
  EX: "bg-amber-500", EX1: "bg-amber-400", EX2: "bg-amber-600",
  MEM: "bg-violet-500", MEM1: "bg-violet-400", MEM2: "bg-violet-600",
  WB: "bg-rose-500", WB1: "bg-rose-400", WB2: "bg-rose-600",
}
const INSTRS = ["ADD R1,R2,R3", "SUB R4,R5,R6", "LW  R7,0(R1)", "OR  R8,R7,R2", "BEQ R8,R0,L"]

function build(stages: readonly string[], n: number, total: number) {
  return Array.from({ length: total }, (_, c) =>
    Array.from({ length: n }, (_, i) => c >= i && c < i + stages.length ? { stage: stages[c - i], idx: i } : null
  ).filter(Boolean) as { stage: string; idx: number }[])
}

export function SuperpipelineDemo() {
  const n = INSTRS.length
  const nCyc = n + 4, sCyc = n + 9, sFull = Math.ceil(sCyc / 2)
  const [c, setC] = useState(0), [play, setPlay] = useState(false), [spd, setSpd] = useState(600)
  const ref = useRef<ReturnType<typeof setInterval> | null>(null)
  const max = Math.max(nCyc, sCyc)
  const nGrid = build(N_STAGES, n, nCyc), sGrid = build(S_STAGES, n, sCyc)
  const reset = useCallback(() => { setPlay(false); setC(0) }, [])
  useEffect(() => {
    if (play) ref.current = setInterval(() => setC(p => { if (p >= max - 1) { setPlay(false); return p } return p + 1 }), spd)
    return () => { if (ref.current) clearInterval(ref.current) }
  }, [play, spd, max])

  const Row = ({ entries, half }: { entries: { stage: string; idx: number }[]; half?: boolean }) => (
    <div className="flex gap-0.5 flex-1">
      {entries.length ? entries.map((e, i) => (
        <span key={i} className={`inline-block px-1.5 py-0.5 text-[9px] font-bold text-white rounded ${SC[e.stage]}`}>{e.stage}</span>
      )) : <span className="text-[9px] text-slate-300 dark:text-slate-600">-</span>}
    </div>
  )

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-lg font-bold mb-1">超流水线演示</h3>
      <p className="text-sm text-slate-500 dark:text-slate-400 mb-4">对比普通流水线（1 发射/周期）与超流水线（2 发射/半周期）</p>
      <div className="flex items-center gap-3 mb-5">
        <button onClick={() => setPlay(p => !p)} disabled={c >= max - 1}
          className="flex items-center gap-1.5 px-3 py-1.5 text-sm rounded-lg bg-violet-600 text-white hover:bg-violet-700 disabled:opacity-40 transition-colors">
          {play ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}{play ? "暂停" : "播放"}
        </button>
        <button onClick={reset} className="flex items-center gap-1.5 px-3 py-1.5 text-sm rounded-lg bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-200 hover:bg-slate-300 dark:hover:bg-slate-600 transition-colors">
          <RotateCcw className="w-4 h-4" />重置
        </button>
        <div className="flex items-center gap-2 ml-2">
          <Timer className="w-4 h-4 text-slate-500" />
          <input type="range" min={200} max={1200} step={100} value={spd} onChange={e => setSpd(+e.target.value)} className="w-24 accent-violet-500" />
          <span className="text-xs text-slate-500 w-12">{spd}ms</span>
        </div>
        <span className="ml-auto text-sm font-mono font-semibold text-slate-700 dark:text-slate-300">半周期: {c + 1}</span>
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-5">
        <div className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
          <div className="flex items-center gap-2 mb-3">
            <Cpu className="w-4 h-4 text-sky-500" />
            <span className="text-sm font-semibold text-slate-700 dark:text-slate-200">普通流水线（5 级）</span>
            {c >= nCyc && <span className="ml-auto text-[10px] px-1.5 py-0.5 bg-emerald-100 dark:bg-emerald-900/40 text-emerald-700 dark:text-emerald-300 rounded">完成</span>}
          </div>
          <div className="space-y-1">
            {Array.from({ length: Math.min(c + 1, nCyc) }, (_, ci) => (
              <motion.div key={ci} initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} className="flex items-center gap-1">
                <span className="text-[10px] font-mono text-slate-400 w-5 text-right shrink-0">{ci + 1}</span>
                <Row entries={nGrid[ci]} />
              </motion.div>
            ))}
          </div>
          <p className="text-[10px] text-slate-400 mt-2">完成时间: {nCyc} 全周期</p>
        </div>
        <div className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-violet-200 dark:border-violet-800">
          <div className="flex items-center gap-2 mb-3">
            <Zap className="w-4 h-4 text-violet-500" />
            <span className="text-sm font-semibold text-slate-700 dark:text-slate-200">超流水线（10 级，半周期）</span>
            {c >= sCyc && <span className="ml-auto text-[10px] px-1.5 py-0.5 bg-emerald-100 dark:bg-emerald-900/40 text-emerald-700 dark:text-emerald-300 rounded">完成</span>}
          </div>
          <div className="space-y-1">
            {Array.from({ length: Math.min(c + 1, sCyc) }, (_, ci) => (
              <motion.div key={ci} initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} className="flex items-center gap-1">
                <span className="text-[10px] font-mono text-slate-400 w-5 text-right shrink-0">{ci % 2 === 0 ? `${ci / 2 + 1}.0` : `${(ci - 1) / 2 + 1}.5`}</span>
                <Row entries={sGrid[ci]} half />
              </motion.div>
            ))}
          </div>
          <p className="text-[10px] text-slate-400 mt-2">完成时间: {sCyc} 半周期 = {sFull} 全周期</p>
        </div>
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {[
          { label: "普通 CPI", val: "1.0", sub: "IPC = 1", cls: "bg-slate-50 dark:bg-slate-900 border-slate-200 dark:border-slate-700", tc: "text-sky-600" },
          { label: "超流水线 CPI", val: "0.5", sub: "IPC = 2", cls: "bg-violet-50 dark:bg-violet-900/30 border-violet-200 dark:border-violet-800", tc: "text-violet-600" },
          { label: "普通频率", val: "f", sub: "基准频率", cls: "bg-slate-50 dark:bg-slate-900 border-slate-200 dark:border-slate-700", tc: "text-emerald-600" },
          { label: "超流水线频率", val: "2f", sub: "时钟频率翻倍", cls: "bg-violet-50 dark:bg-violet-900/30 border-violet-200 dark:border-violet-800", tc: "text-violet-600" },
        ].map((d, i) => (
          <div key={i} className={`p-3 rounded-lg border text-center ${d.cls}`}>
            <p className="text-[10px] text-slate-400 mb-0.5">{d.label}</p>
            <p className={`text-xl font-bold ${d.tc} dark:${d.tc.replace("600", "400")}`}>{d.val}</p>
            <p className="text-[10px] text-slate-400">{d.sub}</p>
          </div>
        ))}
      </div>
      <div className="mt-4 p-3 bg-violet-50 dark:bg-violet-900/20 rounded-lg border border-violet-200 dark:border-violet-800">
        <p className="text-xs text-violet-700 dark:text-violet-300">
          超流水线将每个流水级再细分，使时钟周期减半，每个完整时钟周期可发射 2 条指令。虽然级数增加到 10 级，但每级延迟减半，CPI 从 1.0 降为 0.5，等效 IPC 提升至 2。
        </p>
      </div>
    </div>
  )
}
