"use client"

import { useState, useEffect, useRef, useCallback } from "react"
import { motion } from "framer-motion"
import { Play, Pause, RotateCcw, ChevronRight, Gauge } from "lucide-react"

const STAGES = ["IF", "ID", "EX", "MEM", "WB"] as const
const REGISTERS = ["IF/ID", "ID/EX", "EX/MEM", "MEM/WB"] as const
const SC: Record<string, string> = {
  IF: "bg-sky-500", ID: "bg-emerald-500", EX: "bg-amber-500", MEM: "bg-violet-500", WB: "bg-rose-500",
}
const SL: Record<string, string> = {
  IF: "bg-sky-100 dark:bg-sky-900/40", ID: "bg-emerald-100 dark:bg-emerald-900/40",
  EX: "bg-amber-100 dark:bg-amber-900/40", MEM: "bg-violet-100 dark:bg-violet-900/40",
  WB: "bg-rose-100 dark:bg-rose-900/40",
}
const INSTRS = ["ADD R1,R2,R3", "SUB R4,R1,R5", "LW  R6,0(R4)", "SW  R6,4(R1)", "BEQ R6,R0,L1", "ORI R7,R6,1", "AND R8,R7,R2"]

export function PipelineStageViz() {
  const n = 7, total = n + 4
  const [cycle, setCycle] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(800)
  const ref = useRef<ReturnType<typeof setInterval> | null>(null)
  const grid = Array.from({ length: n }, (_, i) => Array.from({ length: total }, (_, c) => c >= i && c < i + 5 ? STAGES[c - i] : null))

  const reset = useCallback(() => { setPlaying(false); setCycle(0) }, [])
  useEffect(() => {
    if (playing) ref.current = setInterval(() => { setCycle(p => { if (p >= total - 1) { setPlaying(false); return p } return p + 1 }) }, speed)
    return () => { if (ref.current) clearInterval(ref.current) }
  }, [playing, speed, total])

  const active = (s: string) => grid.some((row, i) => row[cycle] === s)
  const occupant = (s: string) => { for (let i = 0; i < n; i++) if (grid[i][cycle] === s) return INSTRS[i]; return null }

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-lg font-bold mb-1">流水线阶段可视化</h3>
      <p className="text-sm text-slate-500 dark:text-slate-400 mb-4">5 级流水线逐周期执行指令，展示每条指令在各阶段的流动</p>
      <div className="flex items-center gap-3 mb-5">
        <button onClick={() => setPlaying(p => !p)} disabled={cycle >= total - 1}
          className="flex items-center gap-1.5 px-3 py-1.5 text-sm rounded-lg bg-sky-600 text-white hover:bg-sky-700 disabled:opacity-40 transition-colors">
          {playing ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}{playing ? "暂停" : "播放"}
        </button>
        <button onClick={reset} className="flex items-center gap-1.5 px-3 py-1.5 text-sm rounded-lg bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-200 hover:bg-slate-300 dark:hover:bg-slate-600 transition-colors">
          <RotateCcw className="w-4 h-4" />重置
        </button>
        <div className="flex items-center gap-2 ml-2">
          <Gauge className="w-4 h-4 text-slate-500" />
          <input type="range" min={200} max={1500} step={100} value={speed} onChange={e => setSpeed(Number(e.target.value))} className="w-24 accent-sky-500" />
          <span className="text-xs text-slate-500 w-12">{speed}ms</span>
        </div>
        <span className="ml-auto text-sm font-mono font-semibold text-slate-700 dark:text-slate-300">周期: {cycle + 1} / {total}</span>
      </div>
      <div className="flex gap-1 mb-4">
        {STAGES.map(s => (
          <div key={s} className={`flex-1 p-2 rounded-lg border text-center transition-all ${active(s) ? `${SC[s]} text-white border-transparent shadow-md` : `${SL[s]} border-slate-200 dark:border-slate-700`}`}>
            <div className="text-xs font-bold">{s}</div>
            <div className="text-[10px] mt-0.5 truncate min-h-[14px]">{occupant(s) && <span className={active(s) ? "text-white/90" : "text-slate-600 dark:text-slate-400"}>{occupant(s)}</span>}</div>
          </div>
        ))}
      </div>
      <div className="flex items-center gap-0.5 mb-4">
        {REGISTERS.map((r, i) => (
          <div key={r} className="flex items-center">
            <div className="px-2 py-1 text-[10px] font-mono bg-slate-100 dark:bg-slate-800 rounded border border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-400">{r}</div>
            {i < REGISTERS.length - 1 && <ChevronRight className="w-3 h-3 text-slate-400 mx-0.5" />}
          </div>
        ))}
      </div>
      <div className="overflow-x-auto">
        <table className="w-full border-collapse text-xs">
          <thead>
            <tr>
              <th className="text-left p-1.5 border-b border-slate-200 dark:border-slate-700 text-slate-500 font-medium w-[120px]">指令</th>
              {Array.from({ length: total }, (_, c) => (
                <th key={c} className={`p-1.5 border-b text-center font-mono min-w-[36px] ${c === cycle ? "border-sky-500 text-sky-600 dark:text-sky-400 font-bold" : "border-slate-200 dark:border-slate-700 text-slate-500"}`}>C{c + 1}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {INSTRS.map((instr, i) => (
              <tr key={i}>
                <td className="p-1.5 border-b border-slate-200 dark:border-slate-700 font-mono text-slate-700 dark:text-slate-300 whitespace-nowrap">{instr}</td>
                {Array.from({ length: total }, (_, c) => {
                  const st = c <= cycle ? grid[i][c] : null
                  return (
                    <td key={c} className={`p-1.5 border-b border-slate-100 dark:border-slate-800 text-center ${c === cycle ? "bg-sky-50/50 dark:bg-sky-900/20" : ""}`}>
                      {st && <motion.span initial={{ scale: 0.6, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} className={`inline-block w-7 h-5 leading-5 rounded text-[10px] font-bold text-white ${SC[st]}`}>{st}</motion.span>}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="mt-3 flex flex-wrap gap-2">
        {STAGES.map(s => <span key={s} className="flex items-center gap-1 text-[10px] text-slate-500"><span className={`w-3 h-3 rounded ${SC[s]}`} />{s}</span>)}
      </div>
    </div>
  )
}
