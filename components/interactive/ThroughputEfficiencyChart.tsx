"use client"

import { useState, useMemo } from "react"
import { motion } from "framer-motion"
import { LineChart, TrendingUp, BarChart2, Activity } from "lucide-react"

type Mode = "speedup" | "throughput" | "efficiency"
const KS = [2, 4, 6, 8, 10, 12, 16]
const COLS = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899", "#06b6d4"]
const W = 640, H = 300, P = { t: 20, r: 20, b: 40, l: 56 }

function yVal(k: number, n: number, m: Mode) {
  if (m === "throughput") return n / (k + n - 1)
  if (m === "speedup") return (n * k) / (k + n - 1)
  return n / ((k + n - 1) * k)
}
function yLabel(m: Mode) { return m === "throughput" ? "吞吐率 (TP)" : m === "speedup" ? "加速比 (S)" : "效率 (E)" }
function yRange(m: Mode): [number, number] { return m === "throughput" ? [0, 1] : m === "speedup" ? [0, 16] : [0, 1] }

export function ThroughputEfficiencyChart() {
  const [mode, setMode] = useState<Mode>("speedup")
  const [maxN, setMaxN] = useState(40)
  const [hov, setHov] = useState<number | null>(null)
  const [yMin, yMax] = yRange(mode)
  const ns = useMemo(() => Array.from({ length: maxN }, (_, i) => i + 1), [maxN])
  const xS = (n: number) => P.l + ((n - 1) / (maxN - 1)) * (W - P.l - P.r)
  const yS = (v: number) => H - P.b - ((v - yMin) / (yMax - yMin)) * (H - P.t - P.b)
  const path = (k: number) => `M${ns.map(n => `${xS(n)},${yS(Math.min(yVal(k, n, mode), yMax))}`).join("L")}`
  const yTicks = useMemo(() => { const t: number[] = []; const s = yMax <= 1 ? 0.2 : 2; for (let v = 0; v <= yMax + .001; v += s) t.push(Math.round(v * 100) / 100); return t }, [yMax])

  const btns: { k: Mode; l: string; i: React.ReactNode }[] = [
    { k: "speedup", l: "加速比", i: <TrendingUp className="w-3.5 h-3.5" /> },
    { k: "throughput", l: "吞吐率", i: <BarChart2 className="w-3.5 h-3.5" /> },
    { k: "efficiency", l: "效率", i: <Activity className="w-3.5 h-3.5" /> },
  ]

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-lg font-bold mb-1">吞吐率 / 效率图表</h3>
      <p className="text-sm text-slate-500 dark:text-slate-400 mb-4">不同流水线深度 (k) 下，性能随任务数 n 的变化趋势</p>
      <div className="flex flex-wrap items-center gap-2 mb-5">
        {btns.map(b => (
          <button key={b.k} onClick={() => setMode(b.k)}
            className={`flex items-center gap-1.5 px-3 py-1.5 text-xs rounded-lg border transition-colors ${mode === b.k ? "bg-sky-600 text-white border-sky-600" : "bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300 border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-700"}`}>
            {b.i}{b.l}
          </button>
        ))}
        <div className="ml-auto flex items-center gap-2">
          <span className="text-xs text-slate-500">n 范围:</span>
          <input type="range" min={10} max={100} step={10} value={maxN} onChange={e => setMaxN(+e.target.value)} className="w-20 accent-sky-500" />
          <span className="text-xs font-mono text-slate-600 dark:text-slate-300 w-6">{maxN}</span>
        </div>
      </div>
      <motion.svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
        <rect x={P.l} y={P.t} width={W - P.l - P.r} height={H - P.t - P.b} className="fill-slate-50 dark:fill-slate-900/50" rx={4} />
        {yTicks.map(v => (
          <g key={v}>
            <line x1={P.l} y1={yS(v)} x2={W - P.r} y2={yS(v)} className="stroke-slate-200 dark:stroke-slate-700" strokeWidth={0.5} />
            <text x={P.l - 6} y={yS(v) + 3} textAnchor="end" className="fill-slate-400" fontSize={9}>{v.toFixed(mode === "speedup" ? 0 : 1)}</text>
          </g>
        ))}
        {[1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100].filter(v => v <= maxN).map(v => (
          <g key={v}>
            <line x1={xS(v)} y1={P.t} x2={xS(v)} y2={H - P.b} className="stroke-slate-200 dark:stroke-slate-700" strokeWidth={0.5} />
            <text x={xS(v)} y={H - P.b + 14} textAnchor="middle" className="fill-slate-400" fontSize={9}>{v}</text>
          </g>
        ))}
        <text x={W / 2} y={H - 4} textAnchor="middle" className="fill-slate-500" fontSize={10}>任务数 (n)</text>
        <text x={12} y={H / 2} textAnchor="middle" className="fill-slate-500" fontSize={10} transform={`rotate(-90,12,${H / 2})`}>{yLabel(mode)}</text>
        {KS.map((k, i) => (
          <path key={k} d={path(k)} fill="none" stroke={COLS[i]} strokeWidth={hov === k ? 3 : 1.5} strokeLinecap="round" strokeLinejoin="round"
            opacity={hov !== null && hov !== k ? 0.2 : 1} onMouseEnter={() => setHov(k)} onMouseLeave={() => setHov(null)} style={{ cursor: "pointer" }} />
        ))}
      </motion.svg>
      <div className="mt-3 flex flex-wrap gap-2">
        {KS.map((k, i) => (
          <span key={k} onMouseEnter={() => setHov(k)} onMouseLeave={() => setHov(null)}
            className={`flex items-center gap-1 text-[10px] px-2 py-0.5 rounded cursor-pointer transition-opacity ${hov !== null && hov !== k ? "opacity-30" : ""}`}>
            <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: COLS[i] }} />k={k}
          </span>
        ))}
      </div>
      <div className="mt-4 grid grid-cols-2 sm:grid-cols-4 gap-3">
        {KS.filter((_, i) => i % 2 === 0 || i === KS.length - 1).map((k, i) => {
          const v = yVal(k, maxN, mode)
          const pct = (v / yMax) * 100
          return (
            <div key={k} className="p-2 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
              <p className="text-[10px] text-slate-400 mb-1">k={k}, n={maxN}</p>
              <p className="text-lg font-bold text-slate-700 dark:text-slate-200">{v.toFixed(mode === "efficiency" ? 1 : 2)}{mode === "efficiency" ? "%" : mode === "throughput" ? "" : "x"}</p>
              <div className="h-1.5 mt-1 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
                <motion.div initial={{ width: 0 }} animate={{ width: `${Math.min(pct, 100)}%` }}
                  transition={{ duration: 0.4 }} className="h-full rounded-full" style={{ backgroundColor: COLS[KS.indexOf(k)] }} />
              </div>
            </div>
          )
        })}
      </div>
      <div className="mt-4 p-3 bg-amber-50 dark:bg-amber-900/20 rounded-lg border border-amber-200 dark:border-amber-800">
        <div className="flex items-start gap-2">
          <LineChart className="w-4 h-4 text-amber-600 dark:text-amber-400 mt-0.5 shrink-0" />
          <p className="text-xs text-amber-700 dark:text-amber-300">
            {mode === "speedup" && "增加流水线深度 k 对大任务数 n 效果显著，但 k 过大时锁存器开销增加，对小 n 反而降低性能。当 n→∞ 时，加速比趋近于 k。"}
            {mode === "throughput" && "吞吐率 = n/(k+n-1)，当 n 远大于 k 时趋近于 1，即每个周期完成一条指令。k 增大时需要更多任务才能达到高吞吐。"}
            {mode === "efficiency" && "效率衡量流水线段的利用率。k 越大、n 越小时效率越低，因为流水线填充和排空阶段占比增大。"}
          </p>
        </div>
      </div>
    </div>
  )
}
