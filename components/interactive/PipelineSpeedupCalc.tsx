"use client"

import { useState, useMemo } from "react"
import { motion } from "framer-motion"
import { Calculator, Zap, BarChart3, Activity } from "lucide-react"

function compute(k: number, n: number, s: number) {
  const ideal = k + n - 1
  const actual = ideal + (n - 1) * s
  const seq = n * k
  return {
    tp: n / actual, sp: seq / actual, eff: (n / (actual * k)) * 100,
    ideal, actual, seq,
    iTP: n / ideal, iSP: seq / ideal, iEff: (n / (ideal * k)) * 100,
  }
}

function Card({ title, formula, iv, av, unit, icon }: {
  title: string; formula: string; iv: string; av: string; unit: string; icon: React.ReactNode
}) {
  return (
    <div className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
      <div className="flex items-center gap-2 mb-2">{icon}<span className="text-sm font-semibold text-slate-700 dark:text-slate-200">{title}</span></div>
      <div className="text-xs font-mono text-slate-500 dark:text-slate-400 mb-3 bg-slate-50 dark:bg-slate-900 px-2 py-1 rounded">{formula}</div>
      <div className="grid grid-cols-2 gap-2">
        <div><p className="text-[10px] text-slate-400 mb-0.5">理想</p><p className="text-lg font-bold text-emerald-600 dark:text-emerald-400">{iv}<span className="text-xs font-normal text-slate-400 ml-0.5">{unit}</span></p></div>
        <div><p className="text-[10px] text-slate-400 mb-0.5">实际</p><p className="text-lg font-bold text-amber-600 dark:text-amber-400">{av}<span className="text-xs font-normal text-slate-400 ml-0.5">{unit}</span></p></div>
      </div>
    </div>
  )
}

export function PipelineSpeedupCalc() {
  const [k, setK] = useState(5)
  const [n, setN] = useState(10)
  const [s, setS] = useState(0)
  const m = useMemo(() => compute(k, n, s), [k, n, s])

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-lg font-bold mb-1">流水线加速比计算器</h3>
      <p className="text-sm text-slate-500 dark:text-slate-400 mb-5">输入流水线级数 k、任务数 n 和停顿周期数，计算性能指标</p>
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-5 mb-6">
        <div>
          <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">流水线级数 (k): <span className="font-bold text-sky-600">{k}</span></label>
          <input type="range" min={2} max={20} value={k} onChange={e => setK(+e.target.value)} className="w-full accent-sky-500" />
          <div className="flex justify-between text-[10px] text-slate-400 mt-0.5"><span>2</span><span>20</span></div>
        </div>
        <div>
          <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">任务数 (n): <span className="font-bold text-emerald-600">{n}</span></label>
          <input type="range" min={1} max={100} value={n} onChange={e => setN(+e.target.value)} className="w-full accent-emerald-500" />
          <div className="flex justify-between text-[10px] text-slate-400 mt-0.5"><span>1</span><span>100</span></div>
        </div>
        <div>
          <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">每条停顿周期: <span className="font-bold text-amber-600">{s}</span></label>
          <input type="range" min={0} max={5} value={s} onChange={e => setS(+e.target.value)} className="w-full accent-amber-500" />
          <div className="flex justify-between text-[10px] text-slate-400 mt-0.5"><span>0</span><span>5</span></div>
        </div>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-5">
        <Card title="吞吐率" formula="TP = n / (k+n-1+(n-1)·s)" iv={m.iTP.toFixed(3)} av={m.tp.toFixed(3)} unit="IPC" icon={<Zap className="w-4 h-4 text-sky-500" />} />
        <Card title="加速比" formula="S = (n·k) / (k+n-1+(n-1)·s)" iv={m.iSP.toFixed(2)} av={m.sp.toFixed(2)} unit="x" icon={<BarChart3 className="w-4 h-4 text-emerald-500" />} />
        <Card title="效率" formula="E = n / (k+n-1+(n-1)·s) / k" iv={m.iEff.toFixed(1)} av={m.eff.toFixed(1)} unit="%" icon={<Activity className="w-4 h-4 text-amber-500" />} />
      </div>
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}
        className="p-4 bg-slate-50 dark:bg-slate-900 rounded-lg border border-slate-200 dark:border-slate-700">
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-center">
          <div><p className="text-[10px] text-slate-400 mb-0.5">非流水线周期</p><p className="text-lg font-bold text-rose-600">{m.seq}</p></div>
          <div><p className="text-[10px] text-slate-400 mb-0.5">理想流水线周期</p><p className="text-lg font-bold text-emerald-600">{m.ideal}</p></div>
          <div><p className="text-[10px] text-slate-400 mb-0.5">实际流水线周期</p><p className="text-lg font-bold text-amber-600">{m.actual}</p></div>
          <div><p className="text-[10px] text-slate-400 mb-0.5">停顿周期总数</p><p className="text-lg font-bold text-red-500">{(n - 1) * s}</p></div>
        </div>
      </motion.div>
      <div className="mb-4 p-3 bg-slate-50 dark:bg-slate-900 rounded-lg border border-slate-200 dark:border-slate-700">
        <p className="text-[10px] text-slate-400 mb-2">加速比对比</p>
        <div className="space-y-2">
          <div>
            <div className="flex items-center justify-between text-xs mb-0.5">
              <span className="text-emerald-600 dark:text-emerald-400">理想加速比</span>
              <span className="font-mono font-bold text-emerald-600">{m.iSP.toFixed(2)}x</span>
            </div>
            <div className="h-3 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
              <motion.div initial={{ width: 0 }} animate={{ width: `${Math.min((m.iSP / k) * 100, 100)}%` }}
                transition={{ duration: 0.5 }} className="h-full bg-emerald-500 rounded-full" />
            </div>
          </div>
          <div>
            <div className="flex items-center justify-between text-xs mb-0.5">
              <span className="text-amber-600 dark:text-amber-400">实际加速比</span>
              <span className="font-mono font-bold text-amber-600">{m.sp.toFixed(2)}x</span>
            </div>
            <div className="h-3 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
              <motion.div initial={{ width: 0 }} animate={{ width: `${Math.min((m.sp / k) * 100, 100)}%` }}
                transition={{ duration: 0.5 }} className="h-full bg-amber-500 rounded-full" />
            </div>
          </div>
        </div>
        <p className="text-[10px] text-slate-400 mt-2">最大理论加速比: {k}x（流水线深度）</p>
      </div>
      <div className="mt-4 p-3 bg-sky-50 dark:bg-sky-900/20 rounded-lg border border-sky-200 dark:border-sky-800">
        <div className="flex items-start gap-2">
          <Calculator className="w-4 h-4 text-sky-600 dark:text-sky-400 mt-0.5 shrink-0" />
          <div className="text-xs text-sky-700 dark:text-sky-300">
            <p className="font-medium mb-1">公式说明</p>
            <p>当 k={k}, n={n}, s={s} 时：理想完成时间 {k}+{n}-1={m.ideal} 周期。非流水线需 {n}×{k}={m.seq} 周期，理想加速比 {m.seq}/{m.ideal}={m.iSP.toFixed(2)}。{s > 0 && `加入 ${s} 个停顿后，实际完成 ${m.actual} 周期。`}</p>
          </div>
        </div>
      </div>
    </div>
  )
}
