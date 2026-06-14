'use client'
import React, { useState, useMemo } from 'react'

const ALGORITHMS = [
  {
    id: 'floyd',
    name: 'Floyd-Warshall',
    formula: 'Θ(V³)',
    color: '#6366f1',
    dark: '#818cf8',
    compute: (V: number, _E: number) => V * V * V,
    note: '不依赖 E，始终立方',
    pros: '负权边 ✓ · 代码极简 · 常数小',
    cons: '稠密/稀疏图相同 · V>500 实际太慢',
  },
  {
    id: 'johnson',
    name: 'Johnson',
    formula: 'O(VE + V²logV)',
    color: '#10b981',
    dark: '#34d399',
    compute: (V: number, E: number) => V * E + V * V * Math.log2(V),
    note: '稀疏图优势显著',
    pros: '负权边 ✓ · 稀疏图极快 · 实用算法',
    cons: '实现较复杂 · 需跑一次 BF',
  },
  {
    id: 'vdijkstra',
    name: 'V × Dijkstra',
    formula: 'O((V+E)V logV)',
    color: '#f59e0b',
    dark: '#fbbf24',
    compute: (V: number, E: number) => (V + E) * V * Math.log2(V),
    note: '无负权时无需 BF 预处理',
    pros: '无负权时实现简单 · 稀疏图可接受',
    cons: '不支持负权边 · 比 Johnson 多常数',
  },
  {
    id: 'repSquare',
    name: '(min,+) 重复平方',
    formula: 'O(V³ logV)',
    color: '#f87171',
    dark: '#f87171',
    compute: (V: number, _E: number) => V * V * V * Math.log2(V),
    note: '比 Floyd 多 logV 因子',
    pros: '理论意义 · 推广到代数半环',
    cons: '实践慢于 Floyd · 常数大',
  },
]

const DENSITY_PRESETS = [
  { label: '树/极稀疏', key: 'tree', eRatio: 1, eDesc: 'E ≈ V' },
  { label: '稀疏图', key: 'sparse', eRatio: 3, eDesc: 'E ≈ 3V' },
  { label: '中等稠密', key: 'medium', eRatio: 10, eDesc: 'E ≈ V^1.5' },
  { label: '稠密图', key: 'dense', eRatio: -1, eDesc: 'E ≈ V²/4' },
]

const SI = (n: number) => {
  if (n >= 1e18) return (n / 1e18).toFixed(1) + 'E'
  if (n >= 1e15) return (n / 1e15).toFixed(1) + 'P'
  if (n >= 1e12) return (n / 1e12).toFixed(1) + 'T'
  if (n >= 1e9)  return (n / 1e9).toFixed(1) + 'G'
  if (n >= 1e6)  return (n / 1e6).toFixed(1) + 'M'
  if (n >= 1e3)  return (n / 1e3).toFixed(1) + 'K'
  return Math.round(n).toString()
}

export default function APSPComplexityComparison() {
  const [V, setV] = useState(200)
  const [densityKey, setDensityKey] = useState('sparse')

  const density = DENSITY_PRESETS.find(d => d.key === densityKey)!
  const E = density.eRatio < 0
    ? Math.floor(V * V / 4)
    : Math.floor(density.eRatio * V)

  const values = useMemo(() =>
    ALGORITHMS.map(a => ({ ...a, val: a.compute(V, E) })),
    [V, E]
  )

  const maxVal = Math.max(...values.map(v => v.val))

  // Bar chart scale
  const chartH = 160

  return (
    <div className="w-full max-w-3xl mx-auto rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 shadow-xl overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-slate-700 via-slate-800 to-slate-900 px-5 py-4">
        <h3 className="text-white font-bold text-lg tracking-tight">APSP 算法复杂度对比</h3>
        <p className="text-slate-400 text-sm mt-0.5">拖动滑块调整参数，实时观察各算法计算量差异</p>
      </div>

      <div className="p-4 space-y-4">
        {/* Controls */}
        <div className="flex flex-col sm:flex-row gap-4">
          <div className="flex-1 space-y-2">
            <div className="flex items-center justify-between text-sm">
              <span className="font-bold text-slate-700 dark:text-slate-200">顶点数 V</span>
              <span className="font-mono font-black text-slate-800 dark:text-white text-lg">{V}</span>
            </div>
            <input type="range" min={10} max={1000} step={10} value={V}
              onChange={e => setV(Number(e.target.value))}
              className="w-full h-2 accent-slate-600 cursor-pointer" />
            <div className="flex justify-between text-[10px] text-slate-400">
              <span>10</span><span>500</span><span>1000</span>
            </div>
          </div>

          <div className="w-px bg-slate-200 dark:bg-slate-700 hidden sm:block" />

          <div className="flex-1 space-y-2">
            <div className="text-sm font-bold text-slate-700 dark:text-slate-200">图的稠密度</div>
            <div className="grid grid-cols-2 gap-1.5">
              {DENSITY_PRESETS.map(d => (
                <button key={d.key} onClick={() => setDensityKey(d.key)}
                  className={`py-1.5 rounded-lg text-[11px] font-bold transition-all ${
                    densityKey === d.key
                      ? 'bg-slate-800 dark:bg-slate-200 text-white dark:text-slate-900 shadow'
                      : 'bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-400 hover:bg-slate-200 dark:hover:bg-slate-700'
                  }`}>
                  {d.label}
                  <div className="font-normal opacity-70 text-[10px] mt-0.5">{d.eDesc}</div>
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Current parameters */}
        <div className="flex gap-3 text-sm">
          <div className="flex-1 rounded-xl bg-slate-50 dark:bg-slate-800/50 border border-slate-200 dark:border-slate-700 px-3 py-2">
            <div className="text-xs text-slate-400">当前参数</div>
            <div className="font-mono font-black text-slate-700 dark:text-slate-200">
              V = {V.toLocaleString()}，E = {E.toLocaleString()}
            </div>
          </div>
          <div className="flex-1 rounded-xl bg-slate-50 dark:bg-slate-800/50 border border-slate-200 dark:border-slate-700 px-3 py-2">
            <div className="text-xs text-slate-400">图的密度</div>
            <div className="font-mono font-bold text-slate-700 dark:text-slate-200">
              E/V = {(E / V).toFixed(1)}
              <span className="text-xs font-normal text-slate-400 ml-1">
                ({E.toLocaleString()} 条边)
              </span>
            </div>
          </div>
        </div>

        {/* Bar chart */}
        <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/30 p-4">
          <div className="text-[11px] text-slate-500 dark:text-slate-400 font-bold uppercase tracking-wide mb-3">
            相对计算量（对数尺度）
          </div>
          <div className="flex items-end gap-3">
            {values.map(a => {
              const logVal = Math.log10(Math.max(1, a.val))
              const logMax = Math.log10(Math.max(1, maxVal))
              const barH = Math.max(4, (logVal / logMax) * chartH)
              const isMin = a.val === Math.min(...values.map(v => v.val))

              return (
                <div key={a.id} className="flex-1 flex flex-col items-center gap-1.5">
                  <div className="font-mono text-[10px] font-bold text-slate-600 dark:text-slate-300">
                    {SI(a.val)}
                  </div>
                  <div className="relative w-full flex items-end justify-center"
                    style={{ height: `${chartH}px` }}>
                    <div className="w-full rounded-t-lg transition-all duration-500 relative"
                      style={{ height: `${barH}px`, background: `linear-gradient(to top, ${a.color}cc, ${a.color})` }}>
                      {isMin && (
                        <div className="absolute -top-5 left-0 right-0 flex justify-center">
                          <span className="text-[9px] font-bold text-emerald-600 dark:text-emerald-400 bg-emerald-50 dark:bg-emerald-900/40 px-1.5 py-0.5 rounded-full border border-emerald-200 dark:border-emerald-700/50">
                            最优 ⭐
                          </span>
                        </div>
                      )}
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="text-[10px] font-bold text-slate-700 dark:text-slate-200 leading-tight">
                      {a.name.split(' ').map((w, i) => <span key={i} className="block">{w}</span>)}
                    </div>
                    <div className="text-[9px] text-slate-400 dark:text-slate-500 font-mono mt-0.5">{a.formula}</div>
                  </div>
                </div>
              )
            })}
          </div>
        </div>

        {/* Algorithm cards */}
        <div className="grid grid-cols-2 gap-2">
          {values.map(a => {
            const rank = [...values].sort((x, y) => x.val - y.val).findIndex(x => x.id === a.id) + 1
            return (
              <div key={a.id}
                className="rounded-xl border border-slate-200 dark:border-slate-700 p-3 space-y-1.5 transition-all hover:border-slate-300 dark:hover:border-slate-600">
                <div className="flex items-start justify-between gap-1">
                  <span className="text-sm font-bold text-slate-800 dark:text-slate-100">{a.name}</span>
                  <span className="text-[10px] font-bold px-1.5 py-0.5 rounded-full"
                    style={{ color: a.color, background: a.color + '18' }}>
                    #{rank}
                  </span>
                </div>
                <div className="font-mono text-xs font-bold"
                  style={{ color: a.color }}>
                  {a.formula}
                </div>
                <div className="flex items-center gap-1">
                  <div className="flex-1 bg-slate-200 dark:bg-slate-700 rounded-full h-1">
                    <div className="h-1 rounded-full transition-all duration-500"
                      style={{
                        width: `${100 - ((rank - 1) / 3) * 100}%`,
                        background: a.color,
                      }} />
                  </div>
                  <span className="font-mono text-[10px] text-slate-500">{SI(a.val)}</span>
                </div>
                <div className="text-[10px] text-slate-500 dark:text-slate-400">{a.note}</div>
              </div>
            )
          })}
        </div>

        {/* Recommendation */}
        <div className="rounded-xl bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-700/40 px-4 py-3 text-xs text-amber-800 dark:text-amber-300">
          <strong>当前推荐：</strong>
          {(() => {
            const best = [...values].sort((a, b) => a.val - b.val)[0]
            if (best.id === 'floyd') return ' Floyd-Warshall — 当前参数下 V³ 复杂度领先，适合中小规模稠密图。'
            if (best.id === 'johnson') return ' Johnson 算法 — 稀疏图条件下显著优于 Floyd，实际工程首选（含负权时）。'
            if (best.id === 'vdijkstra') return ' V × Dijkstra — 无负权边时比 Johnson 更简单直接，节省 BF 预处理开销。'
            return ' (min,+) 重复平方 — 在极端大 V 小常数场景下有理论优势（实践中较少）。'
          })()}
        </div>
      </div>
    </div>
  )
}
