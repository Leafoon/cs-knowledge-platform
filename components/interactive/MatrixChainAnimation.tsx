'use client'
import { useState, useEffect } from 'react'

/* ─── 矩阵链乘法 DP ────────────────────────────────────────── */
interface Step {
  len: number; i: number; j: number; k: number
  cost: number; leftCost: number; rightCost: number; multCost: number
  isBest: boolean; prevBest: number
}

function computeMC(p: number[]): { dp: number[][]; split: number[][]; steps: Step[] } {
  const n = p.length - 1
  const INF = 1e9
  const dp: number[][] = Array.from({ length: n + 1 }, () => new Array(n + 1).fill(INF))
  const split: number[][] = Array.from({ length: n + 1 }, () => new Array(n + 1).fill(0))
  for (let i = 1; i <= n; i++) dp[i][i] = 0
  const steps: Step[] = []
  // replay local dp as we build steps
  const ld: number[][] = Array.from({ length: n + 1 }, () => new Array(n + 1).fill(INF))
  for (let i = 1; i <= n; i++) ld[i][i] = 0
  for (let len = 2; len <= n; len++) {
    for (let i = 1; i <= n - len + 1; i++) {
      const j = i + len - 1
      let best = INF
      for (let k = i; k < j; k++) {
        const mc = p[i - 1] * p[k] * p[j]
        const cost = ld[i][k] + ld[k + 1][j] + mc
        const isBest = cost < best
        steps.push({ len, i, j, k, cost, leftCost: ld[i][k], rightCost: ld[k + 1][j], multCost: mc, isBest, prevBest: best })
        if (isBest) { best = cost; split[i][j] = k }
      }
      ld[i][j] = best
      dp[i][j] = best
    }
  }
  return { dp, split, steps }
}

function buildStepDp(steps: Step[], upTo: number, n: number): { val: number; visited: boolean }[][] {
  const dp: { val: number; visited: boolean }[][] = Array.from({ length: n + 1 }, () =>
    Array.from({ length: n + 1 }, () => ({ val: 1e9, visited: false }))
  )
  for (let i = 1; i <= n; i++) { dp[i][i] = { val: 0, visited: true } }
  for (let si = 0; si <= upTo; si++) {
    const st = steps[si]
    if (st.cost < dp[st.i][st.j].val) dp[st.i][st.j] = { val: st.cost, visited: true }
    dp[st.i][st.j].visited = true
  }
  return dp
}

function buildParens(split: number[][], i: number, j: number, names: string[]): string {
  if (i === j) return names[i - 1]
  const k = split[i][j]
  return `(${buildParens(split, i, k, names)}×${buildParens(split, k + 1, j, names)})`
}

const PRESETS = [
  { label: '4矩阵', p: [30, 35, 15, 5, 10], names: ['A₁','A₂','A₃','A₄'] },
  { label: '3矩阵', p: [10, 100, 5, 50],    names: ['A₁','A₂','A₃'] },
  { label: '5矩阵', p: [5, 10, 3, 12, 5, 50], names: ['A₁','A₂','A₃','A₄','A₅'] },
]

export default function MatrixChainAnimation() {
  const [preIdx, setPreIdx] = useState(0)
  const [step, setStep] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(600)

  const { p, names } = PRESETS[preIdx]
  const n = p.length - 1
  const { dp, split, steps } = computeMC(p)
  const maxStep = steps.length - 1

  useEffect(() => { setStep(0); setPlaying(false) }, [preIdx])
  useEffect(() => {
    if (!playing) return
    if (step >= maxStep) { setPlaying(false); return }
    const id = setTimeout(() => setStep(s => s + 1), speed)
    return () => clearTimeout(id)
  }, [playing, step, maxStep, speed])

  const cur = steps[step]
  const curDp = buildStepDp(steps, step, n)
  const isDone = step >= maxStep

  const fmt = (v: number) => v >= 1e9 ? '∞' : v.toLocaleString()

  function cellBg(i: number, j: number) {
    if (i > j) return 'bg-slate-50 dark:bg-zinc-900 opacity-30'
    if (i === j) return 'bg-emerald-50 dark:bg-emerald-950/60 text-emerald-700 dark:text-emerald-300 border-emerald-300 dark:border-emerald-700'
    const { visited } = curDp[i][j]
    if (cur && i === cur.i && j === cur.j)
      return 'bg-amber-400 dark:bg-amber-500 text-white font-bold ring-2 ring-amber-300 dark:ring-amber-400 shadow-lg z-10 scale-105'
    if (cur && ((i === cur.i && j === cur.k) || (i === cur.k + 1 && j === cur.j)))
      return 'bg-sky-100 dark:bg-sky-900/50 text-sky-800 dark:text-sky-200 border-sky-300 dark:border-sky-700'
    if (visited)
      return 'bg-purple-50 dark:bg-purple-950/50 text-purple-800 dark:text-purple-200 border-purple-200 dark:border-purple-800'
    return 'bg-white dark:bg-zinc-800 text-slate-300 dark:text-zinc-600 border-slate-200 dark:border-zinc-700'
  }

  return (
    <div className="rounded-2xl border border-slate-200 dark:border-zinc-800 bg-white dark:bg-zinc-950 overflow-hidden shadow-sm">
      <div className="px-6 py-4 bg-gradient-to-r from-purple-600 to-violet-600">
        <h3 className="text-white font-bold text-base">矩阵链乘法 — 区间 DP 填表动画</h3>
        <p className="text-purple-200 text-sm mt-0.5">按区间长度从小到大枚举，高亮分割点 k 与依赖格</p>
      </div>

      <div className="p-5 space-y-4">
        {/* 顶部控制栏 */}
        <div className="flex flex-wrap items-center gap-2">
          <div className="flex gap-1">
            {PRESETS.map((pr, idx) => (
              <button key={idx} onClick={() => setPreIdx(idx)}
                className={`px-3 py-1 rounded-lg text-xs font-semibold transition-all ${preIdx === idx ? 'bg-purple-600 text-white shadow' : 'bg-slate-100 dark:bg-zinc-800 text-slate-600 dark:text-zinc-300 hover:bg-slate-200 dark:hover:bg-zinc-700'}`}>
                {pr.label}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-1.5 ml-auto">
            <button onClick={() => setStep(s => Math.max(0, s - 1))} disabled={step === 0}
              className="px-2.5 py-1.5 text-xs bg-slate-100 dark:bg-zinc-800 hover:bg-slate-200 dark:hover:bg-zinc-700 disabled:opacity-40 rounded-lg text-slate-700 dark:text-zinc-200">←</button>
            <button onClick={() => setPlaying(p => !p)}
              className={`px-3 py-1.5 text-xs rounded-lg text-white font-medium ${playing ? 'bg-orange-500' : 'bg-purple-600 hover:bg-purple-500'}`}>
              {playing ? '⏸ 暂停' : '▶ 播放'}
            </button>
            <button onClick={() => setStep(s => Math.min(maxStep, s + 1))} disabled={step >= maxStep}
              className="px-2.5 py-1.5 text-xs bg-slate-100 dark:bg-zinc-800 hover:bg-slate-200 dark:hover:bg-zinc-700 disabled:opacity-40 rounded-lg text-slate-700 dark:text-zinc-200">→</button>
            <button onClick={() => { setStep(0); setPlaying(false) }}
              className="px-2.5 py-1.5 text-xs bg-slate-100 dark:bg-zinc-800 hover:bg-slate-200 dark:hover:bg-zinc-700 rounded-lg text-slate-700 dark:text-zinc-200">↺</button>
          </div>
          <div className="flex items-center gap-1 border-l border-slate-200 dark:border-zinc-700 pl-2">
            {([['慢', 900], ['中', 600], ['快', 250]] as [string, number][]).map(([label, ms]) => (
              <button key={ms} onClick={() => setSpeed(ms)}
                className={`px-2 py-1 text-xs rounded ${speed === ms ? 'bg-purple-600 text-white' : 'bg-slate-100 dark:bg-zinc-800 text-slate-500 dark:text-zinc-400'}`}>
                {label}
              </button>
            ))}
          </div>
        </div>

        {/* 矩阵维度标签 */}
        <div className="flex flex-wrap gap-2">
          {names.map((name, i) => (
            <div key={i} className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-purple-50 dark:bg-purple-950/40 border border-purple-200 dark:border-purple-800">
              <span className="text-purple-700 dark:text-purple-300 font-bold text-sm">{name}</span>
              <span className="text-purple-400 dark:text-purple-500 text-xs">{p[i]}×{p[i+1]}</span>
            </div>
          ))}
        </div>

        {/* DP 表 + 步骤说明 */}
        <div className="flex flex-col lg:flex-row gap-4 items-start">
          {/* DP 表格 */}
          <div className="overflow-auto flex-1">
            <div className="inline-block">
              <div className="flex">
                <div className="w-10 h-10" />
                {Array.from({ length: n }, (_, j) => (
                  <div key={j} className="w-16 h-8 flex items-center justify-center text-xs font-bold text-slate-400 dark:text-zinc-500">j={j+1}</div>
                ))}
              </div>
              {Array.from({ length: n }, (_, row) => {
                const i = row + 1
                return (
                  <div key={i} className="flex items-center">
                    <div className="w-10 flex items-center justify-center text-xs font-bold text-slate-400 dark:text-zinc-500">i={i}</div>
                    {Array.from({ length: n }, (_, col) => {
                      const j = col + 1
                      const { val } = curDp[i][j]
                      return (
                        <div key={j}
                          className={`w-16 h-16 m-0.5 flex flex-col items-center justify-center border rounded-xl text-xs transition-all duration-200 ${cellBg(i, j)}`}>
                          {i <= j && (
                            <>
                              <span className={`font-bold ${i === j ? 'text-base' : 'text-sm'}`}>
                                {i === j ? '0' : val < 1e9 ? fmt(val) : '?'}
                              </span>
                              {i < j && cur && i === cur.i && j === cur.j && (
                                <span className="text-[9px] mt-0.5 opacity-80">k={cur.k}</span>
                              )}
                            </>
                          )}
                        </div>
                      )
                    })}
                  </div>
                )
              })}
            </div>
          </div>

          {/* 右侧：图例 + 当前步骤 */}
          <div className="lg:w-52 space-y-3 flex-shrink-0">
            {/* 图例 */}
            <div className="rounded-xl border border-slate-200 dark:border-zinc-700 bg-slate-50 dark:bg-zinc-900 p-3 space-y-2">
              <p className="text-[10px] font-bold uppercase tracking-widest text-slate-400 dark:text-zinc-500">图例</p>
              {[
                { cls: 'bg-amber-400', label: '当前 dp[i][j]' },
                { cls: 'bg-sky-200 dark:bg-sky-800', label: '依赖格 dp[i][k], dp[k+1][j]' },
                { cls: 'bg-purple-100 dark:bg-purple-900', label: '已完成格' },
                { cls: 'bg-emerald-100 dark:bg-emerald-900', label: 'dp[i][i] = 0' },
              ].map(({ cls, label }) => (
                <div key={label} className="flex items-center gap-2 text-xs text-slate-600 dark:text-zinc-300">
                  <div className={`w-3 h-3 rounded flex-shrink-0 ${cls}`} />
                  {label}
                </div>
              ))}
            </div>

            {/* 当前步骤计算板 */}
            {cur && (
              <div className={`rounded-xl border p-3 text-xs font-mono space-y-1 ${cur.isBest ? 'bg-emerald-50 dark:bg-emerald-950/40 border-emerald-200 dark:border-emerald-800' : 'bg-slate-50 dark:bg-zinc-900 border-slate-200 dark:border-zinc-700'}`}>
                <p className="font-bold text-slate-600 dark:text-zinc-200 not-italic font-sans">步骤 {step+1}/{maxStep+1}</p>
                <div className="text-slate-600 dark:text-zinc-300 space-y-0.5">
                  <p>len={cur.len}  i={cur.i}  j={cur.j}</p>
                  <p>k = {cur.k}</p>
                  <p className="border-t border-slate-200 dark:border-zinc-700 pt-1">
                    dp[{cur.i}][{cur.k}]&nbsp;= {fmt(cur.leftCost)}
                  </p>
                  <p>dp[{cur.k+1}][{cur.j}]= {fmt(cur.rightCost)}</p>
                  <p>p[{cur.i-1}]·p[{cur.k}]·p[{cur.j}]</p>
                  <p className="pl-4">={p[cur.i-1]}·{p[cur.k]}·{p[cur.j]}={cur.multCost}</p>
                  <p className={`border-t border-slate-200 dark:border-zinc-700 pt-1 font-bold ${cur.isBest ? 'text-emerald-600 dark:text-emerald-400' : 'text-slate-500 dark:text-zinc-500'}`}>
                    = {fmt(cur.cost)} {cur.isBest ? '✓ 更优' : '✗ 不优'}
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* 完成横幅 */}
        {isDone && (
          <div className="rounded-xl border border-purple-200 dark:border-purple-800 bg-purple-50 dark:bg-purple-950/40 p-4 flex flex-wrap gap-4 items-center">
            <div>
              <p className="text-xs text-purple-400 dark:text-purple-500 mb-0.5">最优括号化</p>
              <p className="font-mono font-bold text-purple-700 dark:text-purple-300">{buildParens(split, 1, n, names)}</p>
            </div>
            <div className="h-8 w-px bg-purple-200 dark:bg-purple-800" />
            <div>
              <p className="text-xs text-purple-400 dark:text-purple-500 mb-0.5">最少乘法次数</p>
              <p className="font-mono font-bold text-xl text-purple-700 dark:text-purple-300">{dp[1][n].toLocaleString()}</p>
            </div>
            <p className="ml-auto text-xs text-purple-400 dark:text-purple-500">共 {steps.length} 次 k 枚举</p>
          </div>
        )}

        {/* 进度条 */}
        <div className="w-full h-1.5 bg-slate-100 dark:bg-zinc-800 rounded-full overflow-hidden">
          <div className="h-full bg-purple-500 rounded-full transition-all duration-150"
            style={{ width: `${((step + 1) / (maxStep + 1)) * 100}%` }} />
        </div>
      </div>
    </div>
  )
}
