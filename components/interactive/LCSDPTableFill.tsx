'use client'
import { useState, useEffect } from 'react'

/* ─── LCS DP 计算 ──────────────────────────────────────────────────────── */
type Cell = { val: number; from: 'diag' | 'up' | 'left' | 'init' }

function computeLCS(X: string, Y: string): { dp: Cell[][]; steps: [number, number][] } {
  const m = X.length, n = Y.length
  const dp: Cell[][] = Array.from({ length: m + 1 }, () =>
    Array.from({ length: n + 1 }, () => ({ val: 0, from: 'init' as const }))
  )
  const steps: [number, number][] = []
  for (let i = 0; i <= m; i++) steps.push([i, 0])
  for (let j = 1; j <= n; j++) steps.push([0, j])

  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      steps.push([i, j])
      if (X[i - 1] === Y[j - 1]) {
        dp[i][j] = { val: dp[i - 1][j - 1].val + 1, from: 'diag' }
      } else if (dp[i - 1][j].val >= dp[i][j - 1].val) {
        dp[i][j] = { val: dp[i - 1][j].val, from: 'up' }
      } else {
        dp[i][j] = { val: dp[i][j - 1].val, from: 'left' }
      }
    }
  }
  return { dp, steps }
}

function backtrack(dp: Cell[][], X: string, Y: string): Set<string> {
  const path = new Set<string>()
  let i = X.length, j = Y.length
  while (i > 0 && j > 0) {
    path.add(`${i},${j}`)
    if (dp[i][j].from === 'diag') { i--; j-- }
    else if (dp[i][j].from === 'up') i--
    else j--
  }
  path.add(`${i},${j}`)
  return path
}

function getLCSString(dp: Cell[][], X: string, Y: string): string {
  let i = X.length, j = Y.length, res = ''
  while (i > 0 && j > 0) {
    if (dp[i][j].from === 'diag') { res = X[i - 1] + res; i--; j-- }
    else if (dp[i][j].from === 'up') i--
    else j--
  }
  return res
}

const PRESETS = [
  { X: 'ABCBDAB', Y: 'BDCABA', label: 'DNA 序列' },
  { X: 'AGGTAB',  Y: 'GXTXAYB', label: '经典示例' },
  { X: 'ABCDE',   Y: 'ACE', label: '简单' },
]

export default function LCSDPTableFill() {
  const [X, setX] = useState('ABCBDAB')
  const [Y, setY] = useState('BDCABA')
  const [step, setStep] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [showBacktrack, setShowBacktrack] = useState(false)

  // 限制输入长度以免表格过大
  const Xc = X.slice(0, 8).toUpperCase()
  const Yc = Y.slice(0, 8).toUpperCase()
  const { dp, steps } = computeLCS(Xc, Yc)
  const maxStep = steps.length - 1
  const done = step >= maxStep
  const pathSet = done ? backtrack(dp, Xc, Yc) : new Set<string>()
  const lcsStr = done ? getLCSString(dp, Xc, Yc) : ''

  useEffect(() => { setStep(0); setShowBacktrack(false); setPlaying(false) }, [X, Y])
  useEffect(() => {
    if (!playing) return
    if (step >= maxStep) { setPlaying(false); setShowBacktrack(true); return }
    const id = setTimeout(() => setStep(s => s + 1), 200)
    return () => clearTimeout(id)
  }, [playing, step, maxStep])

  // 当前可见格子集合
  const visible = new Set(steps.slice(0, step + 1).map(([i, j]) => `${i},${j}`))
  const [ci, cj] = steps[Math.min(step, maxStep)]

  const cellKey = (i: number, j: number) => `${i},${j}`
  const isVisible = (i: number, j: number) => visible.has(cellKey(i, j))
  const isCurrent = (i: number, j: number) => i === ci && j === cj
  const isPath = (i: number, j: number) => showBacktrack && pathSet.has(cellKey(i, j))
  const isDep = (i: number, j: number) => {
    if (!isCurrent(ci, cj) || ci === 0 || cj === 0) return false
    return (i === ci - 1 && j === cj) || (i === ci && j === cj - 1) || (i === ci - 1 && j === cj - 1)
  }

  // 检查当前格子是否 match
  const currentMatch = ci > 0 && cj > 0 && Xc[ci - 1] === Yc[cj - 1]

  const cellBg = (i: number, j: number) => {
    if (!isVisible(i, j)) return 'bg-slate-50 dark:bg-zinc-900 border-slate-200 dark:border-zinc-700 text-slate-300 dark:text-zinc-600'
    if (isPath(i, j)) return 'bg-amber-400 dark:bg-amber-500 border-amber-500 dark:border-amber-400 text-white font-bold scale-105 z-10'
    if (isCurrent(i, j)) return currentMatch
      ? 'bg-emerald-500 border-emerald-400 text-white scale-110 shadow-lg z-10 ring-2 ring-emerald-300'
      : 'bg-sky-500 border-sky-400 text-white scale-105 shadow z-10 ring-2 ring-sky-300'
    if (isDep(i, j)) return 'bg-violet-100 dark:bg-violet-900/60 border-violet-400 text-violet-800 dark:text-violet-200'
    return 'bg-white dark:bg-zinc-800 border-slate-300 dark:border-zinc-600 text-slate-700 dark:text-zinc-200'
  }

  const CELL = 38

  return (
    <div className="rounded-2xl border border-slate-200 dark:border-zinc-800 bg-white dark:bg-zinc-950 overflow-hidden">
      {/* 头部 */}
      <div className="px-6 py-4 bg-gradient-to-r from-sky-600 to-blue-600">
        <h3 className="text-white font-bold text-base">LCS DP 填表可视化</h3>
        <p className="text-sky-200 text-sm mt-0.5">最长公共子序列 — 逐格填充二维 DP 表 + 回溯路径高亮</p>
      </div>

      <div className="p-5 space-y-4">
        {/* 预设 & 输入 */}
        <div className="space-y-3">
          <div className="flex flex-wrap gap-2">
            {PRESETS.map(p => (
              <button key={p.label} onClick={() => { setX(p.X); setY(p.Y) }}
                className="px-3 py-1 text-xs rounded-lg bg-sky-50 dark:bg-sky-950 text-sky-700 dark:text-sky-300 border border-sky-200 dark:border-sky-800 hover:bg-sky-100 dark:hover:bg-sky-900 transition-colors">
                {p.label}：{p.X} vs {p.Y}
              </button>
            ))}
          </div>
          <div className="flex gap-3 flex-wrap">
            {[{ label: '序列 X', val: X, set: setX }, { label: '序列 Y', val: Y, set: setY }].map(({ label, val, set }) => (
              <div key={label} className="flex items-center gap-2">
                <span className="text-sm text-slate-500 dark:text-zinc-400 whitespace-nowrap">{label}:</span>
                <input value={val} onChange={e => set(e.target.value.toUpperCase().slice(0, 8))} maxLength={8}
                  className="w-28 bg-white dark:bg-zinc-800 border border-slate-300 dark:border-zinc-600 rounded-lg px-3 py-1.5 text-sm font-mono uppercase text-slate-800 dark:text-zinc-100 focus:outline-none focus:ring-2 focus:ring-sky-400" />
              </div>
            ))}
          </div>
        </div>

        {/* 控制 */}
        <div className="flex flex-wrap items-center gap-2">
          <button onClick={() => setStep(s => Math.max(0, s - 1))} disabled={step === 0}
            className="px-3 py-1.5 text-xs bg-slate-100 dark:bg-zinc-800 hover:bg-slate-200 dark:hover:bg-zinc-700 disabled:opacity-40 rounded-lg text-slate-700 dark:text-zinc-200 transition-colors">← 上一格</button>
          <button onClick={() => setPlaying(p => !p)}
            className={`px-3 py-1.5 text-xs rounded-lg text-white font-medium transition-colors ${playing ? 'bg-orange-500 hover:bg-orange-400' : 'bg-sky-600 hover:bg-sky-500'}`}>
            {playing ? '⏸ 暂停' : '▶ 自动填表'}
          </button>
          <button onClick={() => setStep(s => Math.min(maxStep, s + 1))} disabled={step >= maxStep}
            className="px-3 py-1.5 text-xs bg-slate-100 dark:bg-zinc-800 hover:bg-slate-200 dark:hover:bg-zinc-700 disabled:opacity-40 rounded-lg text-slate-700 dark:text-zinc-200 transition-colors">下一格 →</button>
          <button onClick={() => { setStep(maxStep); setShowBacktrack(true) }}
            className="px-3 py-1.5 text-xs bg-emerald-600 hover:bg-emerald-500 rounded-lg text-white transition-colors">⚡ 直接完成</button>
          <button onClick={() => setShowBacktrack(s => !s)} disabled={!done}
            className="px-3 py-1.5 text-xs bg-amber-500 hover:bg-amber-400 disabled:opacity-40 rounded-lg text-white transition-colors">
            {showBacktrack ? '隐藏' : '显示'} 回溯路径
          </button>
          <button onClick={() => { setStep(0); setPlaying(false); setShowBacktrack(false) }}
            className="px-3 py-1.5 text-xs bg-slate-100 dark:bg-zinc-800 hover:bg-slate-200 dark:hover:bg-zinc-700 rounded-lg text-slate-700 dark:text-zinc-200 transition-colors">↺ 重置</button>
          <span className="text-xs text-slate-400 dark:text-zinc-500 ml-auto">格 {step + 1} / {maxStep + 1}</span>
        </div>

        {/* 当前步骤说明 */}
        {ci > 0 && cj > 0 && (
          <div className={`px-4 py-2.5 rounded-xl text-sm font-mono border flex items-center gap-3 ${
            currentMatch
              ? 'bg-emerald-50 dark:bg-emerald-950 border-emerald-200 dark:border-emerald-800 text-emerald-800 dark:text-emerald-200'
              : 'bg-sky-50 dark:bg-sky-950 border-sky-200 dark:border-sky-800 text-sky-800 dark:text-sky-200'
          }`}>
            <span className="font-bold">[{ci},{cj}]</span>
            {currentMatch
              ? <span>X[{ci-1}]=<b>'{Xc[ci-1]}'</b> == Y[{cj-1}]=<b>'{Yc[cj-1]}'</b> → dp[{ci}][{cj}] = dp[{ci-1}][{cj-1}] + 1 = <b>{dp[ci][cj].val}</b></span>
              : <span>X[{ci-1}]='{Xc[ci-1]}' ≠ Y[{cj-1}]='{Yc[cj-1]}' → max(dp[{ci-1}][{cj}]={dp[ci-1][cj]?.val}, dp[{ci}][{cj-1}]={dp[ci][cj-1]?.val}) = <b>{dp[ci][cj].val}</b></span>
            }
          </div>
        )}

        {/* DP 表格 */}
        <div className="overflow-auto rounded-xl bg-slate-50 dark:bg-zinc-900 border border-slate-200 dark:border-zinc-700 p-4">
          <table className="border-separate" style={{ borderSpacing: 3 }}>
            <thead>
              <tr>
                <td style={{ width: CELL, height: CELL }} />
                <td className="text-center text-xs font-mono text-slate-400 dark:text-zinc-500 pb-1" style={{ width: CELL }}>ε</td>
                {Yc.split('').map((c, j) => (
                  <td key={j} className="text-center pb-1" style={{ width: CELL }}>
                    <span className={`text-xs font-bold font-mono px-1 py-0.5 rounded ${cj === j + 1 ? 'bg-sky-100 dark:bg-sky-900 text-sky-700 dark:text-sky-300' : 'text-slate-600 dark:text-zinc-300'}`}>{c}</span>
                  </td>
                ))}
              </tr>
            </thead>
            <tbody>
              {Array.from({ length: Xc.length + 1 }, (_, i) => (
                <tr key={i}>
                  <td className="text-center pr-1" style={{ height: CELL }}>
                    <span className={`text-xs font-bold font-mono px-1 py-0.5 rounded ${ci === i && i > 0 ? 'bg-sky-100 dark:bg-sky-900 text-sky-700 dark:text-sky-300' : 'text-slate-600 dark:text-zinc-300'}`}>
                      {i === 0 ? 'ε' : Xc[i - 1]}
                    </span>
                  </td>
                  {Array.from({ length: Yc.length + 1 }, (_, j) => (
                    <td key={j} style={{ width: CELL, height: CELL }}>
                      <div className={`w-full h-full rounded-lg border-2 flex items-center justify-center text-sm font-bold font-mono transition-all duration-200 ${cellBg(i, j)}`}>
                        {isVisible(i, j) ? dp[i][j].val : ''}
                      </div>
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* 图例 */}
        <div className="flex flex-wrap gap-3 text-xs">
          {[
            { cls: 'bg-emerald-500 border-emerald-400', label: '字符匹配（dp[i-1][j-1]+1）' },
            { cls: 'bg-sky-500 border-sky-400', label: '取上/左最大值' },
            { cls: 'bg-violet-200 dark:bg-violet-900 border-violet-400', label: '依赖格子' },
            { cls: 'bg-amber-400 border-amber-500', label: '回溯路径（LCS）' },
          ].map(({ cls, label }) => (
            <div key={label} className="flex items-center gap-1.5">
              <div className={`w-4 h-4 rounded border-2 ${cls}`} />
              <span className="text-slate-600 dark:text-zinc-400">{label}</span>
            </div>
          ))}
        </div>

        {/* 结果 */}
        {done && lcsStr && (
          <div className="bg-amber-50 dark:bg-amber-950 border border-amber-300 dark:border-amber-700 rounded-xl px-4 py-3 flex items-center gap-3">
            <span className="text-2xl">🏆</span>
            <div>
              <div className="text-sm font-bold text-amber-800 dark:text-amber-200">LCS 长度 = {dp[Xc.length][Yc.length].val}</div>
              <div className="text-sm font-mono text-amber-700 dark:text-amber-300 mt-0.5">一条 LCS: <b>"{lcsStr}"</b></div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
