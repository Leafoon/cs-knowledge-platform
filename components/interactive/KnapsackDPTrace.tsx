'use client'
import { useState, useEffect } from 'react'

/* ─── 0/1 背包 DP 计算 ──────────────────────────────────────────────────── */
interface Item { w: number; v: number; name: string }
type Decision = 'skip' | 'take' | 'init'
interface Cell { val: number; dec: Decision }

function computeKnapsack(items: Item[], W: number): { dp: Cell[][]; steps: [number, number][] } {
  const n = items.length
  const dp: Cell[][] = Array.from({ length: n + 1 }, (_, i) =>
    Array.from({ length: W + 1 }, () => ({ val: 0, dec: 'init' as Decision }))
  )
  const steps: [number, number][] = []
  for (let j = 0; j <= W; j++) steps.push([0, j])
  for (let i = 1; i <= n; i++) {
    for (let j = 0; j <= W; j++) {
      steps.push([i, j])
      const { w, v } = items[i - 1]
      const skipVal = dp[i - 1][j].val
      if (j < w) {
        dp[i][j] = { val: skipVal, dec: 'skip' }
      } else {
        const takeVal = dp[i - 1][j - w].val + v
        if (takeVal > skipVal) dp[i][j] = { val: takeVal, dec: 'take' }
        else dp[i][j] = { val: skipVal, dec: 'skip' }
      }
    }
  }
  return { dp, steps }
}

// 回溯找选了哪些物品
function traceback(dp: Cell[][], items: Item[], W: number): Set<number> {
  const selected = new Set<number>()
  let j = W
  for (let i = items.length; i >= 1; i--) {
    if (dp[i][j].dec === 'take') {
      selected.add(i - 1)
      j -= items[i - 1].w
    }
  }
  return selected
}

// 1D 滚动数组追踪
function compute1D(items: Item[], W: number): { snapshots: number[][]; actions: string[] } {
  const dp = new Array(W + 1).fill(0)
  const snapshots: number[][] = [[...dp]]
  const actions: string[] = ['初始化 dp[0..W] = 0']
  for (const { w, v, name } of items) {
    for (let j = W; j >= w; j--) {
      if (dp[j - w] + v > dp[j]) {
        dp[j] = dp[j - w] + v
        actions.push(`处理 ${name}(w=${w},v=${v}): dp[${j}] = dp[${j-w}]+${v} = ${dp[j]} （选）`)
      } else {
        actions.push(`处理 ${name}(w=${w},v=${v}): dp[${j}] 保持 ${dp[j]}（不选）`)
      }
    }
    snapshots.push([...dp])
  }
  return { snapshots, actions }
}

const DEFAULT_ITEMS: Item[] = [
  { w: 2, v: 3, name: '💎 宝石' },
  { w: 3, v: 4, name: '📿 项链' },
  { w: 4, v: 5, name: '🎭 面具' },
  { w: 5, v: 6, name: '👑 王冠' },
]
const DEFAULT_W = 8

const PRESETS = [
  { items: DEFAULT_ITEMS, W: DEFAULT_W, label: '经典 4 件' },
  { items: [{ w: 1, v: 1, name: 'A' }, { w: 3, v: 4, name: 'B' }, { w: 4, v: 5, name: 'C' }, { w: 5, v: 7, name: 'D' }], W: 7, label: 'LeetCode 风格' },
]

export default function KnapsackDPTrace() {
  const [items, setItems] = useState<Item[]>(DEFAULT_ITEMS)
  const [W, setW] = useState(DEFAULT_W)
  const [step, setStep] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [view, setView] = useState<'2d' | '1d'>('2d')
  const [showTrace, setShowTrace] = useState(false)

  const { dp, steps } = computeKnapsack(items, W)
  const maxStep = steps.length - 1
  const done = step >= maxStep
  const selectedItems = done ? traceback(dp, items, W) : new Set<number>()
  const { snapshots, actions } = compute1D(items, W)
  const snap1dStep = Math.min(step, snapshots.length - 1)

  const [ci, cj] = steps[Math.min(step, maxStep)]

  useEffect(() => { setStep(0); setPlaying(false) }, [items, W])
  useEffect(() => {
    if (!playing) return
    if (step >= maxStep) { setPlaying(false); return }
    const id = setTimeout(() => setStep(s => s + 1), 150)
    return () => clearTimeout(id)
  }, [playing, step, maxStep])

  const visible = new Set(steps.slice(0, step + 1).map(([i, j]) => `${i},${j}`))
  const isVis = (i: number, j: number) => visible.has(`${i},${j}`)
  const isCurr = (i: number, j: number) => i === ci && j === cj
  const isDep = (i: number, j: number) => {
    if (ci === 0 || !isVis(i, j)) return false
    const { w } = items[ci - 1]
    return (i === ci - 1 && j === cj) || (i === ci - 1 && j === cj - w)
  }

  const cellCls = (i: number, j: number) => {
    if (!isVis(i, j)) return 'bg-slate-50 dark:bg-zinc-900 border-slate-200 dark:border-zinc-700 text-slate-300 dark:text-zinc-600'
    const curr = isCurr(i, j)
    const cell = dp[i][j]
    if (curr) return cell.dec === 'take'
      ? 'bg-emerald-500 border-emerald-400 text-white scale-110 shadow-lg ring-2 ring-emerald-300 z-10'
      : 'bg-sky-500 border-sky-400 text-white scale-105 shadow ring-2 ring-sky-300 z-10'
    if (isDep(i, j)) return 'bg-violet-100 dark:bg-violet-900/60 border-violet-400 text-violet-800 dark:text-violet-200'
    if (done && showTrace && i > 0 && selectedItems.has(i - 1) && j === W) return 'bg-amber-200 dark:bg-amber-900 border-amber-400 text-amber-900 dark:text-amber-100'
    const dec = cell.dec
    if (dec === 'take' && i > 0 && isVis(i, j)) return 'bg-emerald-50 dark:bg-emerald-950 border-emerald-300 dark:border-emerald-700 text-emerald-800 dark:text-emerald-200'
    return 'bg-white dark:bg-zinc-800 border-slate-300 dark:border-zinc-600 text-slate-700 dark:text-zinc-200'
  }

  const CELL = Math.max(32, Math.min(40, Math.floor(460 / (W + 2))))

  return (
    <div className="rounded-2xl border border-slate-200 dark:border-zinc-800 bg-white dark:bg-zinc-950 overflow-hidden">
      {/* 头部 */}
      <div className="px-6 py-4 bg-gradient-to-r from-amber-600 to-orange-600">
        <h3 className="text-white font-bold text-base">0/1 背包 DP 决策追踪</h3>
        <p className="text-amber-200 text-sm mt-0.5">逐格追踪"选或不选"决策 — 二维表 与 一维滚动数组 对比</p>
      </div>

      <div className="p-5 space-y-4">
        {/* 预设 */}
        <div className="flex flex-wrap gap-2">
          {PRESETS.map(p => (
            <button key={p.label} onClick={() => { setItems(p.items); setW(p.W) }}
              className="px-3 py-1 text-xs rounded-lg bg-amber-50 dark:bg-amber-950 text-amber-700 dark:text-amber-300 border border-amber-200 dark:border-amber-800 hover:bg-amber-100 dark:hover:bg-amber-900 transition-colors">
              {p.label}（容量 W={p.W}）
            </button>
          ))}
        </div>

        {/* 物品列表 */}
        <div className="bg-slate-50 dark:bg-zinc-900 rounded-xl border border-slate-200 dark:border-zinc-700 p-4">
          <div className="flex items-center gap-2 mb-3">
            <span className="text-xs font-semibold text-slate-600 dark:text-zinc-300">物品清单</span>
            <span className="text-xs text-slate-400 dark:text-zinc-500">（背包容量 W = {W}）</span>
          </div>
          <div className="flex flex-wrap gap-2">
            {items.map((item, i) => (
              <div key={i} className={`flex items-center gap-2 px-3 py-2 rounded-xl border-2 text-sm font-medium transition-all ${
                done && selectedItems.has(i)
                  ? 'bg-emerald-100 dark:bg-emerald-900 border-emerald-400 text-emerald-800 dark:text-emerald-200 scale-105'
                  : ci > 0 && ci - 1 === i
                  ? 'bg-amber-100 dark:bg-amber-900 border-amber-400 text-amber-800 dark:text-amber-200 scale-105'
                  : 'bg-white dark:bg-zinc-800 border-slate-200 dark:border-zinc-700 text-slate-700 dark:text-zinc-200'
              }`}>
                <span>{item.name}</span>
                <span className="font-mono text-xs opacity-75">w={item.w} v={item.v}</span>
                {done && selectedItems.has(i) && <span className="text-xs">✓ 选</span>}
              </div>
            ))}
          </div>
        </div>

        {/* 视图切换 */}
        <div className="flex items-center gap-2">
          {[{ key: '2d', label: '二维 DP 表' }, { key: '1d', label: '一维滚动数组' }].map(({ key, label }) => (
            <button key={key} onClick={() => setView(key as '2d' | '1d')}
              className={`px-4 py-1.5 text-xs rounded-lg font-medium transition-all ${view === key ? 'bg-amber-600 text-white shadow' : 'bg-slate-100 dark:bg-zinc-800 text-slate-600 dark:text-zinc-300 hover:bg-slate-200 dark:hover:bg-zinc-700'}`}>
              {label}
            </button>
          ))}
        </div>

        {/* 控制 */}
        <div className="flex flex-wrap items-center gap-2">
          <button onClick={() => setStep(s => Math.max(0, s - 1))} disabled={step === 0}
            className="px-3 py-1.5 text-xs bg-slate-100 dark:bg-zinc-800 hover:bg-slate-200 dark:hover:bg-zinc-700 disabled:opacity-40 rounded-lg text-slate-700 dark:text-zinc-200">← 上一格</button>
          <button onClick={() => setPlaying(p => !p)}
            className={`px-3 py-1.5 text-xs rounded-lg text-white font-medium transition-colors ${playing ? 'bg-orange-500 hover:bg-orange-400' : 'bg-amber-600 hover:bg-amber-500'}`}>
            {playing ? '⏸ 暂停' : '▶ 自动填表'}
          </button>
          <button onClick={() => setStep(s => Math.min(maxStep, s + 1))} disabled={step >= maxStep}
            className="px-3 py-1.5 text-xs bg-slate-100 dark:bg-zinc-800 hover:bg-slate-200 dark:hover:bg-zinc-700 disabled:opacity-40 rounded-lg text-slate-700 dark:text-zinc-200">下一格 →</button>
          <button onClick={() => { setStep(maxStep); setShowTrace(true) }}
            className="px-3 py-1.5 text-xs bg-emerald-600 hover:bg-emerald-500 rounded-lg text-white">⚡ 完成 + 回溯</button>
          <button onClick={() => { setStep(0); setPlaying(false); setShowTrace(false) }}
            className="px-3 py-1.5 text-xs bg-slate-100 dark:bg-zinc-800 hover:bg-slate-200 dark:hover:bg-zinc-700 rounded-lg text-slate-700 dark:text-zinc-200">↺ 重置</button>
          <span className="text-xs text-slate-400 dark:text-zinc-500 ml-auto">格 {step + 1}/{maxStep + 1}</span>
        </div>

        {/* 当前步骤说明 */}
        {ci > 0 && (
          <div className={`px-4 py-2.5 rounded-xl text-sm font-mono border ${
            dp[ci][cj].dec === 'take'
              ? 'bg-emerald-50 dark:bg-emerald-950 border-emerald-200 dark:border-emerald-800 text-emerald-800 dark:text-emerald-200'
              : 'bg-sky-50 dark:bg-sky-950 border-sky-200 dark:border-sky-800 text-sky-800 dark:text-sky-200'
          }`}>
            {(() => {
              const { w, v, name } = items[ci - 1]
              const skip = dp[ci - 1][cj].val
              const take = cj >= w ? dp[ci - 1][cj - w].val + v : -Infinity
              return dp[ci][cj].dec === 'take'
                ? <span>[i={ci} j={cj}] {name}(w={w},v={v})：<b>选</b>，{skip}(不选) vs {take}(选) → <b>{dp[ci][cj].val}</b></span>
                : cj < w
                ? <span>[i={ci} j={cj}] {name}(w={w}&gt;j={cj})：放不下，<b>跳过</b> → {dp[ci][cj].val}</span>
                : <span>[i={ci} j={cj}] {name}(w={w},v={v})：{skip}(不选) ≥ {take}(选)，<b>不选</b> → {dp[ci][cj].val}</span>
            })()}
          </div>
        )}

        {/* 二维表格 */}
        {view === '2d' && (
          <div className="overflow-auto rounded-xl bg-slate-50 dark:bg-zinc-900 border border-slate-200 dark:border-zinc-700 p-4">
            <table className="border-separate" style={{ borderSpacing: 2 }}>
              <thead>
                <tr>
                  <td className="text-xs text-slate-400 dark:text-zinc-500 font-mono pr-2" style={{ minWidth: 60 }}>i＼w</td>
                  {Array.from({ length: W + 1 }, (_, j) => (
                    <td key={j} style={{ width: CELL }} className="text-center">
                      <span className={`text-xs font-mono ${cj === j ? 'text-amber-600 dark:text-amber-400 font-bold' : 'text-slate-400 dark:text-zinc-500'}`}>{j}</span>
                    </td>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Array.from({ length: items.length + 1 }, (_, i) => (
                  <tr key={i}>
                    <td className="pr-2 py-0.5" style={{ minWidth: 60 }}>
                      <span className={`text-xs font-mono ${ci === i ? 'text-amber-600 dark:text-amber-400 font-bold' : 'text-slate-500 dark:text-zinc-400'}`}>
                        {i === 0 ? '∅' : `${items[i-1].name.split(' ')[0]}`}
                      </span>
                    </td>
                    {Array.from({ length: W + 1 }, (_, j) => (
                      <td key={j} style={{ width: CELL, height: CELL }}>
                        <div className={`w-full h-full rounded-md border flex items-center justify-center text-xs font-bold font-mono transition-all duration-150 ${cellCls(i, j)}`}>
                          {isVis(i, j) ? dp[i][j].val : ''}
                        </div>
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* 一维滚动数组 */}
        {view === '1d' && (
          <div className="bg-slate-50 dark:bg-zinc-900 rounded-xl border border-slate-200 dark:border-zinc-700 p-4 space-y-3">
            <div className="text-xs text-slate-500 dark:text-zinc-400 font-semibold">dp[0..W] 一维滚动数组（容量倒序枚举）</div>
            <div className="flex flex-wrap gap-2">
              {snapshots[snap1dStep].map((v, j) => (
                <div key={j} className="flex flex-col items-center gap-1">
                  <div className={`rounded-xl border-2 flex flex-col items-center justify-center font-mono font-bold text-sm transition-all duration-200 ${
                    v > 0 ? 'bg-amber-500 border-amber-400 text-white' : 'bg-slate-100 dark:bg-zinc-800 border-slate-300 dark:border-zinc-600 text-slate-400 dark:text-zinc-500'
                  }`} style={{ width: CELL, height: CELL }}>
                    <span className="text-[9px] opacity-60">j={j}</span>
                    <span>{v}</span>
                  </div>
                </div>
              ))}
            </div>
            <div className="bg-amber-50 dark:bg-amber-950 border border-amber-200 dark:border-amber-800 rounded-lg px-3 py-2 text-xs text-amber-800 dark:text-amber-200 font-mono">
              ⚠️ <b>关键</b>：j 必须从大到小（W→w_i）枚举，否则同一件物品会被多次选取！
            </div>
          </div>
        )}

        {/* 图例 */}
        <div className="flex flex-wrap gap-3 text-xs">
          {[
            { cls: 'bg-emerald-500 border-emerald-400 text-white', label: '✓ 选择该物品' },
            { cls: 'bg-sky-500 border-sky-400 text-white', label: '− 跳过该物品' },
            { cls: 'bg-violet-200 dark:bg-violet-900 border-violet-400', label: '依赖格子' },
          ].map(({ cls, label }) => (
            <div key={label} className="flex items-center gap-1.5">
              <div className={`w-5 h-5 rounded border-2 ${cls}`} />
              <span className="text-slate-600 dark:text-zinc-400">{label}</span>
            </div>
          ))}
        </div>

        {/* 最终结果 */}
        {done && (
          <div className="bg-amber-50 dark:bg-amber-950 border border-amber-200 dark:border-amber-800 rounded-xl px-4 py-3">
            <div className="flex items-center gap-3 mb-2">
              <span className="text-2xl">🎒</span>
              <div>
                <div className="text-sm font-bold text-amber-800 dark:text-amber-200">最优装袋方案，总价值 = {dp[items.length][W].val}</div>
                <div className="text-xs text-amber-600 dark:text-amber-400">
                  选择物品：{[...selectedItems].map(i => items[i].name).join('、')}
                  （总重 {[...selectedItems].reduce((s, i) => s + items[i].w, 0)} / {W}）
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
