'use client'

import React, { useState, useCallback, useEffect, useRef } from 'react'
import { Play, Pause, SkipBack, ChevronLeft, ChevronRight } from 'lucide-react'

interface Item { id: string; value: number; weight: number }

const ITEMS_RAW: Item[] = [
  { id: 'A', value: 60,  weight: 10 },
  { id: 'B', value: 100, weight: 20 },
  { id: 'C', value: 120, weight: 30 },
  { id: 'D', value: 50,  weight: 25 },
]
const CAPACITY = 50

// Sort by v/w descending
const SORTED_ITEMS = [...ITEMS_RAW].sort((a, b) => b.value / b.weight - a.value / a.weight)

interface FStep {
  desc: string
  detail: string
  processedIdx: number     // which sorted item we just handled (-1 = init)
  fractions: number[]      // fraction taken for each sorted item (0..1)
  remain: number
  totalValue: number
  mode: 'init' | 'take-full' | 'take-partial' | 'skip' | 'done'
}

function buildFractionSteps(): FStep[] {
  const steps: FStep[] = []
  const n = SORTED_ITEMS.length
  const fractions = Array(n).fill(0)

  steps.push({
    desc: '初始化：按 v/w 密度降序排列',
    detail: `将 ${n} 件物品按单位价值 v/w 从高到低排序：${SORTED_ITEMS.map(it => `${it.id}(${(it.value/it.weight).toFixed(1)})`).join(' > ')}。背包容量 W=${CAPACITY}。分数背包允许取物品的任意分数，因此策略是：先取密度最高的物品，取满为止；剩余再取次高的物品，以此类推。`,
    processedIdx: -1,
    fractions: [...fractions],
    remain: CAPACITY,
    totalValue: 0,
    mode: 'init',
  })

  let remain = CAPACITY
  let totalValue = 0

  for (let i = 0; i < n; i++) {
    const it = SORTED_ITEMS[i]
    const density = it.value / it.weight
    if (remain <= 0) {
      steps.push({
        desc: `跳过 ${it.id}：背包已满`,
        detail: `物品 ${it.id}（v=${it.value}, w=${it.weight}, v/w=${density.toFixed(1)}）无法放入，剩余容量 = 0。算法终止。`,
        processedIdx: i,
        fractions: [...fractions],
        remain,
        totalValue,
        mode: 'skip',
      })
      continue
    }

    if (it.weight <= remain) {
      // Take full
      fractions[i] = 1.0
      totalValue += it.value
      remain -= it.weight
      steps.push({
        desc: `✅ 完整取 ${it.id}（v=${it.value}, w=${it.weight}, v/w=${density.toFixed(1)}）`,
        detail: `${it.id} 的重量 ${it.weight} ≤ 剩余容量 ${remain + it.weight}，可以完整装入。获得价值 ${it.value}，剩余容量变为 ${remain}。累计价值 = ${totalValue.toFixed(1)}。`,
        processedIdx: i,
        fractions: [...fractions],
        remain,
        totalValue,
        mode: 'take-full',
      })
    } else {
      // Take partial
      const frac = remain / it.weight
      const gain = it.value * frac
      fractions[i] = frac
      totalValue += gain
      const oldRemain = remain
      remain = 0
      steps.push({
        desc: `⚡ 部分取 ${it.id}：装 ${frac.toFixed(2)} = ${(frac * 100).toFixed(0)}%（v/w=${density.toFixed(1)}）`,
        detail: `${it.id} 的重量 ${it.weight} > 剩余容量 ${oldRemain}，只能取其 ${oldRemain}/${it.weight} = ${(frac*100).toFixed(0)}%。获得价值 ${gain.toFixed(1)}，背包满。累计价值 = ${totalValue.toFixed(1)}。这是分数背包的核心操作——0/1 背包无法做到此步。`,
        processedIdx: i,
        fractions: [...fractions],
        remain: 0,
        totalValue,
        mode: 'take-partial',
      })
    }
  }

  steps.push({
    desc: '算法完成 — 最优分数背包解',
    detail: `贪心策略获得最优总价值 ${totalValue.toFixed(1)}。可证明：对于分数背包，按 v/w 降序贪心是最优策略（反证法：若存在更优解，则必然在某个位置用了密度更低的物品或空余容量，可通过交换提升价值，矛盾）。`,
    processedIdx: n - 1,
    fractions: [...fractions],
    remain: 0,
    totalValue,
    mode: 'done',
  })

  return steps
}

const STEPS = buildFractionSteps()

// 0/1 Knapsack DP for contrast
function knapsack01(items: Item[], W: number) {
  const n = items.length
  const dp = Array.from({ length: n + 1 }, () => Array(W + 1).fill(0))
  for (let i = 1; i <= n; i++) {
    for (let w = 0; w <= W; w++) {
      dp[i][w] = dp[i-1][w]
      if (items[i-1].weight <= w) {
        dp[i][w] = Math.max(dp[i][w], dp[i-1][w - items[i-1].weight] + items[i-1].value)
      }
    }
  }
  // Backtrack
  const chosen: string[] = []
  let w = W
  for (let i = n; i > 0; i--) {
    if (dp[i][w] !== dp[i-1][w]) { chosen.push(items[i-1].id); w -= items[i-1].weight }
  }
  return { value: dp[n][W], chosen }
}

const OPT_01 = knapsack01(ITEMS_RAW, CAPACITY)

const ITEM_COLORS = ['from-cyan-500 to-blue-500', 'from-blue-500 to-indigo-500', 'from-violet-500 to-purple-500', 'from-rose-500 to-pink-500']
const ITEM_SOLID  = ['bg-cyan-500', 'bg-blue-500', 'bg-violet-500', 'bg-rose-500']
const SPEEDS = [0.5, 1, 1.5, 2]

export default function FractionalKnapsackPicker() {
  const [idx, setIdx] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speedIdx, setSpeedIdx] = useState(1)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const cur = STEPS[idx]
  const total = STEPS.length - 1

  const stop = useCallback(() => {
    if (timerRef.current) clearInterval(timerRef.current)
    timerRef.current = null
    setPlaying(false)
  }, [])

  useEffect(() => {
    if (!playing) return
    timerRef.current = setInterval(() => {
      setIdx(prev => {
        if (prev >= total) { stop(); return prev }
        return prev + 1
      })
    }, 1200 / SPEEDS[speedIdx])
    return () => { if (timerRef.current) clearInterval(timerRef.current) }
  }, [playing, speedIdx, total, stop])

  useEffect(() => { if (idx >= total && playing) stop() }, [idx, total, playing, stop])
  const goto = (n: number) => { stop(); setIdx(Math.max(0, Math.min(total, n))) }

  // Build knapsack fill bar segments
  const usedCapacity = CAPACITY - cur.remain
  const usedPct = (usedCapacity / CAPACITY) * 100

  return (
    <div className="w-full max-w-5xl mx-auto rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 shadow-xl overflow-hidden">
      <div className="bg-gradient-to-r from-emerald-600 via-teal-600 to-cyan-600 px-5 py-4">
        <h3 className="text-white font-bold text-lg">分数背包贪心演示 — 逐步装填可视化</h3>
        <p className="text-emerald-100 text-sm mt-0.5">按 v/w 密度降序贪心选取，支持分割；对比 0/1 背包的差异</p>
      </div>

      <div className="p-4 space-y-4">
        {/* Controls */}
        <div className="flex items-center gap-2 flex-wrap">
          <button onClick={() => goto(0)} className="p-1.5 rounded-lg border border-slate-200 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-600 dark:text-slate-300"><SkipBack size={16}/></button>
          <button onClick={() => goto(idx - 1)} className="p-1.5 rounded-lg border border-slate-200 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-600 dark:text-slate-300"><ChevronLeft size={16}/></button>
          <button onClick={playing ? stop : () => setPlaying(true)} className="px-3 py-1.5 rounded-lg bg-teal-600 hover:bg-teal-700 text-white flex items-center gap-1.5 text-sm font-medium">
            {playing ? <Pause size={14}/> : <Play size={14}/>}{playing ? '暂停' : '播放'}
          </button>
          <button onClick={() => goto(idx + 1)} className="p-1.5 rounded-lg border border-slate-200 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-600 dark:text-slate-300"><ChevronRight size={16}/></button>
          <div className="ml-auto flex items-center gap-1.5 text-xs text-slate-500 dark:text-slate-400">
            速度: {SPEEDS.map((s, i) => (
              <button key={i} onClick={() => setSpeedIdx(i)} className={`px-2 py-0.5 rounded ${speedIdx === i ? 'bg-teal-600 text-white' : 'border border-slate-200 dark:border-slate-700'}`}>{s}x</button>
            ))}
          </div>
        </div>

        <div className="w-full h-1.5 rounded-full bg-slate-100 dark:bg-slate-800">
          <div className="h-1.5 rounded-full bg-teal-500 transition-all" style={{ width: `${(idx / total) * 100}%` }}/>
        </div>

        {/* Density table */}
        <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/40 p-3">
          <div className="text-xs font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-2">物品密度排行（按 v/w 降序）</div>
          <div className="grid grid-cols-4 gap-2">
            {SORTED_ITEMS.map((item, i) => {
              const frac = cur.fractions[i]
              const isCurrent = i === cur.processedIdx
              return (
                <div key={item.id} className={`rounded-xl p-3 transition-all ${isCurrent ? 'ring-2 ring-teal-400 scale-105 shadow-lg bg-white dark:bg-slate-900' : 'bg-white dark:bg-slate-900'} border border-slate-200 dark:border-slate-700`}>
                  <div className={`text-2xl font-black bg-gradient-to-br ${ITEM_COLORS[i]} bg-clip-text text-transparent`}>#{i+1} {item.id}</div>
                  <div className="text-xs text-slate-500 dark:text-slate-400 mt-1">v={item.value} w={item.weight}</div>
                  <div className="text-sm font-bold text-teal-600 dark:text-teal-400">v/w = {(item.value/item.weight).toFixed(1)}</div>
                  {/* Fill bar */}
                  <div className="mt-2 h-2 rounded-full bg-slate-100 dark:bg-slate-800">
                    <div className={`h-2 rounded-full ${ITEM_SOLID[i]} transition-all duration-700`} style={{ width: `${frac * 100}%` }}/>
                  </div>
                  <div className="text-[10px] text-slate-500 dark:text-slate-400 mt-0.5">
                    {frac === 0 ? '未取' : frac === 1 ? '完整取 ✅' : `取 ${(frac*100).toFixed(0)}% ⚡`}
                  </div>
                </div>
              )
            })}
          </div>
        </div>

        {/* Knapsack bar */}
        <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/40 p-3">
          <div className="flex items-center justify-between text-xs text-slate-500 dark:text-slate-400 mb-2">
            <span className="font-bold uppercase tracking-wider">背包装填状态（容量 W={CAPACITY}）</span>
            <span className="font-mono">已用 {CAPACITY - cur.remain} / {CAPACITY}（{usedPct.toFixed(0)}%）</span>
          </div>
          <div className="h-10 rounded-xl overflow-hidden bg-slate-200 dark:bg-slate-700 relative">
            {SORTED_ITEMS.map((item, i) => {
              const frac = cur.fractions[i]
              if (frac === 0) return null
              const taken = item.weight * frac
              const widthPct = (taken / CAPACITY) * 100
              // Calculate left offset as sum of previous items
              const leftPct = SORTED_ITEMS.slice(0, i).reduce((acc, it, j) => acc + (it.weight * cur.fractions[j] / CAPACITY) * 100, 0)
              return (
                <div
                  key={item.id}
                  className={`absolute top-0 h-full flex items-center justify-center text-white text-xs font-bold ${ITEM_SOLID[i]} transition-all duration-700`}
                  style={{ left: `${leftPct}%`, width: `${widthPct}%` }}
                  title={`${item.id}: ${(frac*100).toFixed(0)}%`}
                >
                  {widthPct > 5 ? `${item.id}${frac < 1 ? `(${(frac*100).toFixed(0)}%)` : ''}` : ''}
                </div>
              )
            })}
            {/* Empty space */}
            {cur.remain > 0 && (
              <div
                className="absolute top-0 h-full flex items-center justify-center text-slate-400 dark:text-slate-500 text-xs"
                style={{ left: `${usedPct}%`, width: `${(cur.remain/CAPACITY)*100}%` }}
              >
                {(cur.remain/CAPACITY)*100 > 10 ? `空余${cur.remain}` : ''}
              </div>
            )}
          </div>
          <div className="mt-2 flex gap-3 text-xs text-slate-500 dark:text-slate-400">
            <span className="font-bold text-teal-600 dark:text-teal-400">当前价值: {cur.totalValue.toFixed(1)}</span>
            <span>|</span>
            <span>剩余容量: {cur.remain}</span>
          </div>
        </div>

        {/* Step description */}
        <div className={`rounded-xl border px-4 py-3 ${
          cur.mode === 'take-full' ? 'border-emerald-300 dark:border-emerald-700 bg-emerald-50 dark:bg-emerald-950/30'
          : cur.mode === 'take-partial' ? 'border-blue-300 dark:border-blue-700 bg-blue-50 dark:bg-blue-950/30'
          : cur.mode === 'skip' ? 'border-slate-300 dark:border-slate-600 bg-slate-50 dark:bg-slate-800/40'
          : cur.mode === 'done' ? 'border-indigo-300 dark:border-indigo-700 bg-indigo-50 dark:bg-indigo-950/30'
          : 'border-teal-300 dark:border-teal-700 bg-teal-50 dark:bg-teal-950/30'
        }`}>
          <div className="font-bold text-sm text-slate-800 dark:text-slate-100">{cur.desc}</div>
          <div className="mt-1 text-sm text-slate-600 dark:text-slate-300 leading-relaxed">{cur.detail}</div>
        </div>

        {/* Comparison with 0/1 */}
        {idx >= 2 && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div className="rounded-xl border border-teal-300 dark:border-teal-700 bg-teal-50 dark:bg-teal-950/30 px-4 py-3">
              <div className="text-xs font-bold text-teal-600 dark:text-teal-400 uppercase tracking-wider mb-1">分数背包（贪心）</div>
              <div className="text-2xl font-black text-teal-700 dark:text-teal-300">
                {STEPS[STEPS.length - 1].totalValue.toFixed(1)}
              </div>
              <div className="text-xs text-teal-600 dark:text-teal-400">最优解 ✅</div>
            </div>
            <div className="rounded-xl border border-slate-300 dark:border-slate-600 bg-slate-50 dark:bg-slate-800/40 px-4 py-3">
              <div className="text-xs font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-1">0/1 背包（DP 最优）</div>
              <div className="text-2xl font-black text-slate-700 dark:text-slate-200">{OPT_01.value}</div>
              <div className="text-xs text-slate-500 dark:text-slate-400">选: {OPT_01.chosen.join('+')}（不可分割）</div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
