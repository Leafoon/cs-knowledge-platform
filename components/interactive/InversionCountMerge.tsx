'use client'
import React, { useEffect, useRef, useState } from 'react'

const LEFT = [2, 4, 5]
const RIGHT = [1, 3, 6]

interface Step {
  title: string
  desc: string
  i: number
  j: number
  merged: number[]
  picked?: { side: 'L' | 'R'; value: number }
  added?: number
  total: number
  phase: 'compare' | 'count' | 'done'
}

const STEPS: Step[] = [
  {
    title: '初始：左右两半已经有序',
    desc: '归并排序递归返回后，左半 [2,4,5] 与右半 [1,3,6] 都已经排好序。现在要在 merge 的同时统计“跨左右子数组”的逆序对。',
    i: 0,
    j: 0,
    merged: [],
    total: 0,
    phase: 'compare',
  },
  {
    title: '比较 2 和 1：发现跨区间逆序对',
    desc: '因为 2 > 1，所以右侧元素 1 要先进入 merged。同时，左侧当前指针及其后面所有元素 [2,4,5] 都比 1 大，因此一次性新增 3 个逆序对。',
    i: 0,
    j: 1,
    merged: [1],
    picked: { side: 'R', value: 1 },
    added: 3,
    total: 3,
    phase: 'count',
  },
  {
    title: '比较 2 和 3：左侧较小，正常合并',
    desc: '现在 2 ≤ 3，所以取左侧 2。此时不会新增逆序对，因为逆序对只在 left[i] > right[j] 时产生。',
    i: 1,
    j: 1,
    merged: [1, 2],
    picked: { side: 'L', value: 2 },
    total: 3,
    phase: 'compare',
  },
  {
    title: '比较 4 和 3：再次批量计数',
    desc: '由于 4 > 3，右侧 3 先进入 merged，并且左侧剩余元素 [4,5] 全都大于 3，所以一次性新增 2 个逆序对。',
    i: 1,
    j: 2,
    merged: [1, 2, 3],
    picked: { side: 'R', value: 3 },
    added: 2,
    total: 5,
    phase: 'count',
  },
  {
    title: '比较 4 和 6：继续正常合并',
    desc: '4 ≤ 6，取左侧 4，不新增逆序对。',
    i: 2,
    j: 2,
    merged: [1, 2, 3, 4],
    picked: { side: 'L', value: 4 },
    total: 5,
    phase: 'compare',
  },
  {
    title: '比较 5 和 6：继续正常合并',
    desc: '5 ≤ 6，取左侧 5，不新增逆序对。接下来左侧耗尽，只需把右侧剩余元素直接拼接。',
    i: 3,
    j: 2,
    merged: [1, 2, 3, 4, 5],
    picked: { side: 'L', value: 5 },
    total: 5,
    phase: 'compare',
  },
  {
    title: '收尾：追加右侧剩余元素 6',
    desc: '右侧剩余元素 6 直接加入 merged，最终合并数组为 [1,2,3,4,5,6]，总逆序对数量为 5。',
    i: 3,
    j: 3,
    merged: [1, 2, 3, 4, 5, 6],
    picked: { side: 'R', value: 6 },
    total: 5,
    phase: 'done',
  },
]

export default function InversionCountMerge() {
  const [cur, setCur] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1800)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const step = STEPS[cur]

  useEffect(() => {
    if (playing) {
      timerRef.current = setInterval(() => {
        setCur(prev => {
          if (prev >= STEPS.length - 1) {
            setPlaying(false)
            return prev
          }
          return prev + 1
        })
      }, speed)
    } else if (timerRef.current) {
      clearInterval(timerRef.current)
    }

    return () => {
      if (timerRef.current) clearInterval(timerRef.current)
    }
  }, [playing, speed])

  return (
    <div className="w-full max-w-4xl mx-auto rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 shadow-xl overflow-hidden">
      <div className="bg-gradient-to-r from-amber-500 via-orange-500 to-red-500 px-5 py-4">
        <h3 className="text-white font-bold text-lg tracking-tight">逆序对归并计数追踪</h3>
        <p className="text-orange-100 text-sm mt-0.5">merge 时一次性统计跨区间逆序对 · O(n log n)</p>
      </div>

      <div className="px-4 py-3 flex flex-wrap gap-2 items-center border-b border-slate-100 dark:border-slate-800 bg-slate-50 dark:bg-slate-800/40">
        <div className="flex rounded-lg overflow-hidden text-[10px] font-bold border border-slate-200 dark:border-slate-600">
          <div className={`px-3 py-1.5 ${step.phase === 'compare' ? 'bg-amber-500 text-white' : 'bg-slate-100 dark:bg-slate-700 text-slate-400'}`}>比较</div>
          <div className={`px-3 py-1.5 ${step.phase === 'count' ? 'bg-orange-500 text-white' : 'bg-slate-100 dark:bg-slate-700 text-slate-400'}`}>计数</div>
          <div className={`px-3 py-1.5 ${step.phase === 'done' ? 'bg-emerald-600 text-white' : 'bg-slate-100 dark:bg-slate-700 text-slate-400'}`}>完成</div>
        </div>
        <div className="flex gap-1.5 ml-auto">
          <button onClick={() => { setPlaying(false); setCur(0) }} className="px-2 py-1.5 rounded-lg text-xs bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-300">⏮</button>
          <button onClick={() => { setPlaying(false); setCur(v => Math.max(0, v - 1)) }} disabled={cur === 0} className="px-2 py-1.5 rounded-lg text-xs disabled:opacity-40 bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-300">◀</button>
          <button onClick={() => setPlaying(v => !v)} disabled={cur >= STEPS.length - 1} className="px-4 py-1.5 rounded-lg text-xs font-bold bg-orange-500 hover:bg-orange-600 text-white disabled:opacity-40">{playing ? '⏸ 暂停' : '▶ 播放'}</button>
          <button onClick={() => { setPlaying(false); setCur(v => Math.min(STEPS.length - 1, v + 1)) }} disabled={cur >= STEPS.length - 1} className="px-2 py-1.5 rounded-lg text-xs disabled:opacity-40 bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-300">▶</button>
        </div>
        <div className="flex items-center gap-1.5 text-[10px] text-slate-400">
          <input type="range" min={900} max={3200} step={300} value={speed} onChange={e => setSpeed(Number(e.target.value))} className="w-16 accent-orange-500" />
          <span>{(speed / 1000).toFixed(1)}s</span>
        </div>
      </div>

      <div className="p-4 space-y-4">
        <div className="flex items-center gap-2">
          <span className="px-2 py-0.5 rounded-full text-[10px] font-bold text-white bg-orange-500">{cur + 1}/{STEPS.length}</span>
          <span className="font-semibold text-sm text-slate-700 dark:text-slate-200">{step.title}</span>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">
          <div className="lg:col-span-3 rounded-2xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/30 p-4 space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="rounded-2xl border border-blue-200 dark:border-blue-700/60 bg-blue-50 dark:bg-blue-950/30 p-4">
                <div className="text-[10px] uppercase tracking-wider text-blue-600 dark:text-blue-400 font-bold mb-2">左半（已排序）</div>
                <div className="flex gap-2 flex-wrap">
                  {LEFT.map((v, idx) => (
                    <div key={idx} className={`w-12 h-12 rounded-xl flex items-center justify-center text-sm font-black border ${step.i === idx ? 'border-blue-500 bg-blue-500 text-white' : idx < step.i ? 'border-blue-200 dark:border-blue-700 bg-blue-100 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300' : 'border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 text-slate-700 dark:text-slate-200'}`}>{v}</div>
                  ))}
                </div>
                <div className="text-[10px] text-slate-500 dark:text-slate-400 mt-2">指针 i = {Math.min(step.i, LEFT.length)}</div>
              </div>

              <div className="rounded-2xl border border-violet-200 dark:border-violet-700/60 bg-violet-50 dark:bg-violet-950/30 p-4">
                <div className="text-[10px] uppercase tracking-wider text-violet-600 dark:text-violet-400 font-bold mb-2">右半（已排序）</div>
                <div className="flex gap-2 flex-wrap">
                  {RIGHT.map((v, idx) => (
                    <div key={idx} className={`w-12 h-12 rounded-xl flex items-center justify-center text-sm font-black border ${step.j === idx ? 'border-violet-500 bg-violet-500 text-white' : idx < step.j ? 'border-violet-200 dark:border-violet-700 bg-violet-100 dark:bg-violet-900/40 text-violet-700 dark:text-violet-300' : 'border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 text-slate-700 dark:text-slate-200'}`}>{v}</div>
                  ))}
                </div>
                <div className="text-[10px] text-slate-500 dark:text-slate-400 mt-2">指针 j = {Math.min(step.j, RIGHT.length)}</div>
              </div>
            </div>

            <div className="rounded-2xl border border-emerald-200 dark:border-emerald-700/60 bg-emerald-50 dark:bg-emerald-950/30 p-4">
              <div className="text-[10px] uppercase tracking-wider text-emerald-600 dark:text-emerald-400 font-bold mb-2">merged 数组</div>
              <div className="flex gap-2 flex-wrap min-h-[52px] items-center">
                {step.merged.length === 0 && <span className="text-[11px] text-slate-400 italic">当前还没有合并任何元素</span>}
                {step.merged.map((v, idx) => (
                  <div key={idx} className={`w-12 h-12 rounded-xl flex items-center justify-center text-sm font-black border ${step.picked?.value === v && idx === step.merged.length - 1 ? 'border-emerald-500 bg-emerald-500 text-white' : 'border-emerald-200 dark:border-emerald-700 bg-white dark:bg-slate-900 text-emerald-700 dark:text-emerald-300'}`}>{v}</div>
                ))}
              </div>
            </div>
          </div>

          <div className="lg:col-span-2 space-y-3">
            <div className="rounded-2xl border border-orange-200 dark:border-orange-700/60 bg-orange-50 dark:bg-orange-950/30 p-4 text-center">
              <div className="text-[10px] uppercase tracking-wider text-orange-600 dark:text-orange-400 font-bold mb-2">当前逆序对总数</div>
              <div className="text-4xl font-black text-orange-600 dark:text-orange-300">{step.total}</div>
              {step.added ? <div className="mt-2 inline-flex px-3 py-1 rounded-full bg-white dark:bg-slate-900 border border-orange-200 dark:border-orange-700 text-orange-700 dark:text-orange-300 text-xs font-bold">本步 +{step.added}</div> : <div className="mt-2 text-[11px] text-slate-400">本步未新增</div>}
            </div>

            <div className="rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-4">
              <div className="text-[10px] uppercase tracking-wider text-slate-500 dark:text-slate-400 font-bold mb-2">批量计数规则</div>
              <div className="space-y-2 text-[11px] text-slate-600 dark:text-slate-300">
                <p>若 <span className="font-mono">left[i] ≤ right[j]</span>：取左边，不新增逆序对。</p>
                <p>若 <span className="font-mono">left[i] &gt; right[j]</span>：右边先出列，并且左边从 i 到末尾的所有元素都与 right[j] 构成逆序对。</p>
                <div className="rounded-xl bg-slate-50 dark:bg-slate-800/60 px-3 py-2 font-mono text-xs">新增数量 = 左半剩余元素数</div>
              </div>
            </div>

            <div className="rounded-2xl border border-rose-200 dark:border-rose-700/60 bg-rose-50 dark:bg-rose-950/30 p-4">
              <div className="text-[10px] uppercase tracking-wider text-rose-600 dark:text-rose-400 font-bold mb-2">为什么能一次加一段？</div>
              <p className="text-[11px] text-slate-600 dark:text-slate-300">因为左右两半已经排好序。一旦发现 left[i] 比 right[j] 大，那么 left[i+1]、left[i+2] ... 也都不会更小，所以无需逐个再比较。</p>
            </div>
          </div>
        </div>

        <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/30 px-4 py-3 text-[12px] text-slate-600 dark:text-slate-300">
          {step.desc}
        </div>

        <div className="h-1.5 rounded-full bg-slate-200 dark:bg-slate-700 overflow-hidden">
          <div className="h-full rounded-full bg-gradient-to-r from-amber-500 via-orange-500 to-red-500 transition-all duration-500" style={{ width: `${(cur / (STEPS.length - 1)) * 100}%` }} />
        </div>
      </div>
    </div>
  )
}
