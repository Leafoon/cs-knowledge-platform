'use client'

import React, { useState, useCallback, useEffect, useRef } from 'react'
import { Play, Pause, SkipBack, ChevronLeft, ChevronRight } from 'lucide-react'

interface Activity { id: string; s: number; f: number; label: string }

const ACTIVITIES: Activity[] = [
  { id: 'a₁', s: 1, f: 4,  label: '[1, 4)' },
  { id: 'a₂', s: 3, f: 5,  label: '[3, 5)' },
  { id: 'a₃', s: 0, f: 6,  label: '[0, 6)' },
  { id: 'a₄', s: 5, f: 7,  label: '[5, 7)' },
  { id: 'a₅', s: 3, f: 9,  label: '[3, 9)' },
  { id: 'a₆', s: 5, f: 9,  label: '[5, 9)' },
  { id: 'a₇', s: 6, f: 10, label: '[6,10)' },
  { id: 'a₈', s: 8, f: 11, label: '[8,11)' },
  { id: 'a₉', s: 2, f: 13, label: '[2,13)' },
]

type Status = 'unvisited' | 'current' | 'accepted' | 'rejected'

interface Step {
  actIdx: number
  sorted: Activity[]
  statuses: Status[]
  chosen: string[]
  lastEnd: number
  decision: 'accepted' | 'rejected' | 'init'
  desc: string
  detail: string
}

const TMAX = 14

function buildSteps(acts: Activity[]): Step[] {
  const sorted = [...acts].sort((a, b) => a.f - b.f || a.s - b.s)
  const steps: Step[] = []

  // Step 0: initial state – show sorted order
  steps.push({
    actIdx: -1,
    sorted,
    statuses: sorted.map(() => 'unvisited'),
    chosen: [],
    lastEnd: 0,
    decision: 'init',
    desc: '初始状态：按结束时间升序排列',
    detail: `贪心策略：每次选结束时间最早、且与已选活动不冲突的活动。这样可以给后续活动留出最大余地。已将 ${sorted.length} 个活动排好序，lastEnd = 0（虚拟起始）。`,
  })

  const chosen: string[] = []
  let lastEnd = 0
  const statuses: Status[] = sorted.map(() => 'unvisited')

  for (let i = 0; i < sorted.length; i++) {
    const act = sorted[i]
    statuses[i] = 'current'
    const compatible = act.s >= lastEnd
    const prevChosen = [...chosen]

    if (compatible) {
      statuses[i] = 'accepted'
      chosen.push(act.id)
      const prevEnd = lastEnd
      lastEnd = act.f
      steps.push({
        actIdx: i,
        sorted: [...sorted],
        statuses: [...statuses],
        chosen: [...chosen],
        lastEnd,
        decision: 'accepted',
        desc: `✅ 选择 ${act.id} ${act.label}`,
        detail: `${act.id} 的开始时间 s=${act.s} ≥ lastEnd=${prevEnd}，不与已选活动冲突。将其加入最优解。更新 lastEnd = ${act.f}。当前已选：{${chosen.join(', ')}}。`,
      })
    } else {
      statuses[i] = 'rejected'
      steps.push({
        actIdx: i,
        sorted: [...sorted],
        statuses: [...statuses],
        chosen: [...prevChosen],
        lastEnd,
        decision: 'rejected',
        desc: `❌ 跳过 ${act.id} ${act.label}`,
        detail: `${act.id} 的开始时间 s=${act.s} < lastEnd=${lastEnd}，与上一个已选活动冲突（重叠）。贪心直接跳过，不影响已选集合。`,
      })
    }
  }

  return steps
}

const STEPS = buildSteps(ACTIVITIES)

const STATUS_COLOR: Record<Status, string> = {
  unvisited: 'bg-slate-200 dark:bg-slate-700',
  current:   'bg-yellow-400 dark:bg-yellow-500',
  accepted:  'bg-emerald-500 dark:bg-emerald-400',
  rejected:  'bg-rose-400 dark:bg-rose-500',
}

const SPEEDS = [0.5, 1, 1.5, 2]

export default function ActivitySelectionTimeline() {
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

  const play = useCallback(() => {
    if (idx >= total) { setIdx(0); }
    setPlaying(true)
  }, [idx, total])

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

  return (
    <div className="w-full max-w-5xl mx-auto rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 shadow-xl overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-cyan-600 via-blue-600 to-indigo-600 px-5 py-4">
        <h3 className="text-white font-bold text-lg">活动选择时间轴 — 贪心逐步演示</h3>
        <p className="text-cyan-100 text-sm mt-0.5">按结束时间排序后，依次检查每个活动是否与已选集合相容</p>
      </div>

      <div className="p-4 space-y-4">
        {/* Controls */}
        <div className="flex items-center gap-2 flex-wrap">
          <button onClick={() => goto(0)} className="p-1.5 rounded-lg border border-slate-200 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-600 dark:text-slate-300">
            <SkipBack size={16}/>
          </button>
          <button onClick={() => goto(idx - 1)} className="p-1.5 rounded-lg border border-slate-200 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-600 dark:text-slate-300">
            <ChevronLeft size={16}/>
          </button>
          <button onClick={playing ? stop : play} className="px-3 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-700 text-white flex items-center gap-1.5 text-sm font-medium">
            {playing ? <Pause size={14}/> : <Play size={14}/>}
            {playing ? '暂停' : '播放'}
          </button>
          <button onClick={() => goto(idx + 1)} className="p-1.5 rounded-lg border border-slate-200 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-600 dark:text-slate-300">
            <ChevronRight size={16}/>
          </button>
          <div className="ml-auto flex items-center gap-1.5 text-xs text-slate-500 dark:text-slate-400">
            速度:
            {SPEEDS.map((s, i) => (
              <button key={i} onClick={() => setSpeedIdx(i)} className={`px-2 py-0.5 rounded ${speedIdx === i ? 'bg-blue-600 text-white' : 'border border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-300'}`}>{s}x</button>
            ))}
          </div>
        </div>

        {/* Progress bar */}
        <div className="space-y-1">
          <div className="w-full h-1.5 rounded-full bg-slate-100 dark:bg-slate-800">
            <div className="h-1.5 rounded-full bg-blue-500 transition-all" style={{ width: `${(idx / total) * 100}%` }}/>
          </div>
          <div className="text-xs text-slate-500 dark:text-slate-400">步骤 {idx} / {total}</div>
        </div>

        {/* Timeline visualization */}
        <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/40 p-3 space-y-2">
          {/* Timeline ruler */}
          <div className="flex items-end h-5 pl-[5.5rem]">
            {Array.from({ length: TMAX + 1 }).map((_, t) => (
              <div key={t} className="flex-1 text-center text-[9px] text-slate-400 dark:text-slate-500 font-mono">{t}</div>
            ))}
          </div>
          {cur.sorted.map((act, i) => {
            const status = cur.statuses[i]
            const isCurrent = i === cur.actIdx
            return (
              <div key={act.id} className="flex items-center gap-2">
                <div className={`w-20 shrink-0 text-xs font-mono font-bold text-right pr-2 transition-colors ${
                  status === 'accepted' ? 'text-emerald-600 dark:text-emerald-400'
                  : status === 'rejected' ? 'text-rose-500 dark:text-rose-400'
                  : status === 'current' ? 'text-yellow-600 dark:text-yellow-400'
                  : 'text-slate-400 dark:text-slate-500'
                }`}>
                  {act.id} {act.label}
                </div>
                <div className="flex-1 h-7 rounded bg-slate-200/60 dark:bg-slate-700/40 relative">
                  {/* Tick marks */}
                  {Array.from({ length: TMAX + 1 }).map((_, t) => (
                    <div key={t} className="absolute top-0 h-full border-l border-slate-300/50 dark:border-slate-600/30" style={{ left: `${(t / TMAX) * 100}%` }}/>
                  ))}
                  {/* Activity bar */}
                  <div
                    className={`absolute top-1 bottom-1 rounded ${STATUS_COLOR[status]} transition-all duration-500 ${isCurrent ? 'ring-2 ring-yellow-400 ring-offset-1' : ''}`}
                    style={{
                      left: `${(act.s / TMAX) * 100}%`,
                      width: `${((act.f - act.s) / TMAX) * 100}%`
                    }}
                  />
                  {/* lastEnd marker */}
                  {cur.lastEnd > 0 && cur.lastEnd <= TMAX && (
                    <div
                      className="absolute top-0 h-full border-l-2 border-dashed border-blue-500 z-10"
                      style={{ left: `${(cur.lastEnd / TMAX) * 100}%` }}
                      title={`lastEnd = ${cur.lastEnd}`}
                    />
                  )}
                </div>
              </div>
            )
          })}
          {/* Legend */}
          <div className="flex flex-wrap gap-3 pt-1 text-xs text-slate-500 dark:text-slate-400">
            {([['bg-slate-300 dark:bg-slate-600', '未处理'], ['bg-yellow-400 dark:bg-yellow-500', '当前检查'], ['bg-emerald-500', '已选（✅）'], ['bg-rose-400', '已跳过（❌）']] as [string,string][]).map(([cls, lbl]) => (
              <span key={lbl} className="flex items-center gap-1"><span className={`inline-block w-4 h-3 rounded ${cls}`}/>{lbl}</span>
            ))}
            <span className="flex items-center gap-1"><span className="inline-block w-0.5 h-4 border-l-2 border-dashed border-blue-500"/>lastEnd</span>
          </div>
        </div>

        {/* Step description */}
        <div className={`rounded-xl border px-4 py-3 ${
          cur.decision === 'accepted' ? 'border-emerald-300 dark:border-emerald-700 bg-emerald-50 dark:bg-emerald-950/30'
          : cur.decision === 'rejected' ? 'border-rose-300 dark:border-rose-700 bg-rose-50 dark:bg-rose-950/30'
          : 'border-blue-300 dark:border-blue-700 bg-blue-50 dark:bg-blue-950/30'
        }`}>
          <div className={`font-bold text-sm ${cur.decision === 'accepted' ? 'text-emerald-700 dark:text-emerald-300' : cur.decision === 'rejected' ? 'text-rose-700 dark:text-rose-300' : 'text-blue-700 dark:text-blue-300'}`}>
            {cur.desc}
          </div>
          <div className="mt-1 text-sm text-slate-600 dark:text-slate-300 leading-relaxed">{cur.detail}</div>
        </div>

        {/* Chosen set */}
        <div className="flex items-center gap-3 flex-wrap">
          <span className="text-xs font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider">已选集合</span>
          {cur.chosen.length === 0
            ? <span className="text-xs text-slate-400 dark:text-slate-500 italic">（空集合）</span>
            : cur.chosen.map(id => (
              <span key={id} className="px-2.5 py-0.5 rounded-full text-xs font-bold bg-emerald-100 dark:bg-emerald-900/50 text-emerald-700 dark:text-emerald-300 border border-emerald-300 dark:border-emerald-700">{id}</span>
            ))
          }
        </div>

        {/* Final state info box */}
        {idx === total && (
          <div className="rounded-xl border border-indigo-300 dark:border-indigo-700 bg-indigo-50 dark:bg-indigo-950/30 px-4 py-3">
            <div className="font-bold text-indigo-700 dark:text-indigo-300 text-sm">算法完成 🎉</div>
            <div className="text-indigo-600 dark:text-indigo-300 text-sm mt-1">
              最优解大小 = <strong>{cur.chosen.length}</strong>，选出活动：{cur.chosen.join(', ')}
            </div>
            <div className="text-xs text-indigo-500 dark:text-indigo-400 mt-1">时间复杂度 O(n log n)（排序主导），空间 O(n)。对于本例，最优解=3个活动。</div>
          </div>
        )}
      </div>
    </div>
  )
}
