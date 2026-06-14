'use client'
import React, { useEffect, useRef, useState } from 'react'

const POINTS = [
  { id: 'A', x: 90, y: 90, side: 'L' },
  { id: 'B', x: 120, y: 180, side: 'L' },
  { id: 'C', x: 180, y: 130, side: 'L' },
  { id: 'D', x: 260, y: 105, side: 'R' },
  { id: 'E', x: 300, y: 165, side: 'R' },
  { id: 'F', x: 330, y: 220, side: 'R' },
]

const MID_X = 220
const DELTA = 82
const STRIP_IDS = ['C', 'D', 'E']

interface Step {
  title: string
  desc: string
  activePoints: string[]
  activePair?: [string, string]
  bestPair?: [string, string]
  leftBest?: number
  rightBest?: number
  delta?: number
  phase: 'divide' | 'strip' | 'done'
}

const STEPS: Step[] = [
  {
    title: '先按 x 坐标排序并从中线拆分',
    desc: 'Divide：按 x 坐标排序后，以中线 x=220 把点集分成左右两半。左侧与右侧可以独立递归求最近点对。',
    activePoints: [],
    phase: 'divide',
  },
  {
    title: '左半部分得到最近距离 dL',
    desc: '左半部分最近点是 A-C，距离约 98；这里为了演示，重点是“左右子问题彼此独立”。',
    activePoints: ['A', 'C'],
    activePair: ['A', 'C'],
    leftBest: 98,
    phase: 'divide',
  },
  {
    title: '右半部分得到最近距离 dR',
    desc: '右半部分最近点是 D-E，距离约 72。于是全局候选距离 δ = min(dL, dR) = 72。',
    activePoints: ['D', 'E'],
    activePair: ['D', 'E'],
    leftBest: 98,
    rightBest: 72,
    delta: 72,
    phase: 'divide',
  },
  {
    title: '构造中线附近宽度 2δ 的 strip',
    desc: 'Combine：现在只需检查落在中线附近的带状区域。因为离中线太远的点，不可能与另一侧形成更短距离。',
    activePoints: STRIP_IDS,
    leftBest: 98,
    rightBest: 72,
    delta: 72,
    phase: 'strip',
  },
  {
    title: '按 y 坐标检查 strip 中的候选点对',
    desc: 'strip 中按 y 排序后，每个点只需向后检查常数个点。这里比较 C-D，可得到更短距离约 83；还没有打破 δ。',
    activePoints: ['C', 'D'],
    activePair: ['C', 'D'],
    leftBest: 98,
    rightBest: 72,
    delta: 72,
    phase: 'strip',
  },
  {
    title: '继续检查 D-E，保持当前最好值',
    desc: 'D-E 的距离仍是 72，所以当前最优值没有被 strip 中其他点对刷新。',
    activePoints: ['D', 'E'],
    activePair: ['D', 'E'],
    bestPair: ['D', 'E'],
    leftBest: 98,
    rightBest: 72,
    delta: 72,
    phase: 'strip',
  },
  {
    title: '算法完成：最近点对为 D-E',
    desc: '最终答案来自右半部分：D-E。这个例子说明了最近点对分治的关键：左右递归后，只需在线性时间检查 strip，而不用重新枚举所有跨边界点对。',
    activePoints: ['D', 'E'],
    bestPair: ['D', 'E'],
    leftBest: 98,
    rightBest: 72,
    delta: 72,
    phase: 'done',
  },
]

function dist(a: { x: number; y: number }, b: { x: number; y: number }) {
  return Math.hypot(a.x - b.x, a.y - b.y)
}

export default function ClosestPairStrip() {
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

  const pointMap = Object.fromEntries(POINTS.map(p => [p.id, p])) as Record<string, (typeof POINTS)[number]>
  const activeDistance = step.activePair ? dist(pointMap[step.activePair[0]], pointMap[step.activePair[1]]) : null

  return (
    <div className="w-full max-w-4xl mx-auto rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 shadow-xl overflow-hidden">
      <div className="bg-gradient-to-r from-sky-600 via-cyan-600 to-blue-600 px-5 py-4">
        <h3 className="text-white font-bold text-lg tracking-tight">最近点对：Strip 合并动画</h3>
        <p className="text-sky-100 text-sm mt-0.5">左右递归 + 中线带状区域检查 · O(n log n)</p>
      </div>

      <div className="px-4 py-3 flex flex-wrap gap-2 items-center border-b border-slate-100 dark:border-slate-800 bg-slate-50 dark:bg-slate-800/40">
        <div className="flex rounded-lg overflow-hidden text-[10px] font-bold border border-slate-200 dark:border-slate-600">
          <div className={`px-3 py-1.5 ${step.phase === 'divide' ? 'bg-sky-600 text-white' : 'bg-slate-100 dark:bg-slate-700 text-slate-400'}`}>Divide</div>
          <div className={`px-3 py-1.5 ${step.phase === 'strip' ? 'bg-cyan-600 text-white' : 'bg-slate-100 dark:bg-slate-700 text-slate-400'}`}>Strip</div>
          <div className={`px-3 py-1.5 ${step.phase === 'done' ? 'bg-emerald-600 text-white' : 'bg-slate-100 dark:bg-slate-700 text-slate-400'}`}>Done</div>
        </div>
        <div className="flex gap-1.5 ml-auto">
          <button onClick={() => { setPlaying(false); setCur(0) }} className="px-2 py-1.5 rounded-lg text-xs bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-300">⏮</button>
          <button onClick={() => { setPlaying(false); setCur(v => Math.max(0, v - 1)) }} disabled={cur === 0} className="px-2 py-1.5 rounded-lg text-xs disabled:opacity-40 bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-300">◀</button>
          <button onClick={() => setPlaying(v => !v)} disabled={cur >= STEPS.length - 1} className="px-4 py-1.5 rounded-lg text-xs font-bold bg-sky-600 hover:bg-sky-700 text-white disabled:opacity-40">{playing ? '⏸ 暂停' : '▶ 播放'}</button>
          <button onClick={() => { setPlaying(false); setCur(v => Math.min(STEPS.length - 1, v + 1)) }} disabled={cur >= STEPS.length - 1} className="px-2 py-1.5 rounded-lg text-xs disabled:opacity-40 bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-300">▶</button>
        </div>
        <div className="flex items-center gap-1.5 text-[10px] text-slate-400">
          <input type="range" min={900} max={3200} step={300} value={speed} onChange={e => setSpeed(Number(e.target.value))} className="w-16 accent-sky-500" />
          <span>{(speed / 1000).toFixed(1)}s</span>
        </div>
      </div>

      <div className="p-4 space-y-4">
        <div className="flex items-center gap-2">
          <span className="px-2 py-0.5 rounded-full text-[10px] font-bold text-white bg-sky-500">{cur + 1}/{STEPS.length}</span>
          <span className="font-semibold text-sm text-slate-700 dark:text-slate-200">{step.title}</span>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">
          <div className="lg:col-span-3 rounded-2xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/30 overflow-hidden">
            <svg viewBox="40 40 360 240" className="w-full" style={{ maxHeight: 300 }}>
              {step.delta && (
                <rect
                  x={MID_X - step.delta}
                  y={45}
                  width={step.delta * 2}
                  height={220}
                  fill="#06b6d4"
                  opacity={0.08}
                  stroke="#06b6d4"
                  strokeDasharray="6 4"
                />
              )}
              <line x1={MID_X} y1={45} x2={MID_X} y2={265} stroke="#64748b" strokeDasharray="6 4" strokeWidth={1.5} />
              <text x={MID_X} y={38} textAnchor="middle" fontSize={10} fill="#64748b">中线</text>

              {step.activePair && (
                <line
                  x1={pointMap[step.activePair[0]].x}
                  y1={pointMap[step.activePair[0]].y}
                  x2={pointMap[step.activePair[1]].x}
                  y2={pointMap[step.activePair[1]].y}
                  stroke={step.phase === 'done' ? '#10b981' : '#f59e0b'}
                  strokeWidth={3.5}
                  opacity={0.9}
                />
              )}

              {step.bestPair && (
                <line
                  x1={pointMap[step.bestPair[0]].x}
                  y1={pointMap[step.bestPair[0]].y}
                  x2={pointMap[step.bestPair[1]].x}
                  y2={pointMap[step.bestPair[1]].y}
                  stroke="#10b981"
                  strokeWidth={5}
                  opacity={0.25}
                />
              )}

              {POINTS.map(p => {
                const active = step.activePoints.includes(p.id)
                const inStrip = step.delta ? Math.abs(p.x - MID_X) < step.delta : false
                const best = step.bestPair?.includes(p.id)
                return (
                  <g key={p.id}>
                    {best && <circle cx={p.x} cy={p.y} r={18} fill="#10b981" opacity={0.18} />}
                    <circle
                      cx={p.x}
                      cy={p.y}
                      r={11}
                      fill={best ? '#10b981' : active ? '#f59e0b' : inStrip ? '#06b6d4' : p.side === 'L' ? '#3b82f6' : '#8b5cf6'}
                    />
                    <text x={p.x} y={p.y + 4} textAnchor="middle" fontSize={10} fontWeight="bold" fill="white">{p.id}</text>
                    <text x={p.x} y={p.y - 18} textAnchor="middle" fontSize={8} fill="#64748b">({p.x},{p.y})</text>
                  </g>
                )
              })}
            </svg>
          </div>

          <div className="lg:col-span-2 space-y-3">
            <div className="rounded-2xl border border-blue-200 dark:border-blue-700/60 bg-blue-50 dark:bg-blue-950/30 p-4">
              <div className="text-[10px] uppercase tracking-wider text-blue-600 dark:text-blue-400 font-bold mb-2">左右子问题</div>
              <div className="space-y-2 text-[11px] text-slate-600 dark:text-slate-300">
                <div className="flex items-center justify-between rounded-xl bg-white dark:bg-slate-900 border border-blue-200 dark:border-blue-700 px-3 py-2"><span>左半 dL</span><span className="font-black text-blue-600 dark:text-blue-400">{step.leftBest ?? '—'}</span></div>
                <div className="flex items-center justify-between rounded-xl bg-white dark:bg-slate-900 border border-purple-200 dark:border-purple-700 px-3 py-2"><span>右半 dR</span><span className="font-black text-purple-600 dark:text-purple-400">{step.rightBest ?? '—'}</span></div>
                <div className="flex items-center justify-between rounded-xl bg-white dark:bg-slate-900 border border-cyan-200 dark:border-cyan-700 px-3 py-2"><span>δ = min(dL,dR)</span><span className="font-black text-cyan-600 dark:text-cyan-400">{step.delta ?? '—'}</span></div>
              </div>
            </div>

            <div className="rounded-2xl border border-cyan-200 dark:border-cyan-700/60 bg-cyan-50 dark:bg-cyan-950/30 p-4">
              <div className="text-[10px] uppercase tracking-wider text-cyan-600 dark:text-cyan-400 font-bold mb-2">当前候选点对</div>
              {step.activePair ? (
                <div className="space-y-2 text-[11px] text-slate-600 dark:text-slate-300">
                  <div className="inline-flex px-3 py-1.5 rounded-full bg-white dark:bg-slate-900 border border-cyan-200 dark:border-cyan-700 font-bold text-cyan-700 dark:text-cyan-300">{step.activePair[0]} - {step.activePair[1]}</div>
                  <div>距离 ≈ <span className="font-black text-cyan-700 dark:text-cyan-300">{activeDistance?.toFixed(1)}</span></div>
                </div>
              ) : (
                <div className="text-[11px] text-slate-400 italic">当前还没有选中候选点对</div>
              )}
            </div>

            <div className="rounded-2xl border border-emerald-200 dark:border-emerald-700/60 bg-emerald-50 dark:bg-emerald-950/30 p-4">
              <div className="text-[10px] uppercase tracking-wider text-emerald-600 dark:text-emerald-400 font-bold mb-2">7 点引理的意义</div>
              <p className="text-[11px] text-slate-600 dark:text-slate-300">strip 中的点虽然跨越左右边界，但因为已经按 y 排序且几何上过于密集会违反 δ 的定义，所以每个点只需比较后面常数个点，而不是全部重扫。</p>
            </div>
          </div>
        </div>

        <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/30 px-4 py-3 text-[12px] text-slate-600 dark:text-slate-300">
          {step.desc}
        </div>

        <div className="h-1.5 rounded-full bg-slate-200 dark:bg-slate-700 overflow-hidden">
          <div className="h-full rounded-full bg-gradient-to-r from-sky-500 via-cyan-500 to-blue-500 transition-all duration-500" style={{ width: `${(cur / (STEPS.length - 1)) * 100}%` }} />
        </div>
      </div>
    </div>
  )
}
