'use client'
import React, { useState, useCallback, useEffect, useRef } from 'react'
import { Play, Pause, SkipForward, RotateCcw, ChevronRight, ChevronLeft } from 'lucide-react'

// ── 4-节点图（无负权环）─────────────────────────────────────────────
// 顶点: A(0) B(1) C(2) D(3)
// 边: A→B:1, A→C:5, B→C:2, B→D:4, C→D:1, D→A:3
const LABELS = ['A', 'B', 'C', 'D']
const N = 4
const INF = Infinity

// SVG 布局
const NODE_POS = [
  { x: 65,  y: 105 }, // A
  { x: 175, y: 42  }, // B
  { x: 175, y: 162 }, // C
  { x: 290, y: 105 }, // D
]
const EDGES = [
  { u: 0, v: 1, w: 1 },
  { u: 0, v: 2, w: 5 },
  { u: 1, v: 2, w: 2 },
  { u: 1, v: 3, w: 4 },
  { u: 2, v: 3, w: 1 },
  { u: 3, v: 0, w: 3 },
]

// ── 预计算所有 Floyd-Warshall 步骤 ─────────────────────────────────
interface FWStep {
  k: number           // 当前中转站(-1=初始化)
  dist: number[][]
  pred: number[][]
  updated: [number, number][] // 本步更新的 (i,j)
  desc: string
}

function buildSteps(): FWStep[] {
  const steps: FWStep[] = []

  // ── 初始化矩阵 ────
  const dist: number[][] = Array.from({ length: N }, (_, i) =>
    Array.from({ length: N }, (_, j) => (i === j ? 0 : INF))
  )
  const pred: number[][] = Array.from({ length: N }, () => Array(N).fill(-1))
  for (const { u, v, w } of EDGES) {
    dist[u][v] = w
    pred[u][v] = u
  }

  steps.push({
    k: -1,
    dist: dist.map(r => [...r]),
    pred: pred.map(r => [...r]),
    updated: [],
    desc: '初始化：矩阵 D⁽⁰⁾ = 原图权值。d[i][j]=w(i,j) 若有直接边，d[i][j]=∞ 若无边，d[i][i]=0。不允许任何中间节点。',
  })

  // ── k = 0..N-1 ────
  for (let k = 0; k < N; k++) {
    const updated: [number, number][] = []
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        if (dist[i][k] !== INF && dist[k][j] !== INF) {
          const via = dist[i][k] + dist[k][j]
          if (via < dist[i][j]) {
            dist[i][j] = via
            pred[i][j] = pred[k][j]
            updated.push([i, j])
          }
        }
      }
    }
    const label = LABELS[k]
    steps.push({
      k,
      dist: dist.map(r => [...r]),
      pred: pred.map(r => [...r]),
      updated,
      desc: updated.length > 0
        ? `引入中转站 ${label}（k=${k}）：检查所有路径 i→${label}→j。${updated.map(([i, j]) => `d[${LABELS[i]}][${LABELS[j]}]`).join('、')} 被更新。`
        : `引入中转站 ${label}（k=${k}）：检查所有路径 i→${label}→j。本轮无更新，原有路径已最优。`,
    })
  }
  return steps
}

const STEPS = buildSteps()

// ── 辅助：箭头边 ─────────────────────────────────────────────────
function Arrow({
  u, v, w, highlight,
}: {
  u: number; v: number; w: number; highlight: 'active' | 'normal' | 'dim'
}) {
  const { x: x1, y: y1 } = NODE_POS[u]
  const { x: x2, y: y2 } = NODE_POS[v]
  const dx = x2 - x1, dy = y2 - y1
  const len = Math.hypot(dx, dy)
  const R = 20
  const sx = x1 + (dx / len) * R, sy = y1 + (dy / len) * R
  const ex = x2 - (dx / len) * R, ey = y2 - (dy / len) * R
  const mx = (sx + ex) / 2, my = (sy + ey) / 2

  const color = highlight === 'active' ? '#f59e0b' : highlight === 'dim' ? '#e2e8f0' : '#94a3b8'
  const sw = highlight === 'active' ? 2.5 : 1.5
  const mid = `arr-fw-${u}-${v}`

  return (
    <g>
      <defs>
        <marker id={mid} markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
          <path d="M0,0 L7,3.5 L0,7 Z" fill={color} />
        </marker>
      </defs>
      <line x1={sx} y1={sy} x2={ex} y2={ey} stroke={color} strokeWidth={sw} markerEnd={`url(#${mid})`} />
      <text x={mx + dy * 0.12} y={my - dx * 0.12} textAnchor="middle"
        fontSize={10} fontWeight="bold" fill={color}>{w}</text>
    </g>
  )
}

// ── 矩阵单元格 ──────────────────────────────────────────────────
function Cell({
  value, i, j, k, currentStep, updated,
}: {
  value: number; i: number; j: number; k: number; currentStep: number; updated: [number,number][]
}) {
  const isUpdated = currentStep > 0 && updated.some(([a, b]) => a === i && b === j)
  const isPivotRow = currentStep > 0 && k >= 0 && i === k
  const isPivotCol = currentStep > 0 && k >= 0 && j === k
  const isDiag = i === j
  const val = value === INF ? '∞' : String(value)

  let bg = ''
  if (isDiag) bg = 'bg-slate-100 dark:bg-slate-800'
  else if (isUpdated) bg = 'bg-emerald-100 dark:bg-emerald-900/40 ring-2 ring-emerald-400'
  else if (isPivotRow || isPivotCol) bg = 'bg-amber-50 dark:bg-amber-900/20'

  return (
    <td className={`w-12 h-10 text-center text-sm font-mono font-bold transition-all duration-300 ${bg} ${
      isUpdated
        ? 'text-emerald-700 dark:text-emerald-300'
        : isDiag
          ? 'text-slate-400 dark:text-slate-500'
          : value === INF
            ? 'text-slate-300 dark:text-slate-600'
            : (isPivotRow || isPivotCol)
              ? 'text-amber-700 dark:text-amber-300'
              : 'text-slate-700 dark:text-slate-200'
    }`}>
      {val}
    </td>
  )
}

// ── 主组件 ──────────────────────────────────────────────────────
export default function FloydWarshallDP() {
  const [idx, setIdx] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1800)
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const step = STEPS[idx]

  const advance = useCallback(() => {
    setIdx(p => {
      if (p >= STEPS.length - 1) { setPlaying(false); return p }
      return p + 1
    })
  }, [])

  useEffect(() => {
    if (playing) intervalRef.current = setInterval(advance, speed)
    else { if (intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = null } }
    return () => { if (intervalRef.current) clearInterval(intervalRef.current) }
  }, [playing, speed, advance])

  const isDone = idx === STEPS.length - 1
  const k = step.k

  function nodeColor(ni: number) {
    if (k >= 0 && ni === k) return { fill: '#f59e0b', stroke: '#d97706', text: '#fff' }
    if (isDone) return { fill: '#10b981', stroke: '#059669', text: '#fff' }
    return { fill: '#6366f1', stroke: '#4f46e5', text: '#fff' }
  }

  return (
    <div className="w-full max-w-3xl mx-auto rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 shadow-xl overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-sky-600 via-blue-600 to-indigo-600 px-5 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-white font-bold text-lg tracking-tight">Floyd-Warshall — DP 矩阵逐步推导</h3>
            <p className="text-sky-200 text-sm mt-0.5">4 顶点示例 · 引入中转站 k=0→3，观察矩阵变化</p>
          </div>
          <div className="bg-white/20 backdrop-blur-sm rounded-xl px-3 py-1.5 text-white text-sm font-mono">
            {k < 0 ? '⚙️ 初始化' : isDone ? '✅ 完成' : `k = ${LABELS[k]}`}
          </div>
        </div>
      </div>

      <div className="p-4 space-y-4">
        {/* Stage pills */}
        <div className="flex gap-1.5">
          {STEPS.map((s, i) => (
            <button key={i} onClick={() => { setPlaying(false); setIdx(i) }}
              className={`px-2.5 py-1 rounded-lg text-[11px] font-bold transition-all ${
                i === idx
                  ? isDone ? 'bg-emerald-500 text-white scale-110 shadow' : 'bg-blue-600 text-white scale-110 shadow'
                  : i < idx ? 'bg-slate-200 dark:bg-slate-700 text-slate-500' : 'bg-slate-100 dark:bg-slate-800 text-slate-300 dark:text-slate-600'
              }`}>
              {s.k < 0 ? '初始' : `k=${LABELS[s.k]}`}
            </button>
          ))}
        </div>

        <div className="flex gap-4 items-start">
          {/* Graph SVG */}
          <div className="w-[340px] shrink-0 rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50 overflow-hidden">
            <div className="px-3 py-2 text-[11px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wide border-b border-slate-200 dark:border-slate-700">
              原图
            </div>
            <svg viewBox="0 0 360 215" className="w-full">
              {EDGES.map((e, i) => {
                const hl = k >= 0 && (e.u === k || e.v === k) ? 'active' : k >= 0 ? 'dim' : 'normal'
                return <Arrow key={i} {...e} highlight={hl} />
              })}
              {NODE_POS.map((pos, ni) => {
                const c = nodeColor(ni)
                const isK = k >= 0 && ni === k
                return (
                  <g key={ni}>
                    {isK && <circle cx={pos.x} cy={pos.y} r={30} fill="#fde68a44" stroke="#f59e0b60" strokeWidth={2} />}
                    <circle cx={pos.x} cy={pos.y} r={20} fill={c.fill} stroke={c.stroke} strokeWidth={2.5}
                      className="transition-all duration-400" />
                    <text x={pos.x} y={pos.y + 5} textAnchor="middle" fontSize={14} fontWeight="bold" fill={c.text}>
                      {LABELS[ni]}
                    </text>
                    {isK && (
                      <text x={pos.x} y={pos.y - 28} textAnchor="middle" fontSize={9} fontWeight="bold" fill="#d97706">
                        中转站 k
                      </text>
                    )}
                  </g>
                )
              })}
            </svg>
          </div>

          {/* Matrix */}
          <div className="flex-1 min-w-0">
            <div className="rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
              <div className="px-3 py-2 text-[11px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wide border-b border-slate-200 dark:border-slate-700 flex items-center justify-between">
                <span>距离矩阵 D{k < 0 ? '⁽⁰⁾' : `⁽${k+1}⁾`}</span>
                <div className="flex gap-3 text-[10px] font-medium">
                  <span className="flex items-center gap-1">
                    <span className="w-3 h-3 rounded bg-emerald-400 inline-block" /> 更新
                  </span>
                  <span className="flex items-center gap-1">
                    <span className="w-3 h-3 rounded bg-amber-200 inline-block" /> 辅助行/列
                  </span>
                </div>
              </div>
              <table className="w-full border-collapse">
                <thead>
                  <tr>
                    <th className="w-8 py-2 text-center text-[10px] text-slate-400 font-medium bg-slate-50 dark:bg-slate-800/50"></th>
                    {LABELS.map((l, j) => (
                      <th key={j} className={`py-2 text-center text-xs font-bold ${
                        k >= 0 && j === k ? 'text-amber-600 dark:text-amber-400' : 'text-slate-500 dark:text-slate-400'
                      }`}>{l}</th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 dark:divide-slate-700/50">
                  {LABELS.map((rl, i) => (
                    <tr key={i} className="divide-x divide-slate-100 dark:divide-slate-700/50">
                      <td className={`pl-3 text-xs font-bold ${
                        k >= 0 && i === k ? 'text-amber-600 dark:text-amber-400' : 'text-slate-500 dark:text-slate-400'
                      }`}>{rl}</td>
                      {LABELS.map((_, j) => (
                        <Cell key={j} value={step.dist[i][j]} i={i} j={j} k={k} currentStep={idx} updated={step.updated} />
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Updated summary */}
            <div className="mt-2 flex flex-wrap gap-1">
              {step.updated.map(([i, j], x) => (
                <span key={x} className="px-2 py-0.5 bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-300 rounded-md text-[10px] font-mono font-bold border border-emerald-200 dark:border-emerald-700">
                  d[{LABELS[i]}][{LABELS[j]}] → {step.dist[i][j]}
                </span>
              ))}
              {step.updated.length === 0 && idx > 0 && (
                <span className="px-2 py-0.5 bg-slate-100 dark:bg-slate-800 text-slate-400 rounded-md text-[10px] font-mono">
                  本轮无更新
                </span>
              )}
            </div>
          </div>
        </div>

        {/* Description */}
        <div className="rounded-xl bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-700/40 px-4 py-2.5 text-sm text-blue-800 dark:text-blue-300 leading-relaxed">
          {step.desc}
        </div>

        {/* Controls */}
        <div className="flex flex-wrap items-center gap-2">
          <button onClick={() => { setPlaying(false); setIdx(0) }}
            className="p-1.5 rounded-lg border border-slate-300 dark:border-slate-600 text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors">
            <RotateCcw size={14} />
          </button>
          <button onClick={() => { setPlaying(false); setIdx(i => Math.max(0, i - 1)) }} disabled={idx === 0}
            className="p-1.5 rounded-lg border border-slate-300 dark:border-slate-600 text-slate-600 dark:text-slate-300 disabled:opacity-40 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors">
            <ChevronLeft size={14} />
          </button>
          <button onClick={() => setPlaying(p => !p)} disabled={isDone}
            className={`flex items-center gap-1.5 px-4 py-1.5 rounded-lg text-sm font-bold transition-colors ${
              playing ? 'bg-amber-500 hover:bg-amber-600 text-white' : 'bg-blue-600 hover:bg-blue-700 text-white'
            } disabled:opacity-40`}>
            {playing ? <><Pause size={14} />暂停</> : <><Play size={14} />{idx === 0 ? '开始' : '继续'}</>}
          </button>
          <button onClick={() => { setPlaying(false); setIdx(i => Math.min(STEPS.length - 1, i + 1)) }} disabled={isDone}
            className="p-1.5 rounded-lg border border-slate-300 dark:border-slate-600 text-slate-600 dark:text-slate-300 disabled:opacity-40 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors">
            <ChevronRight size={14} />
          </button>
          <span className="text-xs text-slate-400 font-mono ml-1">{idx + 1}/{STEPS.length}</span>
          <div className="flex items-center gap-1.5 ml-auto text-xs text-slate-500">
            <span>速度</span>
            <input type="range" min={600} max={2500} step={200} value={speed}
              onChange={e => setSpeed(Number(e.target.value))} className="w-20 accent-blue-500" />
          </div>
        </div>
        <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-1">
          <div className="bg-blue-500 h-1 rounded-full transition-all duration-300"
            style={{ width: `${(idx / (STEPS.length - 1)) * 100}%` }} />
        </div>
      </div>
    </div>
  )
}
