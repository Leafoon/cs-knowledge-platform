'use client'
import React, { useState, useEffect, useRef } from 'react'
import { Play, Pause, SkipForward, RotateCcw, Zap } from 'lucide-react'

// ── 数据定义 ──────────────────────────────────────────────────────
const NODES = ['s', 't', 'x', 'y', 'z']
const INF = 9999

// SVG 节点坐标（CLRS 经典布局）
const NODE_POS: Record<string, { cx: number; cy: number }> = {
  s: { cx: 80,  cy: 150 },
  t: { cx: 210, cy: 65  },
  x: { cx: 340, cy: 65  },
  y: { cx: 210, cy: 235 },
  z: { cx: 340, cy: 235 },
}

// 有向边定义
const EDGES = [
  { from: 's', to: 't', w: 10 },
  { from: 's', to: 'y', w: 5  },
  { from: 't', to: 'x', w: 1  },
  { from: 't', to: 'y', w: 2  },
  { from: 'x', to: 'z', w: 4  },
  { from: 'y', to: 't', w: 3  },
  { from: 'y', to: 'x', w: 9  },
  { from: 'y', to: 'z', w: 2  },
  { from: 'z', to: 'x', w: 6  },
  { from: 'z', to: 's', w: 7  },
]

type DijkstraStep = {
  d: number[]            // d[s,t,x,y,z]
  confirmed: string[]    // 已确定最短路径的节点
  queue: { node: string; d: number }[]  // 优先队列内容（已排序）
  extracted: string | null              // 本步提取的节点
  relaxed: { from: string; to: string; newVal: number; updated: boolean }[]
  description: string
  phase: 'init' | 'extract' | 'done'
}

// 预计算所有步骤
const STEPS: DijkstraStep[] = [
  {
    d: [0, INF, INF, INF, INF],
    confirmed: [],
    queue: [{ node: 's', d: 0 }, { node: 't', d: INF }, { node: 'x', d: INF }, { node: 'y', d: INF }, { node: 'z', d: INF }],
    extracted: null,
    relaxed: [],
    description: '初始化：令 d[s]=0，其余节点 d=∞。将所有节点加入优先队列 Q（按 d 值排序）。',
    phase: 'init',
  },
  {
    d: [0, 10, INF, 5, INF],
    confirmed: ['s'],
    queue: [{ node: 'y', d: 5 }, { node: 't', d: 10 }, { node: 'x', d: INF }, { node: 'z', d: INF }],
    extracted: 's',
    relaxed: [
      { from: 's', to: 't', newVal: 10,  updated: true  },
      { from: 's', to: 'y', newVal: 5,   updated: true  },
    ],
    description: 'EXTRACT-MIN: s (d=0)。s 已确定！\n松弛 s→t: 0+10=10 < ∞ ✓ 更新\n松弛 s→y: 0+5=5 < ∞ ✓ 更新',
    phase: 'extract',
  },
  {
    d: [0, 8, 14, 5, 7],
    confirmed: ['s', 'y'],
    queue: [{ node: 'z', d: 7 }, { node: 't', d: 8 }, { node: 'x', d: 14 }],
    extracted: 'y',
    relaxed: [
      { from: 'y', to: 't', newVal: 8,  updated: true  },
      { from: 'y', to: 'x', newVal: 14, updated: true  },
      { from: 'y', to: 'z', newVal: 7,  updated: true  },
    ],
    description: 'EXTRACT-MIN: y (d=5)。y 已确定！\n松弛 y→t: 5+3=8 < 10 ✓ 更新\n松弛 y→x: 5+9=14 < ∞ ✓ 更新\n松弛 y→z: 5+2=7 < ∞ ✓ 更新',
    phase: 'extract',
  },
  {
    d: [0, 8, 13, 5, 7],
    confirmed: ['s', 'y', 'z'],
    queue: [{ node: 't', d: 8 }, { node: 'x', d: 13 }],
    extracted: 'z',
    relaxed: [
      { from: 'z', to: 'x', newVal: 13, updated: true  },
      { from: 'z', to: 's', newVal: 7,  updated: false },
    ],
    description: 'EXTRACT-MIN: z (d=7)。z 已确定！\n松弛 z→x: 7+6=13 < 14 ✓ 更新\n松弛 z→s: s 已确定，跳过',
    phase: 'extract',
  },
  {
    d: [0, 8, 9, 5, 7],
    confirmed: ['s', 'y', 'z', 't'],
    queue: [{ node: 'x', d: 9 }],
    extracted: 't',
    relaxed: [
      { from: 't', to: 'x', newVal: 9, updated: true  },
      { from: 't', to: 'y', newVal: 10, updated: false },
    ],
    description: 'EXTRACT-MIN: t (d=8)。t 已确定！\n松弛 t→x: 8+1=9 < 13 ✓ 更新\n松弛 t→y: y 已确定，跳过',
    phase: 'extract',
  },
  {
    d: [0, 8, 9, 5, 7],
    confirmed: ['s', 'y', 'z', 't', 'x'],
    queue: [],
    extracted: 'x',
    relaxed: [
      { from: 'x', to: 'z', newVal: 13, updated: false },
    ],
    description: 'EXTRACT-MIN: x (d=9)。x 已确定！\n松弛 x→z: z 已确定，跳过\n优先队列已空，算法结束！',
    phase: 'extract',
  },
  {
    d: [0, 8, 9, 5, 7],
    confirmed: ['s', 'y', 'z', 't', 'x'],
    queue: [],
    extracted: null,
    relaxed: [],
    description: '🎉 算法完成！所有节点的最短路径已全部确定。\nd[s]=0, d[t]=8, d[x]=9, d[y]=5, d[z]=7',
    phase: 'done',
  },
]

// ── 辅助：绘制带箭头的有向边 ──────────────────────────────────
function ArrowEdge({
  from, to, weight, color, opacity = 1,
}: {
  from: string; to: string; weight: number; color: string; opacity?: number
}) {
  const f = NODE_POS[from]
  const t = NODE_POS[to]
  const r = 22
  const dx = t.cx - f.cx
  const dy = t.cy - f.cy
  const len = Math.sqrt(dx * dx + dy * dy)
  const ux = dx / len
  const uy = dy / len

  // 边偏移（避免双向箭头重叠）
  const perpX = -uy * 6
  const perpY = ux * 6
  const x1 = f.cx + ux * r + perpX
  const y1 = f.cy + uy * r + perpY
  const x2 = t.cx - ux * r + perpX
  const y2 = t.cy - uy * r + perpY

  // 权值标签位置（中点偏移）
  const midX = (x1 + x2) / 2 + perpX * 1.5
  const midY = (y1 + y2) / 2 + perpY * 1.5

  const id = `arrow-${from}-${to}`
  return (
    <g opacity={opacity}>
      <defs>
        <marker id={id} markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
          <path d="M0,0 L0,6 L8,3 z" fill={color} />
        </marker>
      </defs>
      <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={color} strokeWidth="2" markerEnd={`url(#${id})`} />
      <text x={midX} y={midY} textAnchor="middle" dominantBaseline="middle"
        fontSize="11" fill={color} fontWeight="700"
        className="select-none">{weight}</text>
    </g>
  )
}

// ── 主组件 ────────────────────────────────────────────────────────
export default function DijkstraPriorityQueueViz() {
  const [step, setStep] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1200)
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const cur = STEPS[step]
  const isLast = step === STEPS.length - 1

  useEffect(() => {
    if (playing && !isLast) {
      timerRef.current = setTimeout(() => setStep(s => s + 1), speed)
    } else if (isLast) {
      setPlaying(false)
    }
    return () => { if (timerRef.current) clearTimeout(timerRef.current) }
  }, [playing, step, speed, isLast])

  const handlePlay = () => { if (!isLast) setPlaying(p => !p) }
  const handleStep = () => { setPlaying(false); if (!isLast) setStep(s => s + 1) }
  const handleReset = () => { setPlaying(false); setStep(0) }

  // 判断某节点是否是本步被松弛的目标
  const isRelaxedTarget = (node: string) =>
    cur.relaxed.some(r => r.to === node && r.updated)

  // 获取某条边的颜色
  const getEdgeColor = (from: string, to: string) => {
    const r = cur.relaxed.find(r => r.from === from && r.to === to)
    if (!r) return '#6b7280'  // gray
    return r.updated ? '#10b981' : '#f59e0b'  // green if updated, amber if skipped
  }
  const getEdgeOpacity = (from: string, to: string) => {
    if (cur.relaxed.length === 0) return 0.35
    return cur.relaxed.some(r => r.from === from && r.to === to) ? 1 : 0.25
  }

  // 格式化 d 值
  const fmtD = (v: number) => v === INF ? '∞' : v.toString()

  return (
    <div className="w-full max-w-4xl mx-auto my-6 rounded-2xl overflow-hidden border border-slate-200 dark:border-slate-700 shadow-xl">
      {/* Header */}
      <div className="bg-gradient-to-r from-orange-500 via-amber-500 to-yellow-500 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-xl font-bold text-white">Dijkstra 优先队列动画</h3>
            <p className="text-orange-100 text-sm mt-0.5">CLRS 经典 5 节点图 · 源点 s</p>
          </div>
          <div className="flex items-center gap-2 bg-white/20 rounded-lg px-3 py-1.5">
            <Zap className="w-4 h-4 text-yellow-200" />
            <span className="text-white text-sm font-medium">步骤 {step + 1} / {STEPS.length}</span>
          </div>
        </div>
      </div>

      {/* Progress bar */}
      <div className="h-1.5 bg-slate-200 dark:bg-slate-700">
        <div
          className="h-full bg-gradient-to-r from-orange-400 to-yellow-400 transition-all duration-500"
          style={{ width: `${((step) / (STEPS.length - 1)) * 100}%` }}
        />
      </div>

      {/* Main content */}
      <div className="bg-white dark:bg-slate-900 p-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Left: SVG Graph */}
          <div className="bg-slate-50 dark:bg-slate-800 rounded-xl p-3 border border-slate-200 dark:border-slate-700">
            <div className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-2">图结构</div>
            <svg viewBox="0 0 430 300" className="w-full">
              {/* 边 */}
              {EDGES.map(e => (
                <ArrowEdge
                  key={`${e.from}-${e.to}`}
                  from={e.from} to={e.to} weight={e.w}
                  color={getEdgeColor(e.from, e.to)}
                  opacity={getEdgeOpacity(e.from, e.to)}
                />
              ))}
              {/* 节点 */}
              {NODES.map((n, i) => {
                const { cx, cy } = NODE_POS[n]
                const isConfirmed = cur.confirmed.includes(n)
                const isExtracted = cur.extracted === n
                const isRelaxed = isRelaxedTarget(n)
                const dVal = cur.d[i]

                let fillColor = 'white'
                let strokeColor = '#94a3b8'
                let strokeWidth = 2

                if (isExtracted) { fillColor = '#f97316'; strokeColor = '#ea580c'; strokeWidth = 3.5 }
                else if (isConfirmed) { fillColor = '#10b981'; strokeColor = '#059669'; strokeWidth = 2.5 }
                else if (isRelaxed) { fillColor = '#a78bfa'; strokeColor = '#7c3aed'; strokeWidth = 2.5 }

                return (
                  <g key={n}>
                    <circle cx={cx} cy={cy} r={22}
                      fill={fillColor} stroke={strokeColor} strokeWidth={strokeWidth}
                      className="transition-all duration-500"
                    />
                    <text x={cx} y={cy - 3} textAnchor="middle" dominantBaseline="middle"
                      fontSize="15" fontWeight="800" fill={isConfirmed || isExtracted ? 'white' : '#1e293b'}
                      className="select-none dark:fill-slate-700"
                    >{n}</text>
                    <text x={cx} y={cy + 11} textAnchor="middle" dominantBaseline="middle"
                      fontSize="10" fill={isConfirmed || isExtracted ? '#d1fae5' : '#64748b'}
                      className="select-none"
                    >{fmtD(dVal)}</text>
                  </g>
                )
              })}
            </svg>
            {/* Legend */}
            <div className="flex flex-wrap gap-2 mt-1 px-1">
              {[
                { color: 'bg-orange-500', label: '正在处理' },
                { color: 'bg-emerald-500', label: '已确定' },
                { color: 'bg-violet-400', label: '被更新' },
              ].map(l => (
                <div key={l.label} className="flex items-center gap-1 text-xs text-slate-600 dark:text-slate-400">
                  <span className={`w-3 h-3 rounded-full ${l.color}`} />
                  {l.label}
                </div>
              ))}
            </div>
          </div>

          {/* Right: Priority Queue + Step info */}
          <div className="flex flex-col gap-3">
            {/* Priority Queue */}
            <div className="bg-slate-50 dark:bg-slate-800 rounded-xl p-3 border border-slate-200 dark:border-slate-700">
              <div className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-2">
                优先队列 Q（按 d 值排序）
              </div>
              {cur.queue.length === 0 ? (
                <div className="text-center py-3 text-slate-400 dark:text-slate-500 text-sm">队列已空 ✓</div>
              ) : (
                <div className="flex flex-wrap gap-2">
                  {cur.queue.map((item, idx) => (
                    <div key={item.node}
                      className={`flex items-center gap-2 px-3 py-2 rounded-lg border text-sm font-semibold transition-all duration-300
                        ${idx === 0
                          ? 'bg-orange-100 dark:bg-orange-900/40 border-orange-300 dark:border-orange-600 text-orange-800 dark:text-orange-200'
                          : 'bg-white dark:bg-slate-700 border-slate-200 dark:border-slate-600 text-slate-700 dark:text-slate-200'}`}
                    >
                      {idx === 0 && <span className="text-xs text-orange-500 font-bold">MIN</span>}
                      <span>{item.node}</span>
                      <span className="text-xs opacity-70">d={item.d === INF ? '∞' : item.d}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Relax details */}
            {cur.relaxed.length > 0 && (
              <div className="bg-slate-50 dark:bg-slate-800 rounded-xl p-3 border border-slate-200 dark:border-slate-700">
                <div className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-2">本步松弛操作</div>
                <div className="flex flex-col gap-1.5">
                  {cur.relaxed.map(r => (
                    <div key={`${r.from}-${r.to}`} className={`flex items-center gap-2 text-xs px-2.5 py-1.5 rounded-md
                      ${r.updated
                        ? 'bg-emerald-50 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-300 border border-emerald-200 dark:border-emerald-700'
                        : 'bg-amber-50 dark:bg-amber-900/20 text-amber-700 dark:text-amber-400 border border-amber-200 dark:border-amber-700 line-through opacity-60'}`}
                    >
                      <span className={`font-bold text-sm ${r.updated ? '' : 'no-underline'}`}>
                        {r.from}→{r.to}
                      </span>
                      <span>{r.updated ? `d=${r.newVal} ✓` : '已确定/无需更新'}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Step description */}
            <div className={`rounded-xl p-3 border text-sm leading-relaxed
              ${cur.phase === 'done'
                ? 'bg-emerald-50 dark:bg-emerald-900/20 border-emerald-200 dark:border-emerald-700 text-emerald-800 dark:text-emerald-200'
                : 'bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-700 text-blue-800 dark:text-blue-200'}`}
            >
              {cur.description.split('\n').map((line, i) => <div key={i}>{line}</div>)}
            </div>
          </div>
        </div>

        {/* d[] table */}
        <div className="mt-4 bg-slate-50 dark:bg-slate-800 rounded-xl p-3 border border-slate-200 dark:border-slate-700">
          <div className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-2">最短路径距离 d[]</div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm text-center">
              <thead>
                <tr>
                  {NODES.map(n => (
                    <th key={n} className="px-3 py-1.5 font-bold text-slate-700 dark:text-slate-300 bg-slate-100 dark:bg-slate-700 rounded">d[{n}]</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                <tr>
                  {cur.d.map((v, i) => (
                    <td key={i} className={`px-3 py-2 font-mono font-bold rounded transition-all duration-300
                      ${cur.confirmed.includes(NODES[i])
                        ? 'text-emerald-600 dark:text-emerald-400'
                        : isRelaxedTarget(NODES[i])
                          ? 'text-violet-600 dark:text-violet-400'
                          : 'text-slate-700 dark:text-slate-300'}`}
                    >
                      {fmtD(v)}
                    </td>
                  ))}
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        {/* Controls */}
        <div className="mt-4 flex items-center justify-between flex-wrap gap-2">
          <div className="flex items-center gap-2">
            <button onClick={handleReset}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-600 text-sm transition-colors">
              <RotateCcw className="w-3.5 h-3.5" /> 重置
            </button>
            <button onClick={handlePlay} disabled={isLast}
              className="flex items-center gap-1.5 px-4 py-1.5 rounded-lg bg-orange-500 hover:bg-orange-600 text-white text-sm font-semibold transition-colors disabled:opacity-40 disabled:cursor-not-allowed">
              {playing ? <><Pause className="w-3.5 h-3.5" /> 暂停</> : <><Play className="w-3.5 h-3.5" /> 自动播放</>}
            </button>
            <button onClick={handleStep} disabled={isLast}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-amber-100 dark:bg-amber-900/40 text-amber-700 dark:text-amber-300 hover:bg-amber-200 dark:hover:bg-amber-900/60 text-sm transition-colors disabled:opacity-40 disabled:cursor-not-allowed">
              <SkipForward className="w-3.5 h-3.5" /> 下一步
            </button>
          </div>
          <div className="flex items-center gap-2 text-sm text-slate-500 dark:text-slate-400">
            <span>速度</span>
            {[2000, 1200, 700].map(s => (
              <button key={s} onClick={() => setSpeed(s)}
                className={`px-2 py-1 rounded text-xs ${speed === s ? 'bg-orange-500 text-white' : 'bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-300'}`}>
                {s === 2000 ? '慢' : s === 1200 ? '中' : '快'}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
