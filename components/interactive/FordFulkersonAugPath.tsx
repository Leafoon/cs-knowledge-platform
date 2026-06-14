'use client'
import React, { useState, useEffect, useRef } from 'react'

const NODES = [
  { id: 0, label: 's', x: 50,  y: 110 },
  { id: 1, label: 'A', x: 160, y: 42  },
  { id: 2, label: 'B', x: 290, y: 42  },
  { id: 3, label: 'C', x: 160, y: 178 },
  { id: 4, label: 'D', x: 290, y: 178 },
  { id: 5, label: 't', x: 390, y: 110 },
]

// [u, v, cap]
const ORIG_EDGES = [
  [0,1,10],[0,3,10],[1,2,4],
  [1,3,2], [1,4,8], [3,4,9],
  [2,5,10],[4,5,10],
] as const

// Pre-computed augmentation steps
interface AugStep {
  label: string
  path: number[]            // node ids in augmenting path
  pathEdgeIdx: number[]     // ORIG_EDGES indices on path
  bottleneck: number
  delta: number             // traffic added
  flows: number[]           // cumulative flow AFTER this step
  totalFlow: number
  desc: string
}

const STEPS: AugStep[] = [
  {
    label: '初始状态',
    path: [], pathEdgeIdx: [], bottleneck: 0, delta: 0,
    flows: [0,0,0,0,0,0,0,0], totalFlow: 0,
    desc: '所有流量为 0。BFS 从 s 出发搜索增广路。',
  },
  {
    label: '路径 1: s→A→B→t',
    path: [0,1,2,5], pathEdgeIdx: [0,2,6], bottleneck: 4, delta: 4,
    flows: [4,0,4,0,0,0,4,0], totalFlow: 4,
    desc: 'BFS 找到路径 s→A→B→t，瓶颈边 A→B(cap=4)，增广 Δ=4。A→B 饱和，残差图出现反向弧 B→A(4)。',
  },
  {
    label: '路径 2: s→A→D→t',
    path: [0,1,4,5], pathEdgeIdx: [0,4,7], bottleneck: 6, delta: 6,
    flows: [10,0,4,0,6,0,4,6], totalFlow: 10,
    desc: 'BFS 找到路径 s→A→D→t，瓶颈边 A→D(剩余 8-0=8) 受限于 s→A(剩余 10-4=6)，Δ=6。s→A 饱和。',
  },
  {
    label: '路径 3: s→C→D→t',
    path: [0,3,4,5], pathEdgeIdx: [1,5,7], bottleneck: 4, delta: 4,
    flows: [10,4,4,0,6,4,4,10], totalFlow: 14,
    desc: 'BFS 找到路径 s→C→D→t，瓶颈边受 D→t 剩余 10-6=4 限制，Δ=4。无更多增广路，算法终止。',
  },
]

function edgeColor(idx: number, step: AugStep) {
  if (step.pathEdgeIdx.includes(idx)) return '#f59e0b'
  const f = step.flows[idx], cap = ORIG_EDGES[idx][2]
  if (f >= cap && cap > 0) return '#ef4444'
  if (f > 0) return '#3b82f6'
  return '#94a3b8'
}

function nodeColor(id: number, step: AugStep) {
  if (step.path.includes(id)) return '#f59e0b'
  if (id === 0) return '#3b82f6'
  if (id === 5) return '#8b5cf6'
  return '#475569'
}

export default function FordFulkersonAugPath() {
  const [cur, setCur] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1800)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    if (playing) {
      timerRef.current = setInterval(() => {
        setCur(c => {
          if (c >= STEPS.length - 1) { setPlaying(false); return c }
          return c + 1
        })
      }, speed)
    } else {
      if (timerRef.current) clearInterval(timerRef.current)
    }
    return () => { if (timerRef.current) clearInterval(timerRef.current) }
  }, [playing, speed])

  const step = STEPS[cur]
  const R = 20

  return (
    <div className="w-full max-w-2xl mx-auto rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 shadow-xl overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-amber-500 via-orange-500 to-red-500 px-5 py-4">
        <h3 className="text-white font-bold text-lg tracking-tight">Ford-Fulkerson 增广路演示</h3>
        <p className="text-amber-100 text-sm mt-0.5">BFS(Edmonds-Karp) 逐步找到 3 条增广路，最大流 = 14</p>
      </div>

      {/* Controls */}
      <div className="px-4 py-3 flex flex-wrap gap-2 items-center border-b border-slate-100 dark:border-slate-800 bg-slate-50 dark:bg-slate-800/40">
        <button onClick={() => { setPlaying(false); setCur(0) }}
          className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-200 hover:bg-slate-300 dark:hover:bg-slate-600 transition-colors">
          ⏮ 重置
        </button>
        <button onClick={() => { setPlaying(false); setCur(c => Math.max(0, c-1)) }}
          disabled={cur === 0}
          className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-200 hover:bg-slate-300 dark:hover:bg-slate-600 disabled:opacity-40 transition-colors">
          ◀ 上一步
        </button>
        <button onClick={() => setPlaying(p => !p)}
          disabled={cur >= STEPS.length - 1}
          className="px-4 py-1.5 rounded-lg text-xs font-semibold bg-amber-500 text-white hover:bg-amber-600 disabled:opacity-40 transition-colors">
          {playing ? '⏸ 暂停' : '▶ 播放'}
        </button>
        <button onClick={() => { setPlaying(false); setCur(c => Math.min(STEPS.length-1, c+1)) }}
          disabled={cur >= STEPS.length - 1}
          className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-200 hover:bg-slate-300 dark:hover:bg-slate-600 disabled:opacity-40 transition-colors">
          下一步 ▶
        </button>

        <div className="ml-auto flex items-center gap-2 text-[10px] text-slate-400">
          <span>速度</span>
          <input type="range" min={600} max={3000} step={600} value={speed}
            onChange={e => setSpeed(Number(e.target.value))}
            className="w-16 accent-amber-500" />
          <span>{(speed/1000).toFixed(1)}s</span>
        </div>
      </div>

      <div className="p-4 space-y-3">
        {/* Graph + Info layout */}
        <div className="grid grid-cols-1 md:grid-cols-5 gap-3">
          {/* SVG */}
          <div className="md:col-span-3 rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/30 overflow-hidden">
            <svg viewBox="0 0 440 220" className="w-full" style={{ maxHeight: 220 }}>
              <defs>
                {ORIG_EDGES.map((_, idx) => (
                  <marker key={idx} id={`arr-ff-${idx}`} markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
                    <path d="M0,0 L7,3.5 L0,7 Z" fill={edgeColor(idx, step)} />
                  </marker>
                ))}
              </defs>

              {ORIG_EDGES.map(([u, v, cap], idx) => {
                const n1 = NODES[u], n2 = NODES[v]
                const dx = n2.x-n1.x, dy = n2.y-n1.y, len = Math.hypot(dx,dy)
                const sx = n1.x+dx/len*R, sy = n1.y+dy/len*R
                const ex = n2.x-dx/len*R, ey = n2.y-dy/len*R
                const mx = (sx+ex)/2, my = (sy+ey)/2
                const px = -dy/len*13, py = dx/len*13
                const col = edgeColor(idx, step)
                const onPath = step.pathEdgeIdx.includes(idx)
                return (
                  <g key={idx}>
                    <line x1={sx} y1={sy} x2={ex} y2={ey}
                      stroke={col} strokeWidth={onPath ? 3.5 : 2}
                      markerEnd={`url(#arr-ff-${idx})`}
                      strokeDasharray={onPath ? undefined : undefined}
                    />
                    {onPath && (
                      <line x1={sx} y1={sy} x2={ex} y2={ey}
                        stroke="#fbbf24" strokeWidth={6} opacity={0.25} />
                    )}
                    <text x={mx+px} y={my+py+4} textAnchor="middle"
                      fontSize={10} fontWeight="bold" fill={col}>
                      {step.flows[idx]}/{cap}
                    </text>
                  </g>
                )
              })}

              {NODES.map(n => (
                <g key={n.id}>
                  <circle cx={n.x} cy={n.y} r={R}
                    fill={nodeColor(n.id, step)}
                    stroke={step.path.includes(n.id) ? '#fbbf24' : 'transparent'}
                    strokeWidth={2.5} />
                  <text x={n.x} y={n.y+5} textAnchor="middle" fontSize={14} fontWeight="bold" fill="white">
                    {n.label}
                  </text>
                </g>
              ))}
            </svg>
          </div>

          {/* Right panel */}
          <div className="md:col-span-2 space-y-2">
            {/* Step header */}
            <div className="rounded-lg bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-700/40 px-3 py-2">
              <div className="text-[10px] text-amber-500 font-semibold uppercase tracking-wider">步骤 {cur}/{STEPS.length-1}</div>
              <div className="font-bold text-amber-800 dark:text-amber-300 text-sm mt-0.5">{step.label}</div>
            </div>

            {/* Metrics */}
            <div className="grid grid-cols-2 gap-1.5">
              <div className="rounded-lg bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 p-2 text-center">
                <div className="text-[9px] text-slate-400 uppercase tracking-wider">本次增量 Δ</div>
                <div className="font-bold text-orange-500 text-xl">{cur === 0 ? '—' : step.delta}</div>
              </div>
              <div className="rounded-lg bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 p-2 text-center">
                <div className="text-[9px] text-slate-400 uppercase tracking-wider">累计流量 |f|</div>
                <div className="font-bold text-blue-600 dark:text-blue-400 text-xl">{step.totalFlow}</div>
              </div>
            </div>

            {/* Path display */}
            {step.path.length > 0 && (
              <div className="rounded-lg bg-white dark:bg-slate-800 border border-slate-100 dark:border-slate-700 px-3 py-2">
                <div className="text-[10px] text-slate-400 mb-1">增广路径</div>
                <div className="flex items-center gap-0.5 flex-wrap">
                  {step.path.map((nid, i) => (
                    <React.Fragment key={i}>
                      <span className="px-2 py-0.5 rounded-md bg-amber-100 dark:bg-amber-900/40 text-amber-800 dark:text-amber-300 text-xs font-bold">
                        {NODES[nid].label}
                      </span>
                      {i < step.path.length-1 && <span className="text-orange-400 text-xs">→</span>}
                    </React.Fragment>
                  ))}
                </div>
                <div className="text-[10px] text-slate-400 mt-1">瓶颈边容量 = <b className="text-amber-600">{step.bottleneck}</b></div>
              </div>
            )}

            {/* Flow table */}
            <div className="rounded-lg bg-white dark:bg-slate-800 border border-slate-100 dark:border-slate-700 p-2">
              <div className="text-[9px] text-slate-400 uppercase tracking-wider mb-1">边流量</div>
              <div className="grid grid-cols-2 gap-x-3 gap-y-0.5 text-[9.5px]">
                {ORIG_EDGES.map(([u, v, cap], idx) => (
                  <div key={idx} className={`flex justify-between ${step.pathEdgeIdx.includes(idx) ? 'font-bold text-amber-600 dark:text-amber-400' : 'text-slate-500 dark:text-slate-400'}`}>
                    <span>{NODES[u].label}→{NODES[v].label}</span>
                    <span>{step.flows[idx]}/{cap}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Description */}
        <div className="rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/30 px-3 py-2 text-[11px] text-slate-600 dark:text-slate-300">
          {step.desc}
        </div>

        {/* Progress bar */}
        <div className="space-y-1">
          <div className="flex justify-between text-[9px] text-slate-400">
            <span>进度</span><span>{cur}/{STEPS.length-1} 步</span>
          </div>
          <div className="h-1.5 rounded-full bg-slate-200 dark:bg-slate-700 overflow-hidden">
            <div className="h-full bg-gradient-to-r from-amber-500 to-red-500 rounded-full transition-all duration-500"
              style={{ width: `${(cur/(STEPS.length-1))*100}%` }} />
          </div>
        </div>
      </div>
    </div>
  )
}
