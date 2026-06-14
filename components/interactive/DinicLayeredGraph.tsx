'use client'
import React, { useState, useEffect, useRef } from 'react'

// ── Dinic 算法: BFS 层次图 + DFS 阻塞流 ─────────────────────────
// 对于本章例图，只需 1 轮 BFS 即可结束
// BFS 层次: s=0, A=1, C=1, B=2, D=2, t=3

const NODES = [
  { id: 0, label: 's', x: 50,  y: 110, level: 0 },
  { id: 1, label: 'A', x: 160, y: 42,  level: 1 },
  { id: 2, label: 'B', x: 290, y: 42,  level: 2 },
  { id: 3, label: 'C', x: 160, y: 178, level: 1 },
  { id: 4, label: 'D', x: 290, y: 178, level: 2 },
  { id: 5, label: 't', x: 390, y: 110, level: 3 },
]

// [u, v, cap]
const ORIG_EDGES = [
  [0,1,10],[0,3,10],[1,2,4],
  [1,3,2], [1,4,8], [3,4,9],
  [2,5,10],[4,5,10],
] as const

// 层次图中有效边 (level[v] = level[u]+1, 且残差>0)
// in initial state, all edges valid when level difference = 1
const LAYER_EDGE_IDX = [0,1,2,4,5,6,7] // s→A, s→C, A→B, A→D, C→D, B→t, D→t ← A→C(idx3) skipped since levels equal

// 3 DFS paths to find blocking flow
interface DFSPath { edgeIdx: number[]; nodes: number[]; delta: number; flows: number[] }

const DFS_PATHS: DFSPath[] = [
  {
    nodes: [0,1,2,5], edgeIdx: [0,2,6], delta: 4,
    flows: [4,0,4,0,0,0,4,0],
  },
  {
    nodes: [0,1,4,5], edgeIdx: [0,4,7], delta: 6,
    flows: [10,0,4,0,6,0,4,6],
  },
  {
    nodes: [0,3,4,5], edgeIdx: [1,5,7], delta: 4,
    flows: [10,4,4,0,6,4,4,10],
  },
]

// Animation steps
interface Step {
  phase: 'bfs' | 'dfs'
  bfsRevealed: Set<number>   // node ids with level assigned
  bfsEdges: Set<number>      // layer edges revealed
  dfsPathIdx: number         // -1 = none, 0-2 = path index
  flows: number[]
  label: string
  desc: string
}

const BFS_STAGES: Step[] = [
  {
    phase: 'bfs', bfsRevealed: new Set([0]), bfsEdges: new Set(), dfsPathIdx: -1,
    flows: Array(8).fill(0),
    label: 'BFS 第 0 层', desc: '从源点 s 出发，s 加入层次图，level[s]=0。',
  },
  {
    phase: 'bfs', bfsRevealed: new Set([0,1,3]), bfsEdges: new Set([0,1]), dfsPathIdx: -1,
    flows: Array(8).fill(0),
    label: 'BFS 第 1 层', desc: '扩展 s 的邻居：A、C 满足残差>0，level[A]=level[C]=1。s→A、s→C 纳入层次图。',
  },
  {
    phase: 'bfs', bfsRevealed: new Set([0,1,2,3,4]), bfsEdges: new Set([0,1,2,4,5]), dfsPathIdx: -1,
    flows: Array(8).fill(0),
    label: 'BFS 第 2 层', desc: '扩展 A(→B,D) 和 C(→D)：level[B]=level[D]=2。A→B、A→D、C→D 纳入层次图。A→C 因 level 相同被丢弃。',
  },
  {
    phase: 'bfs', bfsRevealed: new Set([0,1,2,3,4,5]), bfsEdges: new Set([0,1,2,4,5,6,7]), dfsPathIdx: -1,
    flows: Array(8).fill(0),
    label: 'BFS 第 3 层（到达 t）', desc: '扩展 B(→t) 和 D(→t)：level[t]=3，BFS 结束。层次图构建完毕。',
  },
]

const DFS_STAGES: Step[] = DFS_PATHS.map((p, i) => ({
  phase: 'dfs',
  bfsRevealed: new Set(NODES.map(n => n.id)),
  bfsEdges: new Set([0,1,2,4,5,6,7]),
  dfsPathIdx: i,
  flows: p.flows,
  label: `DFS 路径 ${i+1}: ${p.nodes.map(id => NODES[id].label).join('→')}`,
  desc: i === 0 ? `当前弧 DFS 找到路径 s→A→B→t，流量 Δ=${p.delta}。A→B 饱和，当前弧前移。`
      : i === 1 ? `DFS 沿 s→A→D→t 增广 Δ=${p.delta}。s→A 饱和，当前弧从 s 前移。`
      : `DFS 沿 s→C→D→t 增广 Δ=${p.delta}。D→t 饱和。阻塞流完成，总流量=14。`,
}))

const ALL_STEPS: Step[] = [...BFS_STAGES, ...DFS_STAGES]

const LEVEL_COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6']

export default function DinicLayeredGraph() {
  const [cur, setCur] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1800)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    if (playing) {
      timerRef.current = setInterval(() => {
        setCur(c => {
          if (c >= ALL_STEPS.length - 1) { setPlaying(false); return c }
          return c + 1
        })
      }, speed)
    } else {
      if (timerRef.current) clearInterval(timerRef.current)
    }
    return () => { if (timerRef.current) clearInterval(timerRef.current) }
  }, [playing, speed])

  const step = ALL_STEPS[cur]
  const R = 20

  function nodeColor(id: number) {
    if (!step.bfsRevealed.has(id)) return '#e2e8f0'
    if (step.phase === 'dfs' && step.dfsPathIdx >= 0) {
      if (DFS_PATHS[step.dfsPathIdx].nodes.includes(id)) return '#f59e0b'
    }
    return LEVEL_COLORS[NODES[id].level] ?? '#475569'
  }

  function nodeDarkText(id: number) {
    return !step.bfsRevealed.has(id)
  }

  function edgeStyle(idx: number): { color: string; width: number; dash?: string; show: boolean } {
    const [u, v] = ORIG_EDGES[idx]
    const inLayer = step.bfsEdges.has(idx)
    const f = step.flows[idx]
    if (!inLayer) return { color: '#e2e8f0', width: 1.5, show: true }

    if (step.phase === 'dfs' && step.dfsPathIdx >= 0) {
      if (DFS_PATHS[step.dfsPathIdx].edgeIdx.includes(idx))
        return { color: '#f59e0b', width: 3.5, show: true }
    }

    if (f >= ORIG_EDGES[idx][2]) return { color: '#ef4444', width: 2.5, show: true }
    if (f > 0) return { color: '#3b82f6', width: 2.5, show: true }
    return { color: '#10b981', width: 2, show: true }
  }

  return (
    <div className="w-full max-w-2xl mx-auto rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 shadow-xl overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-emerald-600 via-teal-600 to-cyan-600 px-5 py-4">
        <h3 className="text-white font-bold text-lg tracking-tight">Dinic 算法 — 层次图 + 阻塞流演示</h3>
        <p className="text-emerald-100 text-sm mt-0.5">BFS 构建层次图 → DFS 当前弧找阻塞流 → 循环至无增广路</p>
      </div>

      {/* Controls */}
      <div className="px-4 py-3 flex flex-wrap gap-2 items-center border-b border-slate-100 dark:border-slate-800 bg-slate-50 dark:bg-slate-800/40">
        {/* Phase indicator */}
        <div className="flex rounded-lg overflow-hidden text-[10px] font-bold border border-slate-200 dark:border-slate-600">
          <div className={`px-3 py-1.5 ${step.phase === 'bfs' ? 'bg-teal-600 text-white' : 'bg-slate-100 dark:bg-slate-700 text-slate-400'}`}>BFS 层次</div>
          <div className={`px-3 py-1.5 ${step.phase === 'dfs' ? 'bg-emerald-600 text-white' : 'bg-slate-100 dark:bg-slate-700 text-slate-400'}`}>DFS 阻塞流</div>
        </div>

        <div className="flex gap-1.5 ml-auto">
          <button onClick={() => { setPlaying(false); setCur(0) }}
            className="px-2 py-1.5 rounded-lg text-xs bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-300 hover:bg-slate-300 dark:hover:bg-slate-600">⏮</button>
          <button onClick={() => { setPlaying(false); setCur(c => Math.max(0,c-1)) }} disabled={cur===0}
            className="px-2 py-1.5 rounded-lg text-xs bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-300 hover:bg-slate-300 dark:hover:bg-slate-600 disabled:opacity-40">◀</button>
          <button onClick={() => setPlaying(p => !p)} disabled={cur >= ALL_STEPS.length-1}
            className="px-3 py-1.5 rounded-lg text-xs font-bold bg-emerald-600 text-white hover:bg-emerald-700 disabled:opacity-40">
            {playing ? '⏸' : '▶'}
          </button>
          <button onClick={() => { setPlaying(false); setCur(c => Math.min(ALL_STEPS.length-1,c+1)) }} disabled={cur >= ALL_STEPS.length-1}
            className="px-2 py-1.5 rounded-lg text-xs bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-300 hover:bg-slate-300 dark:hover:bg-slate-600 disabled:opacity-40">▶</button>
        </div>

        <div className="flex items-center gap-1.5 text-[10px] text-slate-400">
          <input type="range" min={600} max={3000} step={400} value={speed}
            onChange={e => setSpeed(Number(e.target.value))} className="w-16 accent-emerald-500" />
          <span>{(speed/1000).toFixed(1)}s</span>
        </div>
      </div>

      <div className="p-4 space-y-3">
        {/* Step header */}
        <div className="flex items-center gap-2">
          <span className={`px-2 py-0.5 rounded-full text-[10px] font-bold text-white ${step.phase === 'bfs' ? 'bg-teal-500' : 'bg-emerald-500'}`}>
            {cur+1}/{ALL_STEPS.length}
          </span>
          <span className="font-semibold text-sm text-slate-700 dark:text-slate-200">{step.label}</span>
        </div>

        {/* SVG */}
        <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/30 overflow-hidden">
          {/* Level columns */}
          <div className="relative">
            {step.phase === 'bfs' && (
              <div className="absolute inset-0 flex pointer-events-none">
                {[0,1,2,3].map(lv => (
                  <div key={lv} className="flex-1 flex flex-col items-center pt-1">
                    {step.bfsRevealed.has(NODES.findIndex(n => n.level === lv)) && (
                      <span className="text-[9px] font-bold opacity-60"
                        style={{ color: LEVEL_COLORS[lv] }}>L{lv}</span>
                    )}
                  </div>
                ))}
              </div>
            )}
            <svg viewBox="0 0 440 220" className="w-full" style={{ maxHeight: 220 }}>
              <defs>
                {ORIG_EDGES.map((_, idx) => {
                  const es = edgeStyle(idx)
                  return (
                    <marker key={idx} id={`arr-dinic-${idx}`} markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
                      <path d="M0,0 L7,3.5 L0,7 Z" fill={es.color} />
                    </marker>
                  )
                })}
              </defs>

              {/* Level column shading */}
              {[110, 220, 330, 440].map((x, lv) => (
                step.bfsRevealed.has(NODES.findIndex(n => n.level === lv)) ? (
                  <rect key={lv} x={x-110} y={0} width={110} height={220}
                    fill={LEVEL_COLORS[lv]} opacity={0.04} />
                ) : null
              ))}

              {ORIG_EDGES.map(([u, v, cap], idx) => {
                const es = edgeStyle(idx)
                const n1 = NODES[u], n2 = NODES[v]
                const dx = n2.x-n1.x, dy = n2.y-n1.y, len = Math.hypot(dx,dy)
                const sx = n1.x+dx/len*R, sy = n1.y+dy/len*R
                const ex = n2.x-dx/len*R, ey = n2.y-dy/len*R
                const mx = (sx+ex)/2, my = (sy+ey)/2
                const px = -dy/len*13, py = dx/len*13
                const inLayer = step.bfsEdges.has(idx)
                return (
                  <g key={idx} opacity={!inLayer && step.phase==='bfs' && step.bfsEdges.size>0 ? 0.15 : 1}>
                    <line x1={sx} y1={sy} x2={ex} y2={ey}
                      stroke={es.color} strokeWidth={es.width}
                      markerEnd={`url(#arr-dinic-${idx})`} />
                    {step.phase === 'dfs' && (
                      <text x={mx+px} y={my+py+4} textAnchor="middle" fontSize={9} fill={es.color}>
                        {step.flows[idx]}/{cap}
                      </text>
                    )}
                  </g>
                )
              })}

              {NODES.map(n => {
                const revealed = step.bfsRevealed.has(n.id)
                const col = nodeColor(n.id)
                return (
                  <g key={n.id} opacity={revealed ? 1 : 0.2}>
                    <circle cx={n.x} cy={n.y} r={R} fill={col} />
                    {revealed && <text x={n.x} y={n.y-26} textAnchor="middle" fontSize={9}
                      fill={LEVEL_COLORS[n.level]} fontWeight="bold">L{n.level}</text>}
                    <text x={n.x} y={n.y+5} textAnchor="middle" fontSize={14} fontWeight="bold"
                      fill={nodeDarkText(n.id) ? '#64748b' : 'white'}>{n.label}</text>
                  </g>
                )
              })}
            </svg>
          </div>
        </div>

        {/* Description */}
        <div className="rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/30 px-3 py-2 text-[11px] text-slate-600 dark:text-slate-300">
          {step.desc}
        </div>

        {/* Legend */}
        <div className="flex flex-wrap gap-3 text-[10px] text-slate-500 dark:text-slate-400">
          {[0,1,2,3].map(lv => (
            <span key={lv} className="flex items-center gap-1">
              <span className="w-3 h-3 rounded-full inline-block" style={{ backgroundColor: LEVEL_COLORS[lv] }}/>
              层 {lv}
            </span>
          ))}
          {step.phase === 'dfs' && (
            <span className="flex items-center gap-1 ml-2">
              <span className="w-8 h-0.5 bg-amber-400 inline-block rounded"/> 当前增广路
            </span>
          )}
        </div>

        {/* Progress */}
        <div className="h-1.5 rounded-full bg-slate-200 dark:bg-slate-700 overflow-hidden">
          <div className="h-full rounded-full transition-all duration-400"
            style={{
              width: `${(cur/(ALL_STEPS.length-1))*100}%`,
              background: step.phase === 'bfs'
                ? 'linear-gradient(to right,#0d9488,#06b6d4)'
                : 'linear-gradient(to right,#10b981,#059669)',
            }} />
        </div>
      </div>
    </div>
  )
}
