'use client'
import React, { useState } from 'react'

const LABELS = ['A', 'B', 'C', 'D']
const N = 4
const INF = Infinity

// ── Final Floyd-Warshall state (4-node example) ───────────────────
// Edges: A→B:1, A→C:5, B→C:2, B→D:4, C→D:1, D→A:3
const FINAL_DIST = [
  [0, 1, 3, 4],
  [6, 0, 2, 3],
  [4, 5, 0, 1],
  [3, 4, 6, 0],
]

// pred[i][j] = 从 i 到 j 最短路中，j 的前驱节点编号（-1=NIL）
const FINAL_PRED = [
  [-1,  0,  1,  2],  // A → (-, A, B, C)
  [ 3, -1,  1,  2],  // B → (D, -, B, C)
  [ 3,  0, -1,  2],  // C → (D, A, -, C)
  [ 3,  0,  1, -1],  // D → (D, A, B, -)
]

const EDGES = [
  { u: 0, v: 1, w: 1 },
  { u: 0, v: 2, w: 5 },
  { u: 1, v: 2, w: 2 },
  { u: 1, v: 3, w: 4 },
  { u: 2, v: 3, w: 1 },
  { u: 3, v: 0, w: 3 },
]

const NODE_POS = [
  { x: 65,  y: 105 }, // A
  { x: 175, y: 42  }, // B
  { x: 175, y: 162 }, // C
  { x: 290, y: 105 }, // D
]

function reconstructPath(src: number, dst: number): number[] {
  if (FINAL_DIST[src][dst] === INF) return []
  if (src === dst) return [src]
  const path: number[] = []
  let cur = dst
  const limit = N + 1
  let count = 0
  while (cur !== src && count < limit) {
    path.push(cur)
    cur = FINAL_PRED[src][cur]
    count++
  }
  path.push(src)
  return path.reverse()
}

function Arrow({ u, v, w, highlight }: {
  u: number; v: number; w: number; highlight: 'path' | 'normal' | 'dim'
}) {
  const { x: x1, y: y1 } = NODE_POS[u]
  const { x: x2, y: y2 } = NODE_POS[v]
  const dx = x2 - x1, dy = y2 - y1, len = Math.hypot(dx, dy)
  const R = 20
  const sx = x1 + (dx / len) * R, sy = y1 + (dy / len) * R
  const ex = x2 - (dx / len) * R, ey = y2 - (dy / len) * R
  const mx = (sx + ex) / 2, my = (sy + ey) / 2

  const color = highlight === 'path' ? '#10b981' : highlight === 'dim' ? '#e2e8f0' : '#94a3b8'
  const sw = highlight === 'path' ? 3 : 1.5
  const mid = `arr-pr-${u}-${v}`

  return (
    <g>
      <defs>
        <marker id={mid} markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
          <path d="M0,0 L7,3.5 L0,7 Z" fill={color} />
        </marker>
      </defs>
      <line x1={sx} y1={sy} x2={ex} y2={ey} stroke={color} strokeWidth={sw} markerEnd={`url(#${mid})`}
        style={{ transition: 'stroke 0.3s, stroke-width 0.3s' }} />
      <text x={mx + dy * 0.12} y={my - dx * 0.12} textAnchor="middle"
        fontSize={10} fontWeight="bold" fill={color}>{w}</text>
    </g>
  )
}

export default function FloydPathReconstruct() {
  const [src, setSrc] = useState(0)
  const [dst, setDst] = useState(3)
  const [hoverStep, setHoverStep] = useState<number | null>(null)

  const path = reconstructPath(src, dst)
  const dist = src === dst ? 0 : FINAL_DIST[src][dst]
  const reachable = path.length > 0

  // Which edges are ON the shortest path
  const pathEdgeSet = new Set<string>()
  for (let i = 0; i < path.length - 1; i++) {
    pathEdgeSet.add(`${path[i]}-${path[i + 1]}`)
  }

  // Path nodes to highlight
  const activeStep = hoverStep !== null ? hoverStep : path.length - 1
  const visibleNodes = new Set(path.slice(0, activeStep + 1))
  const visibleEdges = new Set<string>()
  for (let i = 0; i < Math.min(activeStep, path.length - 1); i++) {
    visibleEdges.add(`${path[i]}-${path[i + 1]}`)
  }

  return (
    <div className="w-full max-w-3xl mx-auto rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 shadow-xl overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-emerald-600 via-teal-600 to-cyan-600 px-5 py-4">
        <h3 className="text-white font-bold text-lg tracking-tight">Floyd-Warshall — 路径重建（前驱矩阵）</h3>
        <p className="text-emerald-200 text-sm mt-0.5">选择起点和终点，查看最短路径是如何通过前驱矩阵 Π 逐步回溯的</p>
      </div>

      <div className="p-4 space-y-4">
        <div className="flex gap-4 items-start">
          {/* Left: Controls + Graph */}
          <div className="flex flex-col gap-3 w-[340px] shrink-0">
            {/* Src/Dst picker */}
            <div className="flex gap-3">
              {(['起点', '终点'] as const).map((label, li) => {
                const val = li === 0 ? src : dst
                const setter = li === 0 ? setSrc : setDst
                return (
                  <div key={li} className="flex-1 rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
                    <div className={`px-3 py-1.5 text-[11px] font-bold uppercase tracking-wide border-b border-slate-200 dark:border-slate-700 ${
                      li === 0 ? 'bg-emerald-50 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-400' : 'bg-cyan-50 dark:bg-cyan-900/30 text-cyan-700 dark:text-cyan-400'
                    }`}>
                      {label}（{LABELS[val]}）
                    </div>
                    <div className="flex">
                      {LABELS.map((l, ni) => (
                        <button key={ni} onClick={() => {
                          setter(ni)
                          setHoverStep(null)
                        }}
                          disabled={(li === 0 && ni === dst) || (li === 1 && ni === src)}
                          className={`flex-1 py-2 text-sm font-bold transition-all ${
                            val === ni
                              ? li === 0 ? 'bg-emerald-500 text-white' : 'bg-cyan-500 text-white'
                              : 'hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-600 dark:text-slate-300 disabled:opacity-20'
                          }`}>
                          {l}
                        </button>
                      ))}
                    </div>
                  </div>
                )
              })}
            </div>

            {/* Graph SVG */}
            <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50 overflow-hidden">
              <svg viewBox="0 0 360 215" className="w-full">
                {EDGES.map((e, i) => {
                  const key = `${e.u}-${e.v}`
                  const onPath = visibleEdges.has(key)
                  return (
                    <Arrow key={i} {...e}
                      highlight={onPath ? 'path' : pathEdgeSet.has(key) ? 'normal' : 'dim'} />
                  )
                })}
                {NODE_POS.map((pos, ni) => {
                  const isSrc = ni === src
                  const isDst = ni === dst
                  const onPath = visibleNodes.has(ni)
                  let fill = '#e2e8f0', stroke = '#94a3b8', textFill = '#94a3b8'
                  if (isSrc) { fill = '#10b981'; stroke = '#059669'; textFill = '#fff' }
                  else if (isDst) { fill = '#06b6d4'; stroke = '#0891b2'; textFill = '#fff' }
                  else if (onPath) { fill = '#6ee7b7'; stroke = '#10b981'; textFill = '#064e3b' }
                  return (
                    <g key={ni}>
                      <circle cx={pos.x} cy={pos.y} r={22} fill={fill} stroke={stroke} strokeWidth={2.5}
                        style={{ transition: 'fill 0.3s, stroke 0.3s' }} />
                      <text x={pos.x} y={pos.y + 5} textAnchor="middle" fontSize={14} fontWeight="bold" fill={textFill}>
                        {LABELS[ni]}
                      </text>
                      {isSrc && <text x={pos.x} y={pos.y - 28} textAnchor="middle" fontSize={9} fill="#059669" fontWeight="bold">起点</text>}
                      {isDst && <text x={pos.x} y={pos.y - 28} textAnchor="middle" fontSize={9} fill="#0891b2" fontWeight="bold">终点</text>}
                    </g>
                  )
                })}
              </svg>
            </div>

            {/* Result */}
            <div className={`rounded-xl border px-4 py-3 ${
              reachable && src !== dst
                ? 'border-emerald-200 dark:border-emerald-700/50 bg-emerald-50 dark:bg-emerald-900/20'
                : src === dst
                  ? 'border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50'
                  : 'border-red-200 dark:border-red-700/50 bg-red-50 dark:bg-red-900/20'
            }`}>
              <div className="flex items-center justify-between">
                <span className="text-sm font-bold text-slate-700 dark:text-slate-200">
                  {LABELS[src]} → {LABELS[dst]}
                </span>
                <span className={`text-lg font-mono font-black ${
                  src === dst ? 'text-slate-400' : reachable ? 'text-emerald-600 dark:text-emerald-400' : 'text-red-500'
                }`}>
                  {src === dst ? '0（自身）' : reachable ? `距离 = ${dist}` : '不可达 ∞'}
                </span>
              </div>
              {reachable && src !== dst && (
                <div className="mt-1.5 text-xs text-emerald-700 dark:text-emerald-300 font-mono">
                  {path.map(n => LABELS[n]).join(' → ')}
                </div>
              )}
            </div>
          </div>

          {/* Right: Pred matrix + reconstruction */}
          <div className="flex-1 min-w-0 space-y-3">
            {/* Pred matrix */}
            <div className="rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
              <div className="px-3 py-2 text-[11px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wide border-b border-slate-200 dark:border-slate-700">
                前驱矩阵 Π（π[i][j] = j 的前驱）
              </div>
              <table className="w-full border-collapse">
                <thead>
                  <tr>
                    <th className="py-2 text-center text-[10px] text-slate-400 bg-slate-50 dark:bg-slate-800/50 w-8"></th>
                    {LABELS.map((l, j) => (
                      <th key={j} className={`py-2 text-center text-xs font-bold ${
                        j === dst ? 'text-cyan-600 dark:text-cyan-400' : 'text-slate-500 dark:text-slate-400'
                      }`}>{l}</th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 dark:divide-slate-700/50">
                  {LABELS.map((rl, i) => (
                    <tr key={i} className="divide-x divide-slate-100 dark:divide-slate-700/50">
                      <td className={`pl-3 text-xs font-bold ${
                        i === src ? 'text-emerald-600 dark:text-emerald-400' : 'text-slate-500 dark:text-slate-400'
                      }`}>{rl}</td>
                      {LABELS.map((_, j) => {
                        const val = FINAL_PRED[i][j]
                        const isHighlight = i === src && j === dst
                        const isOnPath = i === src && path.includes(j) && j !== src && val !== -1
                        return (
                          <td key={j} className={`w-12 h-9 text-center text-sm font-mono font-bold transition-all ${
                            i === j ? 'text-slate-300 dark:text-slate-600 bg-slate-50 dark:bg-slate-800/50' :
                            isHighlight ? 'bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300 ring-2 ring-amber-400' :
                            isOnPath ? 'bg-emerald-50 dark:bg-emerald-900/20 text-emerald-600 dark:text-emerald-400' :
                            'text-slate-600 dark:text-slate-300'
                          }`}>
                            {val === -1 ? '—' : LABELS[val]}
                          </td>
                        )
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Step-by-step reconstruction */}
            {reachable && src !== dst && (
              <div className="rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
                <div className="px-3 py-2 text-[11px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wide border-b border-slate-200 dark:border-slate-700">
                  回溯过程（从终点反向追踪）
                </div>
                <div className="p-3 space-y-1.5">
                  {path.map((_, stepI) => {
                    const cur = path[path.length - 1 - stepI]
                    const fromSrc = `PRED[${LABELS[src]}]`
                    if (stepI === 0) {
                      return (
                        <div key={stepI}
                          onMouseEnter={() => setHoverStep(path.length - 1)}
                          onMouseLeave={() => setHoverStep(null)}
                          className="flex items-center gap-2 px-2.5 py-1.5 rounded-lg bg-cyan-50 dark:bg-cyan-900/20 border border-cyan-200 dark:border-cyan-700/50 cursor-default">
                          <span className="w-5 h-5 bg-cyan-500 rounded-full text-white text-[10px] flex items-center justify-center font-bold shrink-0">
                            {stepI + 1}
                          </span>
                          <span className="text-xs text-cyan-700 dark:text-cyan-300">
                            从终点 <strong>{LABELS[cur]}</strong> 出发（cur = {LABELS[dst]}）
                          </span>
                        </div>
                      )
                    }
                    const prev = path[path.length - stepI]
                    const predVal = FINAL_PRED[src][prev]
                    const isLast = cur === src
                    return (
                      <div key={stepI}
                        onMouseEnter={() => setHoverStep(path.length - 1 - stepI)}
                        onMouseLeave={() => setHoverStep(null)}
                        className={`flex items-center gap-2 px-2.5 py-1.5 rounded-lg border cursor-default transition-colors ${
                          isLast
                            ? 'bg-emerald-50 dark:bg-emerald-900/20 border-emerald-200 dark:border-emerald-700/50'
                            : 'bg-slate-50 dark:bg-slate-800/50 border-slate-200 dark:border-slate-700'
                        }`}>
                        <span className={`w-5 h-5 rounded-full text-white text-[10px] flex items-center justify-center font-bold shrink-0 ${
                          isLast ? 'bg-emerald-500' : 'bg-slate-400'
                        }`}>{stepI + 1}</span>
                        <span className="text-xs text-slate-700 dark:text-slate-300 font-mono">
                          {fromSrc}[<span className="text-amber-600 dark:text-amber-400 font-bold">{LABELS[prev]}</span>] = <strong>{LABELS[predVal]}</strong>
                          {isLast
                            ? <span className="text-emerald-600 dark:text-emerald-400 ml-1">← 到达起点 {LABELS[src]}，停止</span>
                            : <span className="text-slate-400 ml-1">← 继续回溯</span>}
                        </span>
                      </div>
                    )
                  })}
                  <div className="mt-2 flex items-center gap-2 px-2 py-1 text-xs text-slate-500 dark:text-slate-400">
                    <span>路径：</span>
                    {path.map((n, i) => (
                      <React.Fragment key={n}>
                        <span className={`font-bold font-mono ${i === 0 ? 'text-emerald-600' : i === path.length - 1 ? 'text-cyan-600' : 'text-slate-600 dark:text-slate-300'}`}>
                          {LABELS[n]}
                        </span>
                        {i < path.length - 1 && <span className="text-slate-300">→</span>}
                      </React.Fragment>
                    ))}
                    <span className="ml-auto font-bold text-slate-600 dark:text-slate-300">
                      总长 = {dist}
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
