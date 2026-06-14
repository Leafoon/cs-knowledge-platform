'use client'
import React, { useState } from 'react'

// ── 重新定权后的图（全非负） ───────────────────────────────────────
// h = [0, 0, -3, -2]，新权 w'(u,v) = w + h[u] - h[v]
const LABELS = ['A', 'B', 'C', 'D']
const H = [0, 0, -3, -2]       // 势能函数
const INF = Infinity

const REWEIGHTED_EDGES = [
  { u: 0, v: 1, wOrig: 2,  wNew: 2 },
  { u: 1, v: 2, wOrig: -3, wNew: 0 },
  { u: 1, v: 3, wOrig: 4,  wNew: 6 },
  { u: 2, v: 3, wOrig: 1,  wNew: 0 },
  { u: 3, v: 0, wOrig: 5,  wNew: 3 },
]

// 每个源点的 Dijkstra 结果（新权图中）与原图还原结果
const DIJKSTRA_RESULTS = [
  { dprime: [0, 2, 2, 2], real: [0, 2, -1, 0] }, // 从 A
  { dprime: [3, 0, 0, 0], real: [3, 0, -3, -2] }, // 从 B
  { dprime: [3, 5, 0, 0], real: [6, 8, 0, 1] },   // 从 C
  { dprime: [3, 5, 5, 0], real: [5, 7, 4, 0] },   // 从 D
]

const NODE_POS = [
  { x: 65,  y: 105 },
  { x: 190, y: 42  },
  { x: 190, y: 165 },
  { x: 315, y: 105 },
]

function Arrow({ u, v, wNew, onPath }: { u: number; v: number; wNew: number; onPath: boolean }) {
  const { x: x1, y: y1 } = NODE_POS[u]
  const { x: x2, y: y2 } = NODE_POS[v]
  const dx = x2 - x1, dy = y2 - y1, len = Math.hypot(dx, dy)
  const R = 22
  const sx = x1 + (dx / len) * R, sy = y1 + (dy / len) * R
  const ex = x2 - (dx / len) * R, ey = y2 - (dy / len) * R

  const color = onPath ? '#8b5cf6' : '#94a3b8'
  const sw = onPath ? 3 : 1.5
  const mid = `arr-jdp-${u}-${v}`

  return (
    <g>
      <defs>
        <marker id={mid} markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
          <path d="M0,0 L7,3.5 L0,7 Z" fill={color} />
        </marker>
      </defs>
      <line x1={sx} y1={sy} x2={ex} y2={ey} stroke={color} strokeWidth={sw}
        markerEnd={`url(#${mid})`} style={{ transition: 'stroke 0.3s' }} />
      <text x={(sx + ex) / 2 + dy * 0.15} y={(sy + ey) / 2 - dx * 0.15 - 3}
        textAnchor="middle" fontSize={10} fontWeight="bold"
        fill={onPath ? '#7c3aed' : '#94a3b8'}>{wNew}</text>
    </g>
  )
}

function fmtD(v: number) { return v === INF ? '∞' : String(v) }

export default function JohnsonDijkstraPhase() {
  const [srcIdx, setSrcIdx] = useState(0)
  const [showOrig, setShowOrig] = useState(false)

  const res = DIJKSTRA_RESULTS[srcIdx]

  // 最短路树边（简化：用 dprime 最小路径）
  const pathEdges = new Set<string>()
  // 对每个可达节点，找到其最短路径（用 dprime 值反推）
  for (let v = 0; v < 4; v++) {
    if (v === srcIdx || res.dprime[v] === INF) continue
    for (const e of REWEIGHTED_EDGES) {
      if (e.v === v && res.dprime[e.u] + e.wNew === res.dprime[v]) {
        pathEdges.add(`${e.u}-${e.v}`)
        break
      }
    }
  }

  return (
    <div className="w-full max-w-3xl mx-auto rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 shadow-xl overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-violet-600 via-purple-600 to-fuchsia-600 px-5 py-4">
        <h3 className="text-white font-bold text-lg tracking-tight">Johnson — Dijkstra 阶段与结果还原</h3>
        <p className="text-violet-200 text-sm mt-0.5">在非负权图上对每个顶点跑 Dijkstra，再通过 d(u,v) = d'(u,v) − h[u] + h[v] 还原</p>
      </div>

      <div className="p-4 space-y-4">
        {/* Source selector */}
        <div className="flex items-center gap-3">
          <span className="text-sm font-bold text-slate-600 dark:text-slate-300 shrink-0">选择 Dijkstra 源点：</span>
          <div className="flex gap-1.5">
            {LABELS.map((l, i) => (
              <button key={i} onClick={() => setSrcIdx(i)}
                className={`w-10 h-10 rounded-xl font-bold text-sm transition-all ${
                  i === srcIdx
                    ? 'bg-violet-600 text-white shadow scale-110'
                    : 'bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-400 hover:bg-slate-200 dark:hover:bg-slate-700'
                }`}>
                {l}
              </button>
            ))}
          </div>
          <div className="ml-auto flex items-center gap-2">
            <label className="flex items-center gap-1.5 text-xs text-slate-500 dark:text-slate-400 cursor-pointer select-none">
              <input type="checkbox" checked={showOrig} onChange={e => setShowOrig(e.target.checked)}
                className="rounded accent-violet-500" />
              显示原始距离
            </label>
          </div>
        </div>

        <div className="flex gap-4 items-start">
          {/* Left: Graph */}
          <div className="w-[340px] shrink-0 rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50 overflow-hidden">
            <div className="px-3 py-2 text-[11px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wide border-b border-slate-200 dark:border-slate-700">
              重新定权图（所有边权 ≥ 0）
            </div>
            <svg viewBox="0 0 380 220" className="w-full">
              {REWEIGHTED_EDGES.map((e, i) => (
                <Arrow key={i} {...e} onPath={pathEdges.has(`${e.u}-${e.v}`)} />
              ))}
              {NODE_POS.map((pos, ni) => {
                const isSrc = ni === srcIdx
                const dp = res.dprime[ni]
                return (
                  <g key={ni}>
                    {isSrc && <circle cx={pos.x} cy={pos.y} r={32}
                      fill="#8b5cf633" stroke="#8b5cf655" strokeWidth={2} />}
                    <circle cx={pos.x} cy={pos.y} r={22}
                      fill={isSrc ? '#7c3aed' : dp === INF ? '#e2e8f0' : '#8b5cf6'}
                      stroke={isSrc ? '#6d28d9' : dp === INF ? '#94a3b8' : '#7c3aed'}
                      strokeWidth={2.5} style={{ transition: 'fill 0.3s' }} />
                    <text x={pos.x} y={pos.y - 2} textAnchor="middle" fontSize={12} fontWeight="bold"
                      fill="#fff">{LABELS[ni]}</text>
                    <text x={pos.x} y={pos.y + 12} textAnchor="middle" fontSize={9}
                      fill="#ddd6fe">{fmtD(dp)}</text>
                  </g>
                )
              })}
            </svg>
            <div className="px-3 pb-2 text-[10px] text-slate-400 text-center">
              紫色 = Dijkstra 最短路树边 | 每节点显示 d'(src, v)
            </div>
          </div>

          {/* Right: Full table */}
          <div className="flex-1 min-w-0 space-y-3">
            {/* All-v table */}
            <div className="rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
              <div className="px-3 py-2 text-[11px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wide border-b border-slate-200 dark:border-slate-700">
                从 {LABELS[srcIdx]} 出发的结果
              </div>
              <table className="w-full text-xs">
                <thead>
                  <tr className="bg-slate-50 dark:bg-slate-800/50 border-b border-slate-100 dark:border-slate-700">
                    <th className="px-3 py-2 text-left text-slate-400 font-medium">终点 v</th>
                    <th className="px-3 py-2 text-center text-violet-500 font-medium">d'(u,v)</th>
                    <th className="px-3 py-2 text-center text-blue-500 font-medium">−h[u]</th>
                    <th className="px-3 py-2 text-center text-teal-500 font-medium">+h[v]</th>
                    <th className="px-3 py-2 text-center text-emerald-600 font-bold">d(u,v)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 dark:divide-slate-700/50">
                  {LABELS.map((l, vi) => {
                    const dp = res.dprime[vi]
                    const dr = res.real[vi]
                    const hSrc = H[srcIdx]
                    const hV = H[vi]
                    const isSelf = vi === srcIdx
                    return (
                      <tr key={vi} className={isSelf ? 'opacity-40' : ''}>
                        <td className="px-3 py-2 font-bold text-slate-700 dark:text-slate-200">{l}</td>
                        <td className="px-3 py-2 text-center font-mono font-bold text-violet-600 dark:text-violet-400">
                          {fmtD(dp)}
                        </td>
                        <td className="px-3 py-2 text-center font-mono text-blue-600 dark:text-blue-400">
                          {isSelf ? '—' : `−(${hSrc})=${-hSrc >= 0 ? '+' : ''}${-hSrc}`}
                        </td>
                        <td className="px-3 py-2 text-center font-mono text-teal-600 dark:text-teal-400">
                          {isSelf ? '—' : `+${hV}`}
                        </td>
                        <td className={`px-3 py-2 text-center font-mono font-black ${
                          isSelf ? 'text-slate-300' :
                          dr < 0 ? 'text-rose-600 dark:text-rose-400' : 'text-emerald-600 dark:text-emerald-400'
                        }`}>
                          {isSelf ? '0' : fmtD(dr)}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>

            {/* Full APSP matrix (optional) */}
            {showOrig && (
              <div className="rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
                <div className="px-3 py-2 text-[11px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wide border-b border-slate-200 dark:border-slate-700">
                  完整 APSP 矩阵（原始距离）
                </div>
                <table className="w-full border-collapse">
                  <thead>
                    <tr>
                      <th className="py-2 text-center text-[10px] text-slate-400 bg-slate-50 dark:bg-slate-800/50 w-8"></th>
                      {LABELS.map((l, j) => (
                        <th key={j} className="py-2 text-center text-xs font-bold text-slate-500 dark:text-slate-400">{l}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 dark:divide-slate-700/50">
                    {DIJKSTRA_RESULTS.map((r, i) => (
                      <tr key={i} className="divide-x divide-slate-100 dark:divide-slate-700/50">
                        <td className={`pl-3 text-xs font-bold ${i === srcIdx ? 'text-violet-600 dark:text-violet-400' : 'text-slate-500 dark:text-slate-400'}`}>
                          {LABELS[i]}
                        </td>
                        {r.real.map((d, j) => (
                          <td key={j} className={`w-12 h-9 text-center text-sm font-mono font-bold ${
                            i === j ? 'text-slate-300 dark:text-slate-600 bg-slate-50 dark:bg-slate-800/50' :
                            i === srcIdx && j !== srcIdx ? 'bg-violet-50 dark:bg-violet-900/20 text-violet-700 dark:text-violet-300' :
                            d < 0 ? 'text-rose-600 dark:text-rose-400' : 'text-slate-700 dark:text-slate-200'
                          }`}>
                            {i === j ? '0' : fmtD(d)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {/* Formula box */}
            <div className="rounded-xl bg-violet-50 dark:bg-violet-900/20 border border-violet-200 dark:border-violet-700/40 px-4 py-2.5 text-xs text-violet-800 dark:text-violet-300">
              <strong>还原公式：</strong> d(u, v) = d'(u, v) − h[u] + h[v]
              <br />
              其中 h[u], h[v] 为 Bellman-Ford 求得的势能（δ(S', ·)）。路径保序性保证了 Dijkstra 在新图中找到的最短路对应原图中的真实最短路。
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
