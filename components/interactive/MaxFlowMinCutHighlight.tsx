'use client'
import React, { useState } from 'react'

const NODES = [
  { id: 0, label: 's', x: 50,  y: 110 },
  { id: 1, label: 'A', x: 160, y: 42  },
  { id: 2, label: 'B', x: 290, y: 42  },
  { id: 3, label: 'C', x: 160, y: 178 },
  { id: 4, label: 'D', x: 290, y: 178 },
  { id: 5, label: 't', x: 390, y: 110 },
]

// [u, v, cap, finalFlow]
const EDGES = [
  [0,1,10,10],[0,3,10,4],[1,2,4,4],
  [1,3,2,0],  [1,4,8,6], [3,4,9,4],
  [2,5,10,4], [4,5,10,10],
] as const

// min-cut: S={s,A,C,D}(0,1,3,4)  T={B,t}(2,5)
// cut edges: A→B(idx2, cap=4), D→t(idx7, cap=10)
const S_SIDE = new Set([0, 1, 3, 4])
const CUT_EDGES = new Set([2, 7]) // A→B and D→t

type View = 'flow' | 'cut'

export default function MaxFlowMinCutHighlight() {
  const [view, setView] = useState<View>('flow')
  const [selEdge, setSelEdge] = useState<number | null>(null)
  const R = 20

  function nodeColor(id: number) {
    if (view === 'cut') {
      return S_SIDE.has(id) ? '#3b82f6' : '#f97316'
    }
    if (id === 0) return '#3b82f6'
    if (id === 5) return '#8b5cf6'
    return '#475569'
  }

  function edgeColor(idx: number) {
    if (selEdge === idx) return '#f59e0b'
    if (view === 'cut') {
      if (CUT_EDGES.has(idx)) return '#ef4444'
      const [u, v] = EDGES[idx]
      // cross edges from T to S are irrelevant (ignored in cut)
      return S_SIDE.has(u) && !S_SIDE.has(v) ? '#94a3b8' : '#e2e8f0'
    }
    const [,,cap,f] = EDGES[idx]
    if (f >= cap) return '#ef4444'
    if (f > 0)    return '#3b82f6'
    return '#94a3b8'
  }

  function edgeWidth(idx: number) {
    if (view === 'cut' && CUT_EDGES.has(idx)) return 3.5
    if (selEdge === idx) return 3
    return 2
  }

  return (
    <div className="w-full max-w-2xl mx-auto rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 shadow-xl overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-rose-600 via-pink-600 to-fuchsia-600 px-5 py-4">
        <h3 className="text-white font-bold text-lg tracking-tight">最大流 · 最小割定理可视化</h3>
        <p className="text-rose-200 text-sm mt-0.5">最大流值 = 最小割容量 = 14，切换两种视角深入理解</p>
      </div>

      {/* View toggle */}
      <div className="px-4 py-3 border-b border-slate-100 dark:border-slate-800 bg-slate-50 dark:bg-slate-800/40 flex gap-3">
        {([
          { k: 'flow', label: '最大流视图', desc: '展示最终最大流分配' },
          { k: 'cut',  label: '最小割视图', desc: '展示切割 (S, T) 与割边' },
        ] as { k: View; label: string; desc: string }[]).map(v => (
          <button key={v.k} onClick={() => { setView(v.k); setSelEdge(null) }}
            className={`flex-1 rounded-xl px-3 py-2 text-sm font-semibold border-2 transition-all ${
              view === v.k
                ? 'border-pink-400 bg-pink-50 dark:bg-pink-900/30 text-pink-700 dark:text-pink-300'
                : 'border-slate-200 dark:border-slate-600 text-slate-500 dark:text-slate-400 hover:border-slate-300'
            }`}>
            {v.label}
            <div className="text-[10px] font-normal opacity-70 mt-0.5">{v.desc}</div>
          </button>
        ))}
      </div>

      <div className="p-4 space-y-3">
        {/* SVG */}
        <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/30 overflow-hidden relative">
          {/* S/T partition labels */}
          {view === 'cut' && (
            <>
              <div className="absolute top-2 left-3 text-[11px] font-bold text-blue-500 dark:text-blue-400 bg-blue-50 dark:bg-blue-900/30 rounded px-1.5 py-0.5">S 侧 {'{s,A,C,D}'}</div>
              <div className="absolute top-2 right-3 text-[11px] font-bold text-orange-500 dark:text-orange-400 bg-orange-50 dark:bg-orange-900/30 rounded px-1.5 py-0.5">T 侧 {'{B,t}'}</div>
            </>
          )}
          <svg viewBox="0 0 440 220" className="w-full" style={{ maxHeight: 220 }}>
            <defs>
              {EDGES.map((_, idx) => (
                <marker key={idx} id={`arr-mc-${idx}`} markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
                  <path d="M0,0 L7,3.5 L0,7 Z" fill={edgeColor(idx)} />
                </marker>
              ))}
              {/* Dashed cut line */}
              {view === 'cut' && (
                <linearGradient id="cutLine" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#ef4444" stopOpacity={0.6} />
                  <stop offset="100%" stopColor="#f97316" stopOpacity={0.6} />
                </linearGradient>
              )}
            </defs>

            {/* Partition divider */}
            {view === 'cut' && (
              <line x1={230} y1={10} x2={230} y2={210}
                stroke="url(#cutLine)" strokeWidth={1.5} strokeDasharray="6 4" opacity={0.7} />
            )}

            {/* Edges */}
            {EDGES.map(([u, v, cap, f], idx) => {
              const n1 = NODES[u], n2 = NODES[v]
              const dx = n2.x-n1.x, dy = n2.y-n1.y, len = Math.hypot(dx,dy)
              const sx = n1.x+dx/len*R, sy = n1.y+dy/len*R
              const ex = n2.x-dx/len*R, ey = n2.y-dy/len*R
              const mx = (sx+ex)/2, my = (sy+ey)/2
              const px = -dy/len*13, py = dx/len*13
              const col = edgeColor(idx)
              const isCut = view === 'cut' && CUT_EDGES.has(idx)
              return (
                <g key={idx} className="cursor-pointer" onClick={() => setSelEdge(selEdge===idx?null:idx)}>
                  {isCut && (
                    <line x1={sx} y1={sy} x2={ex} y2={ey}
                      stroke="#ef4444" strokeWidth={10} opacity={0.15} />
                  )}
                  <line x1={sx} y1={sy} x2={ex} y2={ey}
                    stroke={col} strokeWidth={edgeWidth(idx)}
                    markerEnd={`url(#arr-mc-${idx})`} />
                  <line x1={sx} y1={sy} x2={ex} y2={ey} stroke="transparent" strokeWidth={12} />
                  <text x={mx+px} y={my+py+4} textAnchor="middle" fontSize={10} fontWeight="bold" fill={col}>
                    {view === 'flow' ? `${f}/${cap}` : isCut ? `c=${cap}` : `${cap}`}
                  </text>
                </g>
              )
            })}

            {/* Nodes */}
            {NODES.map(n => (
              <g key={n.id}>
                <circle cx={n.x} cy={n.y} r={R} fill={nodeColor(n.id)} />
                <text x={n.x} y={n.y+5} textAnchor="middle" fontSize={14} fontWeight="bold" fill="white">
                  {n.label}
                </text>
              </g>
            ))}
          </svg>
        </div>

        {/* Theorem box */}
        <div className="rounded-xl border border-pink-200 dark:border-pink-700/50 bg-pink-50 dark:bg-pink-900/20 px-4 py-3">
          <div className="text-[10px] font-bold text-pink-600 dark:text-pink-400 uppercase tracking-widest mb-2">最大流 — 最小割定理</div>
          <div className="text-center space-y-1">
            <div className="text-sm text-slate-600 dark:text-slate-300">
              <span className="font-bold text-blue-600 dark:text-blue-400">|f*| = 14</span>
              <span className="mx-2 text-rose-400">=</span>
              <span className="font-bold text-red-500">c(S,T)</span>
            </div>
            {view === 'cut' && (
              <div className="text-xs text-slate-500 dark:text-slate-400 font-mono">
                c(S,T) = c(A→B) + c(D→t) = 4 + 10 = <b className="text-red-500">14</b>
              </div>
            )}
            {view === 'flow' && (
              <div className="text-xs text-slate-500 dark:text-slate-400">
                总流量 = f(s,A) + f(s,C) = 10 + 4 = <b className="text-blue-600">14</b>
              </div>
            )}
          </div>
        </div>

        {/* Edge detail on click */}
        {selEdge !== null && (
          <div className={`rounded-xl border p-3 text-[11px] transition-all ${
            CUT_EDGES.has(selEdge)
              ? 'border-red-200 dark:border-red-700/50 bg-red-50 dark:bg-red-900/20'
              : 'border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/30'
          }`}>
            <div className="font-bold mb-1.5 text-xs uppercase tracking-wide text-slate-600 dark:text-slate-300">
              {NODES[EDGES[selEdge][0]].label} → {NODES[EDGES[selEdge][1]].label}
              {CUT_EDGES.has(selEdge) && <span className="ml-2 text-red-500">✂ 割边</span>}
            </div>
            <div className="grid grid-cols-3 gap-2">
              <div className="bg-white dark:bg-slate-800 rounded-lg p-2 text-center">
                <div className="text-slate-400 text-[9px]">容量</div>
                <div className="font-bold text-slate-700 dark:text-slate-200 text-base">{EDGES[selEdge][2]}</div>
              </div>
              <div className="bg-white dark:bg-slate-800 rounded-lg p-2 text-center">
                <div className="text-slate-400 text-[9px]">最大流</div>
                <div className="font-bold text-blue-600 dark:text-blue-400 text-base">{EDGES[selEdge][3]}</div>
              </div>
              <div className="bg-white dark:bg-slate-800 rounded-lg p-2 text-center">
                <div className="text-slate-400 text-[9px]">利用率</div>
                <div className={`font-bold text-base ${EDGES[selEdge][3]>=EDGES[selEdge][2]?'text-red-500':'text-emerald-600'}`}>
                  {Math.round(EDGES[selEdge][3]/EDGES[selEdge][2]*100)}%
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Legend */}
        <div className="flex flex-wrap gap-3 text-[10px] text-slate-500 dark:text-slate-400">
          {view === 'flow' ? <>
            <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-red-500 inline-block"/>饱和边</span>
            <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-blue-500 inline-block"/>有流量边</span>
            <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-slate-400 inline-block"/>零流量</span>
          </> : <>
            <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-blue-500 inline-block"/>S 侧节点</span>
            <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-orange-500 inline-block"/>T 侧节点</span>
            <span className="flex items-center gap-1.5"><span className="w-8 h-0.5 bg-red-500 inline-block rounded"/>割边（高亮）</span>
          </>}
        </div>
      </div>
    </div>
  )
}
