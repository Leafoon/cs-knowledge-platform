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

// 原始边 [u, v, capacity]
const ORIG_EDGES = [
  [0, 1, 10], [0, 3, 10], [1, 2, 4],
  [1, 3, 2],  [1, 4, 8],  [3, 4, 9],
  [2, 5, 10], [4, 5, 10],
]

// Edmonds-Karp 三条路径后的最终流量
const FINAL_FLOW = [10, 4, 4, 0, 6, 4, 4, 10]

// 残差网络中的所有有向边（前向 + 后向）
interface ResEdge {
  u: number; v: number
  cap: number        // 残差容量
  type: 'fwd' | 'bwd'
  origIdx: number    // 对应原始边编号
}

function buildResidual(flows: number[]): ResEdge[] {
  const res: ResEdge[] = []
  ORIG_EDGES.forEach(([u, v, cap], idx) => {
    const f = flows[idx]
    if (cap - f > 0) res.push({ u, v, cap: cap - f, type: 'fwd', origIdx: idx })
    if (f > 0)       res.push({ u: v, v: u, cap: f, type: 'bwd', origIdx: idx })
  })
  return res
}

const ZERO_FLOW  = ORIG_EDGES.map(() => 0)
const STEP_FLOWS = [
  ZERO_FLOW,
  [4, 0, 4, 0, 0, 0, 4, 0],   // 路径 s→A→B→t Δ=4
  [10, 0, 4, 0, 6, 0, 4, 6],  // 路径 s→A→D→t Δ=6
  FINAL_FLOW,                   // 路径 s→C→D→t Δ=4
]

const STEP_LABELS = [
  '初始状态：所有流量为 0，残差容量 = 原始容量',
  '路径 s→A→B→t 增广 Δ=4 后，A→B 饱和，出现反向弧 B→A(4)',
  '路径 s→A→D→t 增广 Δ=6 后，s→A 饱和，出现 A→s(10)、D→A(6)',
  '路径 s→C→D→t 增广 Δ=4 后，最大流=14，无增广路可用',
]

function quadBezier(x1: number, y1: number, x2: number, y2: number, bend = 0) {
  const mx = (x1 + x2) / 2, my = (y1 + y2) / 2
  const dx = x2 - x1, dy = y2 - y1
  const len = Math.hypot(dx, dy) || 1
  const cx = mx - (dy / len) * bend, cy = my + (dx / len) * bend
  return `M${x1},${y1} Q${cx},${cy} ${x2},${y2}`
}

function arrowPoint(x1: number, y1: number, x2: number, y2: number, bend: number, R: number) {
  // approx tangent at endpoint of quadratic
  const mx = (x1 + x2) / 2, my = (y1 + y2) / 2
  const dx2 = x2 - x1, dy2 = y2 - y1, len2 = Math.hypot(dx2, dy2) || 1
  const cx = mx - (dy2 / len2) * bend, cy = my + (dx2 / len2) * bend
  // tangent at t=1 of Q(P0,PC,P1): direction = P1 - PC
  const tx = x2 - cx, ty = y2 - cy
  const tlen = Math.hypot(tx, ty) || 1
  return { ex: x2 - tx/tlen*R, ey: y2 - ty/tlen*R }
}

export default function ResidualNetworkBuilder() {
  const [step, setStep] = useState(0)
  const [mode, setMode] = useState<'orig' | 'res'>('orig')
  const [selEdge, setSelEdge] = useState<ResEdge | null>(null)

  const flows = STEP_FLOWS[step]
  const resEdges = buildResidual(flows)

  const hasFwd = (e: ResEdge) => e.type === 'fwd'

  // Check if there's a pair (both fwd and bwd) for given (u,v)
  // to decide bend direction for residual edges
  function getBend(e: ResEdge) {
    const hasOpposite = resEdges.some(r => r.u === e.v && r.v === e.u)
    return hasOpposite ? (e.type === 'fwd' ? -18 : 18) : 0
  }

  function edgeStroke(e: ResEdge) {
    if (selEdge && selEdge.u === e.u && selEdge.v === e.v && selEdge.type === e.type) return '#f59e0b'
    return e.type === 'fwd' ? '#10b981' : '#94a3b8'
  }

  const R = 20

  return (
    <div className="w-full max-w-2xl mx-auto rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 shadow-xl overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-indigo-600 via-violet-600 to-purple-600 px-5 py-4">
        <h3 className="text-white font-bold text-lg tracking-tight">残差网络动态构建器</h3>
        <p className="text-indigo-200 text-sm mt-0.5">观察每次增广后残差网络的演变·切换原图/残差图视角</p>
      </div>

      {/* Mode toggle + step */}
      <div className="p-4 flex flex-wrap gap-3 items-center border-b border-slate-100 dark:border-slate-800 bg-slate-50 dark:bg-slate-800/40">
        <div className="flex rounded-lg overflow-hidden border border-slate-200 dark:border-slate-600 text-xs font-semibold">
          {(['orig', 'res'] as const).map(m => (
            <button key={m} onClick={() => { setMode(m); setSelEdge(null) }}
              className={`px-3 py-1.5 transition-colors ${mode === m
                ? 'bg-violet-600 text-white'
                : 'text-slate-500 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-700'}`}>
              {m === 'orig' ? '原始流图' : '残差图 G_f'}
            </button>
          ))}
        </div>

        <div className="flex gap-1.5 ml-auto">
          {STEP_FLOWS.map((_, i) => (
            <button key={i} onClick={() => { setStep(i); setSelEdge(null) }}
              className={`w-7 h-7 rounded-full text-xs font-bold border transition-all ${step === i
                ? 'bg-violet-600 text-white border-violet-600'
                : 'border-slate-300 dark:border-slate-600 text-slate-500 dark:text-slate-400 hover:border-violet-400'}`}>
              {i}
            </button>
          ))}
          <span className="self-center text-[10px] text-slate-400 ml-1">步骤</span>
        </div>
      </div>

      <div className="p-4 space-y-3">
        {/* Step label */}
        <div className="rounded-lg bg-violet-50 dark:bg-violet-900/20 border border-violet-200 dark:border-violet-700/40 px-3 py-2 text-[11px] text-violet-700 dark:text-violet-300 font-medium">
          {STEP_LABELS[step]}
        </div>

        {/* SVG */}
        <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/30 overflow-hidden">
          <svg viewBox="0 0 440 220" className="w-full" style={{ maxHeight: 220 }}>
            <defs>
              {mode === 'orig'
                ? ORIG_EDGES.map((_, idx) => (
                    <marker key={`o${idx}`} id={`arr-res-o${idx}`} markerWidth="7" markerHeight="7"
                      refX="6" refY="3.5" orient="auto">
                      <path d="M0,0 L7,3.5 L0,7 Z"
                        fill={flows[idx] > 0 ? (flows[idx] >= ORIG_EDGES[idx][2] ? '#ef4444' : '#3b82f6') : '#94a3b8'} />
                    </marker>
                  ))
                : resEdges.map((e, ri) => (
                    <marker key={`r${ri}`} id={`arr-res-r${ri}`} markerWidth="7" markerHeight="7"
                      refX="6" refY="3.5" orient="auto">
                      <path d="M0,0 L7,3.5 L0,7 Z" fill={edgeStroke(e)} />
                    </marker>
                  ))
              }
            </defs>

            {/* Original flow edges */}
            {mode === 'orig' && ORIG_EDGES.map(([u, v, cap], idx) => {
              const { x: x1, y: y1 } = NODES[u]
              const { x: x2, y: y2 } = NODES[v]
              const dx = x2-x1, dy = y2-y1, len = Math.hypot(dx,dy)
              const sx = x1+dx/len*R, sy = y1+dy/len*R
              const ex = x2-dx/len*R, ey = y2-dy/len*R
              const mx = (sx+ex)/2, my = (sy+ey)/2
              const px = -dy/len*14, py = dx/len*14
              const f = flows[idx]
              const color = f > 0 ? (f >= cap ? '#ef4444' : '#3b82f6') : '#94a3b8'
              return (
                <g key={idx} className="cursor-pointer" onClick={() => setSelEdge(null)}>
                  <line x1={sx} y1={sy} x2={ex} y2={ey} stroke={color} strokeWidth={2}
                    markerEnd={`url(#arr-res-o${idx})`} />
                  <text x={mx+px} y={my+py+4} textAnchor="middle" fontSize={10} fontWeight="bold" fill={color}>
                    {f}/{cap}
                  </text>
                </g>
              )
            })}

            {/* Residual edges */}
            {mode === 'res' && resEdges.map((e, ri) => {
              const n1 = NODES[e.u], n2 = NODES[e.v]
              const dx = n2.x-n1.x, dy = n2.y-n1.y, len = Math.hypot(dx,dy)
              const bend = getBend(e)
              const { ex: epx, ey: epy } = arrowPoint(n1.x, n1.y, n2.x, n2.y, bend, R)
              const sx = n1.x + dx/len*R, sy = n1.y + dy/len*R
              const path = quadBezier(sx, sy, epx, epy, bend)
              const mx = (sx+epx)/2 - (dy/len)*bend*0.5
              const my = (sy+epy)/2 + (dx/len)*bend*0.5
              const color = edgeStroke(e)
              const sel = selEdge && selEdge.u===e.u && selEdge.v===e.v && selEdge.type===e.type
              return (
                <g key={ri} className="cursor-pointer" onClick={() => setSelEdge(sel ? null : e)}>
                  <path d={path} stroke={color} strokeWidth={sel ? 3 : e.type==='fwd' ? 2.2 : 1.5}
                    strokeDasharray={e.type === 'bwd' ? '5 3' : undefined}
                    fill="none" markerEnd={`url(#arr-res-r${ri})`} />
                  <text x={mx} y={my+4} textAnchor="middle" fontSize={10} fontWeight="bold" fill={color}>
                    {e.cap}
                  </text>
                </g>
              )
            })}

            {/* Nodes */}
            {NODES.map(n => (
              <g key={n.id}>
                <circle cx={n.x} cy={n.y} r={R} fill={n.id===0?'#3b82f6':n.id===5?'#8b5cf6':'#475569'} opacity={0.9}/>
                <text x={n.x} y={n.y+5} textAnchor="middle" fontSize={14} fontWeight="bold" fill="white">{n.label}</text>
              </g>
            ))}
          </svg>
        </div>

        {/* Edge detail panel */}
        {mode === 'res' && selEdge && (
          <div className={`rounded-xl border p-3 text-[11px] ${selEdge.type === 'fwd'
            ? 'border-emerald-200 dark:border-emerald-700/50 bg-emerald-50 dark:bg-emerald-900/20'
            : 'border-slate-200 dark:border-slate-600 bg-slate-50 dark:bg-slate-800/30'}`}>
            <div className={`font-bold mb-2 uppercase tracking-wide text-xs ${selEdge.type === 'fwd' ? 'text-emerald-700 dark:text-emerald-400' : 'text-slate-500 dark:text-slate-300'}`}>
              {selEdge.type === 'fwd' ? '▶ 前向弧（剩余容量）' : '◀ 后向弧（可撤销流）'}
            </div>
            <div className="space-y-1 text-slate-600 dark:text-slate-300">
              <div>节点对：<b>{NODES[selEdge.u].label} → {NODES[selEdge.v].label}</b></div>
              {selEdge.type === 'fwd' ? (
                <>
                  <div>原容量 c = {ORIG_EDGES[selEdge.origIdx][2]}，当前流量 f = {flows[selEdge.origIdx]}</div>
                  <div className="font-mono bg-white dark:bg-slate-700 rounded px-2 py-1 text-emerald-700 dark:text-emerald-300">
                    残差容量 = c − f = {ORIG_EDGES[selEdge.origIdx][2]} − {flows[selEdge.origIdx]} = {selEdge.cap}
                  </div>
                </>
              ) : (
                <>
                  <div>对应正向边 {NODES[selEdge.v].label}→{NODES[selEdge.u].label} 当前流 f = {flows[selEdge.origIdx]}</div>
                  <div className="font-mono bg-white dark:bg-slate-700 rounded px-2 py-1 text-slate-600 dark:text-slate-300">
                    残差容量 = f = {selEdge.cap}（可向后"退还"的流量）
                  </div>
                </>
              )}
            </div>
          </div>
        )}

        {/* Legend */}
        {mode === 'res' && (
          <div className="flex flex-wrap gap-4 text-[10px] text-slate-500 dark:text-slate-400">
            <span className="flex items-center gap-1.5">
              <span className="w-6 h-0.5 bg-emerald-500 inline-block"/> 前向弧（实线，c−f）
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-6 inline-block" style={{ borderBottom: '2px dashed #94a3b8' }}/> 后向弧（虚线，f）
            </span>
            <span className="flex items-center gap-1.5">
              <span className="text-amber-500 font-bold text-xs">—</span> 已选中边
            </span>
          </div>
        )}
      </div>
    </div>
  )
}
