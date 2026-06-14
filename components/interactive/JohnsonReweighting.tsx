'use client'
import React, { useState, useEffect, useRef } from 'react'

// ── 4-顶点示例（含负权边，无负权环）─────────────────────────────
// 顶点: A(0) B(1) C(2) D(3)，超级源点: S'(4)
// 原始边: A→B:2, B→C:-3, B→D:4, C→D:1, D→A:5
// h[]: [0, 0, -3, -2]（通过 Bellman-Ford 从超级源点求得）
// 新权: A→B:2, B→C:0, B→D:6, C→D:0, D→A:3

const LABELS = ['A', 'B', 'C', 'D']
const H = [0, 0, -3, -2]

const ORIG_EDGES = [
  { u: 0, v: 1, w: 2,  w_new: 2 },
  { u: 1, v: 2, w: -3, w_new: 0 },
  { u: 1, v: 3, w: 4,  w_new: 6 },
  { u: 2, v: 3, w: 1,  w_new: 0 },
  { u: 3, v: 0, w: 5,  w_new: 3 },
]

const NODE_POS = [
  { x: 65,  y: 105 }, // A
  { x: 190, y: 42  }, // B
  { x: 190, y: 170 }, // C
  { x: 315, y: 105 }, // D
]
const SUPER_POS = { x: 190, y: 105 } // S' 中心

const PHASES = [
  {
    id: 0,
    label: '① 原图',
    title: '含负权边的原始图',
    desc: '原图有负权边（B→C: -3），Dijkstra 无法直接使用。我们需要找到一种方法将所有权值变为非负数，同时保持最短路径的相对顺序。',
    color: 'from-rose-600 to-orange-600',
    accent: 'rose',
  },
  {
    id: 1,
    label: '② 超级源点',
    title: '添加超级源点 S\'',
    desc: '新增虚拟顶点 S\'，向所有原图顶点连零权边（S\'→A: 0, S\'→B: 0, …）。这样 Bellman-Ford 可以从 S\' 出发，求到所有顶点的最短路作为"势能"。',
    color: 'from-violet-600 to-purple-600',
    accent: 'violet',
  },
  {
    id: 2,
    label: '③ 势能 h[v]',
    title: '运行 Bellman-Ford 求 h[v]',
    desc: 'Bellman-Ford 以 S\' 为源点运行，利用三角不等式保证：对每条边 (u,v)，有 h[v] ≤ h[u] + w(u,v)，即 w(u,v) + h[u] - h[v] ≥ 0。',
    color: 'from-blue-600 to-cyan-600',
    accent: 'blue',
  },
  {
    id: 3,
    label: '④ 重新定权',
    title: '计算新权值 w\'(u,v) ≥ 0',
    desc: 'w\'(u,v) = w(u,v) + h[u] - h[v]。证明保序性：对任意路径 s→t，新总权 = 原总权 + h[s] - h[t]（望远镜消去）。因此最短路径不变，现在可以安全使用 Dijkstra！',
    color: 'from-emerald-600 to-teal-600',
    accent: 'emerald',
  },
]

function Arrow({
  x1, y1, x2, y2, w, wNew, phase, isNeg, isSuperEdge,
}: {
  x1: number; y1: number; x2: number; y2: number
  w: number; wNew?: number; phase: number; isNeg: boolean; isSuperEdge?: boolean
}) {
  const dx = x2 - x1, dy = y2 - y1, len = Math.hypot(dx, dy)
  const R = 22
  const sx = x1 + (dx / len) * R, sy = y1 + (dy / len) * R
  const ex = x2 - (dx / len) * R, ey = y2 - (dy / len) * R
  const mx = (sx + ex) / 2, my = (sy + ey) / 2

  let color = '#94a3b8'
  let sw = 1.5
  let label = String(w)
  let labelColor = '#94a3b8'

  if (isSuperEdge) {
    color = '#a78bfa'
    sw = 1.2
    label = '0'
    labelColor = '#8b5cf6'
  } else if (phase === 3) {
    color = '#10b981'
    sw = 2.5
    label = String(wNew ?? w)
    labelColor = '#059669'
  } else if (phase >= 2 && isNeg) {
    color = '#f87171'
    sw = 2
    labelColor = '#ef4444'
  } else if (isNeg) {
    color = '#f87171'
    sw = 2
    labelColor = '#ef4444'
  }

  const markId = `arr-jr-${x1.toFixed(0)}-${y1.toFixed(0)}-${x2.toFixed(0)}`
  return (
    <g>
      <defs>
        <marker id={markId} markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
          <path d="M0,0 L7,3.5 L0,7 Z" fill={color} />
        </marker>
      </defs>
      <line x1={sx} y1={sy} x2={ex} y2={ey} stroke={color} strokeWidth={sw}
        strokeDasharray={isSuperEdge ? '4 3' : undefined}
        markerEnd={`url(#${markId})`} style={{ transition: 'stroke 0.4s' }} />
      <text x={mx + dy * 0.15} y={my - dx * 0.15 - 3} textAnchor="middle"
        fontSize={10} fontWeight="bold" fill={labelColor}>{label}</text>
    </g>
  )
}

export default function JohnsonReweighting() {
  const [phase, setPhase] = useState(0)
  const [auto, setAuto] = useState(false)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    if (auto) {
      timerRef.current = setInterval(() => {
        setPhase(p => {
          if (p >= 3) { setAuto(false); return p }
          return p + 1
        })
      }, 2200)
    } else {
      if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null }
    }
    return () => { if (timerRef.current) clearInterval(timerRef.current) }
  }, [auto])

  const cur = PHASES[phase]

  return (
    <div className="w-full max-w-3xl mx-auto rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 shadow-xl overflow-hidden">
      {/* Header */}
      <div className={`bg-gradient-to-r ${cur.color} px-5 py-4 transition-all duration-500`}>
        <h3 className="text-white font-bold text-lg tracking-tight">Johnson 算法 — 重新定权可视化</h3>
        <p className="text-white/80 text-sm mt-0.5">{cur.title}</p>
      </div>

      <div className="p-4 space-y-4">
        {/* Phase tabs */}
        <div className="flex gap-1.5">
          {PHASES.map(p => (
            <button key={p.id} onClick={() => { setAuto(false); setPhase(p.id) }}
              className={`flex-1 py-1.5 rounded-lg text-[11px] font-bold transition-all duration-200 ${
                phase === p.id
                  ? `bg-gradient-to-r ${p.color} text-white shadow scale-[1.03]`
                  : 'bg-slate-100 dark:bg-slate-800 text-slate-400 dark:text-slate-500 hover:bg-slate-200 dark:hover:bg-slate-700'
              }`}>
              {p.label}
            </button>
          ))}
        </div>

        <div className="flex gap-4 items-start">
          {/* SVG */}
          <div className="w-[360px] shrink-0 rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50 overflow-hidden">
            <svg viewBox="0 0 380 225" className="w-full">
              {/* Super source edges (phase ≥ 1) */}
              {phase >= 1 && NODE_POS.map((npos, ni) => (
                <Arrow key={`s-${ni}`}
                  x1={SUPER_POS.x} y1={SUPER_POS.y}
                  x2={npos.x} y2={npos.y}
                  w={0} phase={phase} isNeg={false} isSuperEdge />
              ))}

              {/* Original edges */}
              {ORIG_EDGES.map((e, i) => (
                <Arrow key={i}
                  x1={NODE_POS[e.u].x} y1={NODE_POS[e.u].y}
                  x2={NODE_POS[e.v].x} y2={NODE_POS[e.v].y}
                  w={e.w} wNew={e.w_new} phase={phase} isNeg={e.w < 0} />
              ))}

              {/* Regular nodes */}
              {NODE_POS.map((pos, ni) => {
                const hVal = H[ni]
                const showH = phase >= 2
                return (
                  <g key={ni}>
                    <circle cx={pos.x} cy={pos.y} r={22}
                      fill={phase === 3 ? '#10b981' : '#6366f1'}
                      stroke={phase === 3 ? '#059669' : '#4f46e5'}
                      strokeWidth={2.5} style={{ transition: 'fill 0.4s' }} />
                    <text x={pos.x} y={pos.y + (showH ? 0 : 5)} textAnchor="middle"
                      fontSize={13} fontWeight="bold" fill="#fff">{LABELS[ni]}</text>
                    {showH && (
                      <text x={pos.x} y={pos.y + 12} textAnchor="middle" fontSize={9} fill="#bfdbfe">
                        h={hVal}
                      </text>
                    )}
                  </g>
                )
              })}

              {/* Super source (phase ≥ 1) */}
              {phase >= 1 && (
                <g>
                  <circle cx={SUPER_POS.x} cy={SUPER_POS.y} r={22}
                    fill="#7c3aed" stroke="#6d28d9" strokeWidth={2.5}
                    strokeDasharray="5 3" />
                  <text x={SUPER_POS.x} y={SUPER_POS.y + 5} textAnchor="middle"
                    fontSize={12} fontWeight="bold" fill="#fff">S'</text>
                </g>
              )}
            </svg>
          </div>

          {/* Right panel */}
          <div className="flex-1 min-w-0 space-y-3">
            {/* Edge weight table */}
            <div className="rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
              <div className="px-3 py-2 text-[11px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wide border-b border-slate-200 dark:border-slate-700">
                边权变化
              </div>
              <table className="w-full text-xs">
                <thead>
                  <tr className="bg-slate-50 dark:bg-slate-800/50 border-b border-slate-100 dark:border-slate-700/50">
                    <th className="px-3 py-1.5 text-left text-slate-400 font-medium">边</th>
                    <th className="px-3 py-1.5 text-center text-slate-400 font-medium">w(u,v)</th>
                    {phase >= 2 && <th className="px-3 py-1.5 text-center text-blue-500 font-medium">h[u]-h[v]</th>}
                    {phase === 3 && <th className="px-3 py-1.5 text-center text-emerald-600 font-medium">w' ≥ 0</th>}
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 dark:divide-slate-700/50">
                  {ORIG_EDGES.map((e, i) => {
                    const hDiff = H[e.u] - H[e.v]
                    const isNeg = e.w < 0
                    return (
                      <tr key={i} className={`${isNeg ? 'bg-rose-50 dark:bg-rose-900/10' : ''}`}>
                        <td className="px-3 py-1.5 font-bold text-slate-700 dark:text-slate-200">
                          {LABELS[e.u]}→{LABELS[e.v]}
                        </td>
                        <td className={`px-3 py-1.5 text-center font-mono font-bold ${
                          isNeg ? 'text-rose-600 dark:text-rose-400' : 'text-slate-600 dark:text-slate-300'
                        }`}>{e.w}</td>
                        {phase >= 2 && (
                          <td className="px-3 py-1.5 text-center font-mono text-blue-600 dark:text-blue-400">
                            {hDiff >= 0 ? '+' : ''}{hDiff}
                          </td>
                        )}
                        {phase === 3 && (
                          <td className="px-3 py-1.5 text-center font-mono font-bold text-emerald-600 dark:text-emerald-400">
                            {e.w_new} ✓
                          </td>
                        )}
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>

            {/* h values (phase ≥ 2) */}
            {phase >= 2 && (
              <div className="rounded-xl border border-blue-200 dark:border-blue-700/50 bg-blue-50 dark:bg-blue-900/20 overflow-hidden">
                <div className="px-3 py-2 text-[11px] font-bold text-blue-600 dark:text-blue-400 uppercase tracking-wide border-b border-blue-200 dark:border-blue-700/40">
                  势能函数 h[v] = δ(S', v)
                </div>
                <div className="flex p-2 gap-2">
                  {LABELS.map((l, i) => (
                    <div key={i} className="flex-1 bg-white dark:bg-slate-800 rounded-lg p-2 text-center border border-blue-100 dark:border-blue-700/30">
                      <div className="text-xs font-bold text-slate-500 dark:text-slate-400">{l}</div>
                      <div className="text-lg font-black font-mono text-blue-700 dark:text-blue-300">{H[i]}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Formula (phase 3) */}
            {phase === 3 && (
              <div className="rounded-xl border border-emerald-200 dark:border-emerald-700/50 bg-emerald-50 dark:bg-emerald-900/20 p-3">
                <div className="text-[11px] font-bold text-emerald-700 dark:text-emerald-400 uppercase tracking-wide mb-2">
                  公式验证
                </div>
                <div className="space-y-1 text-[11px] font-mono text-emerald-800 dark:text-emerald-300">
                  <div>w'(B→C) = -3 + h[B] - h[C] = -3 + <strong>0</strong> - (<strong>-3</strong>) = <strong>0 ✓</strong></div>
                  <div>w'(B→D) = 4 + h[B] - h[D] = 4 + <strong>0</strong> - (<strong>-2</strong>) = <strong>6 ✓</strong></div>
                  <div>w'(D→A) = 5 + h[D] - h[A] = 5 + (<strong>-2</strong>) - <strong>0</strong> = <strong>3 ✓</strong></div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Description */}
        <div className={`rounded-xl border px-4 py-3 text-sm leading-relaxed transition-all duration-400 ${
          phase === 0 ? 'border-rose-200 dark:border-rose-700/40 bg-rose-50 dark:bg-rose-900/20 text-rose-800 dark:text-rose-300' :
          phase === 1 ? 'border-violet-200 dark:border-violet-700/40 bg-violet-50 dark:bg-violet-900/20 text-violet-800 dark:text-violet-300' :
          phase === 2 ? 'border-blue-200 dark:border-blue-700/40 bg-blue-50 dark:bg-blue-900/20 text-blue-800 dark:text-blue-300' :
                        'border-emerald-200 dark:border-emerald-700/40 bg-emerald-50 dark:bg-emerald-900/20 text-emerald-800 dark:text-emerald-300'
        }`}>
          {cur.desc}
        </div>

        {/* Controls */}
        <div className="flex items-center gap-2">
          <button onClick={() => { setAuto(false); setPhase(0) }}
            className="px-3 py-1.5 rounded-lg border border-slate-300 dark:border-slate-600 text-xs text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors">
            ↺ 重置
          </button>
          <button onClick={() => setPhase(p => Math.max(0, p - 1))} disabled={phase === 0}
            className="px-3 py-1.5 rounded-lg border border-slate-300 dark:border-slate-600 text-xs text-slate-600 dark:text-slate-300 disabled:opacity-40 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors">
            ← 上一步
          </button>
          <button onClick={() => setAuto(a => !a)} disabled={phase === 3}
            className={`px-4 py-1.5 rounded-lg text-xs font-bold transition-colors disabled:opacity-40 ${
              auto ? 'bg-amber-500 text-white hover:bg-amber-600' : 'bg-indigo-600 text-white hover:bg-indigo-700'
            }`}>
            {auto ? '⏸ 暂停' : phase === 0 ? '▶ 自动演示' : '▶ 继续'}
          </button>
          <button onClick={() => { setAuto(false); setPhase(p => Math.min(3, p + 1)) }} disabled={phase === 3}
            className="px-3 py-1.5 rounded-lg border border-slate-300 dark:border-slate-600 text-xs text-slate-600 dark:text-slate-300 disabled:opacity-40 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors">
            下一步 →
          </button>
          <span className="text-xs text-slate-400 ml-auto font-mono">{phase + 1}/4 步</span>
        </div>
      </div>
    </div>
  )
}
