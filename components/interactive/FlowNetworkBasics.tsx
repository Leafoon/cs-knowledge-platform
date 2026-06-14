'use client'
import React, { useState } from 'react'

// ── 贯穿本章的 6 节点流网络 ─────────────────────────────────────
// s(0) A(1) B(2) C(3) D(4) t(5)
// 边: s→A:10, s→C:10, A→B:4, A→C:2, A→D:8, C→D:9, B→t:10, D→t:10
// Max flow = 14

const NODES = [
  { id: 0, label: 's', x: 50,  y: 110, type: 'source' },
  { id: 1, label: 'A', x: 160, y: 42,  type: 'mid' },
  { id: 2, label: 'B', x: 290, y: 42,  type: 'mid' },
  { id: 3, label: 'C', x: 160, y: 178, type: 'mid' },
  { id: 4, label: 'D', x: 290, y: 178, type: 'mid' },
  { id: 5, label: 't', x: 390, y: 110, type: 'sink' },
]

const EDGES = [
  { id: 0, u: 0, v: 1, cap: 10 }, // s→A
  { id: 1, u: 0, v: 3, cap: 10 }, // s→C
  { id: 2, u: 1, v: 2, cap: 4  }, // A→B
  { id: 3, u: 1, v: 3, cap: 2  }, // A→C
  { id: 4, u: 1, v: 4, cap: 8  }, // A→D
  { id: 5, u: 3, v: 4, cap: 9  }, // C→D
  { id: 6, u: 2, v: 5, cap: 10 }, // B→t
  { id: 7, u: 4, v: 5, cap: 10 }, // D→t
]

// 最终最大流分配
const FINAL_FLOW = [10, 4, 4, 0, 6, 4, 4, 10]

// 流守恒演示：展示每个节点的 in/out
function getNodeBalance(nodeId: number, flows: number[]) {
  let inFlow = 0, outFlow = 0
  EDGES.forEach((e, idx) => {
    if (e.v === nodeId) inFlow += flows[idx]
    if (e.u === nodeId) outFlow += flows[idx]
  })
  return { inFlow, outFlow }
}

type Tab = 'cap' | 'flow' | 'conservation'

export default function FlowNetworkBasics() {
  const [tab, setTab] = useState<Tab>('cap')
  const [selNode, setSelNode] = useState<number | null>(null)
  const [selEdge, setSelEdge] = useState<number | null>(null)

  const flows = tab === 'flow' || tab === 'conservation' ? FINAL_FLOW : EDGES.map(() => 0)

  function edgeColor(idx: number) {
    if (selEdge === idx) return '#f59e0b'
    if (tab === 'flow') {
      const ratio = flows[idx] / EDGES[idx].cap
      if (ratio >= 1) return '#ef4444'
      if (ratio > 0) return '#3b82f6'
      return '#cbd5e1'
    }
    return '#94a3b8'
  }

  function edgeWidth(idx: number) {
    return selEdge === idx ? 3 : tab === 'flow' && flows[idx] > 0 ? 2.5 : 1.8
  }

  function nodeColor(id: number) {
    const n = NODES[id]
    if (selNode === id) return '#f59e0b'
    if (n.type === 'source') return '#3b82f6'
    if (n.type === 'sink') return '#8b5cf6'
    if (tab === 'conservation') {
      const { inFlow, outFlow } = getNodeBalance(id, flows)
      return inFlow === outFlow ? '#10b981' : '#ef4444'
    }
    return '#64748b'
  }

  const selEdgeData = selEdge !== null ? EDGES[selEdge] : null

  return (
    <div className="w-full max-w-2xl mx-auto rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 shadow-xl overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-600 via-cyan-600 to-teal-600 px-5 py-4">
        <h3 className="text-white font-bold text-lg tracking-tight">流网络基础——容量、流量与守恒</h3>
        <p className="text-blue-200 text-sm mt-0.5">点击节点或边查看详情 · 切换三种视角</p>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50">
        {([
          { key: 'cap',          label: '① 容量约束' },
          { key: 'flow',         label: '② 最大流分配' },
          { key: 'conservation', label: '③ 流守恒验证' },
        ] as { key: Tab; label: string }[]).map(t => (
          <button key={t.key} onClick={() => { setTab(t.key); setSelNode(null); setSelEdge(null) }}
            className={`flex-1 py-2.5 text-xs font-semibold transition-all ${
              tab === t.key
                ? 'text-blue-600 dark:text-blue-400 border-b-2 border-blue-500 bg-white dark:bg-slate-900'
                : 'text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-200'
            }`}>
            {t.label}
          </button>
        ))}
      </div>

      <div className="p-4 space-y-3">
        {/* SVG Graph */}
        <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/30 overflow-hidden">
          <svg viewBox="0 0 440 220" className="w-full" style={{ maxHeight: 220 }}>
            <defs>
              {EDGES.map((_, idx) => (
                <marker key={idx} id={`arr-fnb-${idx}`} markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
                  <path d="M0,0 L8,4 L0,8 Z" fill={edgeColor(idx)} />
                </marker>
              ))}
            </defs>

            {/* Edges */}
            {EDGES.map((e, idx) => {
              const { x: x1, y: y1 } = NODES[e.u]
              const { x: x2, y: y2 } = NODES[e.v]
              const dx = x2 - x1, dy = y2 - y1, len = Math.hypot(dx, dy)
              const R = 20
              const sx = x1 + dx/len*R, sy = y1 + dy/len*R
              const ex = x2 - dx/len*R, ey = y2 - dy/len*R
              const mx = (sx+ex)/2, my = (sy+ey)/2
              const perp = { x: -dy/len*14, y: dx/len*14 }

              const flowVal = flows[idx]
              const label = tab === 'cap' ? `${e.cap}`
                : tab === 'flow' ? `${flowVal}/${e.cap}`
                : `${flowVal}/${e.cap}`

              return (
                <g key={idx} className="cursor-pointer" onClick={() => setSelEdge(selEdge === idx ? null : idx)}>
                  <line x1={sx} y1={sy} x2={ex} y2={ey}
                    stroke={edgeColor(idx)} strokeWidth={edgeWidth(idx)}
                    markerEnd={`url(#arr-fnb-${idx})`} />
                  {/* Invisible hitbox */}
                  <line x1={sx} y1={sy} x2={ex} y2={ey} stroke="transparent" strokeWidth={12} />
                  <rect x={mx+perp.x-16} y={my+perp.y-9} width={32} height={16} rx={4}
                    fill={selEdge === idx ? '#fef3c7' : 'white'}
                    stroke={edgeColor(idx)} strokeWidth={0.8}
                    className="dark:hidden" />
                  <rect x={mx+perp.x-16} y={my+perp.y-9} width={32} height={16} rx={4}
                    fill={selEdge === idx ? '#78350f' : '#1e293b'}
                    stroke={edgeColor(idx)} strokeWidth={0.8}
                    className="hidden dark:block" />
                  <text x={mx+perp.x} y={my+perp.y+4} textAnchor="middle"
                    fontSize={10} fontWeight="bold" fill={edgeColor(idx)}>{label}</text>
                </g>
              )
            })}

            {/* Nodes */}
            {NODES.map(n => (
              <g key={n.id} className="cursor-pointer" onClick={() => setSelNode(selNode === n.id ? null : n.id)}>
                <circle cx={n.x} cy={n.y} r={20} fill={nodeColor(n.id)} opacity={0.9} />
                {selNode === n.id && (
                  <circle cx={n.x} cy={n.y} r={24} fill="none" stroke="#f59e0b" strokeWidth={2} strokeDasharray="4 2" />
                )}
                <text x={n.x} y={n.y+5} textAnchor="middle" fontSize={14} fontWeight="bold" fill="white">{n.label}</text>
                {/* Node type label */}
                {(n.type === 'source' || n.type === 'sink') && (
                  <text x={n.x} y={n.y+36} textAnchor="middle" fontSize={9}
                    fill={n.type === 'source' ? '#3b82f6' : '#8b5cf6'}>
                    {n.type === 'source' ? '源点' : '汇点'}
                  </text>
                )}
              </g>
            ))}
          </svg>
        </div>

        {/* Info panel */}
        {selEdge !== null && selEdgeData && (
          <div className="rounded-xl border border-amber-200 dark:border-amber-700/50 bg-amber-50 dark:bg-amber-900/20 p-3">
            <div className="text-xs font-bold text-amber-700 dark:text-amber-400 mb-1.5 uppercase tracking-wide">
              边 {NODES[selEdgeData.u].label} → {NODES[selEdgeData.v].label} 详情
            </div>
            <div className="grid grid-cols-3 gap-2 text-[11px]">
              <div className="bg-white dark:bg-slate-800 rounded-lg p-2 text-center">
                <div className="text-slate-400 mb-0.5">容量上限</div>
                <div className="font-bold text-blue-600 dark:text-blue-400 text-base">{selEdgeData.cap}</div>
              </div>
              <div className="bg-white dark:bg-slate-800 rounded-lg p-2 text-center">
                <div className="text-slate-400 mb-0.5">当前流量</div>
                <div className="font-bold text-emerald-600 dark:text-emerald-400 text-base">{flows[selEdge]}</div>
              </div>
              <div className="bg-white dark:bg-slate-800 rounded-lg p-2 text-center">
                <div className="text-slate-400 mb-0.5">利用率</div>
                <div className={`font-bold text-base ${flows[selEdge]/selEdgeData.cap >= 1 ? 'text-red-500' : 'text-violet-600 dark:text-violet-400'}`}>
                  {Math.round(flows[selEdge]/selEdgeData.cap*100)}%
                </div>
              </div>
            </div>
            <div className="mt-2 text-[10px] text-amber-700 dark:text-amber-400">
              容量约束：0 ≤ f({NODES[selEdgeData.u].label},{NODES[selEdgeData.v].label}) = {flows[selEdge]} ≤ c = {selEdgeData.cap} ✓
            </div>
          </div>
        )}

        {selNode !== null && tab === 'conservation' && selNode !== 0 && selNode !== 5 && (
          <div className="rounded-xl border border-emerald-200 dark:border-emerald-700/50 bg-emerald-50 dark:bg-emerald-900/20 p-3">
            <div className="text-xs font-bold text-emerald-700 dark:text-emerald-400 mb-1.5 uppercase tracking-wide">
              节点 {NODES[selNode].label} 流守恒验证
            </div>
            <div className="flex items-center justify-center gap-4 text-sm">
              <div className="text-center">
                <div className="text-xs text-slate-400">流入 ∑f(v,{NODES[selNode].label})</div>
                <div className="font-bold text-blue-600 text-lg">{getNodeBalance(selNode, flows).inFlow}</div>
              </div>
              <div className="text-2xl text-slate-300">=</div>
              <div className="text-center">
                <div className="text-xs text-slate-400">流出 ∑f({NODES[selNode].label},v)</div>
                <div className="font-bold text-violet-600 text-lg">{getNodeBalance(selNode, flows).outFlow}</div>
              </div>
              <span className="text-emerald-500 text-xl">✓</span>
            </div>
          </div>
        )}

        {/* Legend */}
        <div className="flex flex-wrap gap-3 text-[10px] text-slate-500 dark:text-slate-400">
          {tab === 'cap' && <>
            <span className="flex items-center gap-1"><span className="w-6 h-0.5 bg-slate-400 inline-block rounded"/>&nbsp;边 (容量标注)</span>
            <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-blue-500 inline-block"/> 源点 s</span>
            <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-violet-500 inline-block"/> 汇点 t</span>
          </>}
          {tab === 'flow' && <>
            <span className="flex items-center gap-1"><span className="w-6 h-0.5 bg-blue-500 inline-block rounded"/>&nbsp;有流量（流量/容量）</span>
            <span className="flex items-center gap-1"><span className="w-6 h-0.5 bg-red-500 inline-block rounded"/>&nbsp;已饱和（流量=容量）</span>
            <span className="flex items-center gap-1"><span className="w-6 h-0.5 bg-slate-300 inline-block rounded"/>&nbsp;零流量</span>
          </>}
          {tab === 'conservation' && <>
            <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-emerald-500 inline-block"/> 流守恒成立</span>
            <span className="text-slate-400">（点击中间节点查看详情）</span>
          </>}
          <span className="ml-auto font-semibold text-blue-600 dark:text-blue-400">总流量 |f| = {tab === 'cap' ? '?' : '14'}</span>
        </div>
      </div>
    </div>
  )
}
