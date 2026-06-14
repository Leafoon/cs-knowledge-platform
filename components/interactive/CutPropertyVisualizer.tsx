'use client'

import React, { useState, useMemo } from 'react'

interface GEdge { u: string; v: string; w: number }
interface GNode { id: string; x: number; y: number }

const NODES: GNode[] = [
  { id: 'A', x: 80,  y: 120 },
  { id: 'B', x: 230, y: 55  },
  { id: 'C', x: 210, y: 195 },
  { id: 'D', x: 360, y: 120 },
  { id: 'E', x: 355, y: 220 },
]

const EDGES: GEdge[] = [
  { u: 'A', v: 'B', w: 4  },
  { u: 'A', v: 'C', w: 2  },
  { u: 'B', v: 'C', w: 5  },
  { u: 'B', v: 'D', w: 10 },
  { u: 'C', v: 'D', w: 3  },
  { u: 'C', v: 'E', w: 7  },
  { u: 'D', v: 'E', w: 1  },
]

// Kruskal MST
const MST_EDGES = (() => {
  const sortedE = [...EDGES].sort((a, b) => a.w - b.w)
  const parent: Record<string, string> = {}
  NODES.forEach(n => { parent[n.id] = n.id })
  function find(x: string): string { return parent[x] === x ? x : (parent[x] = find(parent[x])) }
  function union(a: string, b: string) { parent[find(a)] = find(b) }
  const mst: GEdge[] = []
  for (const e of sortedE) {
    if (find(e.u) !== find(e.v)) { union(e.u, e.v); mst.push(e) }
  }
  return mst
})()

const MST_EDGE_SET = new Set(MST_EDGES.map(e => `${[e.u,e.v].sort().join('-')}`))

// Kruskal steps for the walkthrough tab
const KRUSKAL_STEPS = (() => {
  const sortedE = [...EDGES].sort((a, b) => a.w - b.w)
  const parent: Record<string, string> = {}
  NODES.forEach(n => { parent[n.id] = n.id })
  function find(x: string): string { return parent[x] === x ? x : (parent[x] = find(parent[x])) }
  function union(a: string, b: string) { parent[find(a)] = find(b) }
  const mst: string[] = []
  const steps: { edge: GEdge; added: boolean; mst: string[]; reason: string }[] = []
  for (const e of sortedE) {
    const fa = find(e.u), fb = find(e.v)
    const added = fa !== fb
    if (added) { union(e.u, e.v); mst.push(`${[e.u,e.v].sort().join('-')}`) }
    steps.push({
      edge: e,
      added,
      mst: [...mst],
      reason: added
        ? `${e.u} 和 ${e.v} 在不同连通分量，割集存在——此边是最小跨割边，加入 MST。`
        : `${e.u} 和 ${e.v} 已在同一连通分量（${fa}），加入会形成环，跳过。`,
    })
  }
  return steps
})()

function nodePos(id: string): GNode { return NODES.find(n => n.id === id)! }

export default function CutPropertyVisualizer() {
  const [setS, setSetS] = useState<string[]>(['A', 'C'])
  const [tab, setTab] = useState<'cut' | 'kruskal'>('cut')
  const [kStep, setKStep] = useState(0)

  const toggleNode = (id: string) => {
    setSetS(prev => prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id])
  }

  const { crossingEdges, safeEdge } = useMemo(() => {
    const S = new Set(setS)
    const crossing = EDGES.filter(e => S.has(e.u) !== S.has(e.v))
    const safe = crossing.reduce<GEdge | null>((min, e) => (!min || e.w < min.w ? e : min), null)
    return { crossingEdges: crossing, safeEdge: safe }
  }, [setS])

  const kCur = KRUSKAL_STEPS[Math.min(kStep, KRUSKAL_STEPS.length - 1)]
  const kMstSet = kCur ? new Set(kCur.mst) : new Set<string>()
  const kHighlight = kCur ? `${[kCur.edge.u, kCur.edge.v].sort().join('-')}` : ''

  function edgeColor(e: GEdge) {
    if (tab === 'kruskal') {
      const key = `${[e.u, e.v].sort().join('-')}`
      if (key === kHighlight) return kCur?.added ? '#10b981' : '#f43f5e'
      if (kMstSet.has(key)) return '#6366f1'
      return '#94a3b8'
    }
    // Cut tab
    const sSet = new Set(setS)
    const isCrossing = sSet.has(e.u) !== sSet.has(e.v)
    if (!isCrossing) return '#94a3b8'
    const key = `${[safeEdge?.u, safeEdge?.v].sort().join('-')}`
    if (key === `${[e.u, e.v].sort().join('-')}`) return '#f59e0b'
    return '#fb923c'
  }

  function edgeWidth(e: GEdge) {
    if (tab === 'kruskal') {
      const key = `${[e.u, e.v].sort().join('-')}`
      return key === kHighlight ? 4 : kMstSet.has(key) ? 3 : 1.5
    }
    const sSet = new Set(setS)
    const isCrossing = sSet.has(e.u) !== sSet.has(e.v)
    const key = `${[safeEdge?.u, safeEdge?.v].sort().join('-')}`
    const isSafe = key === `${[e.u, e.v].sort().join('-')}`
    return isSafe ? 4 : isCrossing ? 2.5 : 1.5
  }

  const SAfeKey = safeEdge ? `${[safeEdge.u, safeEdge.v].sort().join('-')}` : ''

  return (
    <div className="w-full max-w-5xl mx-auto rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 shadow-xl overflow-hidden">
      <div className="bg-gradient-to-r from-sky-600 via-blue-600 to-indigo-600 px-5 py-4">
        <h3 className="text-white font-bold text-lg">MST 切割性质可视化 — 交互式图探索</h3>
        <p className="text-blue-100 text-sm mt-0.5">点击节点切换割集 S，观察跨割边与最小安全边；或逐步演示 Kruskal 算法</p>
      </div>

      <div className="p-4 space-y-4">
        {/* Tabs */}
        <div className="flex gap-2">
          <button onClick={() => setTab('cut')} className={`px-4 py-2 rounded-lg text-sm font-bold transition-all ${tab === 'cut' ? 'bg-blue-600 text-white' : 'border border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-200'}`}>
            切割性质探索器
          </button>
          <button onClick={() => setTab('kruskal')} className={`px-4 py-2 rounded-lg text-sm font-bold transition-all ${tab === 'kruskal' ? 'bg-indigo-600 text-white' : 'border border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-200'}`}>
            Kruskal 逐步演示
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* SVG Graph */}
          <div className="md:col-span-2 rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/40 p-2">
            <svg viewBox="0 0 440 280" className="w-full h-auto" style={{ minHeight: 220 }}>
              {/* S region background (cut mode) */}
              {tab === 'cut' && setS.length > 0 && (
                <ellipse cx="145" cy="160" rx="110" ry="90" fill="rgba(99,102,241,0.08)" stroke="rgba(99,102,241,0.3)" strokeWidth="1.5" strokeDasharray="6,4"/>
              )}

              {/* Edges */}
              {EDGES.map((e, i) => {
                const a = nodePos(e.u), b = nodePos(e.v)
                const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2
                const col = edgeColor(e)
                const wid = edgeWidth(e)
                const key = `${[e.u, e.v].sort().join('-')}`
                const isMSTedge = MST_EDGE_SET.has(key)
                return (
                  <g key={i}>
                    <line x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke={col} strokeWidth={wid} strokeLinecap="round" className="transition-all duration-400"/>
                    {/* Weight label */}
                    <circle cx={mx} cy={my} r={10} fill="white" stroke={col} strokeWidth={1.5} className="transition-all duration-400"/>
                    <text x={mx} y={my} textAnchor="middle" dominantBaseline="middle" fontSize={10} fontWeight="bold" fill={col} className="transition-all duration-400">{e.w}</text>
                  </g>
                )
              })}

              {/* Nodes */}
              {NODES.map(n => {
                const inS = setS.includes(n.id)
                const isMSTNode = MST_EDGES.some(e => e.u === n.id || e.v === n.id)
                return (
                  <g key={n.id} onClick={() => tab === 'cut' && toggleNode(n.id)} style={{ cursor: tab === 'cut' ? 'pointer' : 'default' }}>
                    <circle
                      cx={n.x} cy={n.y} r={22}
                      fill={inS && tab === 'cut' ? '#6366f1' : tab === 'kruskal' ? '#4f46e5' : '#64748b'}
                      stroke={inS && tab === 'cut' ? '#4338ca' : 'white'}
                      strokeWidth={3}
                      className="transition-all duration-300"
                    />
                    <text x={n.x} y={n.y} textAnchor="middle" dominantBaseline="middle" fontSize={14} fontWeight="bold" fill="white">{n.id}</text>
                    {tab === 'cut' && (
                      <text x={n.x} y={n.y + 33} textAnchor="middle" fontSize={9} fill={inS ? '#6366f1' : '#94a3b8'} fontWeight="bold">
                        {inS ? '∈ S' : '∉ S'}
                      </text>
                    )}
                  </g>
                )
              })}

              {/* Safe edge annotation (cut mode) */}
              {tab === 'cut' && safeEdge && (() => {
                const ua = nodePos(safeEdge.u), va = nodePos(safeEdge.v)
                const mx = (ua.x + va.x) / 2, my = (ua.y + va.y) / 2 - 14
                return (
                  <g>
                    <text x={mx} y={my} textAnchor="middle" fontSize={10} fontWeight="bold" fill="#f59e0b">安全边</text>
                  </g>
                )
              })()}
            </svg>
          </div>

          {/* Right panel */}
          <div className="space-y-3">
            {tab === 'cut' ? (
              <>
                {/* Node toggles */}
                <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/40 p-3">
                  <div className="text-xs font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-2">点击节点切换割集 S</div>
                  <div className="flex flex-wrap gap-2">
                    {NODES.map(n => {
                      const inS = setS.includes(n.id)
                      return (
                        <button key={n.id} onClick={() => toggleNode(n.id)} className={`px-3 py-1.5 rounded-full text-sm font-bold border-2 transition-all ${inS ? 'bg-indigo-600 text-white border-indigo-600' : 'bg-white dark:bg-slate-900 text-slate-700 dark:text-slate-200 border-slate-200 dark:border-slate-700'}`}>
                          {n.id} {inS ? '∈S' : '∉S'}
                        </button>
                      )
                    })}
                  </div>
                </div>

                {/* Crossing edges */}
                <div className="rounded-xl border border-orange-200 dark:border-orange-800 bg-orange-50 dark:bg-orange-950/20 p-3">
                  <div className="text-xs font-bold text-orange-700 dark:text-orange-400 uppercase tracking-wider mb-2">跨割边（crossing edges）</div>
                  {crossingEdges.length === 0
                    ? <div className="text-xs text-slate-400 dark:text-slate-500 italic">无跨割边，请调整 S</div>
                    : <div className="space-y-1">
                        {crossingEdges.sort((a,b) => a.w - b.w).map((e, i) => {
                          const key = `${[e.u, e.v].sort().join('-')}`
                          const isSafe = key === SAfeKey
                          return (
                            <div key={i} className={`flex items-center justify-between px-2 py-1 rounded-lg text-xs ${isSafe ? 'bg-amber-100 dark:bg-amber-950/50 font-black text-amber-700 dark:text-amber-300' : 'text-slate-600 dark:text-slate-300'}`}>
                              <span>{e.u}-{e.v}</span>
                              <span>w={e.w} {isSafe ? '← 安全边 ✅' : ''}</span>
                            </div>
                          )
                        })}
                      </div>
                  }
                </div>

                {/* Safe edge callout */}
                {safeEdge ? (
                  <div className="rounded-xl border border-amber-400 dark:border-amber-600 bg-amber-50 dark:bg-amber-950/30 p-3">
                    <div className="text-xs font-bold text-amber-700 dark:text-amber-400 uppercase tracking-wider">安全边（Safe Edge）</div>
                    <div className="mt-1 text-lg font-black text-amber-800 dark:text-amber-300">{safeEdge.u}-{safeEdge.v} (w={safeEdge.w})</div>
                    <div className="text-xs text-amber-700 dark:text-amber-400 mt-1">最小跨割边，必属于某棵 MST</div>
                    {MST_EDGE_SET.has(`${[safeEdge.u, safeEdge.v].sort().join('-')}`) && (
                      <div className="text-xs text-indigo-600 dark:text-indigo-400 mt-0.5 font-bold">✅ 确实在本图 MST 中</div>
                    )}
                  </div>
                ) : null}
              </>
            ) : (
              /* Kruskal panel */
              <>
                <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/40 p-3">
                  <div className="text-xs font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-2">Kruskal 步骤（按权重升序）</div>
                  <div className="flex items-center gap-2 mb-2">
                    <button onClick={() => setKStep(0)} className="px-2 py-1 rounded text-xs border border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-300">重置</button>
                    <button onClick={() => setKStep(p => Math.max(0, p-1))} className="px-2 py-1 rounded text-xs border border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-300">上一步</button>
                    <button onClick={() => setKStep(p => Math.min(KRUSKAL_STEPS.length-1, p+1))} className="px-2 py-1 rounded text-xs bg-indigo-600 text-white">下一步</button>
                    <span className="text-xs text-slate-400 dark:text-slate-500 ml-auto">{kStep+1}/{KRUSKAL_STEPS.length}</span>
                  </div>
                  {kCur && (
                    <div className="space-y-2">
                      <div className={`rounded-lg border px-3 py-2 text-sm font-bold ${kCur.added ? 'border-emerald-300 dark:border-emerald-700 text-emerald-700 dark:text-emerald-300 bg-emerald-50 dark:bg-emerald-950/30' : 'border-rose-300 dark:border-rose-700 text-rose-700 dark:text-rose-300 bg-rose-50 dark:bg-rose-950/30'}`}>
                        {kCur.added ? '✅' : '❌'} {kCur.edge.u}-{kCur.edge.v} (w={kCur.edge.w})
                      </div>
                      <div className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed">{kCur.reason}</div>
                      <div className="text-xs text-slate-500 dark:text-slate-400">
                        MST 边：{kMstSet.size > 0 ? [...kMstSet].join(', ') : '（空）'}
                      </div>
                    </div>
                  )}
                </div>

                <div className="rounded-xl border border-indigo-200 dark:border-indigo-800 bg-indigo-50 dark:bg-indigo-950/20 p-3 text-xs text-indigo-700 dark:text-indigo-300">
                  <div className="font-bold mb-1">切割性质与 Kruskal 的联系</div>
                  每次 Kruskal 选边时，以"已选 MST 节点"为 S，新边就是当前割集的最小跨割边（安全边），这正是切割性质保证 Kruskal 正确性的本质。
                </div>
              </>
            )}
          </div>
        </div>

        {/* Cut property theorem */}
        <div className="rounded-xl border border-blue-200 dark:border-blue-800 bg-blue-50 dark:bg-blue-950/20 px-4 py-3 text-sm text-blue-700 dark:text-blue-300">
          <div className="font-bold mb-1">切割性质定理（Cut Property）</div>
          <div className="leading-relaxed">
            设 <strong>G=(V,E,w)</strong> 是无向加权连通图，<strong>S ⊂ V</strong> 为任意非空真子集（即"割"），<strong>e</strong> 是 S 与 V\S 之间权重最小的跨割边（safe edge）。则<strong> e 必属于 G 的某棵最小生成树（MST）</strong>。<br/>
            <span className="text-xs opacity-80 mt-1 block">证明思路：若 e 不在任意 MST T 中，将 e 加入 T 会形成环 C；C 中必有另一跨割边 e'（w(e')≥w(e)）；用 e 替换 e' 得 T' 权重不增，T' 仍为 MST，且含 e，矛盾 □</span>
          </div>
        </div>
      </div>
    </div>
  )
}
