'use client'
import React, { useState, useEffect, useRef } from 'react'
import { Play, Pause, SkipForward, RotateCcw, GitMerge } from 'lucide-react'

// ── 图定义 ─────────────────────────────────────────────────────
// 节点 A=0, B=1, C=2, D=3, E=4, F=5
const NODES = ['A','B','C','D','E','F']

const NODE_POS: Record<string, { cx: number; cy: number }> = {
  A: { cx: 60,  cy: 140 },
  B: { cx: 175, cy: 60  },
  C: { cx: 175, cy: 218 },
  D: { cx: 295, cy: 140 },
  E: { cx: 295, cy: 218 },
  F: { cx: 400, cy: 178 },
}

// 全部边（无向）
const ALL_EDGES = [
  { u: 'A', v: 'B', w: 4  },
  { u: 'A', v: 'C', w: 2  },
  { u: 'B', v: 'C', w: 1  },
  { u: 'B', v: 'D', w: 5  },
  { u: 'C', v: 'D', w: 8  },
  { u: 'C', v: 'E', w: 10 },
  { u: 'D', v: 'E', w: 2  },
  { u: 'D', v: 'F', w: 6  },
  { u: 'E', v: 'F', w: 3  },
]

// 排序后处理顺序（Kruskal）
const SORTED_EDGES = [...ALL_EDGES].sort((a, b) => a.w - b.w)

// 并查集颜色（每个初始集合的颜色）
const GROUP_COLORS = ['#f87171', '#fb923c', '#facc15','#4ade80','#60a5fa','#c084fc']

type KruskalStep = {
  currentEdgeIdx: number    // SORTED_EDGES 中当前处理的边索引（-1=初始）
  mstEdges: string[]        // 已加入 MST 的边（"A-B" 格式）
  cyclicEdge: string | null // 本步成环的边
  uf: number[]              // parent数组（实际会转为 find() 根）
  description: string
  processedCount: number    // 已处理边数
  accepted: boolean | null   // 本步是接受还是拒绝
}

// 简单并查集（不需完整路径压缩，只用于预计算）
class SimpleUF {
  parent: number[]
  constructor(n: number) { this.parent = Array.from({ length: n }, (_, i) => i) }
  find(x: number): number {
    while (this.parent[x] !== x) x = this.parent[x]
    return x
  }
  union(x: number, y: number): boolean {
    const rx = this.find(x), ry = this.find(y)
    if (rx === ry) return false
    this.parent[ry] = rx
    return true
  }
  getGroups(): number[][] {
    const groups: Map<number, number[]> = new Map()
    for (let i = 0; i < this.parent.length; i++) {
      const r = this.find(i)
      if (!groups.has(r)) groups.set(r, [])
      groups.get(r)!.push(i)
    }
    return [...groups.values()]
  }
}

// 预计算所有步骤
function buildSteps(): KruskalStep[] {
  const steps: KruskalStep[] = []
  const uf = new SimpleUF(6)

  // 初始状态
  steps.push({
    currentEdgeIdx: -1,
    mstEdges: [],
    cyclicEdge: null,
    uf: [...uf.parent],
    description: `按权值对所有 ${ALL_EDGES.length} 条边排序。\n初始状态：每个节点自成一组（共 6 个独立集合）。`,
    processedCount: 0,
    accepted: null,
  })

  const mstEdges: string[] = []
  let processedCount = 0

  for (let i = 0; i < SORTED_EDGES.length; i++) {
    const e = SORTED_EDGES[i]
    const ui = NODES.indexOf(e.u)
    const vi = NODES.indexOf(e.v)
    const edgeKey = `${e.u}-${e.v}`
    const sameSet = uf.find(ui) === uf.find(vi)
    const accepted = !sameSet
    processedCount++

    if (!sameSet) {
      uf.union(ui, vi)
      mstEdges.push(edgeKey)
    }

    const groupInfo = uf.getGroups()
      .map(g => `{${g.map(x => NODES[x]).join(',')}}`)
      .join(' ')

    steps.push({
      currentEdgeIdx: i,
      mstEdges: [...mstEdges],
      cyclicEdge: sameSet ? edgeKey : null,
      uf: [...uf.parent],
      description: accepted
        ? `处理边 ${e.u}-${e.v} (权=${e.w})：\n${e.u} 和 ${e.v} 不在同一集合 → ✅ 加入 MST！\n执行 UNION(${e.u}, ${e.v})\n当前集合：${groupInfo}`
        : `处理边 ${e.u}-${e.v} (权=${e.w})：\n${e.u} 和 ${e.v} 已在同一集合 → ❌ 成环，跳过！\n当前集合：${groupInfo}`,
      processedCount,
      accepted,
    })

    if (mstEdges.length === 5) {
      steps.push({
        currentEdgeIdx: -1,
        mstEdges: [...mstEdges],
        cyclicEdge: null,
        uf: [...uf.parent],
        description: `🎉 MST 构建完成！已加入 ${mstEdges.length} 条边（= |V|-1 = 5）。\nMST 总权值 = ${mstEdges.reduce((acc, ek) => acc + ALL_EDGES.find(e => `${e.u}-${e.v}` === ek)!.w, 0)}`,
        processedCount,
        accepted: null,
      })
      break
    }
  }

  return steps
}

const STEPS = buildSteps()

// ── 绘制无向边 ─────────────────────────────────────────────────
function UndirectedEdge({
  u, v, w, state,
}: {
  u: string; v: string; w: number; state: 'mst' | 'current' | 'cycle' | 'normal'
}) {
  const f = NODE_POS[u], t = NODE_POS[v]
  const midX = (f.cx + t.cx) / 2
  const midY = (f.cy + t.cy) / 2

  const [color, width, opacity] = {
    mst:     ['#10b981', 3.5, 1],
    current: ['#f59e0b', 3,   1],
    cycle:   ['#ef4444', 3,   1],
    normal:  ['#cbd5e1', 1.5, 0.5],
  }[state] as [string, number, number]

  return (
    <g opacity={opacity}>
      <line x1={f.cx} y1={f.cy} x2={t.cx} y2={t.cy}
        stroke={color} strokeWidth={width}
        strokeDasharray={state === 'cycle' ? '5,4' : 'none'}
        className="transition-all duration-400" />
      {/* Weight label */}
      {state !== 'normal' && (
        <g>
          <rect x={midX - 10} y={midY - 8} width={20} height={16} rx={4}
            fill={color} opacity={0.9} />
          <text x={midX} y={midY + 1} textAnchor="middle" dominantBaseline="middle"
            fontSize="11" fontWeight="800" fill="white"
            className="select-none">{w}</text>
        </g>
      )}
      {state === 'normal' && (
        <text x={midX} y={midY - 5} textAnchor="middle"
          fontSize="10" fill="#94a3b8" className="select-none">{w}</text>
      )}
      {/* Cycle X mark */}
      {state === 'cycle' && (
        <>
          <line x1={midX - 5} y1={midY - 12} x2={midX + 5} y2={midY - 2} stroke="#ef4444" strokeWidth={2} />
          <line x1={midX + 5} y1={midY - 12} x2={midX - 5} y2={midY - 2} stroke="#ef4444" strokeWidth={2} />
        </>
      )}
    </g>
  )
}

// ── 主组件 ────────────────────────────────────────────────────────
export default function KruskalUnionFindTrace() {
  const [step, setStep] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1500)
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const cur = STEPS[step]
  const isLast = step === STEPS.length - 1

  useEffect(() => {
    if (playing && !isLast) {
      timerRef.current = setTimeout(() => setStep(s => s + 1), speed)
    } else if (isLast) setPlaying(false)
    return () => { if (timerRef.current) clearTimeout(timerRef.current) }
  }, [playing, step, speed, isLast])

  // 计算每条全局边的显示状态
  const getEdgeState = (edgeKey: string, e: typeof ALL_EDGES[0]): 'mst' | 'current' | 'cycle' | 'normal' => {
    if (cur.mstEdges.includes(edgeKey)) return 'mst'
    if (cur.cyclicEdge === edgeKey) return 'cycle'
    if (cur.currentEdgeIdx >= 0) {
      const ce = SORTED_EDGES[cur.currentEdgeIdx]
      if (`${ce.u}-${ce.v}` === edgeKey) return 'current'
    }
    return 'normal'
  }

  // 并查集分组（用于右侧显示）
  const uf2 = new SimpleUF(6)
  uf2.parent = [...cur.uf]
  const groups = uf2.getGroups()

  // 每个节点的根→组颜色（按根的字典序分配颜色）
  const rootToColor: Record<number, string> = {}
  groups.forEach((g, idx) => {
    const r = uf2.find(g[0])
    rootToColor[r] = GROUP_COLORS[idx % GROUP_COLORS.length]
  })
  const nodeColor = (n: string) => {
    const idx = NODES.indexOf(n)
    const r = uf2.find(idx)
    return rootToColor[r]
  }

  return (
    <div className="w-full max-w-4xl mx-auto my-6 rounded-2xl overflow-hidden border border-slate-200 dark:border-slate-700 shadow-xl">
      {/* Header */}
      <div className="bg-gradient-to-r from-lime-600 via-green-600 to-teal-600 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-xl font-bold text-white">Kruskal 并查集步进</h3>
            <p className="text-lime-100 text-sm mt-0.5">6 节点图 · 边排序贪心选择 · 并查集判环</p>
          </div>
          <div className="flex items-center gap-2 bg-white/20 rounded-lg px-3 py-1.5">
            <GitMerge className="w-4 h-4 text-lime-200" />
            <span className="text-white text-sm font-medium">步骤 {step + 1}/{STEPS.length}</span>
          </div>
        </div>
      </div>

      {/* Progress */}
      <div className="h-1.5 bg-slate-200 dark:bg-slate-700">
        <div className="h-full bg-gradient-to-r from-lime-400 to-teal-400 transition-all duration-500"
          style={{ width: `${(step / (STEPS.length - 1)) * 100}%` }} />
      </div>

      <div className="bg-white dark:bg-slate-900 p-4">
        <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
          {/* SVG graph (wider) */}
          <div className="md:col-span-3 bg-slate-50 dark:bg-slate-800 rounded-xl p-3 border border-slate-200 dark:border-slate-700">
            <div className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400 mb-2">图可视化</div>
            <svg viewBox="0 0 460 280" className="w-full">
              {/* Edges */}
              {ALL_EDGES.map(e => {
                const key = `${e.u}-${e.v}`
                return (
                  <UndirectedEdge key={key} u={e.u} v={e.v} w={e.w} state={getEdgeState(key, e)} />
                )
              })}
              {/* Weight labels for normal edges */}
              {ALL_EDGES.map(e => {
                const key = `${e.u}-${e.v}`
                const state = getEdgeState(key, e)
                if (state !== 'normal') return null
                const f = NODE_POS[e.u], t = NODE_POS[e.v]
                return (
                  <text key={key + 'lbl'} x={(f.cx + t.cx) / 2} y={(f.cy + t.cy) / 2 - 5}
                    textAnchor="middle" fontSize="10" fill="#94a3b8"
                    className="select-none">{e.w}</text>
                )
              })}
              {/* Nodes */}
              {NODES.map(n => {
                const { cx, cy } = NODE_POS[n]
                const color = nodeColor(n)
                const isMSTNode = cur.mstEdges.some(ek => ek.includes(n))
                return (
                  <g key={n}>
                    <circle cx={cx} cy={cy} r={20}
                      fill={color} stroke={isMSTNode ? '#065f46' : '#94a3b8'} strokeWidth={isMSTNode ? 3 : 2}
                      className="transition-all duration-400" opacity={0.9} />
                    <text x={cx} y={cy + 1} textAnchor="middle" dominantBaseline="middle"
                      fontSize="14" fontWeight="800" fill="white"
                      className="select-none">{n}</text>
                  </g>
                )
              })}
            </svg>
            {/* Legend */}
            <div className="flex flex-wrap gap-3 mt-1 px-1">
              {[
                { color: 'bg-emerald-500', label: 'MST 边' },
                { color: 'bg-amber-400', label: '当前考虑' },
                { color: 'bg-red-400', label: '成环跳过' },
              ].map(l => (
                <div key={l.label} className="flex items-center gap-1 text-xs text-slate-600 dark:text-slate-400">
                  <span className={`w-3 h-3 rounded ${l.color}`} />
                  {l.label}
                </div>
              ))}
            </div>
          </div>

          {/* Right panel: UF state + edge list */}
          <div className="md:col-span-2 flex flex-col gap-3">
            {/* Union-Find groups */}
            <div className="bg-slate-50 dark:bg-slate-800 rounded-xl p-3 border border-slate-200 dark:border-slate-700">
              <div className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400 mb-2">并查集分组</div>
              <div className="flex flex-wrap gap-2">
                {groups.map((g, idx) => (
                  <div key={idx}
                    className="flex items-center gap-1 px-2.5 py-1.5 rounded-lg border text-xs font-bold"
                    style={{ backgroundColor: GROUP_COLORS[idx] + '33', borderColor: GROUP_COLORS[idx] }}>
                    <span style={{ color: GROUP_COLORS[idx] }}>{`{${g.map(i => NODES[i]).join(',')}}`}</span>
                  </div>
                ))}
              </div>
              <div className="mt-2 text-xs text-slate-500 dark:text-slate-400">
                当前 MST 边数：{cur.mstEdges.length} / 5
                {cur.mstEdges.length > 0 && (
                  <span className="ml-1 text-emerald-600 dark:text-emerald-400">
                    (权 = {cur.mstEdges.reduce((acc, ek) => acc + ALL_EDGES.find(e => `${e.u}-${e.v}` === ek)!.w, 0)})
                  </span>
                )}
              </div>
            </div>

            {/* Sorted edge processing list */}
            <div className="bg-slate-50 dark:bg-slate-800 rounded-xl p-3 border border-slate-200 dark:border-slate-700 flex-1">
              <div className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400 mb-2">边处理队列（按权升序）</div>
              <div className="flex flex-col gap-1">
                {SORTED_EDGES.map((e, i) => {
                  const ek = `${e.u}-${e.v}`
                  const isMST = cur.mstEdges.includes(ek)
                  const isCurrent = cur.currentEdgeIdx === i
                  const isCycle = cur.cyclicEdge === ek
                  const isPast = cur.currentEdgeIdx > i && cur.currentEdgeIdx !== -1

                  return (
                    <div key={ek} className={`flex items-center justify-between px-2.5 py-1 rounded-md text-xs border transition-all duration-300
                      ${isCurrent && !isCycle ? 'bg-amber-100 dark:bg-amber-900/40 border-amber-300 dark:border-amber-600 font-bold' :
                        isCurrent && isCycle ? 'bg-red-100 dark:bg-red-900/30 border-red-300 dark:border-red-600 line-through' :
                        isMST ? 'bg-emerald-50 dark:bg-emerald-900/20 border-emerald-200 dark:border-emerald-700 text-emerald-700 dark:text-emerald-300' :
                        isPast ? 'bg-red-50 dark:bg-red-900/10 border-red-100 dark:border-red-900 text-red-400 dark:text-red-500 line-through opacity-60' :
                        'bg-white dark:bg-slate-700 border-slate-200 dark:border-slate-600 text-slate-600 dark:text-slate-300 opacity-50'}`}
                    >
                      <span>{e.u}-{e.v}</span>
                      <span>w={e.w}</span>
                      <span>
                        {isMST ? '✅' : isCycle || (isPast && !isMST) ? '❌' : i > (cur.currentEdgeIdx === -1 ? -1 : cur.currentEdgeIdx) ? '⏳' : ''}
                      </span>
                    </div>
                  )
                })}
              </div>
            </div>
          </div>
        </div>

        {/* Step description */}
        <div className={`mt-3 rounded-xl px-4 py-3 border text-sm leading-5
          ${cur.accepted === true ? 'bg-emerald-50 dark:bg-emerald-900/20 border-emerald-200 dark:border-emerald-700 text-emerald-800 dark:text-emerald-200' :
            cur.accepted === false ? 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-700 text-red-800 dark:text-red-200' :
            isLast ? 'bg-teal-50 dark:bg-teal-900/20 border-teal-200 dark:border-teal-700 text-teal-800 dark:text-teal-200' :
            'bg-slate-50 dark:bg-slate-800 border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-300'}`}>
          {cur.description.split('\n').map((line, i) => <div key={i} className="font-mono">{line}</div>)}
        </div>

        {/* Controls */}
        <div className="mt-4 flex items-center justify-between flex-wrap gap-2">
          <div className="flex items-center gap-2">
            <button onClick={() => { setPlaying(false); setStep(0) }}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-600 text-sm transition-colors">
              <RotateCcw className="w-3.5 h-3.5" /> 重置
            </button>
            <button onClick={() => setPlaying(p => !p)} disabled={isLast}
              className="flex items-center gap-1.5 px-4 py-1.5 rounded-lg bg-green-600 hover:bg-green-700 text-white text-sm font-semibold transition-colors disabled:opacity-40">
              {playing ? <><Pause className="w-3.5 h-3.5" /> 暂停</> : <><Play className="w-3.5 h-3.5" /> 自动播放</>}
            </button>
            <button onClick={() => { setPlaying(false); if (!isLast) setStep(s => s + 1) }} disabled={isLast}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-lime-100 dark:bg-lime-900/40 text-lime-700 dark:text-lime-300 hover:bg-lime-200 dark:hover:bg-lime-900/60 text-sm transition-colors disabled:opacity-40">
              <SkipForward className="w-3.5 h-3.5" /> 下一步
            </button>
          </div>
          <div className="flex items-center gap-2 text-sm text-slate-500 dark:text-slate-400">
            速度
            {[2000, 1500, 800].map(s => (
              <button key={s} onClick={() => setSpeed(s)}
                className={`px-2 py-1 rounded text-xs ${speed === s ? 'bg-green-600 text-white' : 'bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-300'}`}>
                {s === 2000 ? '慢' : s === 1500 ? '中' : '快'}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
