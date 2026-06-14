'use client'
import React, { useState, useEffect, useRef } from 'react'
import { Play, Pause, SkipForward, RotateCcw, Sprout } from 'lucide-react'

// ── 数据 ──────────────────────────────────────────────────────────
const NODES = ['A','B','C','D','E','F']
const INF = 9999

const NODE_POS: Record<string, { cx: number; cy: number }> = {
  A: { cx: 60,  cy: 140 },
  B: { cx: 175, cy: 60  },
  C: { cx: 175, cy: 218 },
  D: { cx: 295, cy: 140 },
  E: { cx: 295, cy: 218 },
  F: { cx: 400, cy: 178 },
}

// 无向边集合
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

type PrimStep = {
  extracted: string | null          // 本步 EXTRACT-MIN 的节点
  inMST: string[]                   // 已在 MST 中的节点
  mstEdges: { u: string; v: string; w: number }[]  // MST 边
  key: Record<string, number>       // 各节点 key 值
  parent: Record<string, string | null>             // 前驱节点
  queue: { node: string; key: number }[]            // 优先队列（排序后）
  candidateEdges: { u: string; v: string }[]        // 本步更新的候选边
  description: string
  phase: 'init' | 'extract' | 'done'
}

// 预计算所有步骤
const STEPS: PrimStep[] = (() => {
  const steps: PrimStep[] = []
  const inMST: string[] = []
  const key: Record<string, number> = {}
  const parent: Record<string, string | null> = {}
  const mstEdges: { u: string; v: string; w: number }[] = []

  NODES.forEach(n => { key[n] = INF; parent[n] = null })
  key['A'] = 0

  // 邻接表
  const adj: Record<string, { v: string; w: number }[]> = {}
  NODES.forEach(n => { adj[n] = [] })
  ALL_EDGES.forEach(e => { adj[e.u].push({ v: e.v, w: e.w }); adj[e.v].push({ v: e.u, w: e.w }) })
  // Fix: undirected edges
  ALL_EDGES.forEach(({ u, v, w }) => { adj[u].push({ v, w }); adj[v].push({ v: u, w }) })
  // deduplicate
  NODES.forEach(n => {
    const seen = new Set()
    adj[n] = adj[n].filter(e => { if (seen.has(e.v)) return false; seen.add(e.v); return true })
  })

  const getQueue = () => NODES
    .filter(n => !inMST.includes(n))
    .map(n => ({ node: n, key: key[n] }))
    .sort((a, b) => a.key - b.key)

  // Initial
  steps.push({
    extracted: null,
    inMST: [],
    mstEdges: [],
    key: { ...key },
    parent: { ...parent },
    queue: getQueue(),
    candidateEdges: [],
    description: '初始化：起点 A 的 key=0，其余节点 key=∞。\n将所有节点加入优先队列 Q。',
    phase: 'init',
  })

  while (inMST.length < NODES.length) {
    const queue = getQueue()
    if (queue.length === 0) break
    const { node: u, key: ku } = queue[0]
    if (ku === INF) break

    // Add to MST
    inMST.push(u)
    if (parent[u] !== null) {
      mstEdges.push({ u: parent[u]!, v: u, w: key[u] })
    }

    const candidateEdges: { u: string; v: string }[] = []
    const prevKey = { ...key }

    // Update neighbors
    adj[u].forEach(({ v, w }) => {
      if (!inMST.includes(v) && w < key[v]) {
        key[v] = w
        parent[v] = u
        candidateEdges.push({ u, v })
      }
    })

    steps.push({
      extracted: u,
      inMST: [...inMST],
      mstEdges: [...mstEdges],
      key: { ...key },
      parent: { ...parent },
      queue: getQueue(),
      candidateEdges,
      description: `EXTRACT-MIN: ${u} (key=${ku === INF ? '∞' : ku})。${u} 已加入 MST！\n` +
        (parent[u] ? `加入 MST 边：${parent[u]}-${u} (权=${ku})\n` : '') +
        (candidateEdges.length > 0
          ? `更新邻居：` + adj[u].filter(e => !inMST.includes(e.v)).map(e => `key[${e.v}]=min(${prevKey[e.v] === INF ? '∞' : prevKey[e.v]},${e.w})=${key[e.v] === INF ? '∞' : key[e.v]}`).join('，')
          : '无邻居需要更新。'),
      phase: inMST.length === NODES.length ? 'done' : 'extract',
    })
  }

  // Final done step
  if (inMST.length === NODES.length) {
    steps.push({
      ...steps[steps.length - 1],
      extracted: null,
      candidateEdges: [],
      queue: [],
      description: `🎉 MST 构建完成！共 ${mstEdges.length} 条边。\n总权值 = ${mstEdges.reduce((s, e) => s + e.w, 0)}`,
      phase: 'done',
    })
  }

  return steps
})()

// ── 主组件 ────────────────────────────────────────────────────────
export default function PrimMSTGrowth() {
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

  const fmtKey = (v: number) => v === INF ? '∞' : v.toString()

  // 获取边状态
  const getEdgeState = (u: string, v: string) => {
    // MST edge
    const inMST = cur.mstEdges.some(e =>
      (e.u === u && e.v === v) || (e.u === v && e.v === u))
    if (inMST) return 'mst'
    // Candidate edge (being updated this step)
    const isCandidate = cur.candidateEdges.some(e =>
      (e.u === u && e.v === v) || (e.u === v && e.v === u))
    if (isCandidate) return 'candidate'
    // Cut edge (connecting MST to non-MST)
    const uInMST = cur.inMST.includes(u)
    const vInMST = cur.inMST.includes(v)
    if (uInMST !== vInMST) return 'cut'
    return 'normal'
  }

  const edgeColorMap = {
    mst: '#10b981',
    candidate: '#f59e0b',
    cut: '#c084fc',
    normal: '#cbd5e1',
  }

  return (
    <div className="w-full max-w-4xl mx-auto my-6 rounded-2xl overflow-hidden border border-slate-200 dark:border-slate-700 shadow-xl">
      {/* Header */}
      <div className="bg-gradient-to-r from-violet-600 via-purple-600 to-fuchsia-600 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-xl font-bold text-white">Prim MST 生长动画</h3>
            <p className="text-violet-100 text-sm mt-0.5">从节点 A 出发 · 切割边高亮 · MST 逐步生长</p>
          </div>
          <div className="flex items-center gap-2 bg-white/20 rounded-lg px-3 py-1.5">
            <Sprout className="w-4 h-4 text-violet-200" />
            <span className="text-white text-sm font-medium">步骤 {step + 1}/{STEPS.length}</span>
          </div>
        </div>
      </div>

      {/* Progress */}
      <div className="h-1.5 bg-slate-200 dark:bg-slate-700">
        <div className="h-full bg-gradient-to-r from-violet-400 to-fuchsia-400 transition-all duration-500"
          style={{ width: `${(step / (STEPS.length - 1)) * 100}%` }} />
      </div>

      <div className="bg-white dark:bg-slate-900 p-4">
        <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
          {/* SVG Graph */}
          <div className="md:col-span-3 bg-slate-50 dark:bg-slate-800 rounded-xl p-3 border border-slate-200 dark:border-slate-700">
            <div className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400 mb-2">MST 生长可视化</div>
            <svg viewBox="0 0 460 280" className="w-full">
              {/* Edges */}
              {ALL_EDGES.map(e => {
                const state = getEdgeState(e.u, e.v)
                const color = edgeColorMap[state]
                const f = NODE_POS[e.u], t = NODE_POS[e.v]
                const midX = (f.cx + t.cx) / 2
                const midY = (f.cy + t.cy) / 2
                const opacity = state === 'normal' ? 0.3 : 1
                return (
                  <g key={`${e.u}${e.v}`} opacity={opacity}>
                    <line x1={f.cx} y1={f.cy} x2={t.cx} y2={t.cy}
                      stroke={color}
                      strokeWidth={state === 'mst' ? 4 : state === 'candidate' ? 3 : state === 'cut' ? 2 : 1.5}
                      strokeDasharray={state === 'cut' ? '6,3' : 'none'}
                      className="transition-all duration-400" />
                    {state !== 'normal' && (
                      <>
                        <rect x={midX - 10} y={midY - 9} width={20} height={18} rx={3}
                          fill={color} opacity={0.9} />
                        <text x={midX} y={midY + 1} textAnchor="middle" dominantBaseline="middle"
                          fontSize="11" fontWeight="800" fill="white"
                          className="select-none">{e.w}</text>
                      </>
                    )}
                    {state === 'normal' && (
                      <text x={midX} y={midY - 5} textAnchor="middle"
                        fontSize="10" fill="#94a3b8" className="select-none">{e.w}</text>
                    )}
                  </g>
                )
              })}
              {/* Nodes */}
              {NODES.map(n => {
                const { cx, cy } = NODE_POS[n]
                const inMST = cur.inMST.includes(n)
                const isExtracted = cur.extracted === n
                const k = cur.key[n]
                let fill = 'white', stroke = '#94a3b8', textColor = '#334155'
                if (isExtracted) { fill = '#f97316'; stroke = '#ea580c'; textColor = 'white' }
                else if (inMST) { fill = '#10b981'; stroke = '#059669'; textColor = 'white' }
                else if (cur.candidateEdges.some(e => e.v === n)) { fill = '#c4b5fd'; stroke = '#7c3aed'; textColor = '#3b0764' }
                return (
                  <g key={n}>
                    <circle cx={cx} cy={cy} r={22}
                      fill={fill} stroke={stroke} strokeWidth={inMST || isExtracted ? 3 : 2}
                      className="transition-all duration-400" />
                    <text x={cx} y={cx === 60 ? cy - 2 : cy - 3} textAnchor="middle" dominantBaseline="middle"
                      fontSize="14" fontWeight="800" fill={textColor}
                      className="select-none">{n}</text>
                    <text x={cx} y={cy + 11} textAnchor="middle" dominantBaseline="middle"
                      fontSize="9" fill={inMST ? '#d1fae5' : '#94a3b8'}
                      className="select-none">{fmtKey(k)}</text>
                  </g>
                )
              })}
            </svg>
            {/* Legend */}
            <div className="flex flex-wrap gap-3 px-1 mt-1">
              {[
                { color: 'bg-emerald-500', label: 'MST 已加入' },
                { color: 'bg-orange-500', label: '当前 EXTRACT' },
                { color: 'bg-violet-400', label: '被更新 key' },
                { color: 'bg-purple-400', label: '切割边（候选）', border: 'dashed' },
              ].map(l => (
                <div key={l.label} className="flex items-center gap-1 text-xs text-slate-600 dark:text-slate-400">
                  <span className={`w-3 h-3 rounded ${l.color}`} />
                  {l.label}
                </div>
              ))}
            </div>
          </div>

          {/* Right: PQ + key table */}
          <div className="md:col-span-2 flex flex-col gap-3">
            {/* Priority Queue */}
            <div className="bg-slate-50 dark:bg-slate-800 rounded-xl p-3 border border-slate-200 dark:border-slate-700">
              <div className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400 mb-2">优先队列 Q（按 key 排序）</div>
              {cur.queue.length === 0 ? (
                <div className="text-center py-2 text-slate-400 text-sm">队列已空 ✓</div>
              ) : (
                <div className="flex flex-col gap-1.5">
                  {cur.queue.map((item, idx) => (
                    <div key={item.node}
                      className={`flex items-center justify-between px-3 py-1.5 rounded-lg border text-sm font-semibold
                        ${idx === 0
                          ? 'bg-orange-100 dark:bg-orange-900/40 border-orange-300 dark:border-orange-700 text-orange-800 dark:text-orange-200'
                          : 'bg-white dark:bg-slate-700 border-slate-200 dark:border-slate-600 text-slate-600 dark:text-slate-300'}`}>
                      <span>{idx === 0 && <span className="text-xs mr-1 text-orange-500">MIN</span>}{item.node}</span>
                      <span className="text-xs opacity-80">key={fmtKey(item.key)}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Key table */}
            <div className="bg-slate-50 dark:bg-slate-800 rounded-xl p-3 border border-slate-200 dark:border-slate-700">
              <div className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400 mb-2">key[] / parent[]</div>
              <table className="w-full text-xs text-center">
                <thead>
                  <tr className="bg-slate-100 dark:bg-slate-700">
                    <th className="py-1 px-1 rounded-l text-slate-600 dark:text-slate-400">节点</th>
                    <th className="py-1 px-1 text-slate-600 dark:text-slate-400">key</th>
                    <th className="py-1 px-1 rounded-r text-slate-600 dark:text-slate-400">parent</th>
                  </tr>
                </thead>
                <tbody>
                  {NODES.map(n => {
                    const inMST = cur.inMST.includes(n)
                    return (
                      <tr key={n} className={`${inMST ? 'text-emerald-600 dark:text-emerald-400 font-bold' : 'text-slate-600 dark:text-slate-400'}`}>
                        <td className="py-1 px-1 font-bold">{n}</td>
                        <td className="py-1 px-1 font-mono">{fmtKey(cur.key[n])}</td>
                        <td className="py-1 px-1">{cur.parent[n] ?? '—'}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>

            {/* MST progress */}
            <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-xl p-3 border border-emerald-200 dark:border-emerald-700">
              <div className="text-xs font-semibold text-emerald-700 dark:text-emerald-300 mb-1.5">MST 边集</div>
              {cur.mstEdges.length === 0 ? (
                <div className="text-xs text-slate-400 dark:text-slate-500">尚无边</div>
              ) : (
                <div className="flex flex-col gap-1">
                  {cur.mstEdges.map(e => (
                    <div key={`${e.u}${e.v}`} className="text-xs text-emerald-700 dark:text-emerald-300 font-mono">
                      {e.u}–{e.v}: {e.w}
                    </div>
                  ))}
                  <div className="text-xs font-bold text-emerald-800 dark:text-emerald-200 border-t border-emerald-200 dark:border-emerald-700 pt-1 mt-1">
                    总权值：{cur.mstEdges.reduce((s, e) => s + e.w, 0)}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Description */}
        <div className={`mt-3 rounded-xl px-4 py-3 border text-sm leading-5
          ${cur.phase === 'done' ? 'bg-violet-50 dark:bg-violet-900/20 border-violet-200 dark:border-violet-700 text-violet-800 dark:text-violet-200' :
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
              className="flex items-center gap-1.5 px-4 py-1.5 rounded-lg bg-violet-600 hover:bg-violet-700 text-white text-sm font-semibold transition-colors disabled:opacity-40">
              {playing ? <><Pause className="w-3.5 h-3.5" /> 暂停</> : <><Play className="w-3.5 h-3.5" /> 自动播放</>}
            </button>
            <button onClick={() => { setPlaying(false); if (!isLast) setStep(s => s + 1) }} disabled={isLast}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-purple-100 dark:bg-purple-900/40 text-purple-700 dark:text-purple-300 hover:bg-purple-200 dark:hover:bg-purple-900/60 text-sm transition-colors disabled:opacity-40">
              <SkipForward className="w-3.5 h-3.5" /> 下一步
            </button>
          </div>
          <div className="flex items-center gap-2 text-sm text-slate-500 dark:text-slate-400">
            速度
            {[2000, 1500, 800].map(s => (
              <button key={s} onClick={() => setSpeed(s)}
                className={`px-2 py-1 rounded text-xs ${speed === s ? 'bg-violet-600 text-white' : 'bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-300'}`}>
                {s === 2000 ? '慢' : s === 1500 ? '中' : '快'}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
