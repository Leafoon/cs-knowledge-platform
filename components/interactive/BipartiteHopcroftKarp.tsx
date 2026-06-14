'use client'
import React, { useState, useEffect, useRef } from 'react'

// ── Hopcroft-Karp 二分图最大匹配演示 ─────────────────────────────
// 左: W0-W3（工人），右: J0-J3（任务）
// 边: W0-J0, W0-J1, W1-J1, W1-J2, W2-J2, W3-J2, W3-J3
// 最大匹配 = 4（W0→J0, W1→J1, W2→J2, W3→J3）
// BFS + DFS 两阶段：一轮 BFS 即找出所有 4 条增广路

const LW = 130 // left x
const RW = 320 // right x
const LEFT_NODES = [
  { id: 0, label: 'W₀', x: LW, y: 52  },
  { id: 1, label: 'W₁', x: LW, y: 112 },
  { id: 2, label: 'W₂', x: LW, y: 172 },
  { id: 3, label: 'W₃', x: LW, y: 232 },
]
const RIGHT_NODES = [
  { id: 0, label: 'J₀', x: RW, y: 52  },
  { id: 1, label: 'J₁', x: RW, y: 112 },
  { id: 2, label: 'J₂', x: RW, y: 172 },
  { id: 3, label: 'J₃', x: RW, y: 232 },
]

// edge: [left_id, right_id]
const ALL_EDGES = [
  [0,0],[0,1],[1,1],[1,2],[2,2],[3,2],[3,3]
]

// The final matching: [left_id, right_id]
const MATCHING = [[0,0],[1,1],[2,2],[3,3]]

interface HKStep {
  phase: 'init'|'bfs'|'dfs'|'done'
  bfsLayers: { left: number[], right: number[] }
  activeEdges: [number,number][]   // currently highlighted edges (l,r)
  matchedEdges: [number,number][]  // confirmed matching so far
  augPaths: [number,number][][]    // multiple augmenting paths shown this phase
  activeNodes: { left: number[], right: number[] }
  matchCount: number
  label: string
  desc: string
}

const STEPS: HKStep[] = [
  {
    phase: 'init', bfsLayers: { left: [], right: [] },
    activeEdges: [], matchedEdges: [], augPaths: [],
    activeNodes: { left: [], right: [] },
    matchCount: 0,
    label: '初始状态：所有节点未匹配',
    desc: 'Hopcroft-Karp 是 BFS + DFS 双阶段算法。BFS 阶段从所有未匹配左节点出发，构建分层图，寻找到未匹配右节点的最短增广路集合。DFS 阶段在分层图上同时找出多条顶点不相交的增广路，一次迭代提升匹配数最多。',
  },
  {
    phase: 'bfs', bfsLayers: { left: [0,1,2,3], right: [] },
    activeEdges: [], matchedEdges: [], augPaths: [],
    activeNodes: { left: [0,1,2,3], right: [] },
    matchCount: 0,
    label: 'BFS 第 1 轮：从所有未匹配左节点出发',
    desc: 'BFS 初始队列 = 所有未匹配左节点 {W₀, W₁, W₂, W₃}（层 0）。沿非匹配边向右邻居扩展。',
  },
  {
    phase: 'bfs', bfsLayers: { left: [0,1,2,3], right: [0,1,2,3] },
    activeEdges: ALL_EDGES as [number,number][],
    matchedEdges: [], augPaths: [],
    activeNodes: { left: [0,1,2,3], right: [0,1,2,3] },
    matchCount: 0,
    label: 'BFS：发现右侧节点（层 1），全部空闲！',
    desc: '所有右侧节点 {J₀, J₁, J₂, J₃} 均未匹配（层 1）。BFS 立即发现增广路长度 = 1（直连）的最短增广路集合，停止 BFS 扩展。分层图构建完成，进入 DFS 阶段。',
  },
  {
    phase: 'dfs', bfsLayers: { left: [0,1,2,3], right: [0,1,2,3] },
    activeEdges: [[0,0]] as [number,number][],
    matchedEdges: [], augPaths: [[[0,0]]],
    activeNodes: { left: [0], right: [0] },
    matchCount: 0,
    label: 'DFS：增广路 W₀ → J₀',
    desc: 'DFS 从 W₀ 出发，找到空闲右节点 J₀，得到增广路 W₀ → J₀（长度 1）。将该路径取反（匹配），W₀↔J₀ 建立匹配。',
  },
  {
    phase: 'dfs', bfsLayers: { left: [0,1,2,3], right: [0,1,2,3] },
    activeEdges: [[1,1]] as [number,number][],
    matchedEdges: [[0,0]],
    augPaths: [[[0,0]],[[1,1]]],
    activeNodes: { left: [1], right: [1] },
    matchCount: 1,
    label: 'DFS：增广路 W₁ → J₁',
    desc: 'DFS 从 W₁ 出发，J₀ 已被 W₀ 使用，尝试 J₁（空闲），得到增广路 W₁ → J₁。W₁↔J₁ 建立匹配。',
  },
  {
    phase: 'dfs', bfsLayers: { left: [0,1,2,3], right: [0,1,2,3] },
    activeEdges: [[2,2]] as [number,number][],
    matchedEdges: [[0,0],[1,1]],
    augPaths: [[[0,0]],[[1,1]],[[2,2]]],
    activeNodes: { left: [2], right: [2] },
    matchCount: 2,
    label: 'DFS：增广路 W₂ → J₂',
    desc: 'DFS 从 W₂ 出发，唯一邻居 J₂ 空闲，得到增广路 W₂ → J₂。W₂↔J₂ 建立匹配。',
  },
  {
    phase: 'dfs', bfsLayers: { left: [0,1,2,3], right: [0,1,2,3] },
    activeEdges: [[3,3]] as [number,number][],
    matchedEdges: [[0,0],[1,1],[2,2]],
    augPaths: [[[0,0]],[[1,1]],[[2,2]],[[3,3]]],
    activeNodes: { left: [3], right: [3] },
    matchCount: 3,
    label: 'DFS：增广路 W₃ → J₃',
    desc: 'DFS 从 W₃ 出发，J₂ 已被 W₂ 使用，尝试 J₃（空闲），得到增广路 W₃ → J₃。W₃↔J₃ 建立匹配。本轮共找到 4 条增广路。',
  },
  {
    phase: 'done', bfsLayers: { left: [], right: [] },
    activeEdges: [],
    matchedEdges: [[0,0],[1,1],[2,2],[3,3]],
    augPaths: [],
    activeNodes: { left: [], right: [] },
    matchCount: 4,
    label: '第 2 轮 BFS：无增广路 → 算法终止',
    desc: '第 2 轮 BFS 从未匹配左节点出发——所有左节点均已匹配，队列为空，无增广路存在。算法终止。\n最大匹配 = 4（与 König 定理结合，最小顶点覆盖大小也为 4）。\n时间复杂度：O(√V · E)。',
  },
]

function edgeKey(l: number, r: number) { return `${l}-${r}` }

export default function BipartiteHopcroftKarp() {
  const [cur, setCur] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1800)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    if (playing) {
      timerRef.current = setInterval(() => {
        setCur(c => {
          if (c >= STEPS.length - 1) { setPlaying(false); return c }
          return c + 1
        })
      }, speed)
    } else {
      if (timerRef.current) clearInterval(timerRef.current)
    }
    return () => { if (timerRef.current) clearInterval(timerRef.current) }
  }, [playing, speed])

  const step = STEPS[cur]
  const R = 18

  const matchedSet = new Set(step.matchedEdges.map(([l,r]) => edgeKey(l,r)))
  const activeSet = new Set(step.activeEdges.map(([l,r]) => edgeKey(l,r)))

  function edgeColor(l: number, r: number) {
    const k = edgeKey(l,r)
    if (matchedSet.has(k)) return '#10b981'
    if (activeSet.has(k)) return '#f59e0b'
    if (step.bfsLayers.left.includes(l) && step.bfsLayers.right.includes(r)) return '#8b5cf6'
    return '#cbd5e1'
  }
  function edgeWidth(l: number, r: number) {
    const k = edgeKey(l,r)
    if (matchedSet.has(k)) return 3.5
    if (activeSet.has(k)) return 3.5
    return 1.5
  }
  function leftNodeFill(id: number) {
    if (step.matchedEdges.some(([l]) => l===id)) return '#10b981'
    if (step.activeNodes.left.includes(id)) return '#f59e0b'
    if (step.bfsLayers.left.includes(id)) return '#8b5cf6'
    return '#64748b'
  }
  function rightNodeFill(id: number) {
    if (step.matchedEdges.some(([,r]) => r===id)) return '#10b981'
    if (step.activeNodes.right.includes(id)) return '#f59e0b'
    if (step.bfsLayers.right.includes(id)) return '#a78bfa'
    return '#64748b'
  }

  return (
    <div className="w-full max-w-2xl mx-auto rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 shadow-xl overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-violet-600 via-purple-600 to-indigo-600 px-5 py-4">
        <h3 className="text-white font-bold text-lg tracking-tight">Hopcroft-Karp 最大二分匹配演示</h3>
        <p className="text-purple-200 text-sm mt-0.5">BFS 分层图 + DFS 并行增广 · O(√V · E)</p>
      </div>

      {/* Phase indicator + controls */}
      <div className="px-4 py-3 flex flex-wrap gap-2 items-center border-b border-slate-100 dark:border-slate-800 bg-slate-50 dark:bg-slate-800/40">
        <div className="flex rounded-lg overflow-hidden text-[10px] font-bold border border-slate-200 dark:border-slate-600">
          <div className={`px-3 py-1.5 ${step.phase==='init'?'bg-slate-500 text-white':'text-slate-400 bg-slate-100 dark:bg-slate-700'}`}>初始</div>
          <div className={`px-3 py-1.5 ${step.phase==='bfs'?'bg-violet-600 text-white':'text-slate-400 bg-slate-100 dark:bg-slate-700'}`}>BFS 分层</div>
          <div className={`px-3 py-1.5 ${step.phase==='dfs'?'bg-purple-600 text-white':'text-slate-400 bg-slate-100 dark:bg-slate-700'}`}>DFS 增广</div>
          <div className={`px-3 py-1.5 ${step.phase==='done'?'bg-emerald-600 text-white':'text-slate-400 bg-slate-100 dark:bg-slate-700'}`}>完成</div>
        </div>
        <div className="flex gap-1.5 ml-auto">
          <button onClick={() => { setPlaying(false); setCur(0) }} className="px-2 py-1.5 rounded-lg text-xs bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-300">⏮</button>
          <button onClick={() => { setPlaying(false); setCur(c => Math.max(0,c-1)) }} disabled={cur===0} className="px-2 py-1.5 rounded-lg text-xs bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-300 disabled:opacity-40">◀</button>
          <button onClick={() => setPlaying(p => !p)} disabled={cur>=STEPS.length-1} className="px-4 py-1.5 rounded-lg text-xs font-bold bg-violet-600 text-white hover:bg-violet-700 disabled:opacity-40">
            {playing?'⏸':'▶'} {playing?'暂停':'播放'}
          </button>
          <button onClick={() => { setPlaying(false); setCur(c => Math.min(STEPS.length-1,c+1)) }} disabled={cur>=STEPS.length-1} className="px-2 py-1.5 rounded-lg text-xs bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-300 disabled:opacity-40">▶</button>
        </div>
        <div className="flex items-center gap-1.5 text-[10px] text-slate-400">
          <input type="range" min={600} max={3000} step={400} value={speed} onChange={e=>setSpeed(Number(e.target.value))} className="w-14 accent-violet-500"/>
          <span>{(speed/1000).toFixed(1)}s</span>
        </div>
      </div>

      <div className="p-4 space-y-3">
        {/* Step header */}
        <div className="flex items-center gap-2">
          <span className="px-2 py-0.5 rounded-full text-[10px] font-bold text-white bg-purple-500">{cur+1}/{STEPS.length}</span>
          <span className="font-semibold text-sm text-slate-700 dark:text-slate-200">{step.label}</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-5 gap-3">
          {/* SVG bipartite graph */}
          <div className="md:col-span-3 rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/30 overflow-hidden">
            <svg viewBox="0 70 450 210" className="w-full" style={{ maxHeight: 210 }}>
              {/* Group labels */}
              <text x={LW} y={32} textAnchor="middle" fontSize={10} fontWeight="bold" fill="#8b5cf6">工人（Left）</text>
              <text x={RW} y={32} textAnchor="middle" fontSize={10} fontWeight="bold" fill="#a78bfa">任务（Right）</text>

              {/* Edges */}
              {ALL_EDGES.map(([l,r]) => {
                const ln = LEFT_NODES[l], rn = RIGHT_NODES[r]
                const isMatch = matchedSet.has(edgeKey(l,r))
                const isActive = activeSet.has(edgeKey(l,r))
                return (
                  <g key={`${l}-${r}`}>
                    {(isMatch || isActive) && (
                      <line x1={ln.x} y1={ln.y} x2={rn.x} y2={rn.y}
                        stroke={isMatch?'#10b981':'#f59e0b'} strokeWidth={10} opacity={0.18}/>
                    )}
                    <line x1={ln.x} y1={ln.y} x2={rn.x} y2={rn.y}
                      stroke={edgeColor(l,r)} strokeWidth={edgeWidth(l,r)} strokeLinecap="round"/>
                  </g>
                )
              })}

              {/* Left nodes */}
              {LEFT_NODES.map(n => {
                const isMatched = step.matchedEdges.some(([l])=>l===n.id)
                const isActive = step.activeNodes.left.includes(n.id)
                return (
                  <g key={`L${n.id}`}>
                    {isActive && <circle cx={n.x} cy={n.y} r={R+4} fill="none" stroke="#f59e0b" strokeWidth={2} strokeDasharray="5 2"/>}
                    <circle cx={n.x} cy={n.y} r={R} fill={leftNodeFill(n.id)}/>
                    <text x={n.x} y={n.y+5} textAnchor="middle" fontSize={12} fontWeight="bold" fill="white">{n.label}</text>
                    {isMatched && (
                      <text x={n.x-30} y={n.y+4} textAnchor="middle" fontSize={9} fill="#10b981">✓</text>
                    )}
                  </g>
                )
              })}

              {/* Right nodes */}
              {RIGHT_NODES.map(n => {
                const isMatched = step.matchedEdges.some(([,r])=>r===n.id)
                const isActive = step.activeNodes.right.includes(n.id)
                return (
                  <g key={`R${n.id}`}>
                    {isActive && <circle cx={n.x} cy={n.y} r={R+4} fill="none" stroke="#f59e0b" strokeWidth={2} strokeDasharray="5 2"/>}
                    <circle cx={n.x} cy={n.y} r={R} fill={rightNodeFill(n.id)}/>
                    <text x={n.x} y={n.y+5} textAnchor="middle" fontSize={12} fontWeight="bold" fill="white">{n.label}</text>
                    {isMatched && (
                      <text x={n.x+30} y={n.y+4} textAnchor="middle" fontSize={9} fill="#10b981">✓</text>
                    )}
                  </g>
                )
              })}
            </svg>
          </div>

          {/* Right panel */}
          <div className="md:col-span-2 space-y-2 text-[11px]">
            {/* Matching count */}
            <div className="rounded-xl bg-gradient-to-br from-violet-50 to-purple-50 dark:from-violet-900/20 dark:to-purple-900/20 border border-violet-200 dark:border-violet-700/50 p-3 text-center">
              <div className="text-[10px] font-bold text-violet-600 dark:text-violet-400 uppercase tracking-wider mb-1">当前匹配数</div>
              <div className="text-4xl font-black text-violet-600 dark:text-violet-300">{step.matchCount}</div>
              <div className="text-[9px] text-slate-400 mt-0.5">/ 4（最大匹配）</div>
              <div className="mt-1.5 h-2 rounded-full bg-violet-100 dark:bg-violet-900/40 overflow-hidden">
                <div className="h-full rounded-full bg-gradient-to-r from-violet-500 to-emerald-500 transition-all duration-500"
                  style={{ width: `${step.matchCount*25}%` }}/>
              </div>
            </div>

            {/* Current matching */}
            <div className="rounded-xl border border-emerald-200 dark:border-emerald-700/50 bg-emerald-50 dark:bg-emerald-900/20 p-2.5">
              <div className="text-[10px] font-bold text-emerald-600 dark:text-emerald-400 uppercase tracking-wider mb-1.5">已确认匹配</div>
              {step.matchedEdges.length === 0
                ? <span className="text-[10px] text-slate-400 italic">空</span>
                : <div className="space-y-1">
                    {step.matchedEdges.map(([l,r]) => (
                      <div key={`${l}-${r}`} className="flex items-center gap-1.5">
                        <span className="px-1.5 py-0.5 rounded text-[10px] font-bold bg-emerald-200 dark:bg-emerald-800 text-emerald-800 dark:text-emerald-100">
                          {LEFT_NODES[l].label} ↔ {RIGHT_NODES[r].label}
                        </span>
                      </div>
                    ))}
                  </div>
              }
            </div>

            {/* BFS layers */}
            <div className="rounded-xl border border-purple-200 dark:border-purple-700/50 bg-purple-50 dark:bg-purple-900/20 p-2.5">
              <div className="text-[10px] font-bold text-purple-600 dark:text-purple-400 uppercase tracking-wider mb-1.5">BFS 分层</div>
              <div className="space-y-1">
                <div className="flex items-center gap-1 text-[10px]">
                  <span className="text-slate-500 w-10">Layer 0</span>
                  <div className="flex gap-0.5">
                    {step.bfsLayers.left.map(id => (
                      <span key={id} className="px-1.5 py-0.5 rounded bg-violet-200 dark:bg-violet-800 text-violet-800 dark:text-violet-100 font-bold">{LEFT_NODES[id].label}</span>
                    ))}
                    {step.bfsLayers.left.length===0 && <span className="text-slate-400 italic text-[9px]">—</span>}
                  </div>
                </div>
                <div className="flex items-center gap-1 text-[10px]">
                  <span className="text-slate-500 w-10">Layer 1</span>
                  <div className="flex gap-0.5">
                    {step.bfsLayers.right.map(id => (
                      <span key={id} className="px-1.5 py-0.5 rounded bg-purple-200 dark:bg-purple-800 text-purple-800 dark:text-purple-100 font-bold">{RIGHT_NODES[id].label}</span>
                    ))}
                    {step.bfsLayers.right.length===0 && <span className="text-slate-400 italic text-[9px]">—</span>}
                  </div>
                </div>
              </div>
            </div>

            {/* Algorithm info */}
            <div className="rounded-lg bg-slate-50 dark:bg-slate-800/30 border border-slate-200 dark:border-slate-700 p-2">
              <div className="text-[9px] text-slate-400 uppercase mb-1">复杂度</div>
              <div className="text-[10px] font-bold text-slate-600 dark:text-slate-300">O(√V · E)</div>
              <div className="text-[9px] text-slate-400 mt-0.5">vs 匈牙利算法 O(V · E)</div>
            </div>
          </div>
        </div>

        {/* Color legend */}
        <div className="flex flex-wrap gap-3 text-[10px] text-slate-500 dark:text-slate-400">
          <span className="flex items-center gap-1"><span className="w-5 h-0.5 bg-slate-300 inline-block"/>普通边</span>
          <span className="flex items-center gap-1"><span className="w-5 h-0.5 bg-purple-500 inline-block"/>BFS 层内边</span>
          <span className="flex items-center gap-1"><span className="w-5 h-0.5 bg-amber-400 inline-block"/>增广路径</span>
          <span className="flex items-center gap-1"><span className="w-5 h-0.5 bg-emerald-500 inline-block"/>匹配边</span>
        </div>

        {/* Description */}
        <div className="rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/30 px-3 py-2 text-[11px] text-slate-600 dark:text-slate-300 whitespace-pre-line">
          {step.desc}
        </div>

        {/* Progress */}
        <div className="h-1.5 rounded-full bg-slate-200 dark:bg-slate-700 overflow-hidden">
          <div className="h-full rounded-full bg-gradient-to-r from-violet-500 to-indigo-500 transition-all duration-400"
            style={{ width: `${(cur/(STEPS.length-1))*100}%` }}/>
        </div>
      </div>
    </div>
  )
}
