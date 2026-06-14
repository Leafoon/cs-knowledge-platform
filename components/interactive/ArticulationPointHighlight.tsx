'use client'
import React, { useState, useEffect, useRef } from 'react'

// ── 无向图：桥与割点 Tarjan 算法演示 ─────────────────────────────
// 节点 0-4，边: 0-1(e0), 1-2(e1), 2-0(e2), 2-3(e3), 3-4(e4)
// DFS 结果: disc=[0,1,2,3,4], low=[0,0,0,3,4]
// 桥: (2,3), (3,4)   割点: {2, 3}

const NODES = [
  { id: 0, x: 80,  y: 110, label: '0' },
  { id: 1, x: 185, y: 50,  label: '1' },
  { id: 2, x: 265, y: 110, label: '2' },
  { id: 3, x: 360, y: 65,  label: '3' },
  { id: 4, x: 420, y: 155, label: '4' },
]

const EDGES = [
  { id: 0, u: 0, v: 1 },
  { id: 1, u: 1, v: 2 },
  { id: 2, u: 2, v: 0 },
  { id: 3, u: 2, v: 3 },
  { id: 4, u: 3, v: 4 },
]

interface TarjanStep {
  visitedNodes: number[]
  activeNode: number | null
  activeEdge: number | null
  parentEdge: number | null
  disc: (number|null)[]
  low:  (number|null)[]
  treeEdges: number[]
  backEdgeActive: number | null
  bridges: number[]
  artPoints: number[]
  dfsPath: number[]        // current DFS traversal stack
  label: string
  desc: string
  updatedLow?: number
}

const STEPS: TarjanStep[] = [
  {
    visitedNodes: [], activeNode: 0, activeEdge: null, parentEdge: null,
    disc: [null,null,null,null,null], low: [null,null,null,null,null],
    treeEdges: [], backEdgeActive: null, bridges: [], artPoints: [], dfsPath: [],
    label: '初始：DFS 从节点 0 开始',
    desc: 'Tarjan 算法从任意节点出发做 DFS。使用 timer 分配 disc（发现时间）和 low（可到达的最小 disc）值。初始所有节点未访问。',
  },
  {
    visitedNodes: [0], activeNode: 0, activeEdge: null, parentEdge: null,
    disc: [0,null,null,null,null], low: [0,null,null,null,null],
    treeEdges: [], backEdgeActive: null, bridges: [], artPoints: [], dfsPath: [0],
    label: '访问节点 0：disc[0]=0, low[0]=0',
    desc: 'timer=0 分配给 0，disc[0]=low[0]=0。DFS 栈=[0]，继续探索邻居 1。',
  },
  {
    visitedNodes: [0,1], activeNode: 1, activeEdge: 0, parentEdge: 0,
    disc: [0,1,null,null,null], low: [0,1,null,null,null],
    treeEdges: [0], backEdgeActive: null, bridges: [], artPoints: [], dfsPath: [0,1],
    label: '树边 0→1：disc[1]=1, low[1]=1',
    desc: '沿树边 e0(0-1) 到达节点 1，timer=1。disc[1]=low[1]=1。DFS 栈=[0→1]，继续探索邻居 2。',
  },
  {
    visitedNodes: [0,1,2], activeNode: 2, activeEdge: 1, parentEdge: 1,
    disc: [0,1,2,null,null], low: [0,1,2,null,null],
    treeEdges: [0,1], backEdgeActive: null, bridges: [], artPoints: [], dfsPath: [0,1,2],
    label: '树边 1→2：disc[2]=2, low[2]=2',
    desc: '沿树边 e1(1-2) 到达节点 2，timer=2。disc[2]=low[2]=2。DFS 栈=[0→1→2]。',
  },
  {
    visitedNodes: [0,1,2], activeNode: 2, activeEdge: 2, parentEdge: 1,
    disc: [0,1,2,null,null], low: [0,1,0,null,null],
    treeEdges: [0,1], backEdgeActive: 2, bridges: [], artPoints: [], dfsPath: [0,1,2],
    label: '回边 2→0：low[2] = min(2, disc[0]) = 0',
    desc: '节点 2 的邻居 0 已访问，且 e2 不是父边（父边为 e1）。发现回边！low[2] = min(low[2], disc[0]) = min(2,0) = 0。这说明 2 可通过回边到达祖先 0。',
    updatedLow: 0,
  },
  {
    visitedNodes: [0,1,2,3], activeNode: 3, activeEdge: 3, parentEdge: 3,
    disc: [0,1,2,3,null], low: [0,1,0,3,null],
    treeEdges: [0,1,3], backEdgeActive: null, bridges: [], artPoints: [], dfsPath: [0,1,2,3],
    label: '树边 2→3：disc[3]=3, low[3]=3',
    desc: '从 2 继续探索邻居 3（尚未访问），沿树边 e3(2-3) 到达节点 3，timer=3。disc[3]=low[3]=3。',
  },
  {
    visitedNodes: [0,1,2,3,4], activeNode: 4, activeEdge: 4, parentEdge: 4,
    disc: [0,1,2,3,4], low: [0,1,0,3,4],
    treeEdges: [0,1,3,4], backEdgeActive: null, bridges: [], artPoints: [], dfsPath: [0,1,2,3,4],
    label: '树边 3→4：disc[4]=4, low[4]=4',
    desc: '从 3 探索邻居 4，沿树边 e4(3-4) 到达节点 4，timer=4。disc[4]=low[4]=4。节点 4 无更多未访问邻居，开始回溯。',
  },
  {
    visitedNodes: [0,1,2,3,4], activeNode: 3, activeEdge: 4, parentEdge: 3,
    disc: [0,1,2,3,4], low: [0,1,0,3,4],
    treeEdges: [0,1,3,4], backEdgeActive: null, bridges: [4], artPoints: [3], dfsPath: [0,1,2,3],
    label: '回溯至 3：检测到桥 (3,4) 和割点 3',
    desc: 'DFS 从 4 返回 3。更新 low[3]=min(low[3],low[4])=min(3,4)=3。\n检查 e4(3-4)：low[4]=4 > disc[3]=3 → 是桥！\n检查节点 3（非根）：low[4]=4 ≥ disc[3]=3 → 3 是割点！',
  },
  {
    visitedNodes: [0,1,2,3,4], activeNode: 2, activeEdge: 3, parentEdge: 1,
    disc: [0,1,2,3,4], low: [0,1,0,3,4],
    treeEdges: [0,1,3,4], backEdgeActive: null, bridges: [4,3], artPoints: [3,2], dfsPath: [0,1,2],
    label: '回溯至 2：检测到桥 (2,3) 和割点 2',
    desc: 'DFS 从 3 返回 2。更新 low[2]=min(low[2],low[3])=min(0,3)=0（不变）。\n检查 e3(2-3)：low[3]=3 > disc[2]=2 → 是桥！\n检查节点 2（非根）：low[3]=3 ≥ disc[2]=2 → 2 是割点！',
  },
  {
    visitedNodes: [0,1,2,3,4], activeNode: 1, activeEdge: 1, parentEdge: 0,
    disc: [0,1,2,3,4], low: [0,0,0,3,4],
    treeEdges: [0,1,3,4], backEdgeActive: null, bridges: [4,3], artPoints: [3,2], dfsPath: [0,1],
    label: '回溯至 1：low[1] = min(1, low[2]) = 0',
    desc: 'DFS 从 2 返回 1。更新 low[1]=min(1,low[2])=min(1,0)=0。\n检查 e1(1-2)：low[2]=0 ≤ disc[1]=1 → 不是桥。\nlow[2]=0 < disc[1]=1 → 1 不是割点。',
  },
  {
    visitedNodes: [0,1,2,3,4], activeNode: 0, activeEdge: 0, parentEdge: null,
    disc: [0,1,2,3,4], low: [0,0,0,3,4],
    treeEdges: [0,1,3,4], backEdgeActive: null, bridges: [4,3], artPoints: [3,2], dfsPath: [0],
    label: '回溯至 0：DFS 根，仅 1 个子节点 → 非割点',
    desc: 'DFS 从 1 返回 0。更新 low[0]=min(0,low[1])=0。\n节点 0 是 DFS 根，仅有 1 个子节点（节点 1）→ 不是割点。\n邻居 2 已访问（回边 0→2 被 2 检测过），算法完成。',
  },
  {
    visitedNodes: [0,1,2,3,4], activeNode: null, activeEdge: null, parentEdge: null,
    disc: [0,1,2,3,4], low: [0,0,0,3,4],
    treeEdges: [0,1,3,4], backEdgeActive: null, bridges: [3,4], artPoints: [2,3], dfsPath: [],
    label: '算法完成',
    desc: '桥（Bridge）: 边 e3(2-3) 和 e4(3-4)\n割点（Articulation Point）: 节点 2 和节点 3\n注意：边 e2(0-2) 是回边（构成回路 0-1-2-0），使得 e0 和 e1 不是桥。',
  },
]

function midpoint(x1: number, y1: number, x2: number, y2: number, offset=0) {
  return { x: (x1+x2)/2 + offset, y: (y1+y2)/2 + offset }
}

export default function ArticulationPointHighlight() {
  const [cur, setCur] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(2000)
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
  const R = 20

  function edgeStroke(id: number) {
    if (step.bridges.includes(id)) return '#ef4444'
    if (step.activeEdge === id) return '#f59e0b'
    if (step.backEdgeActive === id) return '#8b5cf6'
    if (step.treeEdges.includes(id)) return '#3b82f6'
    return '#94a3b8'
  }
  function edgeWidth(id: number) {
    if (step.bridges.includes(id)) return 4
    if (step.activeEdge === id) return 3.5
    return 2
  }
  function nodeFill(id: number) {
    if (step.artPoints.includes(id)) return '#f97316'
    if (step.activeNode === id) return '#f59e0b'
    if (step.visitedNodes.includes(id)) return '#3b82f6'
    return '#64748b'
  }
  function discLowColor(id: number) {
    if (step.artPoints.includes(id)) return '#ea580c'
    if (step.visitedNodes.includes(id)) return '#3b82f6'
    return '#94a3b8'
  }

  // compute edge label positions from edge midpoint displaced perpendicular
  function edgeLabelPos(e: typeof EDGES[number]) {
    const n1 = NODES[e.u], n2 = NODES[e.v]
    const dx = n2.x-n1.x, dy = n2.y-n1.y, len = Math.hypot(dx,dy)
    return { x: (n1.x+n2.x)/2 - dy/len*14, y: (n1.y+n2.y)/2 + dx/len*14 }
  }

  const isComplete = cur === STEPS.length - 1

  return (
    <div className="w-full max-w-2xl mx-auto rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 shadow-xl overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-amber-500 via-orange-500 to-red-500 px-5 py-4">
        <h3 className="text-white font-bold text-lg tracking-tight">Tarjan 桥与割点算法演示</h3>
        <p className="text-orange-100 text-sm mt-0.5">DFS disc/low 双值动态追踪 · O(V+E)</p>
      </div>

      {/* Controls */}
      <div className="px-4 py-3 flex flex-wrap gap-2 items-center border-b border-slate-100 dark:border-slate-800 bg-slate-50 dark:bg-slate-800/40">
        <div className="flex gap-3 text-[10px] font-bold">
          <span className="flex items-center gap-1">
            <span className="w-5 h-0.5 bg-blue-500 inline-block rounded"/>树边
          </span>
          <span className="flex items-center gap-1">
            <span className="w-5 h-0.5 bg-purple-500 inline-block rounded"/>回边
          </span>
          <span className="flex items-center gap-1">
            <span className="w-5 h-0.5 bg-red-500 inline-block rounded"/>桥
          </span>
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 rounded-full bg-orange-500 inline-block"/>割点
          </span>
        </div>
        <div className="flex gap-1.5 ml-auto">
          <button onClick={() => { setPlaying(false); setCur(0) }}
            className="px-2 py-1.5 rounded-lg text-xs bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-300">⏮</button>
          <button onClick={() => { setPlaying(false); setCur(c => Math.max(0,c-1)) }} disabled={cur===0}
            className="px-2 py-1.5 rounded-lg text-xs bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-300 disabled:opacity-40">◀</button>
          <button onClick={() => setPlaying(p => !p)} disabled={cur >= STEPS.length-1}
            className="px-4 py-1.5 rounded-lg text-xs font-bold bg-orange-500 text-white hover:bg-orange-600 disabled:opacity-40">
            {playing ? '⏸' : '▶'} {playing ? '暂停' : '播放'}
          </button>
          <button onClick={() => { setPlaying(false); setCur(c => Math.min(STEPS.length-1,c+1)) }} disabled={cur >= STEPS.length-1}
            className="px-2 py-1.5 rounded-lg text-xs bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-300 disabled:opacity-40">▶</button>
        </div>
        <div className="flex items-center gap-1.5 text-[10px] text-slate-400">
          <input type="range" min={600} max={3000} step={400} value={speed}
            onChange={e => setSpeed(Number(e.target.value))} className="w-14 accent-orange-500"/>
          <span>{(speed/1000).toFixed(1)}s</span>
        </div>
      </div>

      <div className="p-4 space-y-3">
        {/* Step header */}
        <div className="flex items-center gap-2">
          <span className="px-2 py-0.5 rounded-full text-[10px] font-bold text-white bg-orange-500">{cur+1}/{STEPS.length}</span>
          <span className="font-semibold text-sm text-slate-700 dark:text-slate-200">{step.label}</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-5 gap-3">
          {/* SVG Graph */}
          <div className="md:col-span-3 rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/30 overflow-hidden">
            <svg viewBox="0 0 470 220" className="w-full" style={{ maxHeight: 220 }}>
              {/* DFS path hint */}
              {step.dfsPath.length > 1 && (
                <polyline
                  points={step.dfsPath.map(id => `${NODES[id].x},${NODES[id].y}`).join(' ')}
                  fill="none" stroke="#f59e0b" strokeWidth={6} opacity={0.12} strokeLinejoin="round"/>
              )}

              {/* Edges */}
              {EDGES.map(e => {
                const n1 = NODES[e.u], n2 = NODES[e.v]
                const isBridge = step.bridges.includes(e.id)
                const lp = edgeLabelPos(e)
                const isBack = step.backEdgeActive === e.id
                return (
                  <g key={e.id}>
                    {/* Bridge double stroke */}
                    {isBridge && (
                      <line x1={n1.x} y1={n1.y} x2={n2.x} y2={n2.y}
                        stroke="#ef4444" strokeWidth={8} opacity={0.22} strokeLinecap="round"/>
                    )}
                    <line x1={n1.x} y1={n1.y} x2={n2.x} y2={n2.y}
                      stroke={edgeStroke(e.id)} strokeWidth={edgeWidth(e.id)}
                      strokeDasharray={isBack ? '6 3' : undefined}
                      strokeLinecap="round"/>
                    {isBridge && (
                      <text x={lp.x} y={lp.y+4} textAnchor="middle" fontSize={8} fontWeight="bold" fill="#ef4444">桥</text>
                    )}
                    <text x={lp.x} y={lp.y-6} textAnchor="middle" fontSize={8} fill={edgeStroke(e.id)}>e{e.id}</text>
                  </g>
                )
              })}

              {/* Nodes */}
              {NODES.map(n => {
                const isAP = step.artPoints.includes(n.id)
                const isActive = step.activeNode === n.id
                const isVisited = step.visitedNodes.includes(n.id)
                const disc = step.disc[n.id]
                const low = step.low[n.id]
                return (
                  <g key={n.id}>
                    {isAP && (
                      <circle cx={n.x} cy={n.y} r={R+6} fill="#f97316" opacity={0.25}/>
                    )}
                    {isActive && (
                      <circle cx={n.x} cy={n.y} r={R+4} fill="none" stroke="#f59e0b" strokeWidth={2.5} strokeDasharray="5 2"/>
                    )}
                    <circle cx={n.x} cy={n.y} r={R} fill={nodeFill(n.id)}/>
                    <text x={n.x} y={n.y+5} textAnchor="middle" fontSize={14} fontWeight="bold" fill="white">{n.label}</text>

                    {/* disc/low badges below node */}
                    {isVisited && (
                      <g>
                        <text x={n.x} y={n.y+R+14} textAnchor="middle" fontSize={8} fontWeight="bold" fill={discLowColor(n.id)}>
                          d={disc}
                        </text>
                        <text x={n.x} y={n.y+R+23} textAnchor="middle" fontSize={8} fontWeight="bold"
                          fill={isAP ? '#ea580c' : '#a78bfa'}>
                          l={low}
                        </text>
                      </g>
                    )}
                    {/* AP label */}
                    {isAP && (
                      <text x={n.x} y={n.y-R-6} textAnchor="middle" fontSize={8} fontWeight="bold" fill="#ea580c">割点</text>
                    )}
                  </g>
                )
              })}
            </svg>
          </div>

          {/* Right panel */}
          <div className="md:col-span-2 space-y-2 text-[11px]">
            {/* disc/low table */}
            <div className="rounded-xl border border-blue-200 dark:border-blue-700/50 bg-blue-50 dark:bg-blue-900/20 p-3">
              <div className="text-[10px] font-bold text-blue-600 dark:text-blue-400 uppercase tracking-wider mb-2">disc[ ] / low[ ] 数组</div>
              <div className="grid grid-cols-5 gap-1 text-center">
                {[0,1,2,3,4].map(id => (
                  <div key={id} className={`rounded-lg py-1 ${step.visitedNodes.includes(id) ? 'bg-blue-100 dark:bg-blue-800/40' : 'bg-slate-100 dark:bg-slate-800'}`}>
                    <div className="font-bold text-slate-700 dark:text-slate-200 text-xs">v{id}</div>
                    <div className="text-blue-600 dark:text-blue-400 font-mono text-[9px]">d:{step.disc[id]??'—'}</div>
                    <div className={`font-mono text-[9px] ${step.artPoints.includes(id)?'text-orange-500 font-bold':'text-purple-500'}`}>
                      l:{step.low[id]??'—'}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* DFS stack */}
            <div className="rounded-xl border border-amber-200 dark:border-amber-700/50 bg-amber-50 dark:bg-amber-900/20 p-2.5">
              <div className="text-[10px] font-bold text-amber-600 dark:text-amber-400 uppercase tracking-wider mb-1.5">DFS 路径</div>
              <div className="flex flex-wrap gap-1 items-center">
                {step.dfsPath.length === 0
                  ? <span className="text-slate-400 italic text-[10px]">—</span>
                  : step.dfsPath.map((v, i) => (
                    <React.Fragment key={i}>
                      <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                        i === step.dfsPath.length-1 ? 'bg-amber-400 text-white' : 'bg-amber-100 dark:bg-amber-800 text-amber-700 dark:text-amber-200'
                      }`}>{v}</span>
                      {i < step.dfsPath.length-1 && <span className="text-amber-400 text-[10px]">→</span>}
                    </React.Fragment>
                  ))}
              </div>
            </div>

            {/* Results */}
            <div className="rounded-xl border border-red-200 dark:border-red-700/50 bg-red-50 dark:bg-red-900/20 p-2.5 space-y-1.5">
              <div className="text-[10px] font-bold text-red-600 dark:text-red-400 uppercase tracking-wider">检测结果</div>
              <div>
                <div className="text-[9px] text-slate-500 mb-1">桥（Bridge）</div>
                <div className="flex flex-wrap gap-1">
                  {step.bridges.length === 0
                    ? <span className="text-slate-400 italic text-[10px]">尚无</span>
                    : step.bridges.map(eid => (
                      <span key={eid} className="px-2 py-0.5 rounded text-[10px] font-bold bg-red-200 dark:bg-red-800 text-red-800 dark:text-red-100">
                        e{eid}({NODES[EDGES[eid].u].label}-{NODES[EDGES[eid].v].label})
                      </span>
                    ))}
                </div>
              </div>
              <div>
                <div className="text-[9px] text-slate-500 mb-1">割点（AP）</div>
                <div className="flex flex-wrap gap-1">
                  {step.artPoints.length === 0
                    ? <span className="text-slate-400 italic text-[10px]">尚无</span>
                    : step.artPoints.map(v => (
                      <span key={v} className="px-2 py-0.5 rounded text-[10px] font-bold bg-orange-200 dark:bg-orange-800 text-orange-800 dark:text-orange-100">
                        v{v}
                      </span>
                    ))}
                </div>
              </div>
            </div>

            {/* Detection condition reminder */}
            <div className="rounded-lg bg-slate-50 dark:bg-slate-800/30 border border-slate-200 dark:border-slate-700 p-2">
              <div className="text-[9px] text-slate-400 uppercase mb-1">判定条件</div>
              <div className="space-y-0.5">
                <div className={`text-[9px] ${isComplete ? 'text-red-500 font-bold' : 'text-slate-500'}`}>• 桥：low[v] &gt; disc[u]</div>
                <div className={`text-[9px] ${isComplete ? 'text-orange-500 font-bold' : 'text-slate-500'}`}>• 割点（非根）：low[v] ≥ disc[u]</div>
                <div className={`text-[9px] ${isComplete ? 'text-orange-500 font-bold' : 'text-slate-500'}`}>• 割点（根）：DFS 子节点数 ≥ 2</div>
              </div>
            </div>
          </div>
        </div>

        {/* Description */}
        <div className="rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/30 px-3 py-2 text-[11px] text-slate-600 dark:text-slate-300 whitespace-pre-line">
          {step.desc}
        </div>

        {/* Progress */}
        <div className="h-1.5 rounded-full bg-slate-200 dark:bg-slate-700 overflow-hidden">
          <div className="h-full rounded-full bg-gradient-to-r from-amber-500 to-red-500 transition-all duration-400"
            style={{ width: `${(cur/(STEPS.length-1))*100}%` }} />
        </div>
      </div>
    </div>
  )
}
