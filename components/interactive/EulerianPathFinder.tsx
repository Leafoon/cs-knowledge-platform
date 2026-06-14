'use client'
import React, { useState, useEffect, useRef } from 'react'

// ── 有向图欧拉路径示例 ──────────────────────────────────────────
// 节点: A(0) B(1) C(2) D(3) E(4)
// 边: A→B(0), B→C(1), C→A(2), A→D(3), D→E(4), E→B(5)
// 出度-入度: A=+1(起点), B=-1(终点), C=D=E=0
// 欧拉路径: A→B→C→A→D→E→B

const NODES = [
  { id: 0, label: 'A', x: 90,  y: 105 },
  { id: 1, label: 'B', x: 230, y: 45  },
  { id: 2, label: 'C', x: 230, y: 165 },
  { id: 3, label: 'D', x: 360, y: 45  },
  { id: 4, label: 'E', x: 360, y: 165 },
]

// 有向边 [u, v, curveBend]
const EDGES = [
  { id: 0, u: 0, v: 1, bend: -12 }, // A→B
  { id: 1, u: 1, v: 2, bend:   0 }, // B→C
  { id: 2, u: 2, v: 0, bend: -18 }, // C→A
  { id: 3, u: 0, v: 3, bend:  18 }, // A→D
  { id: 4, u: 3, v: 4, bend:   0 }, // D→E
  { id: 5, u: 4, v: 1, bend: -18 }, // E→B
]

// 每一步的状态：当前访问的节点、已用边、栈、后序结果、说明
interface AlgoStep {
  activeNode: number       // 当前 DFS 顶端节点
  activeEdge: number | null // 刚刚走的边 id
  visitedEdges: number[]    // 已访问（标记完成）的边
  resultEdges: number[]     // 已加入最终路径的边（逆序后）
  stack: string[]
  result: string[]
  poppedNode: string | null // 本步弹出的节点
  label: string
  desc: string
  phase: 'dfs' | 'pop' | 'done'
}

const STEPS: AlgoStep[] = [
  {
    activeNode: 0, activeEdge: null, visitedEdges: [], resultEdges: [], poppedNode: null,
    stack: ['A'], result: [],
    phase: 'dfs',
    label: '初始：从 A 出发',
    desc: '检查度数条件：A 的出度比入度多 1（起点），B 的入度比出度多 1（终点），其余节点平衡。将 A 压栈。',
  },
  {
    activeNode: 1, activeEdge: 0, visitedEdges: [0], resultEdges: [], poppedNode: null,
    stack: ['A','B'], result: [],
    phase: 'dfs',
    label: 'DFS：A → B（边 e0）',
    desc: '从 A 出发，沿 A→B 前进。将 B 压栈，边 e0 标记已访问。',
  },
  {
    activeNode: 2, activeEdge: 1, visitedEdges: [0,1], resultEdges: [], poppedNode: null,
    stack: ['A','B','C'], result: [],
    phase: 'dfs',
    label: 'DFS：B → C（边 e1）',
    desc: '继续沿 B→C 前进。将 C 压栈，边 e1 标记已访问。',
  },
  {
    activeNode: 0, activeEdge: 2, visitedEdges: [0,1,2], resultEdges: [], poppedNode: null,
    stack: ['A','B','C','A'], result: [],
    phase: 'dfs',
    label: 'DFS：C → A（边 e2，子回路閉合）',
    desc: 'C→A 形成一个子回路 A→B→C→A。将 A 再次压栈。',
  },
  {
    activeNode: 3, activeEdge: 3, visitedEdges: [0,1,2,3], resultEdges: [], poppedNode: null,
    stack: ['A','B','C','A','D'], result: [],
    phase: 'dfs',
    label: 'DFS：A → D（边 e3）',
    desc: '此时 A 还有未访问的出边 A→D，继续前进。将 D 压栈。',
  },
  {
    activeNode: 4, activeEdge: 4, visitedEdges: [0,1,2,3,4], resultEdges: [], poppedNode: null,
    stack: ['A','B','C','A','D','E'], result: [],
    phase: 'dfs',
    label: 'DFS：D → E（边 e4）',
    desc: '继续 D→E。将 E 压栈。',
  },
  {
    activeNode: 1, activeEdge: 5, visitedEdges: [0,1,2,3,4,5], resultEdges: [], poppedNode: null,
    stack: ['A','B','C','A','D','E','B'], result: [],
    phase: 'dfs',
    label: 'DFS：E → B（边 e5，所有边已用完）',
    desc: 'E→B 是最后一条边。B 被压栈后，B 没有更多未访问出边——死胡同！',
  },
  {
    activeNode: 4, activeEdge: null, visitedEdges: [0,1,2,3,4,5], resultEdges: [], poppedNode: 'B',
    stack: ['A','B','C','A','D','E'], result: ['B'],
    phase: 'pop',
    label: '回溯：弹出 B',
    desc: 'B 无出边，将 B 弹出栈并加入后序结果列表 result=[B]。回到 E。',
  },
  {
    activeNode: 3, activeEdge: null, visitedEdges: [0,1,2,3,4,5], resultEdges: [], poppedNode: 'E',
    stack: ['A','B','C','A','D'], result: ['B','E'],
    phase: 'pop',
    label: '回溯：弹出 E、D',
    desc: 'E 也无更多出边，弹出 E（result=[B,E]）；D 同样，弹出 D（result=[B,E,D]）。',
  },
  {
    activeNode: 2, activeEdge: null, visitedEdges: [0,1,2,3,4,5], resultEdges: [], poppedNode: 'D',
    stack: ['A','B','C'], result: ['B','E','D','A'],
    phase: 'pop',
    label: '回溯：弹出 A、C、B、A',
    desc: '继续回溯：弹出 A→C→B→A，result=[B,E,D,A,C,B,A]。栈清空，DFS 结束。',
  },
  {
    activeNode: -1, activeEdge: null, visitedEdges: [0,1,2,3,4,5], resultEdges: [0,3,4,5,2,1], poppedNode: null,
    stack: [], result: ['B','E','D','A','C','B','A'],
    phase: 'done',
    label: '完成：逆序 → 欧拉路径',
    desc: '将 result 逆序得最终欧拉路径：A → B → C → A → D → E → B（共 6 条边，覆盖所有边一次）。',
  },
]

function quadPath(x1: number, y1: number, x2: number, y2: number, bend: number, R: number) {
  const dx = x2-x1, dy = y2-y1, len = Math.hypot(dx,dy)
  const mx = (x1+x2)/2 - (dy/len)*bend
  const my = (y1+y2)/2 + (dx/len)*bend
  const tx = x2 - mx, ty = y2 - my
  const tlen = Math.hypot(tx,ty) || 1
  return {
    d: `M${x1+dx/len*R},${y1+dy/len*R} Q${mx},${my} ${x2-tx/tlen*R},${y2-ty/tlen*R}`,
    mx, my,
  }
}

export default function EulerianPathFinder() {
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

  function edgeColor(id: number) {
    if (step.phase === 'done') {
      if (step.resultEdges.includes(id)) return '#10b981'
      return '#94a3b8'
    }
    if (step.activeEdge === id) return '#f59e0b'
    if (step.visitedEdges.includes(id)) return '#3b82f6'
    return '#94a3b8'
  }

  function edgeWidth(id: number) {
    if (step.activeEdge === id) return 3.5
    if (step.visitedEdges.includes(id)) return 2.5
    return 1.8
  }

  function nodeColor(id: number) {
    if (step.phase === 'done') return id === 0 ? '#3b82f6' : id === 1 ? '#8b5cf6' : '#475569'
    if (step.activeNode === id) return '#f59e0b'
    if (step.stack.includes(NODES[id].label)) return '#3b82f6'
    return '#475569'
  }

  // Final path edge sequence: A→B→C→A→D→E→B = edges [0,1,2,3,4,5]
  const FINAL_PATH = ['A','B','C','A','D','E','B']

  return (
    <div className="w-full max-w-2xl mx-auto rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 shadow-xl overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-600 via-indigo-600 to-violet-600 px-5 py-4">
        <h3 className="text-white font-bold text-lg tracking-tight">Hierholzer 欧拉路径算法演示</h3>
        <p className="text-blue-200 text-sm mt-0.5">DFS 栈合并子回路 · 后序弹出逆序得路径 · O(V+E)</p>
      </div>

      {/* Controls */}
      <div className="px-4 py-3 flex flex-wrap gap-2 items-center border-b border-slate-100 dark:border-slate-800 bg-slate-50 dark:bg-slate-800/40">
        <div className="flex rounded-lg overflow-hidden text-[10px] font-bold border border-slate-200 dark:border-slate-600">
          <div className={`px-2.5 py-1.5 ${step.phase === 'dfs' ? 'bg-blue-600 text-white' : 'text-slate-400 bg-slate-100 dark:bg-slate-700'}`}>DFS 前进</div>
          <div className={`px-2.5 py-1.5 ${step.phase === 'pop' ? 'bg-indigo-600 text-white' : 'text-slate-400 bg-slate-100 dark:bg-slate-700'}`}>回溯弹出</div>
          <div className={`px-2.5 py-1.5 ${step.phase === 'done' ? 'bg-emerald-600 text-white' : 'text-slate-400 bg-slate-100 dark:bg-slate-700'}`}>完成</div>
        </div>
        <div className="flex gap-1.5 ml-auto">
          <button onClick={() => { setPlaying(false); setCur(0) }}
            className="px-2 py-1.5 rounded-lg text-xs bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-300 hover:bg-slate-300 dark:hover:bg-slate-600">⏮</button>
          <button onClick={() => { setPlaying(false); setCur(c => Math.max(0,c-1)) }} disabled={cur===0}
            className="px-2 py-1.5 rounded-lg text-xs bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-300 disabled:opacity-40">◀</button>
          <button onClick={() => setPlaying(p => !p)} disabled={cur >= STEPS.length-1}
            className="px-4 py-1.5 rounded-lg text-xs font-bold bg-indigo-600 text-white hover:bg-indigo-700 disabled:opacity-40">
            {playing ? '⏸' : '▶'} {playing ? '暂停' : '播放'}
          </button>
          <button onClick={() => { setPlaying(false); setCur(c => Math.min(STEPS.length-1,c+1)) }} disabled={cur >= STEPS.length-1}
            className="px-2 py-1.5 rounded-lg text-xs bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-300 disabled:opacity-40">▶</button>
        </div>
        <div className="flex items-center gap-1.5 text-[10px] text-slate-400">
          <input type="range" min={600} max={3000} step={400} value={speed}
            onChange={e => setSpeed(Number(e.target.value))} className="w-14 accent-indigo-500" />
          <span>{(speed/1000).toFixed(1)}s</span>
        </div>
      </div>

      <div className="p-4 space-y-3">
        {/* Step header */}
        <div className="flex items-center gap-2">
          <span className="px-2 py-0.5 rounded-full text-[10px] font-bold text-white bg-indigo-500">{cur+1}/{STEPS.length}</span>
          <span className="font-semibold text-sm text-slate-700 dark:text-slate-200">{step.label}</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-5 gap-3">
          {/* SVG Graph */}
          <div className="md:col-span-3 rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/30 overflow-hidden">
            <svg viewBox="0 0 450 215" className="w-full" style={{ maxHeight: 215 }}>
              <defs>
                {EDGES.map(e => (
                  <marker key={e.id} id={`arr-ep-${e.id}`} markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
                    <path d="M0,0 L8,4 L0,8 Z" fill={edgeColor(e.id)} />
                  </marker>
                ))}
              </defs>

              {/* Final path overlay */}
              {step.phase === 'done' && (
                <polyline
                  points={FINAL_PATH.map(l => { const n = NODES.find(n => n.label===l)!; return `${n.x},${n.y}` }).join(' ')}
                  fill="none" stroke="#10b981" strokeWidth={6} opacity={0.15} strokeLinejoin="round" />
              )}

              {/* Edges */}
              {EDGES.map(e => {
                const n1 = NODES[e.u], n2 = NODES[e.v]
                const { d, mx, my } = quadPath(n1.x, n1.y, n2.x, n2.y, e.bend, R)
                const col = edgeColor(e.id)
                return (
                  <g key={e.id}>
                    <path d={d} stroke={col} strokeWidth={edgeWidth(e.id)} fill="none"
                      markerEnd={`url(#arr-ep-${e.id})`} />
                    <text x={mx} y={my+4} textAnchor="middle" fontSize={9}
                      fontWeight="bold" fill={col}>e{e.id}</text>
                  </g>
                )
              })}

              {/* Nodes */}
              {NODES.map(n => (
                <g key={n.id}>
                  <circle cx={n.x} cy={n.y} r={R} fill={nodeColor(n.id)} />
                  {step.activeNode === n.id && (
                    <circle cx={n.x} cy={n.y} r={R+4} fill="none" stroke="#f59e0b" strokeWidth={2} strokeDasharray="4 2" />
                  )}
                  <text x={n.x} y={n.y+5} textAnchor="middle" fontSize={14} fontWeight="bold" fill="white">{n.label}</text>
                  {/* Degree indicator */}
                  <text x={n.x} y={n.y-26} textAnchor="middle" fontSize={8}
                    fill={n.id===0?'#60a5fa':n.id===1?'#a78bfa':'#94a3b8'}>
                    {n.id===0?'起点':n.id===1?'终点':''}
                  </text>
                </g>
              ))}

              {/* Final path label */}
              {step.phase === 'done' && (
                <text x={225} y={205} textAnchor="middle" fontSize={10} fontWeight="bold" fill="#10b981">
                  欧拉路径: A → B → C → A → D → E → B
                </text>
              )}
            </svg>
          </div>

          {/* Right panel: Stack + Result */}
          <div className="md:col-span-2 space-y-2">
            {/* Stack */}
            <div className="rounded-xl border border-blue-200 dark:border-blue-700/50 bg-blue-50 dark:bg-blue-900/20 p-3">
              <div className="text-[10px] font-bold text-blue-600 dark:text-blue-400 uppercase tracking-wider mb-2">DFS 栈（顶→底）</div>
              <div className="flex flex-wrap gap-1">
                {step.stack.length === 0
                  ? <span className="text-[11px] text-slate-400 italic">空</span>
                  : [...step.stack].reverse().map((v, i) => (
                      <span key={i} className={`px-2 py-0.5 rounded-md text-xs font-bold ${
                        i === 0 ? 'bg-amber-400 text-white' : 'bg-blue-200 dark:bg-blue-800 text-blue-800 dark:text-blue-200'
                      }`}>{v}</span>
                    ))}
              </div>
            </div>

            {/* Result (post-order) */}
            <div className="rounded-xl border border-indigo-200 dark:border-indigo-700/50 bg-indigo-50 dark:bg-indigo-900/20 p-3">
              <div className="text-[10px] font-bold text-indigo-600 dark:text-indigo-400 uppercase tracking-wider mb-2">后序 result（待逆序）</div>
              <div className="flex flex-wrap gap-1">
                {step.result.length === 0
                  ? <span className="text-[11px] text-slate-400 italic">空</span>
                  : step.result.map((v, i) => (
                      <span key={i} className="px-2 py-0.5 rounded-md text-xs font-bold bg-indigo-200 dark:bg-indigo-800 text-indigo-800 dark:text-indigo-200">{v}</span>
                    ))}
              </div>
            </div>

            {/* Final path display */}
            {step.phase === 'done' && (
              <div className="rounded-xl border border-emerald-200 dark:border-emerald-700/50 bg-emerald-50 dark:bg-emerald-900/20 p-3">
                <div className="text-[10px] font-bold text-emerald-600 dark:text-emerald-400 uppercase tracking-wider mb-2">✓ 欧拉路径（逆序后）</div>
                <div className="flex flex-wrap gap-1 items-center">
                  {FINAL_PATH.map((v, i) => (
                    <React.Fragment key={i}>
                      <span className="px-2 py-0.5 rounded-md text-xs font-bold bg-emerald-200 dark:bg-emerald-800 text-emerald-800 dark:text-emerald-200">{v}</span>
                      {i < FINAL_PATH.length-1 && <span className="text-emerald-500 text-xs">→</span>}
                    </React.Fragment>
                  ))}
                </div>
              </div>
            )}

            {/* Degree check */}
            <div className="rounded-lg bg-slate-50 dark:bg-slate-800/30 border border-slate-200 dark:border-slate-700 p-2">
              <div className="text-[9px] text-slate-400 uppercase tracking-wider mb-1">出度 − 入度</div>
              <div className="grid grid-cols-5 gap-1 text-center text-[10px]">
                {['A','B','C','D','E'].map((l,i) => (
                  <div key={i}>
                    <div className="font-bold text-slate-600 dark:text-slate-300">{l}</div>
                    <div className={`font-bold ${[1,0,0,0,0][i]===1?'text-blue-500':[-1,0,0,0,0][i]===-1?'text-violet-500':'text-slate-400'}`}>
                      {['+1','-1','0','0','0'][i]}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Description */}
        <div className="rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/30 px-3 py-2 text-[11px] text-slate-600 dark:text-slate-300">
          {step.desc}
        </div>

        {/* Progress */}
        <div className="h-1.5 rounded-full bg-slate-200 dark:bg-slate-700 overflow-hidden">
          <div className="h-full rounded-full bg-gradient-to-r from-blue-500 to-violet-500 transition-all duration-400"
            style={{ width: `${(cur/(STEPS.length-1))*100}%` }} />
        </div>

        {/* Legend */}
        <div className="flex flex-wrap gap-3 text-[10px] text-slate-500 dark:text-slate-400">
          <span className="flex items-center gap-1"><span className="w-6 h-0.5 bg-slate-400 inline-block"/>未访问边</span>
          <span className="flex items-center gap-1"><span className="w-6 h-0.5 bg-blue-500 inline-block"/>已访问边</span>
          <span className="flex items-center gap-1"><span className="w-6 h-0.5 bg-amber-400 inline-block"/>当前边</span>
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-amber-400 inline-block"/>栈顶节点</span>
        </div>
      </div>
    </div>
  )
}
