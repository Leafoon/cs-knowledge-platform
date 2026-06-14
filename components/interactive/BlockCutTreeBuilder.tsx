'use client'
import React, { useState, useEffect, useRef } from 'react'

// ── 块-割点树（Block-Cut Tree）构建演示 ─────────────────────────
// 原图：7 节点，两个双连通分量（BCC），1 个割点
// BCC₁ = {0,1,2,3}（4 节点回路）
// BCC₂ = {3,4,5,6}（另一个 4 节点回路）
// 割点 = {3}
// Block-Cut Tree：B₁ —— 3 —— B₂（路线形）

const ORIG_NODES = [
  { id: 0, x: 70,  y: 120, label: '0' },
  { id: 1, x: 155, y: 65,  label: '1' },
  { id: 2, x: 155, y: 175, label: '2' },
  { id: 3, x: 250, y: 120, label: '3' },
  { id: 4, x: 345, y: 65,  label: '4' },
  { id: 5, x: 345, y: 175, label: '5' },
  { id: 6, x: 428, y: 120, label: '6' },
]

const ORIG_EDGES = [
  { id: 0, u: 0, v: 1 },
  { id: 1, u: 0, v: 2 },
  { id: 2, u: 1, v: 3 },
  { id: 3, u: 2, v: 3 },
  { id: 4, u: 3, v: 4 },
  { id: 5, u: 3, v: 5 },
  { id: 6, u: 4, v: 6 },
  { id: 7, u: 5, v: 6 },
]

const BCC1_NODES = [0, 1, 2, 3]
const BCC2_NODES = [3, 4, 5, 6]
const BCC1_EDGES = [0, 1, 2, 3]
const BCC2_EDGES = [4, 5, 6, 7]
const CUT_POINTS  = [3]

// Block-cut tree layout (in a separate SVG of same viewbox)
// Three nodes: B1 (block), node3 (cut), B2 (block)
const BCT_B1   = { x: 110, y: 120 }
const BCT_CUT  = { x: 250, y: 120 }
const BCT_B2   = { x: 390, y: 120 }

interface BCTStep {
  highlightBCC1: boolean
  highlightBCC2: boolean
  showCutPoint: boolean
  bctEdge1: boolean   // B1 -- 3
  bctEdge2: boolean   // 3 -- B2
  showBCT: boolean
  activeNodes: number[]
  label: string
  desc: string
}

const STEPS: BCTStep[] = [
  {
    highlightBCC1: false, highlightBCC2: false, showCutPoint: false,
    bctEdge1: false, bctEdge2: false, showBCT: false,
    activeNodes: [],
    label: '初始：连通无向图（7 节点，8 条边）',
    desc: '该无向图由两个强连通子结构组成，直觉上可以分成两部分。Tarjan 算法将使用 DFS 与低值（low）节点栈，自动发现所有双连通分量（BCC）和割点。',
  },
  {
    highlightBCC1: true, highlightBCC2: false, showCutPoint: false,
    bctEdge1: false, bctEdge2: false, showBCT: false,
    activeNodes: [0,1,2,3],
    label: '发现 BCC₁ = {0, 1, 2, 3}（蓝色）',
    desc: 'DFS 从节点 0 出发，经 0→1→3→2→0 形成回路，将边 {0-1, 0-2, 1-3, 2-3} 从边栈弹出，识别为第一个双连通分量 BCC₁。BCC 内任意两顶点之间存在至少两条点不相交路径。',
  },
  {
    highlightBCC1: true, highlightBCC2: false, showCutPoint: true,
    bctEdge1: false, bctEdge2: false, showBCT: false,
    activeNodes: [3],
    label: '识别割点 3（橙色高亮）',
    desc: 'DFS 回溯到节点 3：low[4]=4 ≥ disc[3]=3，说明子树 {4,5,6} 无法绕过 3 到达祖先。节点 3 是割点（Articulation Point）——删除 3 后图将分裂为两个连通分量。',
  },
  {
    highlightBCC1: true, highlightBCC2: true, showCutPoint: true,
    bctEdge1: false, bctEdge2: false, showBCT: false,
    activeNodes: [3,4,5,6],
    label: '发现 BCC₂ = {3, 4, 5, 6}（绿色）',
    desc: 'DFS 继续探索 3→4→6→5→3，将边 {3-4, 3-5, 4-6, 5-6} 弹出，识别 BCC₂。注意：割点 3 同时属于 BCC₁ 和 BCC₂——这是 Block-Cut Tree 中割点作为桥接节点的原因。',
  },
  {
    highlightBCC1: true, highlightBCC2: true, showCutPoint: true,
    bctEdge1: true, bctEdge2: false, showBCT: true,
    activeNodes: [],
    label: '构建 Block-Cut Tree：B₁ 连接割点 3',
    desc: '块割点树是两类节点的二部图：方块节点代表每个 BCC，圆形节点代表割点。BCC₁（方块 B₁）与割点节点 3 相连，因为 3 属于 BCC₁。',
  },
  {
    highlightBCC1: true, highlightBCC2: true, showCutPoint: true,
    bctEdge1: true, bctEdge2: true, showBCT: true,
    activeNodes: [],
    label: '完成：B₂ 连接割点 3 → 树结构',
    desc: '同样，BCC₂（方块 B₂）也与割点节点 3 相连。最终 Block-Cut Tree：B₁ — 割点3 — B₂，是一条三节点路径。树的节点数 = #BCC + #割点 = 2+1 = 3。',
  },
  {
    highlightBCC1: true, highlightBCC2: true, showCutPoint: true,
    bctEdge1: true, bctEdge2: true, showBCT: true,
    activeNodes: [],
    label: '完成：算法总结',
    desc: 'Block-Cut Tree 性质：① 是树（无环）② 叶节点是 BCC ③ 若原图有 n 个割点和 b 个 BCC，树有 n+b 个节点 ④ 任意两点间的路径唯一对应原图中的双联通结构层次 ⑤ 时间复杂度 O(V+E)（与 Tarjan DFS 相同）。',
  },
]

export default function BlockCutTreeBuilder() {
  const [cur, setCur] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(2200)
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

  function nodeFill(id: number) {
    if (step.showCutPoint && CUT_POINTS.includes(id)) {
      return step.activeNodes.includes(id) ? '#f97316' : '#f97316'
    }
    if (step.highlightBCC2 && BCC2_NODES.includes(id)) return '#10b981'
    if (step.highlightBCC1 && BCC1_NODES.includes(id)) return '#3b82f6'
    return '#64748b'
  }

  function edgeFill(eid: number) {
    if (step.highlightBCC2 && BCC2_EDGES.includes(eid)) return '#10b981'
    if (step.highlightBCC1 && BCC1_EDGES.includes(eid)) return '#3b82f6'
    return '#94a3b8'
  }

  function edgeWidth(eid: number) {
    if (step.highlightBCC1 && BCC1_EDGES.includes(eid)) return 3
    if (step.highlightBCC2 && BCC2_EDGES.includes(eid)) return 3
    return 2
  }

  return (
    <div className="w-full max-w-2xl mx-auto rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 shadow-xl overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-emerald-600 via-teal-600 to-cyan-600 px-5 py-4">
        <h3 className="text-white font-bold text-lg tracking-tight">块-割点树（Block-Cut Tree）构建演示</h3>
        <p className="text-teal-200 text-sm mt-0.5">双连通分量分解 + 树形层次结构 · O(V+E)</p>
      </div>

      {/* Controls */}
      <div className="px-4 py-3 flex flex-wrap gap-2 items-center border-b border-slate-100 dark:border-slate-800 bg-slate-50 dark:bg-slate-800/40">
        <div className="flex gap-3 text-[10px] font-bold items-center">
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-blue-500 inline-block"/>BCC₁</span>
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-emerald-500 inline-block"/>BCC₂</span>
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-orange-500 inline-block"/>割点</span>
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-sm bg-slate-300 dark:bg-slate-600 inline-block"/>Block节点</span>
        </div>
        <div className="flex gap-1.5 ml-auto">
          <button onClick={() => { setPlaying(false); setCur(0) }} className="px-2 py-1.5 rounded-lg text-xs bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-300">⏮</button>
          <button onClick={() => { setPlaying(false); setCur(c => Math.max(0,c-1)) }} disabled={cur===0} className="px-2 py-1.5 rounded-lg text-xs bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-300 disabled:opacity-40">◀</button>
          <button onClick={() => setPlaying(p => !p)} disabled={cur>=STEPS.length-1} className="px-4 py-1.5 rounded-lg text-xs font-bold bg-teal-600 text-white hover:bg-teal-700 disabled:opacity-40">
            {playing?'⏸':'▶'} {playing?'暂停':'播放'}
          </button>
          <button onClick={() => { setPlaying(false); setCur(c => Math.min(STEPS.length-1,c+1)) }} disabled={cur>=STEPS.length-1} className="px-2 py-1.5 rounded-lg text-xs bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-300 disabled:opacity-40">▶</button>
        </div>
        <div className="flex items-center gap-1.5 text-[10px] text-slate-400">
          <input type="range" min={800} max={3500} step={400} value={speed} onChange={e=>setSpeed(Number(e.target.value))} className="w-14 accent-teal-500"/>
          <span>{(speed/1000).toFixed(1)}s</span>
        </div>
      </div>

      <div className="p-4 space-y-3">
        {/* Step header */}
        <div className="flex items-center gap-2">
          <span className="px-2 py-0.5 rounded-full text-[10px] font-bold text-white bg-teal-600">{cur+1}/{STEPS.length}</span>
          <span className="font-semibold text-sm text-slate-700 dark:text-slate-200">{step.label}</span>
        </div>

        {/* Original graph */}
        <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/30 overflow-hidden">
          <div className="px-3 pt-2 text-[10px] font-bold text-slate-400 uppercase">原始图</div>
          <svg viewBox="40 30 430 185" className="w-full" style={{ maxHeight: 185 }}>
            {/* BCC area backgrounds */}
            {step.highlightBCC1 && (
              <rect x={45} y={40} width={225} height={165} rx={14} fill="#3b82f6" opacity={0.08} stroke="#3b82f6" strokeWidth={1.5} strokeDasharray="6 3"/>
            )}
            {step.highlightBCC2 && (
              <rect x={230} y={40} width={225} height={165} rx={14} fill="#10b981" opacity={0.08} stroke="#10b981" strokeWidth={1.5} strokeDasharray="6 3"/>
            )}

            {/* BCC labels */}
            {step.highlightBCC1 && (
              <text x={140} y={55} textAnchor="middle" fontSize={10} fontWeight="bold" fill="#3b82f6" opacity={0.7}>BCC₁</text>
            )}
            {step.highlightBCC2 && (
              <text x={360} y={55} textAnchor="middle" fontSize={10} fontWeight="bold" fill="#10b981" opacity={0.7}>BCC₂</text>
            )}

            {/* Edges */}
            {ORIG_EDGES.map(e => {
              const n1 = ORIG_NODES[e.u], n2 = ORIG_NODES[e.v]
              const isB1 = step.highlightBCC1 && BCC1_EDGES.includes(e.id)
              const isB2 = step.highlightBCC2 && BCC2_EDGES.includes(e.id)
              return (
                <g key={e.id}>
                  {(isB1||isB2) && (
                    <line x1={n1.x} y1={n1.y} x2={n2.x} y2={n2.y}
                      stroke={isB1?'#3b82f6':'#10b981'} strokeWidth={8} opacity={0.18} strokeLinecap="round"/>
                  )}
                  <line x1={n1.x} y1={n1.y} x2={n2.x} y2={n2.y}
                    stroke={edgeFill(e.id)} strokeWidth={edgeWidth(e.id)} strokeLinecap="round"/>
                </g>
              )
            })}

            {/* Nodes */}
            {ORIG_NODES.map(n => {
              const isCut = step.showCutPoint && CUT_POINTS.includes(n.id)
              const fill = nodeFill(n.id)
              return (
                <g key={n.id}>
                  {isCut && <circle cx={n.x} cy={n.y} r={R+6} fill="#f97316" opacity={0.2}/>}
                  {isCut && <circle cx={n.x} cy={n.y} r={R+3} fill="none" stroke="#f97316" strokeWidth={2.5} strokeDasharray="4 2"/>}
                  <circle cx={n.x} cy={n.y} r={R} fill={fill}/>
                  <text x={n.x} y={n.y+5} textAnchor="middle" fontSize={13} fontWeight="bold" fill="white">{n.label}</text>
                  {isCut && (
                    <text x={n.x} y={n.y-R-7} textAnchor="middle" fontSize={8} fontWeight="bold" fill="#f97316">割点</text>
                  )}
                </g>
              )
            })}
          </svg>
        </div>

        {/* Block-Cut Tree (shown only when step.showBCT) */}
        <div className={`rounded-xl border overflow-hidden transition-all duration-500 ${
          step.showBCT
            ? 'border-teal-300 dark:border-teal-600 bg-teal-50 dark:bg-teal-900/20 opacity-100'
            : 'border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/30 opacity-50'
        }`}>
          <div className="px-3 pt-2 text-[10px] font-bold text-teal-500 dark:text-teal-400 uppercase flex items-center gap-2">
            块-割点树（Block-Cut Tree）
            {!step.showBCT && <span className="text-slate-400 font-normal italic">（尚未构建）</span>}
          </div>
          <svg viewBox="40 70 430 115" className="w-full" style={{ maxHeight: 115 }}>
            {/* Edges of BCT */}
            {step.bctEdge1 && (
              <g>
                <line x1={BCT_B1.x+32} y1={BCT_B1.y} x2={BCT_CUT.x-R} y2={BCT_CUT.y}
                  stroke="#64748b" strokeWidth={2.5} strokeLinecap="round"/>
              </g>
            )}
            {step.bctEdge2 && (
              <g>
                <line x1={BCT_CUT.x+R} y1={BCT_CUT.y} x2={BCT_B2.x-32} y2={BCT_B2.y}
                  stroke="#64748b" strokeWidth={2.5} strokeLinecap="round"/>
              </g>
            )}

            {/* B1 block node (rounded rect) */}
            <g opacity={step.highlightBCC1 ? 1 : 0.35}>
              <rect x={BCT_B1.x-32} y={BCT_B1.y-22} width={64} height={44} rx={10}
                fill={step.highlightBCC1?'#3b82f6':'#64748b'}/>
              <text x={BCT_B1.x} y={BCT_B1.y-3} textAnchor="middle" fontSize={10} fontWeight="bold" fill="white">BCC₁</text>
              <text x={BCT_B1.x} y={BCT_B1.y+11} textAnchor="middle" fontSize={8} fill="#bfdbfe">{'{'}0,1,2,3{'}'}</text>
            </g>

            {/* Cut point 3 (circle) */}
            <g opacity={step.showCutPoint ? 1 : 0.35}>
              <circle cx={BCT_CUT.x} cy={BCT_CUT.y} r={R+2} fill={step.showCutPoint?'#f97316':'#64748b'}/>
              {step.showCutPoint && <circle cx={BCT_CUT.x} cy={BCT_CUT.y} r={R+6} fill="none" stroke="#f97316" strokeWidth={1.5} strokeDasharray="4 2"/>}
              <text x={BCT_CUT.x} y={BCT_CUT.y+5} textAnchor="middle" fontSize={14} fontWeight="bold" fill="white">3</text>
              <text x={BCT_CUT.x} y={BCT_CUT.y+R+14} textAnchor="middle" fontSize={8} fontWeight="bold" fill="#f97316">割点</text>
            </g>

            {/* B2 block node (rounded rect) */}
            <g opacity={step.highlightBCC2 ? 1 : 0.35}>
              <rect x={BCT_B2.x-32} y={BCT_B2.y-22} width={64} height={44} rx={10}
                fill={step.highlightBCC2?'#10b981':'#64748b'}/>
              <text x={BCT_B2.x} y={BCT_B2.y-3} textAnchor="middle" fontSize={10} fontWeight="bold" fill="white">BCC₂</text>
              <text x={BCT_B2.x} y={BCT_B2.y+11} textAnchor="middle" fontSize={8} fill="#a7f3d0">{'{'}3,4,5,6{'}'}</text>
            </g>

            {/* Not yet label */}
            {!step.showBCT && (
              <text x={250} y={125} textAnchor="middle" fontSize={11} fill="#94a3b8" fontStyle="italic">尚未构建…</text>
            )}
          </svg>
        </div>

        {/* Description */}
        <div className="rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/30 px-3 py-2 text-[11px] text-slate-600 dark:text-slate-300">
          {step.desc}
        </div>

        {/* Summary cards - only when last step */}
        {cur >= STEPS.length - 1 && (
          <div className="grid grid-cols-3 gap-2 text-center text-[11px]">
            <div className="rounded-xl bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-700/50 p-2">
              <div className="text-xl font-black text-blue-600 dark:text-blue-300">2</div>
              <div className="text-[9px] text-slate-400">双连通分量</div>
            </div>
            <div className="rounded-xl bg-orange-50 dark:bg-orange-900/20 border border-orange-200 dark:border-orange-700/50 p-2">
              <div className="text-xl font-black text-orange-600 dark:text-orange-300">1</div>
              <div className="text-[9px] text-slate-400">割点</div>
            </div>
            <div className="rounded-xl bg-teal-50 dark:bg-teal-900/20 border border-teal-200 dark:border-teal-700/50 p-2">
              <div className="text-xl font-black text-teal-600 dark:text-teal-300">3</div>
              <div className="text-[9px] text-slate-400">树节点数</div>
            </div>
          </div>
        )}

        {/* Progress */}
        <div className="h-1.5 rounded-full bg-slate-200 dark:bg-slate-700 overflow-hidden">
          <div className="h-full rounded-full bg-gradient-to-r from-emerald-500 to-cyan-500 transition-all duration-400"
            style={{ width: `${(cur/(STEPS.length-1))*100}%` }}/>
        </div>
      </div>
    </div>
  )
}
