'use client'
import { useState, useEffect } from 'react'

/* ─── 换根 DP（Rerooting）动画 ─────────────────────────────
   经典问题：求每个节点到树上所有其他节点的距离之和 ans[v]
   树结构：6 节点；边：0-1, 0-2, 2-3, 2-4, 4-5
   ────────────────────────────────────────────────────────── */

const N = 6
const EDGES: [number, number][] = [[0,1],[0,2],[2,3],[2,4],[4,5]]
const ADJ: number[][] = Array.from({ length: N }, () => [])
EDGES.forEach(([u, v]) => { ADJ[u].push(v); ADJ[v].push(u) })

// 节点在 SVG 中的位置（viewBox 0 0 340 320）
const POS: [number, number][] = [
  [170, 34],  // 0 - root
  [74,  132], // 1
  [266, 132], // 2
  [200, 224], // 3
  [302, 224], // 4
  [302, 300], // 5
]
const NODE_R = 20

// DP 值（预先计算，动画中逐步"揭露"）
const SZ  = [6, 1, 4, 1, 2, 1]
const DOWN = [9, 0, 4, 0, 1, 0]
const ANS  = [9, 13, 7, 11, 9, 13]

// DFS1 处理顺序（自底向上：后序遍历）
const DFS1_ORDER = [1, 3, 5, 4, 2, 0]
// DFS2 处理顺序（自顶向下：前序遍历）
const DFS2_ORDER = [0, 1, 2, 3, 4, 5]

const PHASE1_STEPS = DFS1_ORDER.length // 6
const TOTAL_STEPS  = PHASE1_STEPS + DFS2_ORDER.length // 12

const NODE_COLORS = ['#7c3aed', '#2563eb', '#059669', '#d97706', '#dc2626', '#0891b2']

function getParent(v: number): number {
  if (v === 0) return -1
  if (v === 1 || v === 2) return 0
  if (v === 3 || v === 4) return 2
  return 4 // v === 5
}

export default function TreeDPRerooting() {
  const [step, setStep] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(700)

  useEffect(() => {
    if (!playing) return
    if (step >= TOTAL_STEPS) { setPlaying(false); return }
    const id = setTimeout(() => setStep(s => s + 1), speed)
    return () => clearTimeout(id)
  }, [playing, step, speed])

  // 判断各节点状态
  const phase1Done = DFS1_ORDER.slice(0, Math.min(step, PHASE1_STEPS))
  const phase2Done = step > PHASE1_STEPS
    ? DFS2_ORDER.slice(0, step - PHASE1_STEPS)
    : []

  const isPhase2 = step > PHASE1_STEPS
  const activeNode = !isPhase2 && step > 0
    ? DFS1_ORDER[step - 1]
    : (isPhase2 && step - PHASE1_STEPS <= DFS2_ORDER.length)
      ? DFS2_ORDER[step - PHASE1_STEPS - 1]
      : null

  function nodeKnowsSZ(v: number) { return phase1Done.includes(v) }
  function nodeKnowsANS(v: number) { return phase2Done.includes(v) }

  // 当前高亮边（active → parent 或 parent → active）
  const activeEdges = new Set<string>()
  if (activeNode !== null) {
    const par = getParent(activeNode)
    if (par >= 0) activeEdges.add(`${Math.min(activeNode, par)}-${Math.max(activeNode, par)}`)
  }

  const SVG_W = 340, SVG_H = 320

  return (
    <div className="rounded-2xl border border-lime-200 dark:border-lime-900 bg-white dark:bg-zinc-950 overflow-hidden shadow-sm">
      <div className="px-6 py-4 bg-gradient-to-r from-lime-600 to-green-600 dark:from-lime-700 dark:to-green-700">
        <h3 className="text-white font-bold text-base">换根 DP（Rerooting）两趟 DFS 动画</h3>
        <p className="text-lime-100 text-sm mt-0.5">
          第一趟：以 0 为根自底向上，得 sz[] 与 down[]；第二趟：自顶向下，ans[v] = ans[par] − sz[v] + (n − sz[v])
        </p>
      </div>

      <div className="p-5 space-y-4">
        {/* 进度 & 控制 */}
        <div className="flex flex-wrap items-center gap-2">
          <div className={`px-3 py-1 text-xs font-bold rounded-full ${!isPhase2 ? 'bg-lime-600 text-white' : 'bg-slate-100 dark:bg-zinc-800 text-slate-400'}`}>
            第一趟 DFS（↑ 自底向上）
          </div>
          <div className={`px-3 py-1 text-xs font-bold rounded-full ${isPhase2 ? 'bg-green-600 text-white' : 'bg-slate-100 dark:bg-zinc-800 text-slate-400'}`}>
            第二趟 DFS（↓ 自顶向下）
          </div>
          <div className="flex items-center gap-1.5 ml-auto">
            <button onClick={() => setStep(s => Math.max(0, s - 1))} disabled={step === 0}
              className="px-2.5 py-1.5 text-xs bg-slate-100 dark:bg-zinc-800 disabled:opacity-40 rounded-lg text-slate-700 dark:text-zinc-200">←</button>
            <button onClick={() => setPlaying(p => !p)}
              className={`px-3 py-1.5 text-xs rounded-lg text-white font-medium ${playing ? 'bg-orange-500' : 'bg-lime-600 hover:bg-lime-500'}`}>
              {playing ? '⏸ 暂停' : '▶ 播放'}
            </button>
            <button onClick={() => setStep(s => Math.min(TOTAL_STEPS, s + 1))} disabled={step >= TOTAL_STEPS}
              className="px-2.5 py-1.5 text-xs bg-slate-100 dark:bg-zinc-800 disabled:opacity-40 rounded-lg text-slate-700 dark:text-zinc-200">→</button>
            <button onClick={() => { setStep(0); setPlaying(false) }}
              className="px-2.5 py-1.5 text-xs bg-slate-100 dark:bg-zinc-800 rounded-lg text-slate-700 dark:text-zinc-200">↺</button>
            {([['慢', 900], ['中', 700], ['快', 300]] as [string, number][]).map(([l, ms]) => (
              <button key={ms} onClick={() => setSpeed(ms)}
                className={`px-2 py-1 text-xs rounded ${speed === ms ? 'bg-lime-600 text-white' : 'bg-slate-100 dark:bg-zinc-800 text-slate-500 dark:text-zinc-400'}`}>{l}</button>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">
          {/* SVG 树 */}
          <div className="lg:col-span-3 bg-slate-50 dark:bg-zinc-900 rounded-xl border border-slate-200 dark:border-zinc-700 p-2">
            <svg width="100%" viewBox={`0 0 ${SVG_W} ${SVG_H}`} className="block">
              {/* 边 */}
              {EDGES.map(([u, v]) => {
                const key = `${Math.min(u,v)}-${Math.max(u,v)}`
                const isActive = activeEdges.has(key)
                const [x1, y1] = POS[u], [x2, y2] = POS[v]
                return (
                  <g key={key}>
                    <line x1={x1} y1={y1} x2={x2} y2={y2}
                      stroke={isActive ? (isPhase2 ? '#16a34a' : '#65a30d') : '#cbd5e1'}
                      className={isActive ? '' : 'dark:stroke-zinc-700'}
                      strokeWidth={isActive ? 2.5 : 1.5} />
                    {/* 边上贡献标注 */}
                    {isActive && nodeKnowsSZ(v > u ? v : u) && !isPhase2 && (
                      <text x={(x1+x2)/2+8} y={(y1+y2)/2-4} fontSize={10} fill="#65a30d" fontWeight="bold">
                        +{SZ[v > u ? v : u]}
                      </text>
                    )}
                  </g>
                )
              })}

              {/* 节点 */}
              {Array.from({ length: N }, (_, v) => {
                const [cx, cy] = POS[v]
                const isActive = activeNode === v
                const p1Done = nodeKnowsSZ(v)
                const p2Done = nodeKnowsANS(v)
                return (
                  <g key={v}>
                    <circle cx={cx} cy={cy} r={NODE_R + (isActive ? 3 : 0)}
                      fill={isActive ? NODE_COLORS[v] : p2Done ? '#f0fdf4' : p1Done ? '#eff6ff' : '#f1f5f9'}
                      className={!isActive ? (p2Done ? 'dark:fill-green-950' : p1Done ? 'dark:fill-blue-950' : 'dark:fill-zinc-800') : ''}
                      stroke={NODE_COLORS[v]}
                      strokeWidth={isActive ? 2.5 : 1.5} />
                    {/* 节点编号 */}
                    <text x={cx} y={cy + 1} textAnchor="middle" dominantBaseline="middle"
                      fontSize={13} fontWeight="bold"
                      fill={isActive ? 'white' : NODE_COLORS[v]}>{v}</text>

                    {/* sz 标注（第一趟完成后） */}
                    {p1Done && (
                      <text x={cx - NODE_R - 4} y={cy - 10} textAnchor="end" fontSize={9} fill="#2563eb" fontWeight="600">
                        sz={SZ[v]}
                      </text>
                    )}
                    {/* ans 标注（第二趟完成后） */}
                    {p2Done && (
                      <text x={cx + NODE_R + 4} y={cy + 6} textAnchor="start" fontSize={9} fill="#16a34a" fontWeight="700">
                        ans={ANS[v]}
                      </text>
                    )}
                  </g>
                )
              })}

              {/* 图例 */}
              <g>
                <circle cx={12} cy={SVG_H - 30} r={6} fill="#dbeafe" stroke="#2563eb" strokeWidth={1} />
                <text x={22} y={SVG_H - 26} fontSize={9} fill="#2563eb">DFS1 完成</text>
                <circle cx={12} cy={SVG_H - 15} r={6} fill="#dcfce7" stroke="#16a34a" strokeWidth={1} />
                <text x={22} y={SVG_H - 11} fontSize={9} fill="#16a34a">DFS2 完成</text>
              </g>
            </svg>
          </div>

          {/* 右侧：步骤说明 + 表格 */}
          <div className="lg:col-span-2 space-y-3">
            {/* 当前操作说明 */}
            {activeNode !== null && (
              <div className={`rounded-xl border p-3 text-xs space-y-1 ${isPhase2 ? 'bg-green-50 dark:bg-green-950/30 border-green-200 dark:border-green-800' : 'bg-blue-50 dark:bg-blue-950/30 border-blue-200 dark:border-blue-800'}`}>
                <p className="font-bold" style={{ color: NODE_COLORS[activeNode] }}>
                  {isPhase2 ? '↓ DFS2' : '↑ DFS1'} 处理节点 {activeNode}
                </p>
                {!isPhase2 ? (
                  <div className="font-mono text-slate-600 dark:text-zinc-300 space-y-0.5 text-[11px]">
                    <p>sz[{activeNode}] = {SZ[activeNode]}</p>
                    <p>down[{activeNode}] = Σ (down[c] + sz[c]) = {DOWN[activeNode]}</p>
                  </div>
                ) : (
                  <div className="font-mono text-slate-600 dark:text-zinc-300 space-y-0.5 text-[11px]">
                    {activeNode === 0 ? (
                      <p>ans[0] = down[0] = {ANS[0]}</p>
                    ) : (
                      <>
                        <p>par = {getParent(activeNode)}</p>
                        <p>ans[{activeNode}] = ans[{getParent(activeNode)}]</p>
                        <p className="pl-4">− sz[{activeNode}] + (n − sz[{activeNode}])</p>
                        <p className="pl-4">= {ANS[getParent(activeNode)]} − {SZ[activeNode]} + ({N} − {SZ[activeNode]})</p>
                        <p className="pl-4 text-green-700 dark:text-green-400 font-bold">= {ANS[activeNode]}</p>
                      </>
                    )}
                  </div>
                )}
              </div>
            )}
            {step === 0 && (
              <div className="rounded-xl border border-slate-200 dark:border-zinc-700 bg-slate-50 dark:bg-zinc-900 p-3 text-xs text-slate-500 dark:text-zinc-400">
                点击播放，观察两趟 DFS 如何高效计算所有节点的距离之和。
              </div>
            )}

            {/* DP 值总览表 */}
            <div className="rounded-xl border border-slate-200 dark:border-zinc-700 bg-slate-50 dark:bg-zinc-900 overflow-auto p-3">
              <p className="text-[10px] font-bold uppercase tracking-widest text-slate-400 dark:text-zinc-500 mb-2">DP 状态总览</p>
              <table className="text-[11px] w-full text-center">
                <thead>
                  <tr className="text-slate-400 dark:text-zinc-500">
                    <th className="py-1">节点</th>
                    <th className="py-1 text-blue-500">sz</th>
                    <th className="py-1 text-blue-500">down</th>
                    <th className="py-1 text-green-500">ans</th>
                  </tr>
                </thead>
                <tbody>
                  {Array.from({ length: N }, (_, v) => {
                    const p1 = nodeKnowsSZ(v)
                    const p2 = nodeKnowsANS(v)
                    const isAct = activeNode === v
                    return (
                      <tr key={v} className={isAct ? (isPhase2 ? 'bg-green-50 dark:bg-green-950/30' : 'bg-blue-50 dark:bg-blue-950/30') : ''}>
                        <td className="py-1 font-bold" style={{ color: NODE_COLORS[v] }}>{v}</td>
                        <td className={`py-1 font-mono ${p1 ? 'text-blue-600 dark:text-blue-400 font-bold' : 'text-slate-300 dark:text-zinc-600'}`}>
                          {p1 ? SZ[v] : '?'}
                        </td>
                        <td className={`py-1 font-mono ${p1 ? 'text-blue-600 dark:text-blue-400' : 'text-slate-300 dark:text-zinc-600'}`}>
                          {p1 ? DOWN[v] : '?'}
                        </td>
                        <td className={`py-1 font-mono ${p2 ? 'text-green-600 dark:text-green-400 font-bold' : 'text-slate-300 dark:text-zinc-600'}`}>
                          {p2 ? ANS[v] : '?'}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>

            {/* 最终结果 */}
            {step >= TOTAL_STEPS && (
              <div className="rounded-xl border border-green-200 dark:border-green-800 bg-green-50 dark:bg-green-950/40 p-3">
                <p className="text-xs font-bold text-green-700 dark:text-green-300 mb-1.5">所有节点距离之和</p>
                <div className="grid grid-cols-3 gap-1">
                  {ANS.map((a, v) => (
                    <div key={v} className="text-center">
                      <p className="text-[10px] text-slate-400" style={{ color: NODE_COLORS[v] }}>节点 {v}</p>
                      <p className="text-sm font-bold text-green-700 dark:text-green-300">{a}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* 进度条 */}
        <div className="w-full h-1.5 bg-slate-100 dark:bg-zinc-800 rounded-full overflow-hidden">
          <div className="h-full rounded-full transition-all duration-150"
            style={{
              width: `${(step / TOTAL_STEPS) * 100}%`,
              background: isPhase2 ? '#16a34a' : '#65a30d'
            }} />
        </div>
        <p className="text-xs text-slate-400 dark:text-zinc-500 text-right">
          {!isPhase2 ? `DFS1: ${Math.min(step, PHASE1_STEPS)}/${PHASE1_STEPS}` : `DFS2: ${step - PHASE1_STEPS}/${DFS2_ORDER.length}`}
        </p>
      </div>
    </div>
  )
}
