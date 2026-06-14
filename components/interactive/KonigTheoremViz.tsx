'use client'
import React, { useState } from 'react'

// ── König 定理可视化：最大匹配 = 最小顶点覆盖 ─────────────────────
// 同一个二分图：左 W0-W3，右 J0-J3
// 最大匹配 M = {W0↔J0, W1↔J1, W2↔J2, W3↔J3}（大小 4）
// 构造最小顶点覆盖：
//   U = 未匹配左节点（M 全部匹配，所有左节点均匹配，U=∅）
//   Z = 从 U 出发可交错到达的节点（Z=∅）
//   最小覆盖 = (L\Z) ∪ (R∩Z) = L ∪ ∅ = {W0,W1,W2,W3}
// 等价验证：每条边至少有一个覆盖端点 ✓

// 为了展示更有趣的例子，使用较复杂图（6 节点）：
// 左: A(0),B(1),C(2) 右: X(0),Y(1),Z(2)
// 边: A-X, A-Y, B-Y, B-Z, C-Z
// 最大匹配: A↔X, B↔Y, C↔Z（大小 3）
// 未匹配左节点 U = ∅ → Z = ∅ → 覆盖 = {A,B,C}（左侧全部）
// 
// 使用 3×3 图示：明确有交错路径的情形
// 左: P(0), Q(1), R(2)  右: S(0), T(1), U(2)
// 边: P-S, P-T, Q-T, Q-U, R-U
// 最大匹配: P↔S, Q↔T, R↔U
// 覆盖 = {P,Q,R} = 全部左节点

// 切换到更经典的例子，有反向交错路：
// 左: L0,L1,L2  右: R0,R1,R2
// 边: L0-R0, L0-R1, L1-R0, L2-R1, L2-R2
// 最大匹配: L0↔R1, L1↔R0, L2↔R2 (size 3)
// 未匹配左节点 U = ∅ → 全部覆盖 = {L0,L1,L2}...
// 等等，需要一个部分匹配的例子：L0,L1,L2,L3 vs R0,R1
// 边: L0-R0, L1-R0, L1-R1, L2-R1, L3-R1
// 最大匹配: L0↔R0, L1↔R1（size 2）
// 未匹配左节点: L2, L3
// 从 L2,L3 出发，非匹配边到 R1(已匹配)→匹配边回 L1→非匹配边到 R0(已匹配)→匹配边回 L0→无更多边
// Z_L = {L2, L3, L1, L0}, Z_R = {R1, R0}
// 覆盖 = (L\Z_L) ∪ (R∩Z_R) = ∅ ∪ {R0,R1} = {R0, R1}（大小 2）✓

const LW2 = 120, RW2 = 310
const LEFT = [
  { id: 0, label: 'L₀', x: LW2, y: 70  },
  { id: 1, label: 'L₁', x: LW2, y: 140 },
  { id: 2, label: 'L₂', x: LW2, y: 210 },
  { id: 3, label: 'L₃', x: LW2, y: 280 },
]
const RIGHT = [
  { id: 0, label: 'R₀', x: RW2, y: 100 },
  { id: 1, label: 'R₁', x: RW2, y: 235 },
]
const EDGES = [
  { id: 0, l: 0, r: 0 },
  { id: 1, l: 1, r: 0 },
  { id: 2, l: 1, r: 1 },
  { id: 3, l: 2, r: 1 },
  { id: 4, l: 3, r: 1 },
]
// 最大匹配
const MATCHING_EDGES = [ { l: 0, r: 0 }, { l: 1, r: 1 } ] // L0↔R0, L1↔R1
// König 构造
// U (unmatched left) = {L2, L3}
// Z alternating reachable:
//   L2 → R1 (non-match) → L1 (match) → R0 (non-match) → L0 (match) → no unvisited
//   L3 → R1 (already) 
// Z_L = {L2, L3, L1, L0}, Z_R = {R0, R1}
// Min vertex cover = (L \ Z_L) ∪ (R ∩ Z_R) = {} ∪ {R0,R1} = {R0, R1}
const Z_LEFT  = [0, 1, 2, 3]
const Z_RIGHT = [0, 1]
const COVER_LEFT  = [] as number[]  // L \ Z_L = {}
const COVER_RIGHT = [0, 1]         // R ∩ Z_R = {R0,R1}
const UNMATCH_LEFT = [2, 3]

type Tab = 'matching' | 'alternating' | 'cover' | 'theorem'

export default function KonigTheoremViz() {
  const [tab, setTab] = useState<Tab>('matching')
  const R = 18

  function edgeStroke(l: number, r: number) {
    const isMatch = MATCHING_EDGES.some(m => m.l===l && m.r===r)
    if (tab === 'matching') return isMatch ? '#10b981' : '#cbd5e1'
    if (tab === 'alternating') {
      if (!isMatch && (l===2||l===3||l===0||l===1)) {
        if ((l===2&&r===1)||(l===3&&r===1)||(l===1&&r===0)) return '#f59e0b'  // alternating non-match
        if (l===0&&r===0) return '#f59e0b'
      }
      if (isMatch && (l===0||l===1)) return '#e879f9'  // alternating match
      return '#cbd5e1'
    }
    if (tab === 'cover') {
      if (COVER_RIGHT.includes(r)) return '#10b981'
      return '#cbd5e1'
    }
    return '#cbd5e1'
  }
  function edgeWidth(l: number, r: number) {
    const isMatch = MATCHING_EDGES.some(m=>m.l===l&&m.r===r)
    if (tab==='matching' && isMatch) return 3.5
    if (tab==='alternating') return 2.5
    if (tab==='cover') return 2
    return 1.5
  }
  function edgeDash(l: number, r: number) {
    const isMatch = MATCHING_EDGES.some(m=>m.l===l&&m.r===r)
    if (tab==='alternating' && !isMatch) return '6 3'
    return undefined
  }

  function leftFill(id: number) {
    if (tab==='matching') return MATCHING_EDGES.some(m=>m.l===id)?'#10b981':UNMATCH_LEFT.includes(id)?'#f59e0b':'#64748b'
    if (tab==='alternating') return Z_LEFT.includes(id)?'#f59e0b':'#64748b'
    if (tab==='cover') return COVER_LEFT.includes(id)?'#ef4444':'#64748b'
    return '#64748b'
  }
  function rightFill(id: number) {
    if (tab==='matching') return MATCHING_EDGES.some(m=>m.r===id)?'#10b981':'#64748b'
    if (tab==='alternating') return Z_RIGHT.includes(id)?'#a78bfa':'#64748b'
    if (tab==='cover') return COVER_RIGHT.includes(id)?'#ef4444':'#64748b'
    return '#64748b'
  }

  const TABS = [
    { key: 'matching',    label: '① 最大匹配',     color: 'from-emerald-500 to-teal-500' },
    { key: 'alternating', label: '② 交错路可达集 Z', color: 'from-amber-500 to-orange-500' },
    { key: 'cover',       label: '③ 最小顶点覆盖',  color: 'from-rose-500 to-pink-500' },
    { key: 'theorem',     label: '④ 定理等式',       color: 'from-violet-500 to-indigo-500' },
  ] as const

  return (
    <div className="w-full max-w-2xl mx-auto rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 shadow-xl overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-rose-600 via-pink-600 to-fuchsia-600 px-5 py-4">
        <h3 className="text-white font-bold text-lg tracking-tight">König 定理：最大匹配 = 最小顶点覆盖</h3>
        <p className="text-pink-200 text-sm mt-0.5">二分图专属定理 · 构造性证明 · 视觉化推导</p>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/40">
        {TABS.map(t => (
          <button key={t.key} onClick={() => setTab(t.key as Tab)}
            className={`flex-1 px-2 py-2.5 text-[10px] font-bold transition-all ${
              tab===t.key
                ? `bg-gradient-to-b ${t.color} text-white`
                : 'text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-700'
            }`}>
            {t.label}
          </button>
        ))}
      </div>

      <div className="p-4 space-y-3">
        <div className="grid grid-cols-1 md:grid-cols-5 gap-3">
          {/* SVG */}
          {tab !== 'theorem' && (
            <div className="md:col-span-3 rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/30 overflow-hidden">
              <svg viewBox="50 30 360 290" className="w-full" style={{ maxHeight: 280 }}>
                {/* Column labels */}
                <text x={LW2} y={42} textAnchor="middle" fontSize={10} fontWeight="bold" fill="#8b5cf6">Left</text>
                <text x={RW2} y={42} textAnchor="middle" fontSize={10} fontWeight="bold" fill="#a78bfa">Right</text>

                {/* Edges */}
                {EDGES.map(e => {
                  const ln = LEFT[e.l], rn = RIGHT[e.r]
                  const isCoverEdge = tab==='cover' && COVER_RIGHT.includes(e.r)
                  return (
                    <g key={e.id}>
                      {isCoverEdge && (
                        <line x1={ln.x} y1={ln.y} x2={rn.x} y2={rn.y} stroke="#ef4444" strokeWidth={8} opacity={0.15}/>
                      )}
                      <line x1={ln.x} y1={ln.y} x2={rn.x} y2={rn.y}
                        stroke={edgeStroke(e.l,e.r)} strokeWidth={edgeWidth(e.l,e.r)}
                        strokeDasharray={edgeDash(e.l,e.r)} strokeLinecap="round"/>
                    </g>
                  )
                })}

                {/* Left nodes */}
                {LEFT.map(n => {
                  const isZ = tab==='alternating' && Z_LEFT.includes(n.id)
                  const isCover = COVER_LEFT.includes(n.id)
                  const isUnmatched = UNMATCH_LEFT.includes(n.id)
                  return (
                    <g key={`L${n.id}`}>
                      {isZ && <circle cx={n.x} cy={n.y} r={R+6} fill="#f59e0b" opacity={0.2}/>}
                      {isCover && <circle cx={n.x} cy={n.y} r={R+6} fill="#ef4444" opacity={0.3}/>}
                      <circle cx={n.x} cy={n.y} r={R} fill={leftFill(n.id)}/>
                      <text x={n.x} y={n.y+5} textAnchor="middle" fontSize={11} fontWeight="bold" fill="white">{n.label}</text>
                      {tab==='matching' && isUnmatched && (
                        <text x={n.x-28} y={n.y+4} textAnchor="middle" fontSize={8} fontWeight="bold" fill="#f59e0b">未匹配</text>
                      )}
                      {tab==='alternating' && isZ && (
                        <text x={n.x-26} y={n.y+4} textAnchor="middle" fontSize={8} fontWeight="bold" fill="#f59e0b">Z</text>
                      )}
                      {tab==='cover' && isCover && (
                        <text x={n.x-24} y={n.y+4} textAnchor="middle" fontSize={8} fontWeight="bold" fill="#ef4444">✓</text>
                      )}
                    </g>
                  )
                })}

                {/* Right nodes */}
                {RIGHT.map(n => {
                  const isZ = tab==='alternating' && Z_RIGHT.includes(n.id)
                  const isCover = COVER_RIGHT.includes(n.id)
                  return (
                    <g key={`R${n.id}`}>
                      {isZ && <circle cx={n.x} cy={n.y} r={R+6} fill="#a78bfa" opacity={0.2}/>}
                      {isCover && <circle cx={n.x} cy={n.y} r={R+6} fill="#ef4444" opacity={0.3}/>}
                      <circle cx={n.x} cy={n.y} r={R} fill={rightFill(n.id)}/>
                      <text x={n.x} y={n.y+5} textAnchor="middle" fontSize={11} fontWeight="bold" fill="white">{n.label}</text>
                      {tab==='alternating' && isZ && (
                        <text x={n.x+26} y={n.y+4} textAnchor="middle" fontSize={8} fontWeight="bold" fill="#a78bfa">Z</text>
                      )}
                      {tab==='cover' && isCover && (
                        <text x={n.x+24} y={n.y+4} textAnchor="middle" fontSize={8} fontWeight="bold" fill="#ef4444">✓</text>
                      )}
                    </g>
                  )
                })}

                {/* Alternating path arrows */}
                {tab==='alternating' && (
                  <g fontSize={8} fontWeight="bold">
                    <text x={215} y={80}  fill="#f59e0b" textAnchor="middle">非匹配→</text>
                    <text x={215} y={150} fill="#e879f9" textAnchor="middle">←匹配</text>
                    <text x={215} y={200} fill="#f59e0b" textAnchor="middle">非匹配→</text>
                    <text x={215} y={240} fill="#e879f9" textAnchor="middle">←匹配</text>
                  </g>
                )}
              </svg>
            </div>
          )}

          {/* Right panel / theorem panel */}
          <div className={`${tab==='theorem'?'col-span-5':'md:col-span-2'} space-y-2`}>
            {tab === 'matching' && (
              <>
                <div className="rounded-xl border border-emerald-200 dark:border-emerald-700/50 bg-emerald-50 dark:bg-emerald-900/20 p-3">
                  <div className="text-[10px] font-bold text-emerald-600 dark:text-emerald-400 uppercase tracking-wider mb-2">最大匹配 M（大小 = 2）</div>
                  {MATCHING_EDGES.map(({l,r}) => (
                    <div key={`${l}-${r}`} className="flex items-center gap-2 text-[11px] py-0.5">
                      <span className="w-2 h-2 rounded-full bg-emerald-500 shrink-0"/>
                      <span className="font-bold text-emerald-700 dark:text-emerald-300">{LEFT[l].label} ↔ {RIGHT[r].label}</span>
                    </div>
                  ))}
                </div>
                <div className="rounded-xl border border-amber-200 dark:border-amber-700/50 bg-amber-50 dark:bg-amber-900/20 p-3">
                  <div className="text-[10px] font-bold text-amber-600 dark:text-amber-400 uppercase tracking-wider mb-2">未匹配左节点 U</div>
                  <div className="flex gap-1">
                    {UNMATCH_LEFT.map(id => (
                      <span key={id} className="px-2 py-0.5 rounded text-[10px] font-bold bg-amber-200 dark:bg-amber-800 text-amber-800 dark:text-amber-200">{LEFT[id].label}</span>
                    ))}
                  </div>
                  <div className="text-[9px] text-slate-400 mt-1.5">König 构造从 U 出发寻找交错路径</div>
                </div>
              </>
            )}

            {tab === 'alternating' && (
              <div className="space-y-2">
                <div className="rounded-xl border border-amber-200 dark:border-amber-700/50 bg-amber-50 dark:bg-amber-900/20 p-3">
                  <div className="text-[10px] font-bold text-amber-600 dark:text-amber-400 uppercase tracking-wider mb-2">Z 集合（交错可达）</div>
                  <div className="text-[10px] text-slate-600 dark:text-slate-300 mb-1.5">从 U={'{'}L₂,L₃{'}'} 出发，沿非匹配边→匹配边→…探索：</div>
                  <div className="space-y-0.5 text-[10px]">
                    <div className="text-amber-600">L₂ →<span className="text-slate-400">（非匹配）</span>→ R₁</div>
                    <div className="text-fuchsia-600">R₁ →<span className="text-slate-400">（匹配）</span>→ L₁</div>
                    <div className="text-amber-600">L₁ →<span className="text-slate-400">（非匹配）</span>→ R₀</div>
                    <div className="text-fuchsia-600">R₀ →<span className="text-slate-400">（匹配）</span>→ L₀</div>
                    <div className="text-slate-400">L₃ →<span className="text-slate-400">（非匹配）</span>→ R₁（已访问）</div>
                  </div>
                </div>
                <div className="rounded-xl border border-purple-200 dark:border-purple-700/50 bg-purple-50 dark:bg-purple-900/20 p-2.5">
                  <div className="text-[10px] font-bold text-purple-600 dark:text-purple-400 uppercase tracking-wider mb-1.5">Z 结果</div>
                  <div className="text-[10px]">
                    <div className="text-amber-600 font-bold">Z_L = {'{'} L₀, L₁, L₂, L₃ {'}'}</div>
                    <div className="text-purple-600 font-bold">Z_R = {'{'} R₀, R₁ {'}'}</div>
                  </div>
                </div>
              </div>
            )}

            {tab === 'cover' && (
              <div className="space-y-2">
                <div className="rounded-xl border border-rose-200 dark:border-rose-700/50 bg-rose-50 dark:bg-rose-900/20 p-3">
                  <div className="text-[10px] font-bold text-rose-600 dark:text-rose-400 uppercase tracking-wider mb-2">最小顶点覆盖构造</div>
                  <div className="text-[10px] space-y-1 text-slate-600 dark:text-slate-300">
                    <div>覆盖 = <span className="font-bold">(L \ Z_L) ∪ (R ∩ Z_R)</span></div>
                    <div>L \ Z_L = {'{'} L₀,L₁,L₂,L₃ {'}'} \ {'{'} L₀,L₁,L₂,L₃ {'}'} = <span className="font-bold text-slate-400">∅</span></div>
                    <div>R ∩ Z_R = {'{'} R₀,R₁ {'}'} ∩ {'{'} R₀,R₁ {'}'} = <span className="font-bold text-rose-600">{'{'} R₀, R₁ {'}'}</span></div>
                  </div>
                </div>
                <div className="rounded-xl border border-emerald-200 dark:border-emerald-700/50 bg-emerald-50 dark:bg-emerald-900/20 p-2.5">
                  <div className="text-[10px] font-bold text-emerald-600 dark:text-emerald-400 uppercase mb-1.5">验证：所有边被覆盖？</div>
                  {EDGES.map(e => (
                    <div key={e.id} className="text-[10px] flex items-center gap-1">
                      <span className="text-emerald-500 font-bold">✓</span>
                      <span className="text-slate-600 dark:text-slate-300">
                        {LEFT[e.l].label}–{RIGHT[e.r].label} 被 {RIGHT[e.r].label} 覆盖
                      </span>
                    </div>
                  ))}
                </div>
                <div className="rounded-lg bg-rose-100 dark:bg-rose-900/30 border border-rose-300 dark:border-rose-700 p-2 text-center">
                  <div className="text-[11px] font-black text-rose-700 dark:text-rose-300">|M| = |C| = 2</div>
                  <div className="text-[9px] text-rose-500 mt-0.5">最大匹配大小 = 最小顶点覆盖大小 ✓</div>
                </div>
              </div>
            )}

            {tab === 'theorem' && (
              <div className="max-w-2xl mx-auto space-y-4">
                {/* Theorem statement */}
                <div className="rounded-2xl border-2 border-violet-300 dark:border-violet-600 bg-gradient-to-br from-violet-50 to-indigo-50 dark:from-violet-900/30 dark:to-indigo-900/30 p-6 text-center">
                  <div className="text-[10px] font-bold text-violet-500 uppercase tracking-widest mb-3">König 定理（1931）</div>
                  <div className="text-2xl font-black text-violet-700 dark:text-violet-200 mb-2">
                    ν(G) = τ(G)
                  </div>
                  <div className="text-[11px] text-slate-500 dark:text-slate-400">
                    在任意二分图 G 中，<span className="font-bold text-violet-600 dark:text-violet-300">最大匹配数 ν(G)</span> 等于{' '}
                    <span className="font-bold text-rose-600 dark:text-rose-400">最小顶点覆盖数 τ(G)</span>
                  </div>
                </div>

                {/* Proof sketch */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-[11px]">
                  <div className="rounded-xl border border-emerald-200 dark:border-emerald-700/50 bg-emerald-50 dark:bg-emerald-900/20 p-3">
                    <div className="text-[10px] font-bold text-emerald-600 dark:text-emerald-400 mb-2">① 计算最大匹配 M</div>
                    <div className="text-slate-600 dark:text-slate-300">用 Hopcroft-Karp 等算法得到最大匹配 M，大小 = ν(G)。</div>
                  </div>
                  <div className="rounded-xl border border-amber-200 dark:border-amber-700/50 bg-amber-50 dark:bg-amber-900/20 p-3">
                    <div className="text-[10px] font-bold text-amber-600 dark:text-amber-400 mb-2">② 构造可达集 Z</div>
                    <div className="text-slate-600 dark:text-slate-300">从未匹配左节点 U 出发，沿「非匹配→匹配→…」路径 BFS/DFS，标记交错可达节点为 Z。</div>
                  </div>
                  <div className="rounded-xl border border-rose-200 dark:border-rose-700/50 bg-rose-50 dark:bg-rose-900/20 p-3">
                    <div className="text-[10px] font-bold text-rose-600 dark:text-rose-400 mb-2">③ 取覆盖 C</div>
                    <div className="text-slate-600 dark:text-slate-300">C = (L \ Z_L) ∪ (R ∩ Z_R)。可证 C 是合法顶点覆盖且 |C| = |M|。</div>
                  </div>
                </div>

                {/* Comparison with non-bipartite */}
                <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/30 p-4">
                  <div className="text-[10px] font-bold text-slate-500 uppercase mb-2">为何只适用于二分图？</div>
                  <div className="text-[11px] text-slate-600 dark:text-slate-300 space-y-1">
                    <div>• 一般图中 <span className="font-bold">ν(G) ≤ τ(G)</span>（弱对偶性，永远成立）</div>
                    <div>• 带奇数环的图：ν = 1（三角形）但 τ = 2，等号不成立</div>
                    <div>• König 定理 + LP 对偶性 → 二分图匹配的整数最优性证明</div>
                    <div>• 推论：二分图的最大独立集 = n − ν(G)（反向补集构造）</div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Tab description row */}
        {tab !== 'theorem' && (
          <div className="rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/30 px-3 py-2 text-[11px] text-slate-600 dark:text-slate-300">
            {tab === 'matching' && '步骤 ①：先用 Hopcroft-Karp 计算最大匹配 M（绿色边）。本例最大匹配 |M|=2：L₀↔R₀, L₁↔R₁。左侧 L₂, L₃ 未被匹配（橙色标注），将作为 König 构造的起点 U。'}
            {tab === 'alternating' && '步骤 ②：从 U={L₂,L₃} 出发，交替走「非匹配边（实线橙色）→ 匹配边（虚线紫色）」，DFS/BFS 标记所有可到达节点为 Z 集合（左侧 Z_L，右侧 Z_R）。'}
            {tab === 'cover' && '步骤 ③：取最小顶点覆盖 C = (L\\ Z_L) ∪ (R ∩ Z_R) = {R₀, R₁}（红色高亮节点）。验证：每条边至少有一个端点在 C 中 ✓。|C|=2 = |M|，König 定理得证。'}
          </div>
        )}
      </div>
    </div>
  )
}
