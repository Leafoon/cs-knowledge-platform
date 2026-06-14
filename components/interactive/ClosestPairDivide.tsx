'use client'

import { useState } from 'react'

interface Pt { x: number; y: number; id: number; side: 'left' | 'right' | 'strip' }

const W = 380, H = 280

const DATASETS = [
  {
    label: '标准集 (n=12)',
    pts: [
      {x:40,y:60},{x:70,y:200},{x:110,y:100},{x:150,y:50},{x:160,y:220},
      {x:190,y:130},{x:220,y:70},{x:250,y:200},{x:270,y:130},{x:300,y:60},
      {x:320,y:210},{x:350,y:130},
    ],
  },
  {
    label: '左密右稀 (n=10)',
    pts: [
      {x:30,y:80},{x:50,y:150},{x:70,y:60},{x:90,y:200},{x:110,y:110},
      {x:130,y:180},{x:250,y:100},{x:300,y:180},{x:340,y:60},{x:360,y:220},
    ],
  },
]

function dist(a: {x:number;y:number}, b: {x:number;y:number}): number {
  return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)
}

function closestPair(pts: {x:number;y:number}[]): { i: number; j: number; d: number } {
  let best = { i: 0, j: 1, d: dist(pts[0], pts[1]) }
  for (let i = 0; i < pts.length; i++)
    for (let j = i+1; j < pts.length; j++) {
      const d = dist(pts[i], pts[j]); if (d < best.d) best = { i, j, d }
    }
  return best
}

export function ClosestPairDivide() {
  const [dsIdx, setDsIdx] = useState(0)
  const [showStrip, setShowStrip] = useState(true)
  const [showBest, setShowBest] = useState(true)
  const dataset = DATASETS[dsIdx]
  const raw = dataset.pts

  // Sort by x, find mid line
  const sortedX = [...raw].sort((a,b) => a.x - b.x)
  const mid = Math.floor(sortedX.length / 2)
  const midX = sortedX[mid].x

  // Assign sides
  const leftPts = sortedX.slice(0, mid)
  const rightPts = sortedX.slice(mid)

  const leftBest = leftPts.length >= 2 ? closestPair(leftPts) : null
  const rightBest = rightPts.length >= 2 ? closestPair(rightPts) : null

  const dLeft  = leftBest  ? leftBest.d  : Infinity
  const dRight = rightBest ? rightBest.d : Infinity
  const delta  = Math.min(dLeft, dRight)

  // Strip points
  const stripPts = raw.filter(p => Math.abs(p.x - midX) < delta)

  // Best pair overall (for display)
  const allBest = closestPair(raw)
  const bA = raw[allBest.i], bB = raw[allBest.j]

  // Label points
  const labeled: Pt[] = raw.map((p, i) => ({
    ...p, id: i,
    side: Math.abs(p.x - midX) < delta ? 'strip' : p.x <= midX ? 'left' : 'right',
  }))

  return (
    <div className="rounded-2xl border border-emerald-200 dark:border-emerald-700 overflow-hidden shadow-lg font-sans">
      <div className="bg-gradient-to-r from-emerald-600 to-teal-500 px-5 py-4">
        <h3 className="text-white font-bold text-base">✂️ 最近点对：分治可视化</h3>
        <p className="text-emerald-50 text-xs mt-0.5">
          展示分治线、左/右子问题最优解 δ、以及宽度 2δ 的带形区域跨界点检测
        </p>
        <div className="flex gap-2 mt-3 flex-wrap">
          {DATASETS.map((d,i) => (
            <button key={i} onClick={() => setDsIdx(i)}
              className={`px-2.5 py-1 text-xs rounded-lg transition-colors ${dsIdx===i?'bg-white text-emerald-700 font-bold':'bg-white/20 text-white hover:bg-white/30'}`}>
              {d.label}
            </button>
          ))}
        </div>
      </div>

      <div className="bg-white dark:bg-slate-900 p-4">
        {/* Toggles */}
        <div className="flex gap-2 mb-3 text-xs flex-wrap">
          <button onClick={() => setShowStrip(v => !v)}
            className={`px-3 py-1.5 rounded-lg font-medium transition-colors ${showStrip?'bg-amber-500 text-white':'bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-300'}`}>
            {showStrip ? '✓' : ''} 显示 Strip (2δ)
          </button>
          <button onClick={() => setShowBest(v => !v)}
            className={`px-3 py-1.5 rounded-lg font-medium transition-colors ${showBest?'bg-rose-500 text-white':'bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-300'}`}>
            {showBest ? '✓' : ''} 显示最近点对
          </button>
        </div>

        <div className="flex gap-4 flex-wrap items-start">
          <svg width={W} height={H} className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 flex-shrink-0">
            {/* Strip region */}
            {showStrip && (
              <rect x={midX - delta} y={0} width={delta * 2} height={H}
                fill="#f59e0b18" stroke="#f59e0b" strokeWidth={1} strokeDasharray="5,3" />
            )}

            {/* Divide line */}
            <line x1={midX} y1={0} x2={midX} y2={H} stroke="#64748b" strokeWidth={2} strokeDasharray="6,4" />
            <text x={midX + 4} y={14} fontSize={10} fill="#64748b" fontWeight="bold">mid x={midX}</text>

            {/* Left best pair */}
            {showBest && leftBest && leftPts.length >= 2 && (
              <line x1={leftPts[leftBest.i].x} y1={leftPts[leftBest.i].y}
                    x2={leftPts[leftBest.j].x} y2={leftPts[leftBest.j].y}
                stroke="#3b82f6" strokeWidth={2.5} strokeDasharray="4,2" strokeLinecap="round"/>
            )}
            {/* Right best pair */}
            {showBest && rightBest && rightPts.length >= 2 && (
              <line x1={rightPts[rightBest.i].x} y1={rightPts[rightBest.i].y}
                    x2={rightPts[rightBest.j].x} y2={rightPts[rightBest.j].y}
                stroke="#f97316" strokeWidth={2.5} strokeDasharray="4,2" strokeLinecap="round"/>
            )}
            {/* Global best pair */}
            {showBest && (
              <g>
                <line x1={bA.x} y1={bA.y} x2={bB.x} y2={bB.y} stroke="#ef4444" strokeWidth={3} strokeLinecap="round"/>
                <circle cx={(bA.x+bB.x)/2} cy={(bA.y+bB.y)/2} r={18} fill="#ef444410" stroke="#ef4444" strokeWidth={1} strokeDasharray="4,3"/>
              </g>
            )}

            {/* Points */}
            {labeled.map(p => {
              const fill = p.side === 'left' ? '#3b82f6' : p.side === 'right' ? '#f97316' : '#f59e0b'
              const r = p.side === 'strip' ? 7 : 5
              return (
                <g key={p.id}>
                  {p.side === 'strip' && <circle cx={p.x} cy={p.y} r={13} fill="#f59e0b10"/>}
                  <circle cx={p.x} cy={p.y} r={r} fill={fill} stroke="white" strokeWidth={1.5}/>
                  <text x={p.x+7} y={p.y-6} fontSize={8} fill={fill} fontWeight="bold">P{p.id}</text>
                </g>
              )
            })}
          </svg>

          {/* Stats panel */}
          <div className="flex-1 min-w-[170px] space-y-2.5 text-xs">
            <div className="rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
              <div className="bg-slate-50 dark:bg-slate-800 px-3 py-2 text-[10px] font-bold text-slate-500 dark:text-slate-400 uppercase">分治统计</div>
              {[
                {label:'总点数 n', val:`${raw.length}`, c:'text-slate-700 dark:text-slate-200'},
                {label:'左子集 |L|', val:`${leftPts.length} 点`, c:'text-blue-600 dark:text-blue-400'},
                {label:'δ_left', val:leftBest?`${leftBest.d.toFixed(2)}`:'N/A', c:'text-blue-600 dark:text-blue-400'},
                {label:'右子集 |R|', val:`${rightPts.length} 点`, c:'text-orange-600 dark:text-orange-400'},
                {label:'δ_right', val:rightBest?`${rightBest.d.toFixed(2)}`:'N/A', c:'text-orange-600 dark:text-orange-400'},
                {label:'δ = min(δL, δR)', val:delta.toFixed(2), c:'text-amber-600 dark:text-amber-400 font-bold'},
                {label:'Strip 内点数', val:`${stripPts.length}`, c:'text-amber-600 dark:text-amber-400'},
                {label:'全局最近距离', val:allBest.d.toFixed(2), c:'text-rose-600 dark:text-rose-400 font-bold'},
              ].map((r,i) => (
                <div key={i} className={`flex justify-between items-center px-3 py-1.5 border-t border-slate-100 dark:border-slate-700 ${i%2===1?'bg-slate-50/50 dark:bg-slate-800/30':''}`}>
                  <span className="text-slate-500 dark:text-slate-400">{r.label}</span>
                  <span className={`font-mono font-bold ${r.c}`}>{r.val}</span>
                </div>
              ))}
            </div>

            <div className="rounded-xl bg-amber-50 dark:bg-amber-900/10 border border-amber-200 dark:border-amber-800 p-3 text-[11px] text-amber-700 dark:text-amber-300 leading-relaxed">
              <strong>七点引理：</strong>带形 Strip 内每个点最多需检查 7 个相邻点（按 y 排序），保证 O(n) 合并步骤，总复杂度 O(n log n)。
            </div>

            <div className="flex gap-1.5 text-[10px] flex-wrap">
              {[{c:'#3b82f6',l:'左侧点'},{c:'#f97316',l:'右侧点'},{c:'#f59e0b',l:'Strip 内'},{c:'#ef4444',l:'全局最近对'}].map(({c,l})=>(
                <span key={l} className="flex items-center gap-1 bg-slate-100 dark:bg-slate-700 px-2 py-1 rounded-lg text-slate-600 dark:text-slate-300">
                  <span className="w-2.5 h-2.5 rounded-full" style={{background:c}}/>{l}
                </span>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
