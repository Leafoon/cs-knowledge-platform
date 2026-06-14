'use client'

import { useState } from 'react'

// The 7-point (8-point) lemma visualization
// In a delta x 2delta rectangle, at most 8 points can be placed such that
// any two are >= delta apart. This is split into a 2x4 grid, each cell delta/2 x delta/2.

const CELL_COLS = 2
const CELL_ROWS = 4
const CELL_W = 80
const CELL_H = 55

const BOX_W = CELL_COLS * CELL_W // 160
const BOX_H = CELL_ROWS * CELL_H // 220

const SVG_W = 380
const SVG_H = 300
const OX = 110
const OY = 35

// Default positions: one sample point per cell (centers)
const DEFAULT_POINTS = [
  { row: 0, col: 0 }, { row: 0, col: 1 },
  { row: 1, col: 0 }, { row: 1, col: 1 },
  { row: 2, col: 0 }, { row: 2, col: 1 },
  { row: 3, col: 0 }, { row: 3, col: 1 },
]

function cellToSvg(row: number, col: number, ox: number = 0.5, oy: number = 0.5): { x: number; y: number } {
  return {
    x: OX + col * CELL_W + ox * CELL_W,
    y: OY + row * CELL_H + oy * CELL_H,
  }
}

function ptDist(a: { x: number; y: number }, b: { x: number; y: number }) {
  return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)
}

const DEMO_MODES = [
  { label: '满格（8点）', desc: '标准证明：将 δ×2δ 矩形分为 2×4 = 8 个 (δ/2)×(δ/2) 小方格，每格至多放 1 点（鸽巢原理），故至多 8 点。', points: DEFAULT_POINTS },
  { label: '删去 1 点（7点）', desc: '实际算法中，Strip 内参照点自身不算（或两侧各取半），因此有效邻居最多 7 个，保证线性扫描。', points: DEFAULT_POINTS.slice(0, 7) },
  { label: '对角极值', desc: '极端情形：8 点分布于方格对角处，验证任意两点距离 ≥ δ（在同一侧）或需跨越 δ（跨中线）。', points: [
    {row:0,col:0},{row:0,col:1},{row:1,col:0},{row:1,col:1},
    {row:2,col:0},{row:2,col:1},{row:3,col:0},{row:3,col:1},
  ].map((p, i) => ({ ...p, col: i % 2 === 0 ? 0 : 1 })) },
]

export function SevenPointLemma() {
  const [mode, setMode] = useState(0)
  const [highlightPair, setHighlightPair] = useState<[number, number] | null>(null)
  const demo = DEMO_MODES[mode]
  const pts = demo.points.map((p, i) => ({
    id: i,
    ...cellToSvg(p.row, p.col, 0.5 + (i % 3 - 1) * 0.15, 0.35 + (i % 2) * 0.3),
    row: p.row, col: p.col,
  }))

  const pairDists = pts.map((a, i) =>
    pts.slice(i + 1).map((b, jj) => ({
      i, j: i + 1 + jj,
      d: ptDist(a, b),
      ax: a.x, ay: a.y, bx: b.x, by: b.y,
    }))
  ).flat().sort((a, b) => a.d - b.d)

  const highlighted = highlightPair !== null ? pairDists.find(p => p.i === highlightPair[0] && p.j === highlightPair[1]) : null
  const delta = CELL_W  // visual delta = cell width
  // Distance in "real units": 1 cell = δ/2, so real dist = svgDist / (CELL_W / (delta/2)) = svgDist * 2 / CELL_W * 0.5
  const toRealDelta = (d: number) => (d / CELL_W).toFixed(2) + 'δ'

  return (
    <div className="rounded-2xl border border-amber-200 dark:border-amber-700 overflow-hidden shadow-lg font-sans">
      <div className="bg-gradient-to-r from-amber-500 to-yellow-400 px-5 py-4">
        <h3 className="text-white font-bold text-base">🔒 七点引理可视化</h3>
        <p className="text-amber-50 text-xs mt-0.5">
          在宽 δ、高 2δ 的矩形（Strip 横截面）中，间距 ≥ δ 的点最多只能有 8 个
        </p>
        <div className="flex gap-2 mt-3 flex-wrap">
          {DEMO_MODES.map((m, i) => (
            <button key={i} onClick={() => { setMode(i); setHighlightPair(null) }}
              className={`px-2.5 py-1 text-xs rounded-lg transition-colors ${mode===i?'bg-white text-amber-700 font-bold':'bg-white/25 text-white hover:bg-white/35'}`}>
              {m.label}
            </button>
          ))}
        </div>
      </div>

      <div className="bg-white dark:bg-slate-900 p-4">
        <p className="text-xs text-slate-500 dark:text-slate-400 mb-3 leading-relaxed">{demo.desc}</p>

        <div className="flex gap-4 flex-wrap items-start">
          {/* SVG */}
          <svg width={SVG_W} height={SVG_H} className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 flex-shrink-0">
            {/* δ annotation */}
            <text x={OX - 5} y={OY + BOX_H/2} textAnchor="end" fontSize={12} fill="#f59e0b" fontWeight="bold" dominantBaseline="central">2δ</text>
            <line x1={OX-18} y1={OY} x2={OX-18} y2={OY+BOX_H} stroke="#f59e0b" strokeWidth={1.5} markerEnd="url(#arr)"/>
            <text x={OX + BOX_W/2} y={OY - 10} textAnchor="middle" fontSize={12} fill="#f59e0b" fontWeight="bold">δ</text>
            <line x1={OX} y1={OY-18} x2={OX+BOX_W} y2={OY-18} stroke="#f59e0b" strokeWidth={1.5}/>

            {/* Grid cells */}
            {Array.from({ length: CELL_ROWS }, (_, row) =>
              Array.from({ length: CELL_COLS }, (_, col) => (
                <rect key={`${row}-${col}`}
                  x={OX + col * CELL_W} y={OY + row * CELL_H}
                  width={CELL_W} height={CELL_H}
                  fill={`hsl(${(row * CELL_COLS + col) * 45}, 80%, 96%)`}
                  className="dark:opacity-20"
                  stroke="#d1d5db" strokeWidth={1}/>
              ))
            )}
            {/* Outer rectangle (delta x 2delta) */}
            <rect x={OX} y={OY} width={BOX_W} height={BOX_H} fill="none" stroke="#f59e0b" strokeWidth={2.5} rx={3}/>

            {/* Cell labels */}
            {Array.from({ length: CELL_ROWS }, (_, row) =>
              Array.from({ length: CELL_COLS }, (_, col) => (
                <text key={`lbl-${row}-${col}`}
                  x={OX + col * CELL_W + CELL_W / 2}
                  y={OY + row * CELL_H + 10}
                  textAnchor="middle" fontSize={8.5}
                  fill="#94a3b8">
                  {String.fromCharCode(65 + row * CELL_COLS + col)}
                </text>
              ))
            )}

            {/* Highlighted pair line */}
            {highlighted && (
              <line x1={highlighted.ax} y1={highlighted.ay} x2={highlighted.bx} y2={highlighted.by}
                stroke="#ef4444" strokeWidth={2} strokeDasharray="5,3" opacity={0.8}/>
            )}

            {/* Points */}
            {pts.map(p => (
              <g key={p.id}>
                <circle cx={p.x} cy={p.y} r={10} fill="#f59e0b" stroke="white" strokeWidth={2}
                  style={{ cursor: 'pointer' }}
                  onClick={() => setHighlightPair(null)}/>
                <text x={p.x} y={p.y} textAnchor="middle" dominantBaseline="central"
                  fontSize={9} fill="white" fontWeight="bold">{p.id + 1}</text>
              </g>
            ))}

            {/* Mid line label */}
            <text x={OX + BOX_W + 8} y={OY + BOX_H/4} fontSize={9} fill="#64748b">每格</text>
            <text x={OX + BOX_W + 8} y={OY + BOX_H/4 + 12} fontSize={9} fill="#64748b">δ/2</text>
            <text x={OX + BOX_W + 8} y={OY + BOX_H/4 + 24} fontSize={9} fill="#64748b">×δ/2</text>
          </svg>

          {/* Right panel */}
          <div className="flex-1 min-w-[170px] space-y-3">
            {/* Count badge */}
            <div className={`rounded-xl p-3 text-center border ${
              pts.length <= 8
                ? 'bg-emerald-50 dark:bg-emerald-900/10 border-emerald-200 dark:border-emerald-800'
                : 'bg-rose-50 dark:bg-rose-900/10 border-rose-200 dark:border-rose-800'
            }`}>
              <span className={`text-3xl font-black ${pts.length <= 8 ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600'}`}>
                {pts.length}
              </span>
              <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
                个点 {pts.length <= 8 ? '≤ 8 ✓' : '> 8 矛盾!'}
              </p>
            </div>

            {/* Distances */}
            <div>
              <p className="text-xs font-bold text-slate-600 dark:text-slate-300 mb-1.5">点对距离（前5近）</p>
              <div className="space-y-1">
                {pairDists.slice(0, 5).map((pd, i) => (
                  <button key={i}
                    onClick={() => setHighlightPair(highlightPair?.[0]===pd.i && highlightPair?.[1]===pd.j ? null : [pd.i, pd.j])}
                    className={`w-full flex items-center justify-between px-3 py-1.5 rounded-lg text-xs transition-colors ${
                      highlightPair?.[0]===pd.i && highlightPair?.[1]===pd.j
                        ? 'bg-rose-100 dark:bg-rose-900/30 text-rose-700 dark:text-rose-300'
                        : 'bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-600'
                    }`}>
                    <span className="font-mono">点{pd.i+1} — 点{pd.j+1}</span>
                    <span className="font-bold">{toRealDelta(pd.d)}</span>
                  </button>
                ))}
              </div>
            </div>

            {/* Proof sketch */}
            <div className="rounded-xl bg-amber-50 dark:bg-amber-900/10 border border-amber-200 dark:border-amber-800 p-3 text-[11px] text-amber-700 dark:text-amber-300 space-y-1.5 leading-relaxed">
              <p className="font-bold">引理证明思路：</p>
              <p>① 划分为 {CELL_ROWS}×{CELL_COLS} = 8 个 (δ/2)×(δ/2) 小格</p>
              <p>② 同一格内两点距离 ≤ √2·δ/2 ≈ 0.707δ &lt; δ</p>
              <p>③ 若间距要 ≥ δ，每格至多 1 点（鸽巢原理）</p>
              <p>④ 故 Strip 内满足条件的点最多 <strong>8</strong> 个</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
