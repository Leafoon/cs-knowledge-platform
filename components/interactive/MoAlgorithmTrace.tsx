'use client'

import { useState } from 'react'

const N = 12
const BLOCK_SIZE = Math.floor(Math.sqrt(N))

interface Query { l: number; r: number; idx: number }

const QUERIES_RAW: Query[] = [
  { l: 0,  r: 5,  idx: 0 },
  { l: 7,  r: 11, idx: 1 },
  { l: 1,  r: 4,  idx: 2 },
  { l: 8,  r: 10, idx: 3 },
  { l: 3,  r: 9,  idx: 4 },
  { l: 0,  r: 3,  idx: 5 },
  { l: 6,  r: 11, idx: 6 },
  { l: 2,  r: 7,  idx: 7 },
]

function moSort(qs: Query[]): Query[] {
  return [...qs].sort((a, b) => {
    const ba = Math.floor(a.l / BLOCK_SIZE)
    const bb = Math.floor(b.l / BLOCK_SIZE)
    if (ba !== bb) return ba - bb
    return ba % 2 === 0 ? a.r - b.r : b.r - a.r
  })
}

const SORTED = moSort(QUERIES_RAW)

// Color per L-block
const BLOCK_COLORS = ['#6366f1', '#f97316', '#10b981', '#ec4899', '#f59e0b']
function colorForBlock(l: number) { return BLOCK_COLORS[Math.floor(l / BLOCK_SIZE) % BLOCK_COLORS.length] }

export function MoAlgorithmTrace() {
  const [step, setStep] = useState<number | null>(null)  // which query is highlighted
  const [mode, setMode] = useState<'original' | 'sorted'>('original')

  const queries = mode === 'original' ? QUERIES_RAW : SORTED
  const gridSize = 220
  const margin = 28
  const scale = (gridSize - 2 * margin) / N

  // Compute total pointer moves for each ordering
  function totalMoves(qs: Query[]) {
    let total = 0, cl = 0, cr = 0
    for (const q of qs) { total += Math.abs(q.l - cl) + Math.abs(q.r - cr); cl = q.l; cr = q.r }
    return total
  }
  const origMoves = totalMoves(QUERIES_RAW)
  const sortMoves = totalMoves(SORTED)

  return (
    <div className="rounded-2xl border border-yellow-200 dark:border-yellow-800 overflow-hidden shadow-lg font-sans">
      <div className="bg-gradient-to-r from-yellow-500 to-amber-400 px-5 py-4">
        <h3 className="text-white font-bold text-base">🚛 Mo 算法：查询排序 vs 原始顺序</h3>
        <p className="text-yellow-50 text-xs mt-0.5">
          n={N}, 块大小={BLOCK_SIZE}，{QUERIES_RAW.length} 个区间查询。悬停查询点查看详情
        </p>
        <div className="flex gap-2 mt-3">
          {(['original', 'sorted'] as const).map(m => (
            <button key={m} onClick={() => setMode(m)}
              className={`px-3 py-1 text-xs rounded-lg transition-colors ${
                mode === m ? 'bg-white text-yellow-700 font-bold' : 'bg-white/25 text-white hover:bg-white/35'
              }`}>
              {m === 'original' ? '原始顺序' : 'Mo 排序后'}
            </button>
          ))}
        </div>
      </div>

      <div className="bg-white dark:bg-slate-900 p-5 space-y-4">
        {/* L-R scatter plot */}
        <div className="flex gap-4 flex-wrap">
          <div className="flex-shrink-0">
            <p className="text-xs font-semibold text-slate-600 dark:text-slate-300 mb-2">
              (L, R) 散点图 — 每点是一个区间查询
            </p>
            <svg width={gridSize} height={gridSize} className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800">
              {/* Grid lines */}
              {Array.from({ length: N + 1 }, (_, i) => (
                <g key={i}>
                  <line x1={margin + i * scale} y1={margin} x2={margin + i * scale} y2={gridSize - margin} stroke="#e2e8f0" strokeWidth={0.5} />
                  <line x1={margin} y1={margin + i * scale} x2={gridSize - margin} y2={margin + i * scale} stroke="#e2e8f0" strokeWidth={0.5} />
                </g>
              ))}
              {/* Block vertical separators */}
              {Array.from({ length: Math.ceil(N / BLOCK_SIZE) + 1 }, (_, bi) => (
                <line key={bi}
                  x1={margin + bi * BLOCK_SIZE * scale} y1={margin}
                  x2={margin + bi * BLOCK_SIZE * scale} y2={gridSize - margin}
                  stroke={BLOCK_COLORS[bi % BLOCK_COLORS.length]} strokeWidth={1.5} strokeDasharray="3 2" />
              ))}
              {/* Axes labels */}
              <text x={gridSize / 2} y={gridSize - 4} textAnchor="middle" fontSize={9} fill="#94a3b8">L (左端点)</text>
              <text x={8} y={gridSize / 2} textAnchor="middle" fontSize={9} fill="#94a3b8" transform={`rotate(-90, 8, ${gridSize/2})`}>R (右端点)</text>
              {/* Trajectory line */}
              {queries.map((q, i) => i > 0 && (
                <line key={i}
                  x1={margin + queries[i-1].l * scale} y1={gridSize - margin - queries[i-1].r * scale}
                  x2={margin + q.l * scale} y2={gridSize - margin - q.r * scale}
                  stroke="#94a3b8" strokeWidth={1} strokeDasharray="4 2" />
              ))}
              {/* Points */}
              {queries.map((q, i) => {
                const cx = margin + q.l * scale
                const cy = gridSize - margin - q.r * scale
                const color = colorForBlock(q.l)
                const isSel = step === i
                return (
                  <g key={q.idx}
                    onMouseEnter={() => setStep(i)}
                    onMouseLeave={() => setStep(null)}
                    style={{ cursor: 'pointer' }}>
                    <circle cx={cx} cy={cy} r={isSel ? 8 : 5}
                      fill={color} stroke="white" strokeWidth={2}
                      style={{ transition: 'r 0.15s' }} />
                    <text x={cx + 8} y={cy - 4} fontSize={7} fill={color} fontWeight="bold">
                      {i + 1}
                    </text>
                  </g>
                )
              })}
            </svg>
          </div>

          {/* Query list */}
          <div className="flex-1 min-w-[180px]">
            <p className="text-xs font-semibold text-slate-600 dark:text-slate-300 mb-2">查询列表</p>
            <div className="space-y-1 max-h-[200px] overflow-y-auto pr-1">
              {queries.map((q, i) => {
                const color = colorForBlock(q.l)
                const isSel = step === i
                return (
                  <div key={q.idx}
                    onMouseEnter={() => setStep(i)}
                    onMouseLeave={() => setStep(null)}
                    className={`flex items-center gap-2 px-2 py-1.5 rounded-lg text-xs cursor-pointer transition-all border ${
                      isSel
                        ? 'border-slate-400 dark:border-slate-500 bg-slate-100 dark:bg-slate-700'
                        : 'border-transparent bg-slate-50 dark:bg-slate-800 hover:bg-slate-100 dark:hover:bg-slate-700'
                    }`}>
                    <span className="w-5 h-5 rounded-full flex items-center justify-center text-[10px] text-white font-bold flex-shrink-0"
                      style={{ backgroundColor: color }}>
                      {i + 1}
                    </span>
                    <span className="font-mono text-slate-700 dark:text-slate-300">
                      [{q.l}, {q.r}]
                    </span>
                    <span className="text-[10px] text-slate-400 ml-auto">
                      B{Math.floor(q.l / BLOCK_SIZE)}
                    </span>
                  </div>
                )
              })}
            </div>
          </div>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-2 gap-3">
          <div className={`rounded-xl border p-3 text-center transition-all ${
            mode === 'original'
              ? 'border-red-200 dark:border-red-800 bg-red-50 dark:bg-red-900/10'
              : 'border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50 opacity-60'
          }`}>
            <p className="text-2xl font-bold text-red-600 dark:text-red-400">{origMoves}</p>
            <p className="text-xs text-slate-500 dark:text-slate-400">原始顺序总移动步数</p>
          </div>
          <div className={`rounded-xl border p-3 text-center transition-all ${
            mode === 'sorted'
              ? 'border-emerald-200 dark:border-emerald-800 bg-emerald-50 dark:bg-emerald-900/10'
              : 'border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50 opacity-60'
          }`}>
            <p className="text-2xl font-bold text-emerald-600 dark:text-emerald-400">{sortMoves}</p>
            <p className="text-xs text-slate-500 dark:text-slate-400">Mo 排序后总移动步数</p>
          </div>
        </div>

        <p className="text-[11px] text-slate-400 dark:text-slate-600 text-center border-t border-slate-100 dark:border-slate-800 pt-3">
          Mo 算法时间复杂度：O((n + q)√n)；L 指针按块分组，R 指针在每块内单调移动，总移动量最优化
        </p>
      </div>
    </div>
  )
}
