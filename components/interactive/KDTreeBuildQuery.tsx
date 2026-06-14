'use client'

import { useState } from 'react'

interface Point { x: number; y: number; label: string }

const POINTS: Point[] = [
  { x: 2,  y: 3,  label: 'A' },
  { x: 5,  y: 4,  label: 'B' },
  { x: 9,  y: 6,  label: 'C' },
  { x: 4,  y: 7,  label: 'D' },
  { x: 8,  y: 1,  label: 'E' },
  { x: 7,  y: 2,  label: 'F' },
  { x: 3,  y: 6,  label: 'G' },
  { x: 6,  y: 8,  label: 'H' },
]

interface Split { axis: 'x' | 'y'; value: number; depth: number }

const SPLITS: Split[] = [
  { axis: 'x', value: 5.5, depth: 0 },
  { axis: 'y', value: 4.5, depth: 1 },
  { axis: 'y', value: 2.5, depth: 1 },
  { axis: 'x', value: 3.5, depth: 2 },
  { axis: 'x', value: 7.5, depth: 2 },
]

const W = 300, H = 280
const PAD = 28
const MAX = 10

function toSvg(v: number, axis: 'x' | 'y'): number {
  if (axis === 'x') return PAD + (v / MAX) * (W - 2 * PAD)
  return H - PAD - (v / MAX) * (H - 2 * PAD)
}

const QUERY_POINT: Point = { x: 6, y: 5, label: 'Q' }
function dist(a: Point, b: Point) { return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2) }
const nearest = POINTS.reduce((best, p) => dist(p, QUERY_POINT) < dist(best, QUERY_POINT) ? p : best, POINTS[0])
const nnDist = dist(nearest, QUERY_POINT)

const DEPTH_COLORS = ['#6366f1', '#f97316', '#10b981', '#ec4899']

export function KDTreeBuildQuery() {
  const [showDepth, setShowDepth] = useState(3)
  const [mode, setMode] = useState<'build' | 'query'>('build')

  const visibleSplits = SPLITS.filter(s => s.depth < showDepth)

  return (
    <div className="rounded-2xl border border-cyan-200 dark:border-cyan-700 overflow-hidden shadow-lg font-sans">
      <div className="bg-gradient-to-r from-cyan-600 to-blue-500 px-5 py-4">
        <h3 className="text-white font-bold text-base">🗺️ KD-Tree：空间划分 + 最近邻搜索</h3>
        <p className="text-cyan-50 text-xs mt-0.5">
          {POINTS.length} 个 2D 点，交替按 x/y 轴划分；最近邻搜索通过剪枝跳过远区域
        </p>
        <div className="flex gap-2 mt-3">
          {(['build', 'query'] as const).map(m => (
            <button key={m} onClick={() => setMode(m)}
              className={`px-3 py-1 text-xs rounded-lg transition-colors ${
                mode === m ? 'bg-white text-cyan-700 font-bold' : 'bg-white/25 text-white hover:bg-white/35'
              }`}>
              {m === 'build' ? '构建过程' : '最近邻查询'}
            </button>
          ))}
          {mode === 'build' && (
            <div className="ml-auto flex items-center gap-2">
              <span className="text-xs text-cyan-100">深度 ≤</span>
              {[1, 2, 3].map(d => (
                <button key={d} onClick={() => setShowDepth(d)}
                  className={`w-6 h-6 rounded-full text-xs transition-all ${
                    showDepth === d ? 'bg-white text-cyan-700 font-bold' : 'bg-white/25 text-white hover:bg-white/35'
                  }`}>{d}</button>
              ))}
            </div>
          )}
        </div>
      </div>

      <div className="bg-white dark:bg-slate-900 p-5">
        <div className="flex gap-4 flex-wrap">
          {/* SVG canvas */}
          <div>
            <svg width={W} height={H} className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800">
              {/* Axes */}
              <line x1={PAD} y1={H-PAD} x2={W-PAD} y2={H-PAD} stroke="#94a3b8" strokeWidth={1} />
              <line x1={PAD} y1={PAD} x2={PAD} y2={H-PAD} stroke="#94a3b8" strokeWidth={1} />
              <text x={W/2} y={H-6} textAnchor="middle" fontSize={9} fill="#94a3b8">x</text>
              <text x={10} y={H/2} textAnchor="middle" fontSize={9} fill="#94a3b8" transform={`rotate(-90, 10, ${H/2})`}>y</text>
              {/* Tick labels */}
              {[0,2,4,6,8,10].map(v => (
                <g key={v}>
                  <text x={toSvg(v,'x')} y={H-PAD+12} textAnchor="middle" fontSize={7} fill="#94a3b8">{v}</text>
                  <text x={PAD-4} y={toSvg(v,'y')+3} textAnchor="end" fontSize={7} fill="#94a3b8">{v}</text>
                </g>
              ))}

              {/* Split lines */}
              {visibleSplits.map((s, i) => {
                const color = DEPTH_COLORS[s.depth % DEPTH_COLORS.length]
                if (s.axis === 'x') {
                  const sx = toSvg(s.value, 'x')
                  return <line key={i} x1={sx} y1={PAD} x2={sx} y2={H-PAD} stroke={color} strokeWidth={1.5} strokeDasharray={s.depth > 0 ? "5 3" : ""} />
                } else {
                  const sy = toSvg(s.value, 'y')
                  return <line key={i} x1={PAD} y1={sy} x2={W-PAD} y2={sy} stroke={color} strokeWidth={1.5} strokeDasharray="5 3" />
                }
              })}

              {/* NN circle */}
              {mode === 'query' && (
                <circle cx={toSvg(QUERY_POINT.x, 'x')} cy={toSvg(QUERY_POINT.y, 'y')} r={nnDist * ((W-2*PAD)/MAX)}
                  fill="none" stroke="#f97316" strokeWidth={1.5} strokeDasharray="4 2" />
              )}

              {/* Points */}
              {POINTS.map(p => {
                const cx = toSvg(p.x, 'x')
                const cy = toSvg(p.y, 'y')
                const isNearest = mode === 'query' && p.label === nearest.label
                return (
                  <g key={p.label}>
                    <circle cx={cx} cy={cy} r={isNearest ? 8 : 5}
                      fill={isNearest ? '#10b981' : '#6366f1'} stroke="white" strokeWidth={2} />
                    <text x={cx + 8} y={cy - 5} fontSize={10} fontWeight="bold"
                      fill={isNearest ? '#065f46' : '#4338ca'}>
                      {p.label}
                    </text>
                  </g>
                )
              })}

              {/* Query point */}
              {mode === 'query' && (
                <g>
                  <circle cx={toSvg(QUERY_POINT.x,'x')} cy={toSvg(QUERY_POINT.y,'y')} r={7}
                    fill="#f97316" stroke="white" strokeWidth={2} />
                  <text x={toSvg(QUERY_POINT.x,'x')+9} y={toSvg(QUERY_POINT.y,'y')-5} fontSize={10} fontWeight="bold" fill="#c2410c">Q</text>
                  {/* NN line */}
                  <line x1={toSvg(QUERY_POINT.x,'x')} y1={toSvg(QUERY_POINT.y,'y')}
                    x2={toSvg(nearest.x,'x')} y2={toSvg(nearest.y,'y')}
                    stroke="#10b981" strokeWidth={1.5} strokeDasharray="4 2" />
                </g>
              )}
            </svg>
          </div>

          {/* Info panel */}
          <div className="flex-1 min-w-[180px] space-y-3">
            {mode === 'build' ? (
              <>
                <p className="text-xs font-semibold text-slate-700 dark:text-slate-200">划分层次</p>
                {[0,1,2].map(d => (
                  <div key={d} className={`flex items-center gap-2 text-xs rounded-lg px-2 py-1.5 transition-all ${
                    showDepth > d
                      ? 'opacity-100'
                      : 'opacity-30'
                  }`} style={{ backgroundColor: DEPTH_COLORS[d] + '20', border: '1px solid ' + DEPTH_COLORS[d] + '60' }}>
                    <div className="w-3 h-3 rounded-sm flex-shrink-0" style={{ backgroundColor: DEPTH_COLORS[d] }} />
                    <div>
                      <p className="font-medium" style={{ color: DEPTH_COLORS[d] }}>深度 {d}：按 {d % 2 === 0 ? 'x' : 'y'} 轴划分</p>
                      <p className="text-slate-500 dark:text-slate-400 text-[10px]">{SPLITS.filter(s=>s.depth===d).length} 条分割线</p>
                    </div>
                  </div>
                ))}
                <div className="rounded-lg bg-slate-50 dark:bg-slate-800 p-2 text-[11px] text-slate-600 dark:text-slate-400 border border-slate-200 dark:border-slate-700">
                  构建：每层选方差最大维度（简化版取 x/y 轮流），取中位数为分割点 → O(n log n)
                </div>
              </>
            ) : (
              <>
                <p className="text-xs font-semibold text-slate-700 dark:text-slate-200">最近邻查询 Q=({QUERY_POINT.x},{QUERY_POINT.y})</p>
                <div className="rounded-xl border border-emerald-200 dark:border-emerald-800 bg-emerald-50 dark:bg-emerald-900/10 p-3 text-xs">
                  <p className="font-bold text-emerald-700 dark:text-emerald-300">最近邻：{nearest.label} = ({nearest.x},{nearest.y})</p>
                  <p className="text-slate-500 dark:text-slate-400 mt-1">距离 = {nnDist.toFixed(2)}</p>
                </div>
                <div className="rounded-lg bg-orange-50 dark:bg-orange-900/10 p-2 text-[11px] text-orange-700 dark:text-orange-400 border border-orange-200 dark:border-orange-800">
                  橙色圆 = 当前最优半径，落在圆外的区域（由 KD 树分割线界定）可直接剪枝跳过
                </div>
                <div className="text-[11px] text-slate-500 dark:text-slate-400">
                  期望复杂度 O(log n)（低维），最坏 O(n)（高维退化）
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
