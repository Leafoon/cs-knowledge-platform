'use client'

import { useState } from 'react'

const W = 360, H = 280
const CX = 170, CY = 140

type Pt = { x: number; y: number }

function convexPoly(n: number): Pt[] {
  const pts: Pt[] = []
  for (let i = 0; i < n; i++) {
    const a = (2 * Math.PI * i) / n - Math.PI / 2
    const r = 90 * (0.65 + 0.35 * Math.sin(i * 1.7 + 0.5))
    pts.push({ x: CX + r * Math.cos(a), y: CY + r * Math.sin(a) })
  }
  return pts
}

const HULL = convexPoly(7)

type BBox = { corners: Pt[]; area: number; w: number; h: number }

function projectOnAxis(pts: Pt[], ux: number, uy: number) {
  const projs = pts.map(p => p.x * ux + p.y * uy)
  return { min: Math.min(...projs), max: Math.max(...projs) }
}

function computeBbox(pts: Pt[], edgeIdx: number): BBox {
  const n = pts.length
  const a = pts[edgeIdx], b = pts[(edgeIdx + 1) % n]
  const dx = b.x - a.x, dy = b.y - a.y
  const len = Math.hypot(dx, dy) || 1
  const ex = dx / len, ey = dy / len // edge direction
  const nx = -ey, ny = ex // normal

  const { min: minE, max: maxE } = projectOnAxis(pts, ex, ey)
  const { min: minN, max: maxN } = projectOnAxis(pts, nx, ny)
  const w = maxE - minE, h = maxN - minN

  // 4 corners in world coords
  const corners: Pt[] = [
    { x: minE * ex + minN * nx, y: minE * ey + minN * ny },
    { x: maxE * ex + minN * nx, y: maxE * ey + minN * ny },
    { x: maxE * ex + maxN * nx, y: maxE * ey + maxN * ny },
    { x: minE * ex + maxN * nx, y: minE * ey + maxN * ny },
  ]

  return { corners, area: w * h, w, h }
}

export function MinAreaBoundingRect() {
  const [edgeIdx, setEdgeIdx] = useState(0)
  const n = HULL.length

  const bboxes = HULL.map((_, i) => computeBbox(HULL, i))
  const minArea = Math.min(...bboxes.map(b => b.area))
  const minIdx = bboxes.findIndex(b => b.area === minArea)
  const curBbox = bboxes[edgeIdx]
  const isBest = edgeIdx === minIdx

  return (
    <div className="rounded-2xl border border-orange-200 dark:border-orange-700 overflow-hidden shadow-lg font-sans">
      {/* Header */}
      <div className="bg-gradient-to-r from-orange-500 via-amber-500 to-yellow-500 px-5 py-4">
        <h3 className="text-white font-bold text-base">📦 最小面积外接矩形</h3>
        <p className="text-orange-50 text-xs mt-0.5">
          对凸包每条边建立坐标系，投影求包围盒面积，最小者即为最优解
        </p>
        <div className="flex gap-1.5 mt-3 flex-wrap">
          {HULL.map((_, i) => (
            <button key={i} onClick={() => setEdgeIdx(i)}
              className={`px-2 py-0.5 text-xs rounded-lg transition-colors font-bold ${
                edgeIdx === i
                  ? 'bg-white text-orange-700'
                  : i === minIdx
                  ? 'bg-white/40 text-white ring-2 ring-white'
                  : 'bg-white/20 text-white hover:bg-white/30'
              }`}>
              边 {i}→{(i+1)%n}
            </button>
          ))}
        </div>
      </div>

      <div className="bg-white dark:bg-slate-900 p-4">
        <div className="flex gap-4 flex-wrap items-start">
          {/* SVG */}
          <svg width={W} height={H}
            className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 flex-shrink-0">
            {/* Min bbox (always show faintly in bg) */}
            {!isBest && (
              <polygon
                points={bboxes[minIdx].corners.map(p => `${p.x},${p.y}`).join(' ')}
                fill="#f59e0b12" stroke="#f59e0b60" strokeWidth={1.5} strokeDasharray="5,3"
              />
            )}

            {/* Current bbox */}
            <polygon
              points={curBbox.corners.map(p => `${p.x},${p.y}`).join(' ')}
              fill={isBest ? '#f59e0b18' : '#0ea5e912'}
              stroke={isBest ? '#f59e0b' : '#0ea5e9'}
              strokeWidth={isBest ? 3 : 2}
            />

            {/* Edge direction arrow */}
            {(() => {
              const a = HULL[edgeIdx], b = HULL[(edgeIdx + 1) % n]
              const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2
              const dx = b.x - a.x, dy = b.y - a.y
              const len = Math.hypot(dx, dy) || 1
              const ux = dx / len * 14, uy = dy / len * 14
              return (
                <g>
                  <line x1={a.x} y1={a.y} x2={b.x} y2={b.y}
                    stroke="#ef4444" strokeWidth={3} strokeLinecap="round"/>
                  <polygon
                    points={`${mx + ux},${my + uy} ${mx + uy * 0.5},${my - ux * 0.5} ${mx - uy * 0.5},${my + ux * 0.5}`}
                    fill="#ef4444"
                  />
                </g>
              )
            })()}

            {/* Convex hull */}
            <polygon points={HULL.map(p => `${p.x},${p.y}`).join(' ')}
              fill="#64748b15" stroke="#64748b" strokeWidth={2}/>

            {/* Vertices */}
            {HULL.map((p, i) => (
              <g key={i}>
                <circle cx={p.x} cy={p.y} r={5}
                  fill={i === edgeIdx || i === (edgeIdx + 1) % n ? '#ef4444' : '#64748b'}
                  stroke="white" strokeWidth={1.5}/>
              </g>
            ))}

            {/* Area label in bbox center */}
            {(() => {
              const cx = curBbox.corners.reduce((s, p) => s + p.x, 0) / 4
              const cy = curBbox.corners.reduce((s, p) => s + p.y, 0) / 4
              return (
                <text x={cx} y={cy} textAnchor="middle" dominantBaseline="middle"
                  fontSize={11} fontWeight="bold"
                  fill={isBest ? '#f59e0b' : '#0ea5e9'}>
                  {curBbox.area.toFixed(0)}
                </text>
              )
            })()}
          </svg>

          {/* Info panel */}
          <div className="flex flex-col gap-3 flex-1 min-w-[150px]">
            {/* Current bbox stats */}
            <div className={`rounded-xl border-2 p-3 transition-all ${
              isBest
                ? 'border-amber-400 bg-amber-50 dark:bg-amber-900/15'
                : 'border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800'
            }`}>
              <p className="text-[10px] font-bold text-slate-500 dark:text-slate-400 uppercase mb-2">
                边 {edgeIdx}→{(edgeIdx + 1) % n} 的包围盒
              </p>
              <div className="space-y-1 text-xs font-mono">
                <p className="text-slate-600 dark:text-slate-300">宽: <strong>{curBbox.w.toFixed(1)}</strong></p>
                <p className="text-slate-600 dark:text-slate-300">高: <strong>{curBbox.h.toFixed(1)}</strong></p>
                <p className={`font-black ${isBest ? 'text-amber-600 dark:text-amber-400' : 'text-sky-600 dark:text-sky-400'}`}>
                  面积: {curBbox.area.toFixed(0)} {isBest ? '⭐' : ''}
                </p>
              </div>
            </div>

            {/* All edge areas */}
            <div className="rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden text-xs">
              <div className="bg-slate-100 dark:bg-slate-800 px-2 py-1.5 text-[10px] font-bold text-slate-500 dark:text-slate-400 uppercase">
                各边面积对比
              </div>
              {bboxes.map((b, i) => (
                <button key={i} onClick={() => setEdgeIdx(i)}
                  className={`w-full flex items-center gap-2 px-2 py-1 border-t border-slate-100 dark:border-slate-700 text-left transition-colors ${
                    i === edgeIdx ? 'bg-sky-50 dark:bg-sky-900/10' :
                    i === minIdx ? 'bg-amber-50 dark:bg-amber-900/10' :
                    'hover:bg-slate-50 dark:hover:bg-slate-800/50'
                  }`}>
                  <span className="font-mono text-slate-500 dark:text-slate-400 w-16">边{i}→{(i+1)%n}</span>
                  <div className="flex-1 bg-slate-200 dark:bg-slate-700 rounded-full h-2 overflow-hidden">
                    <div className="h-full rounded-full transition-all"
                      style={{
                        width: `${(b.area / Math.max(...bboxes.map(x => x.area)) * 100).toFixed(0)}%`,
                        background: i === minIdx ? '#f59e0b' : i === edgeIdx ? '#0ea5e9' : '#94a3b8',
                      }}/>
                  </div>
                  <span className={`font-mono font-bold w-14 text-right ${i === minIdx ? 'text-amber-600 dark:text-amber-400' : 'text-slate-600 dark:text-slate-300'}`}>
                    {b.area.toFixed(0)}
                  </span>
                </button>
              ))}
            </div>

            <div className="rounded-xl bg-orange-50 dark:bg-orange-900/10 border border-orange-200 dark:border-orange-800 p-3 text-xs text-orange-700 dark:text-orange-300">
              <p className="font-bold">最小面积 ⭐</p>
              <p className="font-mono text-lg font-black mt-0.5">{minArea.toFixed(0)}</p>
              <p className="text-[10px] mt-1 opacity-70">时间复杂度 O(n) 旋转卡壳</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
