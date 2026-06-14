'use client'

import { useState, useRef, useCallback } from 'react'

const W = 340, H = 280
const CX = 170, CY = 145, R = 105

type Pt = { x: number; y: number }

function convexHull(n: number): Pt[] {
  const pts: Pt[] = []
  // Regular n-gon, CCW
  for (let i = 0; i < n; i++) {
    const a = (2 * Math.PI * i) / n - Math.PI / 2
    pts.push({ x: Math.round(CX + R * Math.cos(a)), y: Math.round(CY + R * Math.sin(a)) })
  }
  return pts
}

function cross2(o: Pt, a: Pt, b: Pt) {
  return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
}

function pointInTriangle(q: Pt, a: Pt, b: Pt, c: Pt) {
  const d1 = cross2(q, a, b)
  const d2 = cross2(q, b, c)
  const d3 = cross2(q, c, a)
  const hasNeg = d1 < 0 || d2 < 0 || d3 < 0
  const hasPos = d1 > 0 || d2 > 0 || d3 > 0
  return !(hasNeg && hasPos)
}

// Binary search: find which triangle from P0 fan contains Q
function bsearchFan(pts: Pt[], q: Pt): { lo: number; hi: number; steps: { lo: number; hi: number; mid: number }[] } {
  const n = pts.length
  const P0 = pts[0]
  let lo = 1, hi = n - 1
  const steps: { lo: number; hi: number; mid: number }[] = []

  while (hi - lo > 1) {
    const mid = Math.floor((lo + hi) / 2)
    steps.push({ lo, hi, mid })
    const c = cross2(P0, pts[mid], q)
    if (c >= 0) lo = mid
    else hi = mid
  }
  steps.push({ lo, hi, mid: lo })
  return { lo, hi, steps }
}

export function ConvexPolygonBinarySearch() {
  const [nPts, setNPts] = useState(8)
  const [query, setQuery] = useState<Pt>({ x: 190, y: 120 })
  const [step, setStep] = useState(0)
  const [dragging, setDragging] = useState(false)
  const svgRef = useRef<SVGSVGElement>(null)

  const pts = convexHull(nPts)
  const { steps } = bsearchFan(pts, query)
  const maxStep = steps.length - 1
  const curStep = Math.min(step, maxStep)
  const { lo, hi, mid } = steps[curStep] || steps[maxStep]
  const P0 = pts[0]
  const found = hi - lo === 1
  const inside = found && pointInTriangle(query, P0, pts[lo], pts[hi])

  const getPos = (e: React.MouseEvent | React.TouchEvent) => {
    if (!svgRef.current) return null
    const rect = svgRef.current.getBoundingClientRect()
    const client = 'touches' in e ? (e as React.TouchEvent).touches[0] : (e as React.MouseEvent)
    return {
      x: Math.max(5, Math.min(W - 5, client.clientX - rect.left)),
      y: Math.max(5, Math.min(H - 5, client.clientY - rect.top)),
    }
  }

  const handleMove = useCallback((e: React.MouseEvent<SVGSVGElement>) => {
    if (!dragging) return
    const p = getPos(e)
    if (p) { setQuery(p); setStep(0) }
  }, [dragging])

  const handleTouch = useCallback((e: React.TouchEvent<SVGSVGElement>) => {
    const p = getPos(e)
    if (p) { setQuery(p); setStep(0) }
  }, [])

  return (
    <div className="rounded-2xl border border-amber-200 dark:border-amber-700 overflow-hidden shadow-lg font-sans">
      {/* Header */}
      <div className="bg-gradient-to-r from-amber-500 via-orange-500 to-yellow-400 px-5 py-4">
        <h3 className="text-white font-bold text-base">🔍 凸多边形内点判断：扇形二分法</h3>
        <p className="text-amber-50 text-xs mt-0.5">以 P0 为中心建扇形，对角度二分 O(log n) 定位目标三角形</p>
        <div className="flex items-center gap-3 mt-2 flex-wrap">
          <label className="text-xs text-white flex items-center gap-2">
            顶点数
            <select value={nPts} onChange={e => { setNPts(+e.target.value); setStep(0) }}
              className="rounded px-1 py-0.5 text-amber-800 bg-white/90 font-bold text-xs">
              {[6, 8, 10, 12].map(n => <option key={n} value={n}>{n}</option>)}
            </select>
          </label>
          <span className="text-white/70 text-xs">步骤 {curStep + 1}/{steps.length}</span>
        </div>
      </div>

      <div className="bg-white dark:bg-slate-900 p-4">
        <div className="flex gap-4 flex-wrap items-start">
          {/* SVG */}
          <svg ref={svgRef} width={W} height={H}
            className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 cursor-crosshair select-none flex-shrink-0"
            onMouseDown={() => setDragging(true)}
            onMouseUp={() => setDragging(false)}
            onMouseLeave={() => setDragging(false)}
            onMouseMove={handleMove}
            onTouchMove={handleTouch}
          >
            {/* All fan triangles (faded) */}
            {Array.from({ length: nPts - 2 }).map((_, i) => (
              <polygon key={i}
                points={[P0, pts[i + 1], pts[i + 2]].map(p => `${p.x},${p.y}`).join(' ')}
                fill="#f59e0b08" stroke="#f59e0b20" strokeWidth={1}
              />
            ))}

            {/* Current search range */}
            <polygon
              points={[P0, pts[lo], pts[hi]].map(p => `${p.x},${p.y}`).join(' ')}
              fill={found ? (inside ? '#10b98125' : '#ef444420') : '#f59e0b20'}
              stroke={found ? (inside ? '#10b981' : '#ef4444') : '#f59e0b'}
              strokeWidth={2}
            />

            {/* Mid ray */}
            {!found && (
              <line x1={P0.x} y1={P0.y} x2={pts[mid].x} y2={pts[mid].y}
                stroke="#a78bfa" strokeWidth={2} strokeDasharray="6,3"/>
            )}

            {/* Polygon outline */}
            <polygon
              points={pts.map(p => `${p.x},${p.y}`).join(' ')}
              fill="none" stroke="#64748b" strokeWidth={2}
            />

            {/* Vertices */}
            {pts.map((p, i) => {
              const isKey = i === 0 || i === lo || i === hi || (!found && i === mid)
              return (
                <g key={i}>
                  <circle cx={p.x} cy={p.y} r={isKey ? 7 : 4}
                    fill={
                      i === 0 ? '#f59e0b' :
                      i === mid && !found ? '#a78bfa' :
                      (i === lo || i === hi) ? '#0ea5e9' : '#64748b'
                    }
                    stroke="white" strokeWidth={1.5}
                  />
                  <text x={p.x + (p.x < CX ? -12 : 8)} y={p.y + (p.y < CY ? -4 : 14)}
                    fontSize={9} fontWeight="bold"
                    fill={i === 0 ? '#f59e0b' : '#64748b'}
                    className="dark:fill-slate-300">P{i}</text>
                </g>
              )
            })}

            {/* Query Q */}
            <circle cx={query.x} cy={query.y} r={10}
              fill={inside ? '#10b981' : '#ef4444'} stroke="white" strokeWidth={2.5}/>
            <text x={query.x} y={query.y} textAnchor="middle" dominantBaseline="middle"
              fontSize={9} fontWeight="bold" fill="white">Q</text>
          </svg>

          {/* Controls & info */}
          <div className="flex flex-col gap-3 flex-1 min-w-[140px]">
            {/* Step controls */}
            <div className="flex gap-1.5">
              <button
                onClick={() => setStep(s => Math.max(0, s - 1))}
                disabled={curStep === 0}
                className="flex-1 py-1.5 rounded-lg bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 text-xs font-bold hover:bg-slate-200 dark:hover:bg-slate-700 disabled:opacity-40 transition-colors">
                ← 上步
              </button>
              <button
                onClick={() => setStep(s => Math.min(maxStep, s + 1))}
                disabled={curStep === maxStep}
                className="flex-1 py-1.5 rounded-lg bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300 text-xs font-bold hover:bg-amber-200 dark:hover:bg-amber-900/50 disabled:opacity-40 transition-colors">
                下步 →
              </button>
            </div>
            <button
              onClick={() => setStep(0)}
              className="py-1.5 rounded-lg bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-400 text-xs hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors">
              重置步骤
            </button>

            {/* Current state */}
            <div className="rounded-xl bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 p-3 text-xs space-y-1.5">
              <p className="font-bold text-slate-600 dark:text-slate-300">当前搜索状态</p>
              <div className="font-mono space-y-1">
                <p className="text-sky-600 dark:text-sky-400">lo = {lo} (P{lo})</p>
                <p className="text-purple-600 dark:text-purple-400">mid = {mid} (P{mid})</p>
                <p className="text-sky-600 dark:text-sky-400">hi = {hi} (P{hi})</p>
              </div>
            </div>

            {found && (
              <div className={`rounded-xl border-2 text-center py-3 ${
                inside
                  ? 'border-emerald-400 bg-emerald-50 dark:bg-emerald-900/20 text-emerald-700 dark:text-emerald-300'
                  : 'border-rose-400 bg-rose-50 dark:bg-rose-900/20 text-rose-700 dark:text-rose-300'
              }`}>
                <p className="text-lg">{inside ? '✅' : '⛔'}</p>
                <p className="font-bold text-sm">{inside ? '多边形内部' : '多边形外部'}</p>
                <p className="text-[10px] mt-0.5">总步数: {steps.length} = O(log {nPts})</p>
              </div>
            )}

            <div className="rounded-xl bg-amber-50 dark:bg-amber-900/10 border border-amber-200 dark:border-amber-800 p-3 text-xs text-amber-700 dark:text-amber-300">
              <p className="font-bold mb-1">时间复杂度</p>
              <p>扇形二分 <strong>O(log n)</strong></p>
              <p className="text-[10px] mt-1 text-amber-600/70 dark:text-amber-400/70">参考: 暴力遍历 O(n)</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
