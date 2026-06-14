'use client'

import { useState, useRef, useCallback } from 'react'

const W = 380, H = 300

type Pt = { x: number; y: number }

function makeRect(cx: number, cy: number, w: number, h: number, angle = 0): Pt[] {
  const hw = w / 2, hh = h / 2
  const cos = Math.cos(angle), sin = Math.sin(angle)
  const corners: Pt[] = [
    { x: -hw, y: -hh }, { x: hw, y: -hh },
    { x: hw, y: hh }, { x: -hw, y: hh },
  ]
  return corners.map(p => ({
    x: cx + p.x * cos - p.y * sin,
    y: cy + p.x * sin + p.y * cos,
  }))
}

function makeTriangle(cx: number, cy: number, r: number, angle = 0): Pt[] {
  const pts: Pt[] = []
  for (let i = 0; i < 3; i++) {
    const a = (2 * Math.PI * i) / 3 + angle
    pts.push({ x: cx + r * Math.cos(a), y: cy + r * Math.sin(a) })
  }
  return pts
}

function edges(poly: Pt[]): { nx: number; ny: number }[] {
  const n = poly.length
  return poly.map((p, i) => {
    const q = poly[(i + 1) % n]
    const dx = q.x - p.x, dy = q.y - p.y
    const len = Math.hypot(dx, dy) || 1
    return { nx: -dy / len, ny: dx / len }
  })
}

function project(poly: Pt[], nx: number, ny: number): { min: number; max: number } {
  const projs = poly.map(p => p.x * nx + p.y * ny)
  return { min: Math.min(...projs), max: Math.max(...projs) }
}

function overlap(a: { min: number; max: number }, b: { min: number; max: number }) {
  return a.max >= b.min && b.max >= a.min
}

type SATResult = {
  colliding: boolean
  axes: { nx: number; ny: number; projA: { min: number; max: number }; projB: { min: number; max: number }; overlaps: boolean }[]
}

function satTest(polyA: Pt[], polyB: Pt[]): SATResult {
  const axesA = edges(polyA)
  const axesB = edges(polyB)
  const allAxes = [...axesA, ...axesB]

  const results = allAxes.map(({ nx, ny }) => {
    const projA = project(polyA, nx, ny)
    const projB = project(polyB, nx, ny)
    return { nx, ny, projA, projB, overlaps: overlap(projA, projB) }
  })

  return { colliding: results.every(r => r.overlaps), axes: results }
}

export function SATCollisionDetection() {
  const [posA, setPosA] = useState<Pt>({ x: 130, y: 145 })
  const [posB, setPosB] = useState<Pt>({ x: 265, y: 145 })
  const [angle, setAngle] = useState(0.3)
  const [dragging, setDragging] = useState<'A' | 'B' | null>(null)
  const [selAxis, setSelAxis] = useState<number | null>(null)
  const svgRef = useRef<SVGSVGElement>(null)

  const polyA = makeRect(posA.x, posA.y, 90, 60, angle)
  const polyB = makeTriangle(posB.x, posB.y, 55, -0.3)
  const sat = satTest(polyA, polyB)

  const getPos = (e: React.MouseEvent | React.TouchEvent) => {
    if (!svgRef.current) return null
    const rect = svgRef.current.getBoundingClientRect()
    const client = 'touches' in e ? (e as React.TouchEvent).touches[0] : (e as React.MouseEvent)
    return {
      x: Math.max(30, Math.min(W - 30, client.clientX - rect.left)),
      y: Math.max(30, Math.min(H - 30, client.clientY - rect.top)),
    }
  }

  const handleMove = useCallback((e: React.MouseEvent<SVGSVGElement>) => {
    if (!dragging) return
    const p = getPos(e)
    if (!p) return
    if (dragging === 'A') setPosA(p)
    else setPosB(p)
  }, [dragging])

  const handleTouch = useCallback((e: React.TouchEvent<SVGSVGElement>) => {
    if (!dragging) return
    const p = getPos(e)
    if (!p) return
    if (dragging === 'A') setPosA(p)
    else setPosB(p)
  }, [dragging])

  // SAP projection bar scale
  const PROJ_SCALE = 1.5

  // Draw projected intervals on selected axis
  const selectedAxis = selAxis !== null ? sat.axes[selAxis] : null

  // Projection visualizer center
  const PROJ_CX = W / 2, PROJ_CY = 55

  return (
    <div className="rounded-2xl border border-red-200 dark:border-red-800 overflow-hidden shadow-lg font-sans">
      {/* Header */}
      <div className={`px-5 py-4 transition-colors duration-300 ${
        sat.colliding
          ? 'bg-gradient-to-r from-red-600 via-rose-600 to-pink-600'
          : 'bg-gradient-to-r from-slate-600 via-slate-700 to-slate-600'
      }`}>
        <div className="flex items-center gap-3">
          <div>
            <h3 className="text-white font-bold text-base">⚡ SAT 碰撞检测（分离轴定理）</h3>
            <p className="text-white/75 text-xs mt-0.5">
              拖动两个形状，点击轴查看投影区间。只要存在一个分离轴 → 无碰撞。
            </p>
          </div>
          <div className={`ml-auto rounded-xl px-4 py-2 text-center flex-shrink-0 ${
            sat.colliding ? 'bg-white/20' : 'bg-white/10'
          }`}>
            <p className="text-2xl">{sat.colliding ? '💥' : '✅'}</p>
            <p className="text-white font-black text-xs">{sat.colliding ? '碰撞!' : '分离'}</p>
          </div>
        </div>
        <div className="flex items-center gap-2 mt-3">
          <span className="text-white/70 text-xs">矩形旋转角</span>
          <input type="range" min={0} max={3.14} step={0.05} value={angle}
            onChange={e => setAngle(+e.target.value)}
            className="flex-1 accent-white h-1.5"/>
          <span className="text-white/70 text-xs w-10">{(angle * 180 / Math.PI).toFixed(0)}°</span>
        </div>
      </div>

      <div className="bg-white dark:bg-slate-900 p-4">
        <div className="flex gap-4 flex-wrap items-start">
          {/* SVG canvas */}
          <div>
            {/* Projection bar (above canvas) */}
            {selectedAxis && (
              <div className="mb-2 rounded-xl bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 p-3">
                <p className="text-[10px] text-slate-500 dark:text-slate-400 font-bold mb-2">
                  轴 {selAxis} 上的投影区间
                </p>
                <div className="relative h-8">
                  {/* Bar background */}
                  <div className="absolute inset-y-2 left-0 right-0 bg-slate-200 dark:bg-slate-700 rounded-full"/>
                  {/* PolyA projection */}
                  {(() => {
                    const scale = 1.5
                    const offset = 190
                    const la = (selectedAxis.projA.min * scale + offset)
                    const wa = (selectedAxis.projA.max - selectedAxis.projA.min) * scale
                    const lb = (selectedAxis.projB.min * scale + offset)
                    const wb = (selectedAxis.projB.max - selectedAxis.projB.min) * scale
                    return (
                      <>
                        <div className="absolute top-1 h-6 rounded-full bg-blue-500/80 border-2 border-blue-600 transition-all"
                          style={{ left: `${Math.max(0, la)}px`, width: `${Math.max(4, wa)}px` }}/>
                        <div className="absolute top-1 h-6 rounded-full bg-orange-500/70 border-2 border-orange-600 transition-all"
                          style={{ left: `${Math.max(0, lb)}px`, width: `${Math.max(4, wb)}px` }}/>
                      </>
                    )
                  })()}
                  <div className="absolute -bottom-4 left-0 text-[9px] text-blue-600 dark:text-blue-400 font-bold">矩形A</div>
                  <div className="absolute -bottom-4 left-16 text-[9px] text-orange-600 dark:text-orange-400 font-bold">三角形B</div>
                  <div className={`absolute -bottom-4 right-0 text-[9px] font-bold ${selectedAxis.overlaps ? 'text-rose-600 dark:text-rose-400' : 'text-emerald-600 dark:text-emerald-400'}`}>
                    {selectedAxis.overlaps ? '重叠' : '分离！'}
                  </div>
                </div>
              </div>
            )}

            <svg ref={svgRef} width={W} height={H}
              className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 cursor-crosshair select-none"
              onMouseDown={() => {}}
              onMouseUp={() => setDragging(null)}
              onMouseLeave={() => setDragging(null)}
              onMouseMove={handleMove}
              onTouchMove={handleTouch}
              onTouchEnd={() => setDragging(null)}
            >
              {/* Poly A (rect) */}
              <polygon
                points={polyA.map(p => `${p.x},${p.y}`).join(' ')}
                fill={sat.colliding ? '#ef444420' : '#3b82f618'}
                stroke={sat.colliding ? '#ef4444' : '#3b82f6'}
                strokeWidth={2.5}
                className="cursor-grab active:cursor-grabbing"
                onMouseDown={() => setDragging('A')}
                onTouchStart={() => setDragging('A')}
              />
              <text x={posA.x} y={posA.y} textAnchor="middle" dominantBaseline="middle"
                fontSize={13} fontWeight="bold" fill={sat.colliding ? '#ef4444' : '#3b82f6'}
                className="pointer-events-none select-none">A</text>

              {/* Poly B (triangle) */}
              <polygon
                points={polyB.map(p => `${p.x},${p.y}`).join(' ')}
                fill={sat.colliding ? '#f9731620' : '#f9731618'}
                stroke={sat.colliding ? '#ef4444' : '#f97316'}
                strokeWidth={2.5}
                className="cursor-grab active:cursor-grabbing"
                onMouseDown={() => setDragging('B')}
                onTouchStart={() => setDragging('B')}
              />
              <text x={posB.x} y={posB.y} textAnchor="middle" dominantBaseline="middle"
                fontSize={13} fontWeight="bold" fill={sat.colliding ? '#ef4444' : '#f97316'}
                className="pointer-events-none select-none">B</text>

              {/* Collision flash */}
              {sat.colliding && (
                <text x={W / 2} y={H - 12} textAnchor="middle"
                  fontSize={12} fontWeight="bold" fill="#ef4444" opacity={0.7}>
                  💥 所有轴均重叠 → 碰撞
                </text>
              )}
            </svg>
            <p className="text-[10px] text-slate-400 dark:text-slate-500 text-center mt-1">
              拖动 A / B · 调节旋转角
            </p>
          </div>

          {/* Axis list */}
          <div className="flex-1 min-w-[160px]">
            <p className="text-xs font-bold text-slate-600 dark:text-slate-300 mb-2">
              分离轴检测（点击查看投影）
            </p>
            <div className="rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden text-xs">
              <div className="bg-slate-100 dark:bg-slate-800 grid grid-cols-3 gap-0 text-[10px] font-bold text-slate-500 dark:text-slate-400 uppercase">
                <div className="px-2 py-1.5">轴来源</div>
                <div className="px-2 py-1.5 text-center">区间</div>
                <div className="px-2 py-1.5 text-center">结果</div>
              </div>
              {sat.axes.map((ax, i) => {
                const isActiveAxis = selAxis === i
                const srcLabel = i < polyA.length ? `A 边 ${i}` : `B 边 ${i - polyA.length}`
                return (
                  <button key={i} onClick={() => setSelAxis(isActiveAxis ? null : i)}
                    className={`w-full grid grid-cols-3 border-t border-slate-100 dark:border-slate-700 text-left transition-colors ${
                      isActiveAxis ? 'bg-slate-100 dark:bg-slate-700' :
                      !ax.overlaps ? 'bg-emerald-50/60 dark:bg-emerald-900/10' :
                      'hover:bg-slate-50 dark:hover:bg-slate-800/50'
                    }`}>
                    <div className="px-2 py-1.5 text-slate-600 dark:text-slate-400 font-mono text-[10px]">
                      {srcLabel}
                    </div>
                    <div className="px-2 py-1.5 text-center font-mono text-[10px] text-slate-500 dark:text-slate-400">
                      [{ax.projA.min.toFixed(0)},{ax.projA.max.toFixed(0)}]
                    </div>
                    <div className="px-2 py-1.5 text-center">
                      {ax.overlaps
                        ? <span className="text-rose-500 dark:text-rose-400 font-bold">重叠</span>
                        : <span className="text-emerald-600 dark:text-emerald-400 font-bold">分离!</span>
                      }
                    </div>
                  </button>
                )
              })}
            </div>

            <div className={`mt-3 rounded-xl border-2 text-center py-3 transition-all ${
              sat.colliding
                ? 'border-red-400 bg-red-50 dark:bg-red-900/20'
                : 'border-emerald-400 bg-emerald-50 dark:bg-emerald-900/10'
            }`}>
              <p className="text-xl">{sat.colliding ? '💥' : '✅'}</p>
              <p className={`font-black text-sm ${sat.colliding ? 'text-red-700 dark:text-red-300' : 'text-emerald-700 dark:text-emerald-300'}`}>
                {sat.colliding ? '碰撞（无分离轴）' : '无碰撞（有分离轴）'}
              </p>
              <p className="text-[10px] text-slate-500 dark:text-slate-400 mt-1">
                共检测 {sat.axes.length} 条候选轴
              </p>
            </div>

            <div className="mt-3 rounded-xl bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 p-3 text-xs text-slate-600 dark:text-slate-300">
              <p className="font-bold mb-1">SAT 定理</p>
              <p className="text-[10px] leading-relaxed">两凸多边形<strong>不相交</strong>，当且仅当存在某条轴使两投影区间不重叠。</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
