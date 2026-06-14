'use client'

import { useState, useRef, useCallback } from 'react'

const W = 340, H = 260

type Pt = { x: number; y: number }

// Star polygon (self-intersecting)
function starPoly(cx: number, cy: number, r1: number, r2: number, n: number): Pt[] {
  const pts: Pt[] = []
  for (let i = 0; i < n * 2; i++) {
    const r = i % 2 === 0 ? r1 : r2
    const angle = (Math.PI / n) * i - Math.PI / 2
    pts.push({ x: cx + r * Math.cos(angle), y: cy + r * Math.sin(angle) })
  }
  return pts
}

const STAR = starPoly(170, 135, 95, 38, 5)

function windingNumber(q: Pt, poly: Pt[]): number {
  const n = poly.length
  let wn = 0
  for (let i = 0; i < n; i++) {
    const a = poly[i], b = poly[(i + 1) % n]
    if (a.y <= q.y) {
      if (b.y > q.y) {
        const cross = (b.x - a.x) * (q.y - a.y) - (q.x - a.x) * (b.y - a.y)
        if (cross > 0) wn++
      }
    } else {
      if (b.y <= q.y) {
        const cross = (b.x - a.x) * (q.y - a.y) - (q.x - a.x) * (b.y - a.y)
        if (cross < 0) wn--
      }
    }
  }
  return wn
}

function windingColor(wn: number): string {
  if (wn === 0) return '#94a3b8'
  if (Math.abs(wn) === 1) return '#ec4899'
  return '#7c3aed'
}

export function WindingNumberDemo() {
  const [query, setQuery] = useState<Pt>({ x: 170, y: 120 })
  const [dragging, setDragging] = useState(false)
  const svgRef = useRef<SVGSVGElement>(null)

  const poly = STAR
  const n = poly.length
  const wn = windingNumber(query, poly)
  const inside = wn !== 0
  const color = windingColor(wn)

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
    if (p) setQuery(p)
  }, [dragging])

  const handleTouch = useCallback((e: React.TouchEvent<SVGSVGElement>) => {
    const p = getPos(e)
    if (p) setQuery(p)
  }, [])

  // Sample background grid for visual winding-number zones (heatmap)
  const samples: { x: number; y: number; wn: number }[] = []
  for (let gx = 20; gx <= W - 20; gx += 30) {
    for (let gy = 20; gy <= H - 20; gy += 30) {
      const gwn = windingNumber({ x: gx, y: gy }, poly)
      if (gwn !== 0) samples.push({ x: gx, y: gy, wn: gwn })
    }
  }

  return (
    <div className="rounded-2xl border border-pink-200 dark:border-pink-800 overflow-hidden shadow-lg font-sans">
      {/* Header */}
      <div className="bg-gradient-to-r from-pink-600 via-rose-500 to-fuchsia-600 px-5 py-4">
        <h3 className="text-white font-bold text-base">🌀 卷绕数算法（Winding Number）</h3>
        <p className="text-pink-100 text-xs mt-0.5">五角星是自相交多边形，不同区域卷绕数不同（0=外 | ±1=内层 | ±2=核心区）</p>
      </div>

      <div className="bg-white dark:bg-slate-900 p-4">
        <div className="flex gap-4 flex-wrap items-start">
          {/* SVG */}
          <div>
            <svg ref={svgRef} width={W} height={H}
              className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 cursor-crosshair select-none"
              onMouseDown={() => setDragging(true)}
              onMouseUp={() => setDragging(false)}
              onMouseLeave={() => setDragging(false)}
              onMouseMove={handleMove}
              onTouchMove={handleTouch}
            >
              {/* Heat map dots */}
              {samples.map((s, i) => (
                <circle key={i} cx={s.x} cy={s.y} r={11}
                  fill={Math.abs(s.wn) >= 2 ? '#7c3aed20' : '#ec489920'}
                />
              ))}

              {/* Star polygon - use even-odd rule visually */}
              <polygon
                points={poly.map(p => `${p.x},${p.y}`).join(' ')}
                fill="none"
                stroke="#ec4899"
                strokeWidth={2.5}
                fillRule="evenodd"
              />
              {/* Filled regions */}
              <polygon
                points={poly.map(p => `${p.x},${p.y}`).join(' ')}
                fill="#ec489910"
                stroke="none"
                fillRule="nonzero"
              />

              {/* Upward crossing edges (CCW winding) */}
              {Array.from({ length: n }).map((_, i) => {
                const a = poly[i], b = poly[(i + 1) % n]
                const contributing =
                  (a.y <= query.y && b.y > query.y) || (b.y <= query.y && a.y > query.y)
                return (
                  <line key={i} x1={a.x} y1={a.y} x2={b.x} y2={b.y}
                    stroke={contributing ? '#ec4899' : '#ec489940'}
                    strokeWidth={contributing ? 2.5 : 1.5}
                  />
                )
              })}

              {/* Horizontal scan line */}
              <line x1={0} y1={query.y} x2={W} y2={query.y}
                stroke="#f59e0b" strokeWidth={1} strokeDasharray="6,3" opacity={0.6}/>

              {/* Star vertices */}
              {poly.map((p, i) => (
                <circle key={i} cx={p.x} cy={p.y} r={3}
                  fill="#ec4899" stroke="white" strokeWidth={1}/>
              ))}

              {/* Query point */}
              <circle cx={query.x} cy={query.y} r={12}
                fill={color} stroke="white" strokeWidth={2.5}/>
              <text x={query.x} y={query.y} textAnchor="middle" dominantBaseline="middle"
                fontSize={9} fontWeight="bold" fill="white">Q</text>

              {/* WN badge */}
              <rect x={6} y={6} width={64} height={28} rx={8} fill={color}/>
              <text x={38} y={20} textAnchor="middle" fontSize={10} fill="white" fontWeight="bold">WN =</text>
              <text x={38} y={30} textAnchor="middle" fontSize={10} fill="white" fontWeight="black">{wn}</text>
            </svg>
            <p className="text-[10px] text-slate-400 dark:text-slate-500 text-center mt-1">拖动 Q 探索不同区域</p>
          </div>

          {/* Right info */}
          <div className="flex flex-col gap-3 flex-1 min-w-[150px]">
            <div className={`rounded-2xl border-2 text-center py-4 transition-all`}
              style={{ borderColor: color, background: `${color}18` }}>
              <p className="text-3xl font-black font-mono" style={{ color }}>{wn}</p>
              <p className="text-xs font-bold mt-1" style={{ color }}>
                {wn === 0 ? '外部（WN=0）' : Math.abs(wn) === 1 ? '内部（单层）' : '中心区域（双层）'}
              </p>
              <p className="text-[10px] text-slate-500 dark:text-slate-400 mt-1">{inside ? '在多边形内' : '在多边形外'}</p>
            </div>

            <div className="rounded-xl bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 p-3 space-y-2">
              <p className="text-xs font-bold text-slate-600 dark:text-slate-300">颜色图例</p>
              {[
                { wn: 0, label: 'WN=0：外部', color: '#94a3b8' },
                { wn: 1, label: 'WN=±1：内层', color: '#ec4899' },
                { wn: 2, label: 'WN=±2：核心', color: '#7c3aed' },
              ].map(item => (
                <div key={item.wn} className="flex items-center gap-2">
                  <span className="w-3 h-3 rounded-full flex-shrink-0" style={{ background: item.color }}/>
                  <span className="text-xs text-slate-600 dark:text-slate-400">{item.label}</span>
                </div>
              ))}
            </div>

            <div className="rounded-xl bg-fuchsia-50 dark:bg-fuchsia-900/10 border border-fuchsia-200 dark:border-fuchsia-800 p-3 text-xs text-fuchsia-700 dark:text-fuchsia-300">
              <p className="font-bold mb-1">vs 射线法</p>
              <p>卷绕数对自相交多边形<strong>更精确</strong>，可区分重叠层数；射线法仅能给出奇偶。</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
