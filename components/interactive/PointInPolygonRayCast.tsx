'use client'

import { useState, useCallback, useRef } from 'react'

const W = 360, H = 280

type Pt = { x: number; y: number }

const DEFAULT_POLY: Pt[] = [
  { x: 80, y: 220 }, { x: 80, y: 100 }, { x: 160, y: 60 },
  { x: 240, y: 100 }, { x: 280, y: 180 }, { x: 200, y: 230 },
]

function raycastHit(q: Pt, a: Pt, b: Pt): boolean {
  // Ray going right (+x) from q
  if ((a.y <= q.y && b.y > q.y) || (b.y <= q.y && a.y > q.y)) {
    const t = (q.y - a.y) / (b.y - a.y)
    const ix = a.x + t * (b.x - a.x)
    if (ix > q.x) return true
  }
  return false
}

function isInsideRayCast(q: Pt, poly: Pt[]): boolean {
  const n = poly.length
  let inside = false
  for (let i = 0; i < n; i++) {
    if (raycastHit(q, poly[i], poly[(i + 1) % n])) inside = !inside
  }
  return inside
}

export function PointInPolygonRayCast() {
  const [query, setQuery] = useState<Pt>({ x: 170, y: 155 })
  const [dragging, setDragging] = useState(false)
  const svgRef = useRef<SVGSVGElement>(null)

  const poly = DEFAULT_POLY
  const n = poly.length

  // Compute per-edge ray intersection
  const rayMaxX = W + 30
  const edges = poly.map((a, i) => {
    const b = poly[(i + 1) % n]
    const hit = raycastHit(query, a, b)
    let ix = 0
    if (hit) {
      const t = (query.y - a.y) / (b.y - a.y)
      ix = a.x + t * (b.x - a.x)
    }
    return { a, b, hit, ix }
  })
  const crossings = edges.filter(e => e.hit).length
  const inside = crossings % 2 === 1

  const handleSvgMove = useCallback((e: React.MouseEvent<SVGSVGElement>) => {
    if (!dragging || !svgRef.current) return
    const rect = svgRef.current.getBoundingClientRect()
    const x = Math.max(5, Math.min(W - 5, e.clientX - rect.left))
    const y = Math.max(5, Math.min(H - 5, e.clientY - rect.top))
    setQuery({ x, y })
  }, [dragging])

  const handleSvgTouch = useCallback((e: React.TouchEvent<SVGSVGElement>) => {
    if (!svgRef.current) return
    const rect = svgRef.current.getBoundingClientRect()
    const touch = e.touches[0]
    const x = Math.max(5, Math.min(W - 5, touch.clientX - rect.left))
    const y = Math.max(5, Math.min(H - 5, touch.clientY - rect.top))
    setQuery({ x, y })
  }, [])

  return (
    <div className="rounded-2xl border border-teal-200 dark:border-teal-700 overflow-hidden shadow-lg font-sans">
      {/* Header */}
      <div className="bg-gradient-to-r from-teal-600 via-cyan-600 to-sky-500 px-5 py-4">
        <h3 className="text-white font-bold text-base">🎯 射线法：点是否在多边形内</h3>
        <p className="text-teal-50 text-xs mt-0.5">拖动查询点 Q，观察射线与各边的交叉次数（奇数=内部，偶数=外部）</p>
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
              onMouseMove={handleSvgMove}
              onTouchMove={handleSvgTouch}
            >
              {/* Polygon fill */}
              <polygon
                points={poly.map(p => `${p.x},${p.y}`).join(' ')}
                fill={inside ? '#14b8a620' : '#94a3b815'}
                stroke="none"
              />

              {/* Ray */}
              <line x1={query.x} y1={query.y} x2={rayMaxX} y2={query.y}
                stroke="#f59e0b" strokeWidth={1.5} strokeDasharray="8,4"
                markerEnd="url(#arrowhead)"
              />
              <defs>
                <marker id="arrowhead" markerWidth="6" markerHeight="4" refX="6" refY="2" orient="auto">
                  <polygon points="0 0, 6 2, 0 4" fill="#f59e0b"/>
                </marker>
              </defs>

              {/* Polygon edges */}
              {edges.map((e, i) => (
                <line key={i} x1={e.a.x} y1={e.a.y} x2={e.b.x} y2={e.b.y}
                  stroke={e.hit ? '#0d9488' : '#64748b'}
                  strokeWidth={e.hit ? 3 : 2}
                  strokeLinecap="round"
                />
              ))}

              {/* Intersection markers */}
              {edges.filter(e => e.hit).map((e, i) => (
                <g key={i}>
                  <circle cx={e.ix} cy={query.y} r={7}
                    fill="#f59e0b" stroke="white" strokeWidth={2}/>
                  <text x={e.ix} y={query.y} textAnchor="middle" dominantBaseline="middle"
                    fontSize={8} fontWeight="bold" fill="white">{i + 1}</text>
                </g>
              ))}

              {/* Polygon vertices */}
              {poly.map((p, i) => (
                <circle key={i} cx={p.x} cy={p.y} r={4}
                  fill="#64748b" stroke="white" strokeWidth={1.5}/>
              ))}

              {/* Query point Q */}
              <circle cx={query.x} cy={query.y} r={10}
                fill={inside ? '#14b8a6' : '#94a3b8'} stroke="white" strokeWidth={2.5}
                className="drop-shadow-md cursor-grab active:cursor-grabbing"
              />
              <text x={query.x} y={query.y} textAnchor="middle" dominantBaseline="middle"
                fontSize={9} fontWeight="bold" fill="white">Q</text>

              {/* Crossings counter near ray */}
              <rect x={rayMaxX - 48} y={query.y - 22} width={44} height={18} rx={5}
                fill={inside ? '#0d9488' : '#64748b'} opacity={0.9}/>
              <text x={rayMaxX - 26} y={query.y - 9} textAnchor="middle"
                fontSize={9} fontWeight="bold" fill="white">
                {crossings}次交叉
              </text>
            </svg>
            <p className="text-[10px] text-slate-400 dark:text-slate-500 text-center mt-1">
              拖动 Q 点 · 亮绿色边 = 射线穿越
            </p>
          </div>

          {/* Right panel */}
          <div className="flex flex-col gap-3 min-w-[150px] flex-1">
            {/* Result badge */}
            <div className={`rounded-2xl border-2 text-center py-4 transition-all ${
              inside
                ? 'border-teal-400 bg-teal-50 dark:bg-teal-900/20'
                : 'border-slate-300 bg-slate-50 dark:bg-slate-800'
            }`}>
              <div className={`text-3xl mb-1`}>{inside ? '✅' : '⛔'}</div>
              <p className={`font-black text-base ${
                inside ? 'text-teal-700 dark:text-teal-300' : 'text-slate-500 dark:text-slate-400'
              }`}>{inside ? '内 部' : '外 部'}</p>
              <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                {crossings} 次交叉 → {crossings % 2 === 1 ? '奇数' : '偶数'}
              </p>
            </div>

            {/* Edge list */}
            <div className="rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden text-xs">
              <div className="bg-slate-100 dark:bg-slate-800 px-2 py-1.5 text-[10px] font-bold text-slate-500 dark:text-slate-400 uppercase">
                各边检测结果
              </div>
              {edges.map((e, i) => (
                <div key={i} className={`flex items-center gap-2 px-2 py-1 border-t border-slate-100 dark:border-slate-700 ${
                  e.hit ? 'bg-teal-50 dark:bg-teal-900/10' : ''
                }`}>
                  <span className="font-mono text-slate-600 dark:text-slate-400 w-14">
                    P{i}→P{(i + 1) % n}
                  </span>
                  {e.hit
                    ? <span className="text-teal-600 dark:text-teal-400 font-bold">✓ 交叉</span>
                    : <span className="text-slate-400">— 未穿越</span>
                  }
                </div>
              ))}
            </div>

            <div className="rounded-xl bg-amber-50 dark:bg-amber-900/10 border border-amber-200 dark:border-amber-800 p-3 text-xs text-amber-700 dark:text-amber-300">
              <p className="font-bold mb-1">奇偶规则</p>
              <p>交叉次数为<strong>奇数</strong> → 在多边形内部</p>
              <p>交叉次数为<strong>偶数</strong> → 在多边形外部</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
