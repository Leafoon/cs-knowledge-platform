'use client'

import { useState, useRef, useCallback } from 'react'

interface Pt { x: number; y: number }

const W = 320, H = 320, O_X = 160, O_Y = 160, SCALE = 50

function toSvg(p: Pt): Pt {
  return { x: O_X + p.x * SCALE, y: O_Y - p.y * SCALE }
}
function fromSvg(sx: number, sy: number): Pt {
  return { x: (sx - O_X) / SCALE, y: -(sy - O_Y) / SCALE }
}
function cross2(a: Pt, b: Pt): number { return a.x * b.y - a.y * b.x }
function norm(p: Pt): number { return Math.sqrt(p.x * p.x + p.y * p.y) }

const PRESETS: { label: string; A: Pt; B: Pt }[] = [
  { label: '左转 ↺', A: { x: 2, y: 0 }, B: { x: 1, y: 2 } },
  { label: '右转 ↻', A: { x: 1, y: 2 }, B: { x: 2.5, y: -0.5 } },
  { label: '共线 —', A: { x: 1, y: 1 }, B: { x: 2.5, y: 2.5 } },
]

type DragTarget = 'A' | 'B' | null

export function CrossProductViz() {
  const [A, setA] = useState<Pt>({ x: 2, y: 0 })
  const [B, setB] = useState<Pt>({ x: 1, y: 2 })
  const [dragging, setDragging] = useState<DragTarget>(null)
  const svgRef = useRef<SVGSVGElement>(null)

  const c = cross2(A, B)
  const absC = Math.abs(c)
  const EPS = 0.05

  const result = absC < EPS * SCALE * SCALE / SCALE / SCALE
    ? { label: '共线（O、A、B 三点共线）', color: '#94a3b8', bg: 'bg-slate-100 dark:bg-slate-800', text: 'text-slate-600 dark:text-slate-300', icon: '—' }
    : c > 0
    ? { label: `左转 ↺  (逆时针)`, color: '#6366f1', bg: 'bg-indigo-50 dark:bg-indigo-900/20', text: 'text-indigo-700 dark:text-indigo-300', icon: '↺' }
    : { label: `右转 ↻  (顺时针)`, color: '#ef4444', bg: 'bg-rose-50 dark:bg-rose-900/20', text: 'text-rose-700 dark:text-rose-300', icon: '↻' }

  function svgCoords(e: React.MouseEvent | React.TouchEvent): Pt | null {
    if (!svgRef.current) return null
    const rect = svgRef.current.getBoundingClientRect()
    const clientX = 'touches' in e ? e.touches[0].clientX : e.clientX
    const clientY = 'touches' in e ? e.touches[0].clientY : e.clientY
    const sx = (clientX - rect.left) / rect.width * W
    const sy = (clientY - rect.top) / rect.height * H
    return fromSvg(sx, sy)
  }

  function snap(v: number): number {
    const g = 0.5
    return Math.round(v / g) * g
  }

  function clamp(p: Pt): Pt {
    const r = 2.8
    const d = norm(p)
    if (d < 0.4) return p
    if (d > r) { const s = r / d; return { x: p.x * s, y: p.y * s } }
    return p
  }

  const onMouseMove = useCallback((e: React.MouseEvent) => {
    if (!dragging) return
    const p = svgCoords(e)
    if (!p) return
    const clamped = clamp({ x: snap(p.x), y: snap(p.y) })
    if (dragging === 'A') setA(clamped)
    else setB(clamped)
  }, [dragging])

  const svgA = toSvg(A)
  const svgB = toSvg(B)
  const o = toSvg({ x: 0, y: 0 })

  // Area parallelogram for fill
  const paraPath = `M${o.x},${o.y} L${svgA.x},${svgA.y} L${svgA.x + svgB.x - o.x},${svgA.y + svgB.y - o.y} L${svgB.x},${svgB.y} Z`

  // Arrow helper
  function arrow(x1: number, y1: number, x2: number, y2: number, color: string, id: string) {
    const dx = x2 - x1, dy = y2 - y1
    const len = Math.sqrt(dx * dx + dy * dy)
    if (len < 5) return null
    const ux = dx / len, uy = dy / len
    const head = 10
    const ax = x2 - ux * head - uy * head * 0.4
    const ay = y2 - uy * head + ux * head * 0.4
    const bx = x2 - ux * head + uy * head * 0.4
    const by = y2 - uy * head - ux * head * 0.4
    return (
      <g key={id}>
        <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={color} strokeWidth={2.5} strokeLinecap="round" />
        <polygon points={`${x2},${y2} ${ax},${ay} ${bx},${by}`} fill={color} />
      </g>
    )
  }

  return (
    <div className="rounded-2xl border border-indigo-200 dark:border-indigo-800 overflow-hidden shadow-lg font-sans">
      <div className="bg-gradient-to-r from-indigo-600 to-purple-600 px-5 py-4">
        <h3 className="text-white font-bold text-base">✕ 叉积方向判断：拖动端点感受"左转/右转/共线"</h3>
        <p className="text-indigo-100 text-xs mt-0.5">
          叉积 OA × OB = {c.toFixed(2)}　平行四边形面积 = |叉积| = {Math.abs(c).toFixed(2)}
        </p>
        <div className="flex gap-2 mt-3">
          {PRESETS.map((p, i) => (
            <button key={i} onClick={() => { setA(p.A); setB(p.B) }}
              className="px-2.5 py-1 text-xs rounded-lg bg-white/20 text-white hover:bg-white/35 transition-colors">
              {p.label}
            </button>
          ))}
        </div>
      </div>

      <div className="bg-white dark:bg-slate-900 p-4 flex gap-4 flex-wrap items-start">
        {/* SVG Canvas */}
        <svg ref={svgRef} width={W} height={H} viewBox={`0 0 ${W} ${H}`}
          className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 flex-shrink-0 cursor-crosshair select-none"
          onMouseMove={onMouseMove}
          onMouseUp={() => setDragging(null)}
          onMouseLeave={() => setDragging(null)}>

          {/* Grid */}
          {Array.from({ length: 12 }, (_, i) => i - 5).map(v => (
            <g key={v}>
              <line x1={O_X + v * SCALE} y1={20} x2={O_X + v * SCALE} y2={H - 20}
                stroke={v === 0 ? '#94a3b8' : '#e2e8f0'} strokeWidth={v === 0 ? 1.5 : 0.7} className="dark:stroke-slate-600" />
              <line x1={20} y1={O_Y + v * SCALE} x2={W - 20} y2={O_Y + v * SCALE}
                stroke={v === 0 ? '#94a3b8' : '#e2e8f0'} strokeWidth={v === 0 ? 1.5 : 0.7} className="dark:stroke-slate-600" />
            </g>
          ))}

          {/* Parallelogram fill (area) */}
          <path d={paraPath} fill={result.color} opacity={0.12} />
          <path d={paraPath} fill="none" stroke={result.color} strokeWidth={1} strokeDasharray="4 2" opacity={0.5} />

          {/* Vectors */}
          {arrow(o.x, o.y, svgA.x, svgA.y, '#6366f1', 'oa')}
          {arrow(o.x, o.y, svgB.x, svgB.y, '#f97316', 'ob')}

          {/* Angle arc */}
          {norm(A) > 0.3 && norm(B) > 0.3 && (() => {
            const r = 28
            const a1 = Math.atan2(-A.y, A.x)
            const a2 = Math.atan2(-B.y, B.x)
            const large = Math.abs(a2 - a1) > Math.PI ? 1 : 0
            const sweep = c > 0 ? 0 : 1
            return (
              <path d={`M ${o.x + r * Math.cos(a1)} ${o.y + r * Math.sin(a1)} A ${r} ${r} 0 ${large} ${sweep} ${o.x + r * Math.cos(a2)} ${o.y + r * Math.sin(a2)}`}
                fill="none" stroke={result.color} strokeWidth={2} />
            )
          })()}

          {/* Origin */}
          <circle cx={o.x} cy={o.y} r={5} fill="#64748b" />
          <text x={o.x - 14} y={o.y + 14} fontSize={12} fill="#64748b" fontWeight="bold">O</text>

          {/* Draggable A */}
          <g onMouseDown={(e) => { e.preventDefault(); setDragging('A') }} style={{ cursor: 'grab' }}>
            <circle cx={svgA.x} cy={svgA.y} r={10} fill="#6366f1" opacity={0.15} />
            <circle cx={svgA.x} cy={svgA.y} r={6} fill="#6366f1" stroke="white" strokeWidth={2} />
            <text x={svgA.x + 10} y={svgA.y - 8} fontSize={12} fill="#6366f1" fontWeight="bold">A</text>
            <text x={svgA.x + 10} y={svgA.y + 4} fontSize={9} fill="#818cf8">({A.x.toFixed(1)},{A.y.toFixed(1)})</text>
          </g>

          {/* Draggable B */}
          <g onMouseDown={(e) => { e.preventDefault(); setDragging('B') }} style={{ cursor: 'grab' }}>
            <circle cx={svgB.x} cy={svgB.y} r={10} fill="#f97316" opacity={0.15} />
            <circle cx={svgB.x} cy={svgB.y} r={6} fill="#f97316" stroke="white" strokeWidth={2} />
            <text x={svgB.x + 10} y={svgB.y - 8} fontSize={12} fill="#f97316" fontWeight="bold">B</text>
            <text x={svgB.x + 10} y={svgB.y + 4} fontSize={9} fill="#fb923c">({B.x.toFixed(1)},{B.y.toFixed(1)})</text>
          </g>
        </svg>

        {/* Info panel */}
        <div className="flex-1 min-w-[180px] space-y-3">
          {/* Result badge */}
          <div className={`rounded-xl border p-4 ${result.bg} ${c > 0 ? 'border-indigo-200 dark:border-indigo-800' : c < 0 ? 'border-rose-200 dark:border-rose-800' : 'border-slate-200 dark:border-slate-700'}`}>
            <div className="text-3xl font-bold text-center mb-1">{result.icon}</div>
            <p className={`text-sm font-bold text-center ${result.text}`}>{result.label}</p>
          </div>

          {/* Formula */}
          <div className="rounded-xl bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 p-3 space-y-2">
            <p className="text-xs font-bold text-slate-600 dark:text-slate-300">叉积公式</p>
            <div className="font-mono text-xs text-slate-700 dark:text-slate-200 space-y-1">
              <p>OA × OB</p>
              <p className="text-slate-400">= A.x × B.y - A.y × B.x</p>
              <p>= {A.x.toFixed(1)} × {B.y.toFixed(1)} - {A.y.toFixed(1)} × {B.x.toFixed(1)}</p>
              <p className={`font-bold text-base ${result.text}`}>= {c.toFixed(3)}</p>
            </div>
          </div>

          {/* Area */}
          <div className="rounded-xl bg-amber-50 dark:bg-amber-900/10 border border-amber-200 dark:border-amber-800 p-3">
            <p className="text-xs font-bold text-amber-700 dark:text-amber-300 mb-1">△OAB 面积</p>
            <p className="font-mono text-lg font-bold text-amber-600 dark:text-amber-400">{(Math.abs(c) / 2).toFixed(3)}</p>
            <p className="text-xs text-amber-600/70 dark:text-amber-500 mt-0.5">= |叉积| / 2 = {Math.abs(c).toFixed(3)} / 2</p>
          </div>

          <p className="text-[10px] text-slate-400 dark:text-slate-600 leading-relaxed">
            拖动蓝色点 A 或橙色点 B 改变向量方向，观察叉积符号和平行四边形面积的变化。
          </p>
        </div>
      </div>
    </div>
  )
}
