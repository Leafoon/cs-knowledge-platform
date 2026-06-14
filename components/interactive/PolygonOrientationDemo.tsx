'use client'

import { useState } from 'react'

const W = 320, H = 240

type Pt = { x: number; y: number }

const POLYGON: Pt[] = [
  { x: 160, y: 35 },
  { x: 260, y: 85 },
  { x: 280, y: 175 },
  { x: 200, y: 210 },
  { x: 120, y: 210 },
  { x: 40, y: 175 },
  { x: 60, y: 85 },
]

function cross2(o: Pt, a: Pt, b: Pt) {
  return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
}

function computeSignedArea(pts: Pt[]) {
  const n = pts.length
  let s = 0
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n
    s += pts[i].x * pts[j].y - pts[j].x * pts[i].y
  }
  return s / 2
}

export function PolygonOrientationDemo() {
  const [cw, setCw] = useState(false)
  const [activeEdge, setActiveEdge] = useState<number | null>(null)

  const pts = cw ? [...POLYGON].reverse() : POLYGON
  const signedArea = computeSignedArea(pts)
  const n = pts.length

  // Mid-edge arrow
  function arrowAt(i: number, highlight: boolean) {
    const a = pts[i], b = pts[(i + 1) % n]
    const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2
    const dx = b.x - a.x, dy = b.y - a.y
    const len = Math.sqrt(dx * dx + dy * dy) || 1
    const ux = dx / len, uy = dy / len
    const sz = 8
    const tip = { x: mx + ux * sz, y: my + uy * sz }
    const l = { x: mx - ux * sz - uy * sz * 0.5, y: my - uy * sz + ux * sz * 0.5 }
    const r = { x: mx - ux * sz + uy * sz * 0.5, y: my - uy * sz - ux * sz * 0.5 }
    const color = highlight ? '#f59e0b' : (cw ? '#ef4444' : '#3b82f6')
    return (
      <polygon key={`arr-${i}`}
        points={`${tip.x},${tip.y} ${l.x},${l.y} ${r.x},${r.y}`}
        fill={color} opacity={0.9}
      />
    )
  }

  return (
    <div className="rounded-2xl border border-slate-200 dark:border-slate-700 overflow-hidden shadow-md font-sans">
      {/* Header */}
      <div className={`px-5 py-4 transition-colors duration-500 ${
        cw
          ? 'bg-gradient-to-r from-rose-600 to-red-500'
          : 'bg-gradient-to-r from-blue-600 to-cyan-500'
      }`}>
        <h3 className="text-white font-bold text-base">🔄 多边形方向：CCW vs CW</h3>
        <p className="text-white/80 text-xs mt-0.5">有向面积的符号由顶点遍历方向决定。切换方向，观察符号变化。</p>
      </div>

      <div className="bg-white dark:bg-slate-900 p-4">
        {/* Toggle */}
        <div className="flex justify-center mb-4">
          <div className="flex bg-slate-100 dark:bg-slate-800 rounded-xl p-1 gap-1">
            <button onClick={() => { setCw(false); setActiveEdge(null) }}
              className={`px-4 py-1.5 rounded-lg text-xs font-bold transition-colors ${
                !cw ? 'bg-blue-600 text-white shadow' : 'text-slate-500 dark:text-slate-400 hover:bg-slate-200 dark:hover:bg-slate-700'
              }`}>
              ↺ 逆时针 CCW（正）
            </button>
            <button onClick={() => { setCw(true); setActiveEdge(null) }}
              className={`px-4 py-1.5 rounded-lg text-xs font-bold transition-colors ${
                cw ? 'bg-rose-600 text-white shadow' : 'text-slate-500 dark:text-slate-400 hover:bg-slate-200 dark:hover:bg-slate-700'
              }`}>
              ↻ 顺时针 CW（负）
            </button>
          </div>
        </div>

        <div className="flex gap-4 flex-wrap justify-center">
          {/* Canvas */}
          <svg width={W} height={H} className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800">
            {/* Fill color changes */}
            <polygon
              points={pts.map(p => `${p.x},${p.y}`).join(' ')}
              fill={cw ? '#ef444418' : '#3b82f618'}
              stroke="none"
            />
            {/* Edges */}
            {pts.map((p, i) => {
              const j = (i + 1) % n
              const q = pts[j]
              const isActive = activeEdge === i
              return (
                <line key={i} x1={p.x} y1={p.y} x2={q.x} y2={q.y}
                  stroke={isActive ? '#f59e0b' : (cw ? '#ef4444' : '#3b82f6')}
                  strokeWidth={isActive ? 3 : 2}
                  strokeLinecap="round"
                  className="cursor-pointer"
                  onClick={() => setActiveEdge(isActive ? null : i)}
                />
              )
            })}
            {/* Arrows on each edge */}
            {pts.map((_, i) => arrowAt(i, activeEdge === i))}
            {/* Points */}
            {pts.map((p, i) => (
              <g key={i}>
                <circle cx={p.x} cy={p.y} r={5}
                  fill={cw ? '#ef4444' : '#3b82f6'} stroke="white" strokeWidth={1.5}/>
                {i === 0 && (
                  <circle cx={p.x} cy={p.y} r={9}
                    fill="none" stroke={cw ? '#ef4444' : '#3b82f6'} strokeWidth={1.5} strokeDasharray="3,2"/>
                )}
                <text x={p.x + 7} y={p.y - 6} fontSize={9} fontWeight="bold"
                  fill={cw ? '#ef4444' : '#3b82f6'}>P{i}</text>
              </g>
            ))}
            {/* Center label */}
            <text x={W/2} y={H/2} textAnchor="middle" dominantBaseline="middle"
              fontSize={12} fontWeight="bold"
              fill={cw ? '#ef4444' : '#3b82f6'} opacity={0.6}>
              {cw ? 'CW ↻' : 'CCW ↺'}
            </text>
          </svg>

          {/* Info panel */}
          <div className="flex flex-col gap-3 min-w-[160px] justify-center">
            <div className={`rounded-xl p-3 border text-center transition-colors ${
              cw
                ? 'bg-rose-50 dark:bg-rose-900/10 border-rose-200 dark:border-rose-800'
                : 'bg-blue-50 dark:bg-blue-900/10 border-blue-200 dark:border-blue-800'
            }`}>
              <p className="text-[10px] text-slate-500 dark:text-slate-400 uppercase font-bold tracking-wide mb-1">有向面积</p>
              <p className={`text-2xl font-black font-mono ${
                cw ? 'text-rose-600 dark:text-rose-400' : 'text-blue-600 dark:text-blue-400'
              }`}>
                {signedArea > 0 ? '+' : ''}{signedArea.toFixed(0)}
              </p>
              <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
                {cw ? '< 0 ⟹ 顺时针' : '> 0 ⟹ 逆时针'}
              </p>
            </div>

            <div className="rounded-xl p-3 bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700">
              <p className="text-[10px] text-slate-500 dark:text-slate-400 uppercase font-bold tracking-wide mb-1">|实际面积|</p>
              <p className="text-xl font-black font-mono text-slate-700 dark:text-slate-200">
                {Math.abs(signedArea).toFixed(0)}
              </p>
              <p className="text-[10px] text-slate-500 dark:text-slate-400 mt-0.5">符号取绝对值</p>
            </div>

            <div className="rounded-xl p-3 bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-xs text-slate-600 dark:text-slate-300">
              <p className="font-bold mb-1">方向判断</p>
              <p className="font-mono text-[10px] leading-relaxed">
                cross(P0,P1,P2)<br/>
                {cross2(pts[0], pts[1], pts[2]) > 0 ? '> 0 → CCW' : '< 0 → CW'}
              </p>
            </div>
          </div>
        </div>

        <p className="text-[10px] text-slate-400 dark:text-slate-500 text-center mt-3">
          点击边可高亮查看 · 起点 P0 以双圆圈标记
        </p>
      </div>
    </div>
  )
}
