'use client'

import { useState, useCallback } from 'react'

const W = 340, H = 260, PAD = 30

// Preset polygons in SVG coordinates (y-down)
const PRESETS = [
  {
    label: '矩形 4×3',
    pts: [{x:60,y:200},{x:260,y:200},{x:260,y:80},{x:60,y:80}],
    expectedArea: 12,
  },
  {
    label: '凹多边形 L 形',
    pts: [{x:60,y:200},{x:200,y:200},{x:200,y:140},{x:140,y:140},{x:140,y:80},{x:60,y:80}],
    expectedArea: 3,
  },
  {
    label: '五边形',
    pts: [
      {x:170,y:50},{x:290,y:145},{x:250,y:240},
      {x:90,y:240},{x:50,y:145},
    ],
    expectedArea: null,
  },
  {
    label: '三角形',
    pts: [{x:170,y:55},{x:290,y:220},{x:50,y:220}],
    expectedArea: null,
  },
]

type Pt = { x: number; y: number }

// Convert SVG coords (y-down) to "math" coords (y-up) for Shoelace
function svgToMath(p: Pt, scale = 40): Pt {
  return { x: (p.x - PAD) / scale, y: -(p.y - H + PAD) / scale }
}

function shoelaceSteps(pts: Pt[]) {
  const n = pts.length
  const math = pts.map(p => svgToMath(p))
  const steps: { i: number; j: number; xi: number; yi: number; xj: number; yj: number; term: number; cumSum: number }[] = []
  let cum = 0
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n
    const term = math[i].x * math[j].y - math[j].x * math[i].y
    cum += term
    steps.push({ i, j, xi: math[i].x, yi: math[i].y, xj: math[j].x, yj: math[j].y, term, cumSum: cum })
  }
  return { steps, area: Math.abs(cum) / 2, signed2: cum }
}

export function ShoelacePolygonArea() {
  const [presetIdx, setPresetIdx] = useState(0)
  const [activeStep, setActiveStep] = useState<number | null>(null)
  const preset = PRESETS[presetIdx]
  const pts = preset.pts
  const { steps, area, signed2 } = shoelaceSteps(pts)
  const n = pts.length

  const handlePreset = useCallback((i: number) => {
    setPresetIdx(i)
    setActiveStep(null)
  }, [])

  const activePt = activeStep !== null ? steps[activeStep] : null

  // For drawing cross-lines of active step "shoelace" visual
  const getEdgeColor = (i: number) => {
    if (activeStep === null) return '#6366f1'
    return i === activeStep ? '#f59e0b' : '#6366f130'
  }

  return (
    <div className="rounded-2xl border border-indigo-200 dark:border-indigo-700 overflow-hidden shadow-lg font-sans">
      {/* Header */}
      <div className="bg-gradient-to-r from-indigo-600 via-purple-600 to-pink-500 px-5 py-4">
        <h3 className="text-white font-bold text-base">📐 Shoelace 公式：逐边累加可视化</h3>
        <p className="text-indigo-100 text-xs mt-0.5">
          点击下方表格中每一行，观察对应边对面积的贡献（正/负/大小）
        </p>
        <div className="flex gap-1.5 mt-3 flex-wrap">
          {PRESETS.map((p, i) => (
            <button key={i} onClick={() => handlePreset(i)}
              className={`px-2.5 py-1 text-xs rounded-lg transition-colors ${
                presetIdx === i ? 'bg-white text-indigo-700 font-bold' : 'bg-white/20 text-white hover:bg-white/30'
              }`}>{p.label}</button>
          ))}
        </div>
      </div>

      <div className="bg-white dark:bg-slate-900 p-4">
        <div className="flex gap-4 flex-wrap items-start">
          {/* SVG Canvas */}
          <div className="flex-shrink-0">
            <svg width={W} height={H} className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800">
              {/* Shoelace cross-lines for active step */}
              {activePt && (() => {
                const a = pts[activePt.i], b = pts[activePt.j]
                return (
                  <g opacity={0.35}>
                    <line x1={a.x} y1={H-PAD} x2={b.x} y2={H-PAD} stroke="#f59e0b" strokeWidth={1} strokeDasharray="3,2"/>
                    <line x1={a.x} y1={H-PAD} x2={a.x} y2={a.y} stroke="#f59e0b" strokeWidth={1} strokeDasharray="3,2"/>
                    <line x1={b.x} y1={H-PAD} x2={b.x} y2={b.y} stroke="#f59e0b" strokeWidth={1} strokeDasharray="3,2"/>
                  </g>
                )
              })()}

              {/* Polygon fill */}
              <polygon
                points={pts.map(p => `${p.x},${p.y}`).join(' ')}
                fill={activePt ? '#6366f108' : '#6366f115'}
                stroke="none"
              />

              {/* Edges */}
              {pts.map((p, i) => {
                const j = (i + 1) % n
                const q = pts[j]
                const isActive = activeStep === i
                return (
                  <line key={i} x1={p.x} y1={p.y} x2={q.x} y2={q.y}
                    stroke={getEdgeColor(i)}
                    strokeWidth={isActive ? 3.5 : 1.5}
                    strokeLinecap="round"
                  />
                )
              })}

              {/* Active edge triangle to x-axis */}
              {activePt && (() => {
                const a = pts[activePt.i], b = pts[activePt.j]
                const sign = activePt.term
                return (
                  <polygon
                    points={`${a.x},${a.y} ${b.x},${b.y} ${b.x},${H-PAD} ${a.x},${H-PAD}`}
                    fill={sign >= 0 ? '#10b98125' : '#ef444425'}
                    stroke={sign >= 0 ? '#10b981' : '#ef4444'}
                    strokeWidth={1}
                    strokeDasharray="4,2"
                  />
                )
              })()}

              {/* Points */}
              {pts.map((p, i) => {
                const isActiveEndpoint = activePt && (activePt.i === i || activePt.j === i)
                return (
                  <g key={i}>
                    <circle cx={p.x} cy={p.y} r={isActiveEndpoint ? 8 : 5}
                      fill={isActiveEndpoint ? '#f59e0b' : '#6366f1'} stroke="white" strokeWidth={1.5}/>
                    <text x={p.x + 8} y={p.y - 7} fontSize={10} fill="#6366f1" fontWeight="bold"
                      className="dark:fill-indigo-300">P{i}</text>
                  </g>
                )
              })}

              {/* x-axis (base line) */}
              <line x1={PAD-5} y1={H-PAD} x2={W-PAD+5} y2={H-PAD} stroke="#94a3b8" strokeWidth={1} strokeDasharray="4,3"/>
              <text x={W-PAD+6} y={H-PAD+4} fontSize={9} fill="#94a3b8">x轴</text>

              {/* Area label */}
              <text x={W/2} y={30} textAnchor="middle" fontSize={13} fontWeight="bold" fill="#6366f1"
                className="dark:fill-indigo-300">
                S = {area.toFixed(2)}
              </text>
            </svg>
          </div>

          {/* Step table */}
          <div className="flex-1 min-w-[200px]">
            <p className="text-xs font-bold text-slate-600 dark:text-slate-300 mb-2">
              逐边累加（点击行高亮）
            </p>
            <div className="rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden text-xs">
              {/* Header */}
              <div className="grid grid-cols-4 bg-slate-100 dark:bg-slate-800 text-[10px] font-bold text-slate-500 dark:text-slate-400 uppercase">
                <div className="px-2 py-1.5">边</div>
                <div className="px-2 py-1.5 text-center">xi·yj</div>
                <div className="px-2 py-1.5 text-center">xj·yi</div>
                <div className="px-2 py-1.5 text-center">贡献</div>
              </div>
              {steps.map((s, idx) => {
                const isActive = activeStep === idx
                const pos = s.term >= 0
                return (
                  <button key={idx} onClick={() => setActiveStep(isActive ? null : idx)}
                    className={`w-full grid grid-cols-4 border-t border-slate-100 dark:border-slate-700 transition-colors text-left ${
                      isActive
                        ? 'bg-amber-50 dark:bg-amber-900/20'
                        : 'hover:bg-slate-50 dark:hover:bg-slate-800/50'
                    }`}>
                    <div className={`px-2 py-1.5 font-mono font-bold ${isActive ? 'text-amber-600 dark:text-amber-400' : 'text-indigo-600 dark:text-indigo-400'}`}>
                      P{s.i}→P{s.j}
                    </div>
                    <div className="px-2 py-1.5 text-center font-mono text-slate-500 dark:text-slate-400">
                      {(s.xi * s.yj).toFixed(1)}
                    </div>
                    <div className="px-2 py-1.5 text-center font-mono text-slate-500 dark:text-slate-400">
                      {(s.xj * s.yi).toFixed(1)}
                    </div>
                    <div className={`px-2 py-1.5 text-center font-mono font-bold ${
                      pos ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400'
                    }`}>
                      {pos ? '+' : ''}{s.term.toFixed(1)}
                    </div>
                  </button>
                )
              })}
              {/* Total */}
              <div className="grid grid-cols-4 border-t-2 border-indigo-200 dark:border-indigo-700 bg-indigo-50 dark:bg-indigo-900/20">
                <div className="px-2 py-2 font-bold text-indigo-700 dark:text-indigo-300 col-span-3">
                  S = |Σ| / 2
                </div>
                <div className="px-2 py-2 text-center font-black text-indigo-700 dark:text-indigo-300">
                  {area.toFixed(2)}
                </div>
              </div>
            </div>

            {/* Active step explanation */}
            {activePt && (
              <div className={`mt-2 rounded-xl p-3 text-xs border leading-relaxed ${
                activePt.term >= 0
                  ? 'bg-emerald-50 dark:bg-emerald-900/10 border-emerald-200 dark:border-emerald-800 text-emerald-700 dark:text-emerald-300'
                  : 'bg-rose-50 dark:bg-rose-900/10 border-rose-200 dark:border-rose-800 text-rose-700 dark:text-rose-300'
              }`}>
                <p className="font-bold mb-0.5">边 P{activePt.i}→P{activePt.j}</p>
                <p className="font-mono">{activePt.xi.toFixed(1)}×{activePt.yj.toFixed(1)} − {activePt.xj.toFixed(1)}×{activePt.yi.toFixed(1)} = <strong>{activePt.term.toFixed(2)}</strong></p>
                <p className="mt-1 text-[10px]">{activePt.term >= 0 ? '✓ 正贡献（逆时针转角，增面积）' : '✗ 负贡献（顺时针转角，减面积）'}</p>
              </div>
            )}
          </div>
        </div>

        {/* Formula reminder */}
        <div className="mt-3 rounded-xl bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 px-4 py-2.5 text-xs text-slate-500 dark:text-slate-400 font-mono text-center">
          S = ½ |Σ (xᵢ·yᵢ₊₁ − xᵢ₊₁·yᵢ)| &nbsp;=&nbsp; ½ · |{(signed2).toFixed(2)}| &nbsp;=&nbsp; <strong className="text-indigo-600 dark:text-indigo-400">{area.toFixed(2)}</strong>
        </div>
      </div>
    </div>
  )
}
