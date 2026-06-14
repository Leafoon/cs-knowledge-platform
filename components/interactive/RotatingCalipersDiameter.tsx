'use client'

import { useState, useEffect, useRef } from 'react'

const W = 360, H = 280
const CX = 180, CY = 140

type Pt = { x: number; y: number }

function dist2(a: Pt, b: Pt) { return Math.hypot(a.x - b.x, a.y - b.y) }

function convexHullFromAngles(n: number, scale = 100): Pt[] {
  const pts: Pt[] = []
  for (let i = 0; i < n; i++) {
    const a = (2 * Math.PI * i) / n - Math.PI / 2
    const r = scale * (0.7 + 0.3 * Math.sin(i * 2.3))
    pts.push({ x: Math.round(CX + r * Math.cos(a)), y: Math.round(CY + r * Math.sin(a)) })
  }
  return pts
}

const HULL = convexHullFromAngles(8)

export function RotatingCalipersDiameter() {
  const [step, setStep] = useState(0)
  const [playing, setPlaying] = useState(false)
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const n = HULL.length

  // Pre-compute antipodalPairs for each edge i (rotate caliper for vertex j)
  // Simplified: for each i, find j that maximizes dist
  const pairs: { i: number; j: number; d: number }[] = []
  for (let i = 0; i < n; i++) {
    let best = -1, bj = -1
    for (let j = 0; j < n; j++) {
      const d = dist2(HULL[i], HULL[j])
      if (d > best) { best = d; bj = j }
    }
    pairs.push({ i, j: bj, d: best })
  }

  const maxD = Math.max(...pairs.map(p => p.d))
  const diamPair = pairs.find(p => p.d === maxD)!
  const maxStep = n - 1
  const cur = pairs[Math.min(step, maxStep)]
  const pA = HULL[cur.i], pB = HULL[cur.j]

  useEffect(() => {
    if (playing) {
      timerRef.current = setTimeout(() => {
        setStep(s => {
          if (s >= maxStep) { setPlaying(false); return s }
          return s + 1
        })
      }, 700)
    }
    return () => { if (timerRef.current) clearTimeout(timerRef.current) }
  }, [playing, step, maxStep])

  // Support lines: perpendicular to current edge direction
  function supportLine(p: Pt, dir: Pt, len = 80) {
    return { x1: p.x - dir.x * len, y1: p.y - dir.y * len, x2: p.x + dir.x * len, y2: p.y + dir.y * len }
  }

  const ea = HULL[cur.i], eb = HULL[(cur.i + 1) % n]
  const dx = eb.x - ea.x, dy = eb.y - ea.y
  const len = Math.sqrt(dx * dx + dy * dy) || 1
  const ux = dx / len, uy = dy / len
  // perpendicular direction
  const px = -uy, py = ux
  const sl1 = supportLine(pA, { x: ux, y: uy })
  const sl2 = supportLine(pB, { x: ux, y: uy })

  return (
    <div className="rounded-2xl border border-emerald-200 dark:border-emerald-700 overflow-hidden shadow-lg font-sans">
      {/* Header */}
      <div className="bg-gradient-to-r from-emerald-600 via-green-600 to-teal-500 px-5 py-4">
        <h3 className="text-white font-bold text-base">📏 旋转卡壳：凸多边形直径</h3>
        <p className="text-emerald-50 text-xs mt-0.5">
          两条平行支撑线旋转一圈，对踵点对之间的最大距离即为直径
        </p>
        <div className="flex items-center gap-3 mt-3">
          <button onClick={() => { setStep(s => Math.max(0, s - 1)); setPlaying(false) }}
            disabled={step === 0}
            className="px-3 py-1 text-xs rounded-lg bg-white/20 text-white hover:bg-white/30 disabled:opacity-40 font-bold">
            ← 上步
          </button>
          <button onClick={() => {
            if (playing) { setPlaying(false) }
            else { if (step >= maxStep) setStep(0); setPlaying(true) }
          }}
            className="px-4 py-1 text-xs rounded-lg bg-white text-emerald-700 font-bold hover:bg-emerald-50">
            {playing ? '⏸ 暂停' : step >= maxStep ? '↺ 重播' : '▶ 播放'}
          </button>
          <button onClick={() => { setStep(s => Math.min(maxStep, s + 1)); setPlaying(false) }}
            disabled={step >= maxStep}
            className="px-3 py-1 text-xs rounded-lg bg-white/20 text-white hover:bg-white/30 disabled:opacity-40 font-bold">
            下步 →
          </button>
          <span className="text-white/70 text-xs ml-auto">步 {step + 1}/{n}</span>
        </div>
      </div>

      <div className="bg-white dark:bg-slate-900 p-4">
        <div className="flex gap-4 flex-wrap items-start">
          {/* SVG */}
          <svg width={W} height={H}
            className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 flex-shrink-0">
            {/* Convex hull */}
            <polygon points={HULL.map(p => `${p.x},${p.y}`).join(' ')}
              fill="#10b98110" stroke="#10b981" strokeWidth={2}/>

            {/* Support lines (calipers) */}
            <line {...sl1} stroke="#0ea5e9" strokeWidth={1.5} strokeDasharray="8,4" opacity={0.8}/>
            <line {...sl2} stroke="#0ea5e9" strokeWidth={1.5} strokeDasharray="8,4" opacity={0.8}/>

            {/* Current diameter */}
            <line x1={pA.x} y1={pA.y} x2={pB.x} y2={pB.y}
              stroke={cur.d === maxD ? '#f59e0b' : '#ef4444'}
              strokeWidth={cur.d === maxD ? 3.5 : 2.5}
              strokeLinecap="round"
              strokeDasharray={cur.d === maxD ? '' : '6,3'}
            />

            {/* Max diameter record */}
            {cur.d !== maxD && (
              <line x1={HULL[diamPair.i].x} y1={HULL[diamPair.i].y}
                x2={HULL[diamPair.j].x} y2={HULL[diamPair.j].y}
                stroke="#f59e0b40" strokeWidth={3} strokeLinecap="round"/>
            )}

            {/* Vertices */}
            {HULL.map((p, i) => {
              const isActive = i === cur.i || i === cur.j
              const isDiam = i === diamPair.i || i === diamPair.j
              return (
                <g key={i}>
                  <circle cx={p.x} cy={p.y} r={isActive ? 8 : isDiam ? 6 : 4}
                    fill={isActive ? (i === cur.i ? '#0ea5e9' : '#ef4444') : isDiam ? '#f59e0b80' : '#64748b'}
                    stroke="white" strokeWidth={1.5}/>
                  <text x={p.x + (p.x < CX ? -14 : 8)} y={p.y + (p.y < CY ? -6 : 16)}
                    fontSize={9} fontWeight="bold"
                    fill={isActive ? (i === cur.i ? '#0ea5e9' : '#ef4444') : '#64748b'}
                    className="dark:fill-slate-300">P{i}</text>
                </g>
              )
            })}

            {/* Labels */}
            <text x={pA.x - 22} y={pA.y} fontSize={11} fontWeight="bold" fill="#0ea5e9">A</text>
            <text x={pB.x + 6} y={pB.y} fontSize={11} fontWeight="bold" fill="#ef4444">B</text>
          </svg>

          {/* Info panel */}
          <div className="flex flex-col gap-3 flex-1 min-w-[140px]">
            <div className={`rounded-xl border-2 p-3 text-center transition-all ${
              cur.d === maxD
                ? 'border-amber-400 bg-amber-50 dark:bg-amber-900/15'
                : 'border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800'
            }`}>
              <p className="text-[10px] text-slate-500 dark:text-slate-400 font-bold uppercase mb-1">当前距离</p>
              <p className={`text-2xl font-black font-mono ${
                cur.d === maxD ? 'text-amber-600 dark:text-amber-400' : 'text-slate-600 dark:text-slate-300'
              }`}>{cur.d.toFixed(1)}</p>
              <p className="text-[10px] mt-0.5 text-slate-500">P{cur.i} ↔ P{cur.j}</p>
              {cur.d === maxD && <p className="text-xs text-amber-600 dark:text-amber-400 font-bold mt-1">⭐ 当前最大</p>}
            </div>

            <div className="rounded-xl border border-amber-200 dark:border-amber-800 bg-amber-50 dark:bg-amber-900/10 p-3">
              <p className="text-[10px] text-amber-600 dark:text-amber-400 font-bold uppercase mb-1">直径（最大距离）</p>
              <p className="text-xl font-black font-mono text-amber-700 dark:text-amber-300">{maxD.toFixed(1)}</p>
              <p className="text-[10px] mt-0.5 text-amber-600/70 dark:text-amber-400/70">P{diamPair.i} ↔ P{diamPair.j}</p>
            </div>

            <div className="rounded-xl bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 p-3 text-xs">
              <p className="font-bold text-slate-600 dark:text-slate-300 mb-2">各步对踵距离</p>
              <div className="space-y-1">
                {pairs.map((p, i) => (
                  <div key={i} className={`flex items-center gap-2 ${i === step ? 'font-bold' : 'opacity-60'}`}>
                    <span className="w-3 h-3 rounded-full flex-shrink-0"
                      style={{ background: p.d === maxD ? '#f59e0b' : i === step ? '#10b981' : '#94a3b8' }}/>
                    <span className="font-mono text-[10px] text-slate-600 dark:text-slate-400">
                      P{p.i}↔P{p.j}: {p.d.toFixed(0)}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            <div className="rounded-xl bg-emerald-50 dark:bg-emerald-900/10 border border-emerald-200 dark:border-emerald-800 p-3 text-xs text-emerald-700 dark:text-emerald-300">
              <p className="font-bold">时间复杂度</p>
              <p className="font-mono text-sm mt-0.5">O(n log n)</p>
              <p className="text-[10px] mt-1 opacity-70">含凸包构建；卡壳本身 O(n)</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
