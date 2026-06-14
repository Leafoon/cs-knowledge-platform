"use client"

import { useState, useEffect, useRef } from 'react'
import { motion } from 'framer-motion'
import { Play, Pause, RotateCcw } from 'lucide-react'

// Compute potential Φ = 2*size - capacity for each push_back step
// We use the convention Φ₀ = 0 (initial state: size=0, cap=1, offset+1)
function buildSteps(n: number) {
  let size = 0
  let cap = 1
  // Φ = 2*size - cap + 1 (adjusted so Φ₀ = 0)
  const phiOf = (s: number, c: number) => 2 * s - c + 1

  const steps: {
    op: number
    size: number
    cap: number
    doubled: boolean
    realCost: number
    phiBefore: number
    phiAfter: number
    deltaPhi: number
    amortized: number
  }[] = []

  for (let i = 1; i <= n; i++) {
    const phiBefore = phiOf(size, cap)
    const doubled = size === cap
    const realCost = doubled ? cap + 1 : 1
    if (doubled) cap = cap * 2
    size++
    const phiAfter = phiOf(size, cap)
    const deltaPhi = phiAfter - phiBefore

    steps.push({
      op: i,
      size,
      cap,
      doubled,
      realCost,
      phiBefore,
      phiAfter,
      deltaPhi,
      amortized: realCost + deltaPhi,
    })
  }
  return steps
}

const N_DEFAULT = 16

export function PotentialMethodVisualizer() {
  const [n, setN] = useState(N_DEFAULT)
  const [step, setStep] = useState(0)
  const [playing, setPlaying] = useState(false)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const steps = buildSteps(n)
  const shown = steps.slice(0, step)

  // Auto-play
  useEffect(() => {
    if (playing) {
      timerRef.current = setInterval(() => {
        setStep(prev => {
          if (prev >= n) { setPlaying(false); return prev }
          return prev + 1
        })
      }, 300)
    }
    return () => { if (timerRef.current) clearInterval(timerRef.current) }
  }, [playing, n])

  const reset = () => { setPlaying(false); setStep(0) }
  const changeN = (newN: number) => { setN(newN); setStep(0); setPlaying(false) }

  // Chart dimensions
  const chartW = 520
  const chartH = 160
  const padL = 50
  const padB = 24
  const plotW = chartW - padL - 10
  const plotH = chartH - padB - 10

  // Scale
  const maxPhi = steps.length > 0 ? Math.max(...steps.map(s => Math.max(s.phiBefore, s.phiAfter, s.realCost, 4))) : 10
  const minPhi = Math.min(0, steps.length > 0 ? Math.min(...steps.map(s => s.phiBefore)) : 0)
  const range = maxPhi - minPhi || 1
  const xStep = plotW / (n + 1)

  const toY = (v: number) => plotH - ((v - minPhi) / range) * plotH + 10

  const phiPoints = shown.map((s, i) => `${padL + (i + 1) * xStep},${toY(s.phiAfter)}`)
  const realPoints = shown.map((s, i) => `${padL + (i + 1) * xStep},${toY(s.realCost)}`)
  const amortPoints = shown.map((s, i) => `${padL + (i + 1) * xStep},${toY(s.amortized)}`)

  return (
    <div className="rounded-xl border border-violet-200 dark:border-violet-800 bg-white dark:bg-gray-900 overflow-hidden shadow-sm my-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-violet-500 to-purple-600 px-5 py-3">
        <h3 className="text-white font-bold text-base">势能法可视化：Φ = 2·size − capacity + 1</h3>
        <p className="text-violet-100 text-xs mt-0.5">
          实际代价 c、势能 Φ、摊销代价 ĉ = c + ΔΦ 的动态曲线
        </p>
      </div>

      <div className="p-4 space-y-4">
        {/* Controls */}
        <div className="flex flex-wrap items-center gap-3">
          <span className="text-sm text-gray-600 dark:text-gray-400">n =</span>
          {[8, 16, 24].map(v => (
            <button key={v} onClick={() => changeN(v)}
              className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
                n === v
                  ? 'bg-violet-600 text-white'
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
              }`}
            >{v}</button>
          ))}
          <div className="flex items-center gap-2 ml-auto">
            <button onClick={reset} className="p-1.5 rounded bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors">
              <RotateCcw size={16} className="text-gray-600 dark:text-gray-300" />
            </button>
            <button onClick={() => setStep(p => Math.max(0, p - 1))} disabled={step === 0}
              className="px-2 py-1 rounded text-sm bg-gray-100 dark:bg-gray-700 disabled:opacity-40 hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors text-gray-700 dark:text-gray-300">‹</button>
            <button
              onClick={() => { if (playing) { setPlaying(false) } else { if (step >= n) setStep(0); setPlaying(true) } }}
              className="flex items-center gap-1 px-3 py-1.5 rounded bg-violet-600 text-white hover:bg-violet-700 transition-colors text-sm"
            >
              {playing ? <Pause size={14} /> : <Play size={14} />}
              {playing ? '暂停' : step >= n ? '重播' : '播放'}
            </button>
            <button onClick={() => setStep(p => Math.min(n, p + 1))} disabled={step >= n}
              className="px-2 py-1 rounded text-sm bg-gray-100 dark:bg-gray-700 disabled:opacity-40 hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors text-gray-700 dark:text-gray-300">›</button>
          </div>
        </div>

        {/* SVG Chart */}
        <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-2 overflow-x-auto">
          <svg width={chartW} height={chartH} className="block">
            {/* Y grid lines & labels */}
            {[-2, 0, 2, 4, 6, 8].filter(v => v >= minPhi && v <= maxPhi).map(v => (
              <g key={v}>
                <line x1={padL} y1={toY(v)} x2={chartW - 10} y2={toY(v)}
                  stroke="#e5e7eb" strokeWidth={1} className="dark:stroke-gray-700" />
                <text x={padL - 4} y={toY(v) + 4} fontSize={9} textAnchor="end" fill="#9ca3af">{v}</text>
              </g>
            ))}

            {/* Zero line */}
            <line x1={padL} y1={toY(0)} x2={chartW - 10} y2={toY(0)}
              stroke="#6b7280" strokeWidth={1} strokeDasharray="3 2" />

            {/* Amortized = 3 reference */}
            {step > 0 && (
              <line x1={padL} y1={toY(3)} x2={padL + step * xStep} y2={toY(3)}
                stroke="#f59e0b" strokeWidth={1.5} strokeDasharray="5 3" opacity={0.7} />
            )}

            {/* Lines */}
            {shown.length >= 2 && (
              <>
                {/* Phi line */}
                <polyline points={phiPoints.join(' ')} fill="none" stroke="#8b5cf6" strokeWidth={2} />
                {/* Real cost line */}
                <polyline points={realPoints.join(' ')} fill="none" stroke="#ef4444" strokeWidth={2} />
                {/* Amortized line */}
                <polyline points={amortPoints.join(' ')} fill="none" stroke="#10b981" strokeWidth={2} strokeDasharray="4 2" />
              </>
            )}

            {/* Dots */}
            {shown.map((s, i) => {
              const cx = padL + (i + 1) * xStep
              return (
                <g key={i}>
                  <circle cx={cx} cy={toY(s.phiAfter)} r={3} fill="#8b5cf6" />
                  <circle cx={cx} cy={toY(s.realCost)} r={3} fill={s.doubled ? '#ef4444' : '#fca5a5'} />
                  <circle cx={cx} cy={toY(s.amortized)} r={3} fill="#10b981" />
                </g>
              )
            })}

            {/* X axis */}
            <line x1={padL} y1={chartH - padB} x2={chartW - 10} y2={chartH - padB}
              stroke="#d1d5db" strokeWidth={1} />
            {steps.map((_, i) =>
              (n <= 16 || i % 4 === 3) && (
                <text key={i} x={padL + (i + 1) * xStep} y={chartH - padB + 12}
                  fontSize={8} textAnchor="middle" fill="#9ca3af">{i + 1}</text>
              )
            )}
            <text x={chartW / 2} y={chartH - 2} fontSize={9} textAnchor="middle" fill="#9ca3af">
              操作编号
            </text>
          </svg>
        </div>

        {/* Legend */}
        <div className="flex flex-wrap gap-4 text-xs text-gray-600 dark:text-gray-400">
          <span className="flex items-center gap-1">
            <span className="w-4 border-t-2 border-purple-500 inline-block" />
            Φ (势能)
          </span>
          <span className="flex items-center gap-1">
            <span className="w-4 border-t-2 border-red-400 inline-block" />
            实际代价 c
          </span>
          <span className="flex items-center gap-1">
            <span className="w-4 border-t-2 border-dashed border-emerald-500 inline-block" />
            摊销代价 ĉ = c + ΔΦ
          </span>
          <span className="flex items-center gap-1">
            <span className="w-4 border-t-2 border-dashed border-amber-400 inline-block" />
            参考线 ĉ = 3
          </span>
        </div>

        {/* Current step detail */}
        {step > 0 && step <= steps.length && (() => {
          const s = steps[step - 1]
          return (
            <div className="bg-violet-50 dark:bg-violet-900/20 border border-violet-200 dark:border-violet-700 rounded-lg p-3 text-sm">
              <div className="font-semibold text-violet-700 dark:text-violet-300 mb-2">
                第 {s.op} 次 push_back {s.doubled ? '（扩容！）' : '（普通）'}
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
                <div>
                  <span className="text-gray-500">size / cap</span>
                  <div className="font-mono font-bold text-gray-700 dark:text-gray-200">
                    {s.size} / {s.cap}
                  </div>
                </div>
                <div>
                  <span className="text-gray-500">实际代价 c</span>
                  <div className={`font-mono font-bold ${s.doubled ? 'text-red-600 dark:text-red-400' : 'text-gray-700 dark:text-gray-200'}`}>
                    {s.realCost}
                  </div>
                </div>
                <div>
                  <span className="text-gray-500">ΔΦ</span>
                  <div className={`font-mono font-bold ${s.deltaPhi < 0 ? 'text-red-500' : 'text-emerald-600 dark:text-emerald-400'}`}>
                    {s.deltaPhi > 0 ? '+' : ''}{s.deltaPhi}
                  </div>
                </div>
                <div>
                  <span className="text-gray-500">ĉ = c + ΔΦ</span>
                  <div className="font-mono font-bold text-emerald-600 dark:text-emerald-400">
                    {s.amortized}
                  </div>
                </div>
              </div>
              {s.doubled && (
                <div className="mt-2 text-violet-700 dark:text-violet-300 text-xs">
                  扩容时 Φ 从 {s.phiBefore} 骤降到 {s.phiAfter} (ΔΦ = {s.deltaPhi})，
                  正好"消耗"积累的势能支付复制代价，使摊销仍 = {s.amortized}。
                </div>
              )}
            </div>
          )
        })()}

        {/* Insight */}
        <div className="bg-purple-50 dark:bg-purple-900/20 border border-purple-200 dark:border-purple-800 rounded-lg p-3 text-sm text-purple-800 dark:text-purple-300">
          <strong>核心洞察：</strong>势能函数 Φ 在普通操作时积累（+2），
          在扩容时大幅释放（−k），始终使 ĉ = c + ΔΦ ≡ 3。
          势能相当于预付款——提前存钱以便在昂贵操作到来时透支。
        </div>
      </div>
    </div>
  )
}
