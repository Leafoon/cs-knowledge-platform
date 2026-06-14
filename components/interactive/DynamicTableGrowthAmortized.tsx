"use client"

import { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Play, Pause, RotateCcw, ChevronRight } from 'lucide-react'

// 预计算 n 次 push_back 的实际代价序列
function buildOperations(n: number) {
  let size = 0
  let cap = 1
  const ops: { op: number; size: number; cap: number; cost: number; doubled: boolean }[] = []

  for (let i = 1; i <= n; i++) {
    const doubled = size === cap
    const cost = doubled ? cap + 1 : 1
    if (doubled) cap = cap * 2
    size++
    ops.push({ op: i, size, cap, cost, doubled })
  }
  return ops
}

const N_DEFAULT = 16
const ALL_OPS = buildOperations(N_DEFAULT)
const MAX_COST = Math.max(...ALL_OPS.map(o => o.cost))
const AMORTIZED = 3

export function DynamicTableGrowthAmortized() {
  const [n, setN] = useState(N_DEFAULT)
  const [step, setStep] = useState(0)       // 已展示的操作数
  const [playing, setPlaying] = useState(false)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const ops = buildOperations(n)
  const maxCost = Math.max(...ops.map(o => o.cost), AMORTIZED + 1)
  const shown = ops.slice(0, step)
  const totalCost = shown.reduce((s, o) => s + o.cost, 0)

  // Auto-play
  useEffect(() => {
    if (playing) {
      timerRef.current = setInterval(() => {
        setStep(prev => {
          if (prev >= n) {
            setPlaying(false)
            return prev
          }
          return prev + 1
        })
      }, 200)
    }
    return () => { if (timerRef.current) clearInterval(timerRef.current) }
  }, [playing, n])

  const reset = () => {
    setPlaying(false)
    setStep(0)
  }

  const changeN = (newN: number) => {
    setN(newN)
    setStep(0)
    setPlaying(false)
  }

  // Bar chart dims
  const chartH = 200
  const barMaxH = 160
  const barW = Math.max(8, Math.floor(480 / n) - 2)
  const gap = 2

  return (
    <div className="rounded-xl border border-teal-200 dark:border-teal-800 bg-white dark:bg-gray-900 overflow-hidden shadow-sm my-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-teal-500 to-emerald-600 px-5 py-3">
        <h3 className="text-white font-bold text-base">动态数组增长：聚合法可视化</h3>
        <p className="text-teal-100 text-xs mt-0.5">
          每次 push_back 的实际代价柱 & 摊销均值线（= 3）
        </p>
      </div>

      <div className="p-4 space-y-4">
        {/* Controls */}
        <div className="flex flex-wrap items-center gap-3">
          <label className="text-sm text-gray-600 dark:text-gray-400 whitespace-nowrap">
            操作次数 n =
          </label>
          {[8, 16, 24, 32].map(v => (
            <button
              key={v}
              onClick={() => changeN(v)}
              className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
                n === v
                  ? 'bg-teal-600 text-white'
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
              }`}
            >
              {v}
            </button>
          ))}

          <div className="flex items-center gap-2 ml-auto">
            <button
              onClick={reset}
              className="p-1.5 rounded bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
              title="重置"
            >
              <RotateCcw size={16} className="text-gray-600 dark:text-gray-300" />
            </button>
            <button
              onClick={() => setStep(prev => Math.max(0, prev - 1))}
              disabled={step === 0}
              className="px-2 py-1 rounded text-sm bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 disabled:opacity-40 transition-colors text-gray-700 dark:text-gray-300"
            >
              ‹
            </button>
            <button
              onClick={() => {
                if (playing) {
                  setPlaying(false)
                } else {
                  if (step >= n) setStep(0)
                  setPlaying(true)
                }
              }}
              className="flex items-center gap-1 px-3 py-1.5 rounded bg-teal-600 text-white hover:bg-teal-700 transition-colors text-sm"
            >
              {playing ? <Pause size={14} /> : <Play size={14} />}
              {playing ? '暂停' : step >= n ? '重播' : '播放'}
            </button>
            <button
              onClick={() => setStep(prev => Math.min(n, prev + 1))}
              disabled={step >= n}
              className="px-2 py-1 rounded text-sm bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 disabled:opacity-40 transition-colors text-gray-700 dark:text-gray-300"
            >
              ›
            </button>
          </div>
        </div>

        {/* Bar chart */}
        <div className="relative bg-gray-50 dark:bg-gray-800 rounded-lg p-3">
          {/* Y axis labels */}
          <div className="absolute left-0 top-3 bottom-8 flex flex-col justify-between text-xs text-gray-400 dark:text-gray-500 pl-1">
            <span>{maxCost}</span>
            <span>{Math.floor(maxCost / 2)}</span>
            <span>0</span>
          </div>

          <div className="ml-7 overflow-x-auto">
            <svg
              width={Math.max(480, n * (barW + gap))}
              height={chartH + 28}
              className="overflow-visible"
            >
              {/* Grid lines */}
              {[0, 0.25, 0.5, 0.75, 1].map(frac => (
                <line
                  key={frac}
                  x1={0} y1={barMaxH * (1 - frac)}
                  x2={n * (barW + gap)} y2={barMaxH * (1 - frac)}
                  stroke="#e5e7eb" strokeWidth={1}
                  className="dark:stroke-gray-700"
                />
              ))}

              {/* Amortized cost line (= 3) */}
              {step > 0 && (
                <line
                  x1={0}
                  y1={barMaxH - (AMORTIZED / maxCost) * barMaxH}
                  x2={step * (barW + gap) - gap}
                  y2={barMaxH - (AMORTIZED / maxCost) * barMaxH}
                  stroke="#f59e0b"
                  strokeWidth={2}
                  strokeDasharray="6 3"
                />
              )}
              {step > 0 && (
                <text
                  x={step * (barW + gap) + 4}
                  y={barMaxH - (AMORTIZED / maxCost) * barMaxH + 4}
                  fontSize={10}
                  fill="#f59e0b"
                >
                  摊销=3
                </text>
              )}

              {/* Bars */}
              {ops.map((op, i) => {
                const barH = (op.cost / maxCost) * barMaxH
                const x = i * (barW + gap)
                const y = barMaxH - barH
                const isShown = i < step
                const isCurrent = i === step - 1

                return (
                  <g key={i}>
                    {isShown && (
                      <motion.rect
                        initial={{ height: 0, y: barMaxH }}
                        animate={{ height: barH, y }}
                        transition={{ duration: 0.15 }}
                        x={x} width={barW}
                        fill={
                          op.doubled
                            ? isCurrent ? '#ef4444' : '#f87171'
                            : isCurrent ? '#14b8a6' : '#99f6e4'
                        }
                        rx={1}
                      />
                    )}
                    {/* Cost label on doubled bars */}
                    {isShown && op.doubled && barH > 20 && (
                      <text
                        x={x + barW / 2}
                        y={y - 2}
                        fontSize={9}
                        textAnchor="middle"
                        fill={op.doubled ? '#ef4444' : '#0d9488'}
                        fontWeight="bold"
                      >
                        {op.cost}
                      </text>
                    )}
                    {/* X axis label */}
                    {n <= 16 && (
                      <text
                        x={x + barW / 2}
                        y={barMaxH + 14}
                        fontSize={9}
                        textAnchor="middle"
                        fill="#9ca3af"
                      >
                        {op.op}
                      </text>
                    )}
                    {n > 16 && i % 4 === 3 && (
                      <text
                        x={x + barW / 2}
                        y={barMaxH + 14}
                        fontSize={8}
                        textAnchor="middle"
                        fill="#9ca3af"
                      >
                        {op.op}
                      </text>
                    )}
                  </g>
                )
              })}

              {/* Baseline */}
              <line x1={0} y1={barMaxH} x2={n * (barW + gap)} y2={barMaxH}
                stroke="#d1d5db" strokeWidth={1} />
            </svg>
          </div>

          {/* Legend */}
          <div className="flex gap-4 mt-2 text-xs text-gray-600 dark:text-gray-400">
            <span className="flex items-center gap-1">
              <span className="w-3 h-3 rounded-sm inline-block bg-teal-300" /> 普通插入 (代价=1)
            </span>
            <span className="flex items-center gap-1">
              <span className="w-3 h-3 rounded-sm inline-block bg-red-400" /> 扩容 (代价=i)
            </span>
            <span className="flex items-center gap-1">
              <span className="w-4 border-t-2 border-dashed border-amber-400 inline-block" /> 摊销均值=3
            </span>
          </div>
        </div>

        {/* Stats */}
        {step > 0 && (
          <div className="grid grid-cols-3 gap-3 text-center">
            <div className="bg-teal-50 dark:bg-teal-900/30 rounded-lg p-3">
              <div className="text-lg font-bold text-teal-700 dark:text-teal-300">{step}</div>
              <div className="text-xs text-gray-500">已执行操作</div>
            </div>
            <div className="bg-red-50 dark:bg-red-900/30 rounded-lg p-3">
              <div className="text-lg font-bold text-red-600 dark:text-red-400">{totalCost}</div>
              <div className="text-xs text-gray-500">总实际代价 T(n)</div>
            </div>
            <div className="bg-amber-50 dark:bg-amber-900/30 rounded-lg p-3">
              <div className="text-lg font-bold text-amber-600 dark:text-amber-400">
                {(totalCost / step).toFixed(2)}
              </div>
              <div className="text-xs text-gray-500">运行均值 T(n)/n</div>
            </div>
          </div>
        )}

        {/* Insight */}
        <div className="bg-emerald-50 dark:bg-emerald-900/20 border border-emerald-200 dark:border-emerald-800 rounded-lg p-3 text-sm text-emerald-800 dark:text-emerald-300">
          <strong>核心结论：</strong> 扩容代价 (红柱) 被均摊到其后的所有普通插入上。
          总代价 T(n) &lt; 3n，每次操作摊销代价 = O(1)。
          {step >= n && (
            <span className="ml-2 font-semibold">
              本例：T({n}) = {totalCost} &lt; 3×{n} = {3 * n} ✓
            </span>
          )}
        </div>
      </div>
    </div>
  )
}
