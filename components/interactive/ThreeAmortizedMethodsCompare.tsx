"use client"

import { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Play, Pause, RotateCcw } from 'lucide-react'

// 4-bit binary counter: compute bit sequences for 16 increments
function buildCounterSteps(maxOp: number) {
  const steps: {
    op: number
    bits: number[]           // [b3,b2,b1,b0] — MSB first
    bitsFlipped: number      // how many bits flipped this step
    flippedMask: boolean[]   // which bits changed
  }[] = []

  let counter = 0
  for (let i = 1; i <= maxOp; i++) {
    const prev = counter
    counter = (counter + 1) & 0xf   // 4-bit wrap
    const changed = prev ^ counter
    const bitsFlipped = bin4(prev ^ counter).filter(b => b).length
    const flippedMask = [3, 2, 1, 0].map(bit => !!(changed & (1 << bit)))
    steps.push({
      op: i,
      bits: [3, 2, 1, 0].map(bit => (counter >> bit) & 1),
      bitsFlipped,
      flippedMask,
    })
  }
  return steps
}

function bin4(n: number) {
  return [3, 2, 1, 0].map(bit => !!(n & (1 << bit)))
}

const MAX_OPS = 16
const COUNTER_STEPS = buildCounterSteps(MAX_OPS)

// Aggregate analysis data
function aggregateData(steps: typeof COUNTER_STEPS) {
  let total = 0
  return steps.map(s => {
    total += s.bitsFlipped
    return { ...s, totalCost: total, avgCost: total / s.op }
  })
}

// Accounting method: credit = 2 per increment.
// Each bit-flip costs 1 credit. Setting a bit to 1 uses 1 credit, stores 1 credit.
// Resetting to 0 uses the stored credit.
function accountingData(steps: typeof COUNTER_STEPS) {
  let creditBalance = 0
  return steps.map(s => {
    // Each INCREMENT gets 2 credits
    const creditsGranted = 2
    // Debits: bitsFlipped credits consumed
    const debits = s.bitsFlipped
    creditBalance = creditBalance + creditsGranted - debits
    return { ...s, creditsGranted, debits, creditBalance }
  })
}

// Potential method: Φ = number of 1-bits in counter
function potentialData(steps: typeof COUNTER_STEPS) {
  let prevPhi = 0
  return steps.map(s => {
    const phi = s.bits.reduce((a, b) => a + b, 0)
    const realCost = s.bitsFlipped
    const deltaPhi = phi - prevPhi
    const amortized = realCost + deltaPhi
    prevPhi = phi
    return { ...s, phi, prevPhi: prevPhi - deltaPhi, deltaPhi, realCost, amortized }
  })
}

type Tab = 'aggregate' | 'accounting' | 'potential'

export function ThreeAmortizedMethodsCompare() {
  const [step, setStep] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [tab, setTab] = useState<Tab>('aggregate')
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const aggData = aggregateData(COUNTER_STEPS)
  const accData = accountingData(COUNTER_STEPS)
  const potData = potentialData(COUNTER_STEPS)

  useEffect(() => {
    if (playing) {
      timerRef.current = setInterval(() => {
        setStep(prev => {
          if (prev >= MAX_OPS) { setPlaying(false); return prev }
          return prev + 1
        })
      }, 400)
    }
    return () => { if (timerRef.current) clearInterval(timerRef.current) }
  }, [playing])

  const reset = () => { setPlaying(false); setStep(0) }

  const tabs: { key: Tab; label: string; color: string; desc: string }[] = [
    { key: 'aggregate', label: '聚合法', color: 'blue', desc: '所有操作总代价 / 操作次数' },
    { key: 'accounting', label: '记账法', color: 'emerald', desc: '每次付 2 代金，1 用1存' },
    { key: 'potential', label: '势能法', color: 'violet', desc: 'Φ = #(1-bits), ĉ = c + ΔΦ' },
  ]

  const colorMap: Record<Tab, string> = { aggregate: 'blue', accounting: 'emerald', potential: 'violet' }
  const c = colorMap[tab]

  // Current step display data
  const curAgg = step > 0 ? aggData[step - 1] : null
  const curAcc = step > 0 ? accData[step - 1] : null
  const curPot = step > 0 ? potData[step - 1] : null

  // Counter display (always show current state)
  const curBits = step > 0 ? COUNTER_STEPS[step - 1].bits : [0, 0, 0, 0]
  const curFlipped = step > 0 ? COUNTER_STEPS[step - 1].flippedMask : [false, false, false, false]

  return (
    <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-gray-900 overflow-hidden shadow-sm my-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-slate-600 to-indigo-700 px-5 py-3">
        <h3 className="text-white font-bold text-base">三种摊销分析方法对比：4 位二进制计数器 INCREMENT</h3>
        <p className="text-slate-200 text-xs mt-0.5">
          同一操作序列，聚合法 / 记账法 / 势能法各自的记账视角
        </p>
      </div>

      <div className="p-4 space-y-4">
        {/* Controls */}
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-2">
            <button onClick={reset} className="p-1.5 rounded bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600">
              <RotateCcw size={16} className="text-gray-600 dark:text-gray-300" />
            </button>
            <button onClick={() => setStep(p => Math.max(0, p - 1))} disabled={step === 0}
              className="px-2 py-1 rounded text-sm bg-gray-100 dark:bg-gray-700 disabled:opacity-40 hover:bg-gray-200 dark:hover:bg-gray-600 text-gray-700 dark:text-gray-300">‹</button>
            <button onClick={() => { if (playing) { setPlaying(false) } else { if (step >= MAX_OPS) setStep(0); setPlaying(true) } }}
              className="flex items-center gap-1 px-3 py-1.5 rounded bg-indigo-600 text-white hover:bg-indigo-700 text-sm">
              {playing ? <Pause size={14} /> : <Play size={14} />}
              {playing ? '暂停' : step >= MAX_OPS ? '重播' : '播放'}
            </button>
            <button onClick={() => setStep(p => Math.min(MAX_OPS, p + 1))} disabled={step >= MAX_OPS}
              className="px-2 py-1 rounded text-sm bg-gray-100 dark:bg-gray-700 disabled:opacity-40 hover:bg-gray-200 dark:hover:bg-gray-600 text-gray-700 dark:text-gray-300">›</button>
          </div>
          <span className="text-sm text-gray-500 dark:text-gray-400 ml-2">
            第 {step} / {MAX_OPS} 次 INCREMENT
          </span>
        </div>

        {/* 4-bit counter display */}
        <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4">
          <div className="text-xs text-gray-500 mb-2 text-center">当前计数器值（二进制，MSB→LSB）</div>
          <div className="flex gap-3 justify-center">
            {curBits.map((b, i) => (
              <motion.div
                key={i}
                animate={{
                  backgroundColor: curFlipped[i] ? '#ef4444' : b ? '#4f46e5' : '#e5e7eb',
                  scale: curFlipped[i] ? [1, 1.2, 1] : 1,
                }}
                transition={{ duration: 0.3 }}
                className="w-12 h-12 rounded-lg flex items-center justify-center text-2xl font-mono font-bold text-white shadow"
                style={{
                  backgroundColor: curFlipped[i] ? '#ef4444' : b ? '#4f46e5' : '#9ca3af',
                }}
              >
                {b}
              </motion.div>
            ))}
          </div>
          {step > 0 && (
            <div className="text-center mt-2 text-xs text-gray-500">
              翻转了 {COUNTER_STEPS[step - 1].bitsFlipped} 位（红色）
              {COUNTER_STEPS[step - 1].bitsFlipped > 2 && (
                <span className="ml-2 text-orange-500">️ ← 进位链较长</span>
              )}
            </div>
          )}
        </div>

        {/* Tabs */}
        <div className="flex gap-2">
          {tabs.map(t => (
            <button key={t.key} onClick={() => setTab(t.key)}
              className={`flex-1 py-2 px-3 rounded-lg text-xs font-medium transition-colors ${
                tab === t.key
                  ? 'bg-indigo-100 dark:bg-indigo-900/40 text-indigo-700 dark:text-indigo-300 border border-indigo-300 dark:border-indigo-600'
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-600'
              }`}
            >
              <div className="font-bold">{t.label}</div>
              <div className="text-gray-400 dark:text-gray-500 mt-0.5 hidden sm:block">{t.desc}</div>
            </button>
          ))}
        </div>

        {/* Tab content */}
        <AnimatePresence mode="wait">
          <motion.div key={tab} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }} transition={{ duration: 0.15 }}>

            {tab === 'aggregate' && curAgg && (
              <div className="space-y-3">
                <div className="rounded-lg border border-blue-200 dark:border-blue-700 p-3 bg-blue-50 dark:bg-blue-900/20">
                  <div className="text-xs text-blue-600 dark:text-blue-400 font-semibold mb-1">聚合法（Aggregate Method）</div>
                  <div className="text-xs text-gray-600 dark:text-gray-400">
                    计算 n 次操作总代价 T(n)，摊销代价 = T(n)/n。
                    对于 k 位计数器，第 i 位每 2ⁱ 次翻转一次，
                    总翻转次数 = Σ ⌊n/2ⁱ⌋ &lt; 2n。
                  </div>
                </div>
                {/* Running table */}
                <div className="overflow-x-auto">
                  <table className="w-full text-xs border-collapse">
                    <thead>
                      <tr className="bg-blue-100 dark:bg-blue-900/30">
                        <th className="px-2 py-1 text-left border border-blue-200 dark:border-blue-700">操作</th>
                        <th className="px-2 py-1 text-left border border-blue-200 dark:border-blue-700">实际代价</th>
                        <th className="px-2 py-1 text-left border border-blue-200 dark:border-blue-700">累计 T(n)</th>
                        <th className="px-2 py-1 text-left border border-blue-200 dark:border-blue-700">均值 T/n</th>
                      </tr>
                    </thead>
                    <tbody>
                      {aggData.slice(0, step).map((row, i) => (
                        <tr key={i} className={i === step - 1 ? 'bg-blue-200 dark:bg-blue-800/40' : 'hover:bg-blue-50 dark:hover:bg-blue-900/10'}>
                          <td className="px-2 py-1 border border-blue-100 dark:border-blue-800 font-mono">{row.op}</td>
                          <td className={`px-2 py-1 border border-blue-100 dark:border-blue-800 font-mono ${row.bitsFlipped > 2 ? 'text-red-500 font-bold' : ''}`}>{row.bitsFlipped}</td>
                          <td className="px-2 py-1 border border-blue-100 dark:border-blue-800 font-mono font-semibold">{row.totalCost}</td>
                          <td className="px-2 py-1 border border-blue-100 dark:border-blue-800 font-mono text-blue-600 dark:text-blue-400">{row.avgCost.toFixed(2)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                {curAgg && (
                  <div className="text-xs text-blue-700 dark:text-blue-300 bg-blue-50 dark:bg-blue-900/20 rounded p-2">
                    n={curAgg.op} 次后：T({curAgg.op}) = {curAgg.totalCost}，
                    均值 = {curAgg.avgCost.toFixed(2)} ≤ 2
                    （总翻转次数 &lt; 2n = {2 * curAgg.op}）
                  </div>
                )}
              </div>
            )}

            {tab === 'accounting' && curAcc && (
              <div className="space-y-3">
                <div className="rounded-lg border border-emerald-200 dark:border-emerald-700 p-3 bg-emerald-50 dark:bg-emerald-900/20">
                  <div className="text-xs text-emerald-600 dark:text-emerald-400 font-semibold mb-1">记账法（Accounting Method）</div>
                  <div className="text-xs text-gray-600 dark:text-gray-400">
                    摊销代价 = 2/次。每次 INCREMENT 时：
                    付 1 代金给翻转为 1 的那位（存储起来），付 1 代金给翻转为 0 的位
                    （但那些代金来自之前存储的余额）。余额始终 ≥ 0。
                  </div>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-xs border-collapse">
                    <thead>
                      <tr className="bg-emerald-100 dark:bg-emerald-900/30">
                        <th className="px-2 py-1 text-left border border-emerald-200 dark:border-emerald-700">操作</th>
                        <th className="px-2 py-1 text-left border border-emerald-200 dark:border-emerald-700">实际代价</th>
                        <th className="px-2 py-1 text-left border border-emerald-200 dark:border-emerald-700">获得代金</th>
                        <th className="px-2 py-1 text-left border border-emerald-200 dark:border-emerald-700">消耗代金</th>
                        <th className="px-2 py-1 text-left border border-emerald-200 dark:border-emerald-700">余额</th>
                      </tr>
                    </thead>
                    <tbody>
                      {accData.slice(0, step).map((row, i) => (
                        <tr key={i} className={i === step - 1 ? 'bg-emerald-200 dark:bg-emerald-800/40' : 'hover:bg-emerald-50 dark:hover:bg-emerald-900/10'}>
                          <td className="px-2 py-1 border border-emerald-100 dark:border-emerald-800 font-mono">{row.op}</td>
                          <td className={`px-2 py-1 border border-emerald-100 dark:border-emerald-800 font-mono ${row.bitsFlipped > 2 ? 'text-red-500 font-bold' : ''}`}>{row.bitsFlipped}</td>
                          <td className="px-2 py-1 border border-emerald-100 dark:border-emerald-800 font-mono text-emerald-600 dark:text-emerald-400">+{row.creditsGranted}</td>
                          <td className="px-2 py-1 border border-emerald-100 dark:border-emerald-800 font-mono text-red-500">-{row.debits}</td>
                          <td className={`px-2 py-1 border border-emerald-100 dark:border-emerald-800 font-mono font-semibold ${row.creditBalance >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-red-600'}`}>{row.creditBalance}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div className="text-xs text-emerald-700 dark:text-emerald-300 bg-emerald-50 dark:bg-emerald-900/20 rounded p-2">
                  余额 = #(1-bits in counter) ≥ 0 始终成立。
                  摊销代价恒 = 2，总代价 ≤ 2n。
                </div>
              </div>
            )}

            {tab === 'potential' && curPot && (
              <div className="space-y-3">
                <div className="rounded-lg border border-violet-200 dark:border-violet-700 p-3 bg-violet-50 dark:bg-violet-900/20">
                  <div className="text-xs text-violet-600 dark:text-violet-400 font-semibold mb-1">势能法（Potential Method）</div>
                  <div className="text-xs text-gray-600 dark:text-gray-400">
                    Φ = 计数器中1的个数。INCREMENT 翻转 t+1 位（t个1→0，1个0→1），
                    实际代价 = t+1，ΔΦ = 1−t，故 ĉ = (t+1)+(1−t) = 2。
                  </div>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-xs border-collapse">
                    <thead>
                      <tr className="bg-violet-100 dark:bg-violet-900/30">
                        <th className="px-2 py-1 text-left border border-violet-200 dark:border-violet-700">操作</th>
                        <th className="px-2 py-1 text-left border border-violet-200 dark:border-violet-700">实际 c</th>
                        <th className="px-2 py-1 text-left border border-violet-200 dark:border-violet-700">Φ(前)</th>
                        <th className="px-2 py-1 text-left border border-violet-200 dark:border-violet-700">Φ(后)</th>
                        <th className="px-2 py-1 text-left border border-violet-200 dark:border-violet-700">ΔΦ</th>
                        <th className="px-2 py-1 text-left border border-violet-200 dark:border-violet-700">ĉ = c+ΔΦ</th>
                      </tr>
                    </thead>
                    <tbody>
                      {potData.slice(0, step).map((row, i) => (
                        <tr key={i} className={i === step - 1 ? 'bg-violet-200 dark:bg-violet-800/40' : 'hover:bg-violet-50 dark:hover:bg-violet-900/10'}>
                          <td className="px-2 py-1 border border-violet-100 dark:border-violet-800 font-mono">{row.op}</td>
                          <td className={`px-2 py-1 border border-violet-100 dark:border-violet-800 font-mono ${row.realCost > 2 ? 'text-red-500 font-bold' : ''}`}>{row.realCost}</td>
                          <td className="px-2 py-1 border border-violet-100 dark:border-violet-800 font-mono">{row.prevPhi}</td>
                          <td className="px-2 py-1 border border-violet-100 dark:border-violet-800 font-mono">{row.phi}</td>
                          <td className={`px-2 py-1 border border-violet-100 dark:border-violet-800 font-mono ${row.deltaPhi < -1 ? 'text-red-500' : 'text-emerald-600 dark:text-emerald-400'}`}>
                            {row.deltaPhi > 0 ? '+' : ''}{row.deltaPhi}
                          </td>
                          <td className="px-2 py-1 border border-violet-100 dark:border-violet-800 font-mono font-bold text-violet-600 dark:text-violet-300">{row.amortized}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div className="text-xs text-violet-700 dark:text-violet-300 bg-violet-50 dark:bg-violet-900/20 rounded p-2">
                  ĉ 始终 = 2，与进位链长度 t 无关。势能函数 Φ 将高代价操作的影响"预先化解"。
                </div>
              </div>
            )}
          </motion.div>
        </AnimatePresence>

        {/* Comparison insight */}
        <div className="bg-slate-50 dark:bg-slate-800/50 border border-slate-200 dark:border-slate-700 rounded-lg p-3 text-xs text-slate-700 dark:text-slate-300">
          <strong>方法对比：</strong>
          三种方法均证明每次 INCREMENT 摊销代价 = 2（即 O(1)），但视角不同：
          <span className="text-blue-600 dark:text-blue-400 mx-1">聚合法</span>统计总代价；
          <span className="text-emerald-600 dark:text-emerald-400 mx-1">记账法</span>给每次操作分配"代金券"；
          <span className="text-violet-600 dark:text-violet-400 mx-1">势能法</span>用全局势能函数建模。
          势能法灵活性最强，适用于更复杂的数据结构。
        </div>
      </div>
    </div>
  )
}
