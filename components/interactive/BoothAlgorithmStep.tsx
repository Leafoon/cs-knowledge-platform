'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { ArrowRight } from 'lucide-react'

export function BoothAlgorithmStep() {
  const [multiplicand, setMultiplicand] = useState(3)
  const [multiplier, setMultiplier] = useState(5)
  const [step, setStep] = useState(0)
  const [playing, setPlaying] = useState(false)
  const bits = 4

  const toTwos = (v: number, n: number) => {
    const u = v < 0 ? (1 << n) + v : v
    return u.toString(2).padStart(n, '0').slice(-n)
  }

  const M = toTwos(multiplicand, bits)
  const negM = toTwos(-multiplicand, bits)

  const boothSteps: { action: string; A: string; Q: string; Q1: string; desc: string }[] = []
  let A = '0'.repeat(bits)
  let Q = toTwos(multiplier, bits)
  let Q_1 = '0'

  const addBin = (x: string, y: string) => {
    let carry = 0
    let res = ''
    for (let i = bits - 1; i >= 0; i--) {
      const s = parseInt(x[i]) + parseInt(y[i]) + carry
      res = (s % 2) + res
      carry = Math.floor(s / 2)
    }
    return res.slice(-bits)
  }

  boothSteps.push({ action: '初始值', A, Q, Q1: Q_1, desc: '初始化 A=0, Q=乘数, Q₋₁=0' })

  for (let i = 0; i < bits; i++) {
    const pair = Q[bits - 1] + Q_1
    if (pair === '10') {
      A = addBin(A, negM)
      boothSteps.push({ action: `步骤${i + 1}a: 10→A=A-M`, A, Q, Q1: Q_1, desc: `Q₀Q₋₁=10, A=A-M=${A}` })
    } else if (pair === '01') {
      A = addBin(A, M)
      boothSteps.push({ action: `步骤${i + 1}a: 01→A=A+M`, A, Q, Q1: Q_1, desc: `Q₀Q₋₁=01, A=A+M=${A}` })
    } else {
      boothSteps.push({ action: `步骤${i + 1}a: ${pair}→无操作`, A, Q, Q1: Q_1, desc: `Q₀Q₋₁=${pair}, 无加减` })
    }
    Q_1 = Q[bits - 1]
    Q = A[bits - 1] + Q.slice(0, bits - 1)
    A = A[0] + A.slice(0, bits - 1)
    boothSteps.push({ action: `步骤${i + 1}b: 算术右移`, A, Q, Q1: Q_1, desc: 'A,Q,Q₋₁ 算术右移一位' })
  }

  const finalResult = parseInt((A + Q).slice(-bits * 2), 2)
  const expectedResult = multiplicand * multiplier

  useEffect(() => {
    if (playing && step < boothSteps.length - 1) {
      const t = setTimeout(() => setStep(s => s + 1), 1000)
      return () => clearTimeout(t)
    }
    setPlaying(false)
  }, [playing, step, boothSteps.length])

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <div className="flex items-center gap-2 mb-4">
        <ArrowRight className="w-5 h-5 text-indigo-500" />
        <h3 className="text-lg font-bold">Booth 算法逐步演示</h3>
      </div>

      <div className="flex flex-wrap gap-4 mb-4">
        <div>
          <label className="text-xs text-slate-500 block">被乘数 M</label>
          <input type="number" min={-7} max={7} value={multiplicand}
            onChange={e => { setMultiplicand(Number(e.target.value)); setStep(0) }}
            className="w-16 px-2 py-1 border rounded font-mono" />
        </div>
        <div>
          <label className="text-xs text-slate-500 block">乘数 Q</label>
          <input type="number" min={-7} max={7} value={multiplier}
            onChange={e => { setMultiplier(Number(e.target.value)); setStep(0) }}
            className="w-16 px-2 py-1 border rounded font-mono" />
        </div>
        <button onClick={() => { setStep(0); setPlaying(true) }}
          className="self-end px-3 py-1 bg-indigo-500 text-white rounded text-sm">开始</button>
      </div>

      <div className="mb-3 text-sm font-mono">M={M} ({multiplicand}), -M={negM} ({-multiplicand})</div>

      <div className="space-y-1 max-h-64 overflow-y-auto">
        {boothSteps.map((s, i) => (
          <motion.div key={i}
            className={`flex items-center gap-2 px-2 py-1 rounded text-xs font-mono ${step === i ? 'bg-indigo-100 dark:bg-indigo-900 font-bold' : step > i ? 'bg-slate-50 dark:bg-slate-800' : 'opacity-30'}`}
            animate={{ opacity: step >= i ? 1 : 0.3 }}>
            <span className="w-32 text-slate-600 truncate">{s.action}</span>
            <span className="w-20">A={s.A}</span>
            <span className="w-20">Q={s.Q}</span>
            <span className="w-12">Q₋₁={s.Q1}</span>
          </motion.div>
        ))}
      </div>

      <motion.div className="mt-3 p-2 bg-green-50 dark:bg-green-950 rounded text-center text-sm"
        animate={{ opacity: step >= boothSteps.length - 1 ? 1 : 0.3 }}>
        结果: {A + Q} = {expectedResult} ({multiplicand} × {multiplier} = {expectedResult})
      </motion.div>
    </div>
  )
}
