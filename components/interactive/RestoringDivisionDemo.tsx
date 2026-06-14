'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Divide } from 'lucide-react'

export function RestoringDivisionDemo() {
  const [dividend, setDividend] = useState(13)
  const [divisor, setDivisor] = useState(3)
  const [step, setStep] = useState(-1)
  const [playing, setPlaying] = useState(false)

  const n = 4
  const maxVal = (1 << n) - 1

  const steps: { A: number; Q: number; action: string; restored?: boolean }[] = []
  let A = 0
  let Q = dividend & maxVal

  steps.push({ A, Q, action: '初始值' })

  for (let i = 0; i < n; i++) {
    A = ((A << 1) | (Q >> (n - 1))) & maxVal
    Q = (Q << 1) & maxVal
    steps.push({ A, Q, action: `左移 AQ` })

    A = A - divisor
    steps.push({ A, Q, action: `A = A - D = ${A < 0 ? A + (1 << (n + 1)) : A}` })

    if (A < 0) {
      A = A + divisor
      steps.push({ A, Q, action: `A < 0, 恢复余数, Q₀ = 0`, restored: true })
    } else {
      Q = Q | 1
      steps.push({ A, Q, action: `A ≥ 0, Q₀ = 1` })
    }
  }

  const quotient = Q & maxVal
  const remainder = A

  useEffect(() => {
    if (playing && step < steps.length - 1) {
      const t = setTimeout(() => setStep(s => s + 1), 800)
      return () => clearTimeout(t)
    }
    setPlaying(false)
  }, [playing, step, steps.length])

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <div className="flex items-center gap-2 mb-4">
        <Divide className="w-5 h-5 text-amber-500" />
        <h3 className="text-lg font-bold">恢复余数法</h3>
      </div>

      <div className="flex flex-wrap gap-4 mb-4">
        <div>
          <label className="text-xs text-slate-500 block">被除数 (0-{maxVal})</label>
          <input type="number" min={0} max={maxVal} value={dividend}
            onChange={e => { setDividend(Math.min(maxVal, Math.max(0, Number(e.target.value)))); setStep(-1) }}
            className="w-16 px-2 py-1 border rounded font-mono" />
        </div>
        <div>
          <label className="text-xs text-slate-500 block">除数 (1-{maxVal})</label>
          <input type="number" min={1} max={maxVal} value={divisor}
            onChange={e => { setDivisor(Math.min(maxVal, Math.max(1, Number(e.target.value)))); setStep(-1) }}
            className="w-16 px-2 py-1 border rounded font-mono" />
        </div>
        <button onClick={() => { setStep(-1); setPlaying(true) }}
          className="self-end px-3 py-1 bg-amber-500 text-white rounded text-sm">开始</button>
      </div>

      <div className="text-sm mb-3">{dividend} ÷ {divisor} = {Math.floor(dividend / divisor)} 余 {dividend % divisor}</div>

      <div className="bg-slate-50 dark:bg-slate-900 rounded-lg p-3 max-h-72 overflow-y-auto">
        <div className="grid grid-cols-4 gap-2 text-xs font-mono text-center font-bold text-slate-400 mb-1">
          <span>步骤</span><span>A(余数)</span><span>Q(商)</span><span>操作</span>
        </div>
        {steps.map((s, i) => (
          <motion.div key={i}
            className={`grid grid-cols-4 gap-2 text-xs font-mono text-center py-1 rounded ${step === i ? 'bg-amber-100 dark:bg-amber-900 font-bold' : step > i ? '' : 'opacity-20'}`}
            animate={{ opacity: step >= i ? 1 : 0.2 }}>
            <span className="text-slate-400">{i}</span>
            <span className="text-blue-600">{s.A.toString(2).padStart(n + 1, '0')}</span>
            <span className="text-green-600">{s.Q.toString(2).padStart(n, '0')}</span>
            <span className={`text-left truncate ${s.restored ? 'text-red-500' : 'text-slate-600'}`}>{s.action}</span>
          </motion.div>
        ))}
      </div>

      <motion.div className="mt-3 p-2 bg-green-50 dark:bg-green-950 rounded text-center text-sm"
        animate={{ opacity: step >= steps.length - 1 ? 1 : 0.3 }}>
        商 = {quotient} ({quotient.toString(2).padStart(n, '0')}), 余数 = {remainder}
      </motion.div>
    </div>
  )
}
