'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Minus } from 'lucide-react'

export function SubtractionCircuit() {
  const [a, setA] = useState(7)
  const [b, setB] = useState(3)
  const [step, setStep] = useState(0)
  const [playing, setPlaying] = useState(false)

  const bits = 4
  const lo = -(1 << (bits - 1))
  const hi = (1 << (bits - 1)) - 1

  const toTwos = (v: number) => {
    const u = v < 0 ? (1 << bits) + v : v
    return u.toString(2).padStart(bits, '0').slice(-bits)
  }

  const bNeg = -b
  const result = a - b
  const bBin = toTwos(b)
  const bInv = bBin.split('').map(c => c === '0' ? '1' : '0').join('')
  const bNegBin = toTwos(bNeg)
  const aBin = toTwos(a)
  const resultBin = toTwos(result)
  const resultInRange = result >= lo && result <= hi

  useEffect(() => {
    if (playing && step < 3) {
      const t = setTimeout(() => setStep(s => s + 1), 1200)
      return () => clearTimeout(t)
    }
    setPlaying(false)
  }, [playing, step])

  const steps = [
    { title: '原始操作数', desc: 'A 的补码和 B 的补码', aBin, bBin },
    { title: 'B 按位取反', desc: '对 B 的每一位取反 (NOT)', bInv },
    { title: '取反后加 1', desc: `得到 -B 的补码 = ${bNegBin}`, bNegBin },
    { title: 'A + (-B)', desc: `补码加法: ${aBin} + ${bNegBin} = ${resultBin}`, resultBin },
  ]

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <div className="flex items-center gap-2 mb-4">
        <Minus className="w-5 h-5 text-red-500" />
        <h3 className="text-lg font-bold">减法器电路 (补码减法)</h3>
      </div>
      <p className="text-sm text-slate-500 mb-4">A - B = A + (-B)，展示取反加1过程</p>

      <div className="flex flex-wrap gap-4 mb-4">
        <div>
          <label className="text-xs text-slate-500 block">A ({lo}~{hi})</label>
          <input type="number" min={lo} max={hi} value={a}
            onChange={e => { setA(Math.max(lo, Math.min(hi, Number(e.target.value)))); setStep(0) }}
            className="w-20 px-2 py-1 border rounded font-mono" />
        </div>
        <div>
          <label className="text-xs text-slate-500 block">B ({lo}~{hi})</label>
          <input type="number" min={lo} max={hi} value={b}
            onChange={e => { setB(Math.max(lo, Math.min(hi, Number(e.target.value)))); setStep(0) }}
            className="w-20 px-2 py-1 border rounded font-mono" />
        </div>
        <button onClick={() => { setStep(0); setPlaying(true) }}
          className="self-end px-3 py-1 bg-red-500 text-white rounded text-sm">演示步骤</button>
      </div>

      <div className="text-center text-lg font-mono mb-4">
        {a} - {b} = {a} + ({bNeg}) = {resultInRange ? result : '溢出'}
      </div>

      <div className="space-y-3">
        {steps.map((s, i) => (
          <motion.div key={i}
            className={`p-3 rounded-lg border ${step >= i ? 'border-red-300 bg-red-50 dark:bg-red-950 dark:border-red-800' : 'border-slate-200 bg-slate-50 dark:bg-slate-900 opacity-30'}`}
            animate={{ opacity: step >= i ? 1 : 0.3, x: step === i ? [0, 5, 0] : 0 }}
            transition={{ duration: 0.5 }}>
            <div className="flex items-center gap-3">
              <motion.div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold ${step >= i ? 'bg-red-500 text-white' : 'bg-slate-300 text-slate-500'}`}
                animate={{ scale: step === i ? 1.2 : 1 }}>
                {i + 1}
              </motion.div>
              <div className="flex-1">
                <div className="text-sm font-bold">{s.title}</div>
                <div className="text-xs text-slate-500">{s.desc}</div>
                {i === 0 ? (
                  <div className="font-mono text-sm mt-1">
                    A = <span className="text-blue-600">{aBin}</span> &nbsp;
                    B = <span className="text-purple-600">{bBin}</span>
                  </div>
                ) : i === 1 ? (
                  <div className="font-mono text-sm mt-1">
                    ~B = <span className="text-orange-600">{bInv}</span>
                  </div>
                ) : i === 2 ? (
                  <div className="font-mono text-sm mt-1">
                    -B = <span className="text-red-600">{bNegBin}</span>
                  </div>
                ) : (
                  <div className="font-mono text-sm mt-1">
                    <span className="text-blue-600">{aBin}</span> + <span className="text-red-600">{bNegBin}</span> = <span className="text-green-600 font-bold">{resultBin}</span>
                  </div>
                )}
              </div>
            </div>
          </motion.div>
        ))}
      </div>

      <motion.div className="mt-4 p-3 bg-green-50 dark:bg-green-950 rounded-lg text-center"
        animate={{ opacity: step >= 3 ? 1 : 0.3 }}>
        <span className="text-sm font-medium text-green-700">
          {aBin} - {bBin} = {resultBin} ({resultInRange ? result : '溢出'})
        </span>
      </motion.div>
    </div>
  )
}
