'use client'

import { useState, useEffect, useCallback } from 'react'
import { motion } from 'framer-motion'
import { Play, RotateCcw } from 'lucide-react'

export function RippleCarryAdder() {
  const [a, setA] = useState(0b0110)
  const [b, setB] = useState(0b0101)
  const [step, setStep] = useState(-1)
  const [running, setRunning] = useState(false)

  const bits = 4
  const aBits = Array.from({ length: bits }, (_, i) => (a >> (bits - 1 - i)) & 1)
  const bBits = Array.from({ length: bits }, (_, i) => (b >> (bits - 1 - i)) & 1)

  const computeStep = useCallback((idx: number) => {
    let carry = 0
    for (let i = bits - 1; i >= 0; i--) {
      const ab = (a >> i) & 1
      const bb = (b >> i) & 1
      const total = ab + bb + carry
      if (i === idx) {
        return { aBit: ab, bBit: bb, cin: carry, sum: total % 2, cout: Math.floor(total / 2) }
      }
      carry = Math.floor(total / 2)
    }
    return { aBit: 0, bBit: 0, cin: 0, sum: 0, cout: 0 }
  }, [a, b])

  const getCarryUpTo = useCallback((idx: number) => {
    let carry = 0
    for (let i = bits - 1; i >= 0; i--) {
      const ab = (a >> i) & 1
      const bb = (b >> i) & 1
      carry = Math.floor((ab + bb + carry) / 2)
      if (i === idx) return carry
    }
    return 0
  }, [a, b])

  const finalResult = a + b

  useEffect(() => {
    if (running && step < bits - 1) {
      const timer = setTimeout(() => setStep(s => s + 1), 800)
      return () => clearTimeout(timer)
    }
    if (step >= bits - 1) setRunning(false)
  }, [running, step, bits])

  const startAnim = () => { setStep(-1); setRunning(true) }
  const reset = () => { setStep(-1); setRunning(false) }

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <div className="flex items-center gap-2 mb-4">
        <Play className="w-5 h-5 text-green-500" />
        <h3 className="text-lg font-bold">串行进位加法器 (4-bit Ripple Carry)</h3>
      </div>

      <div className="flex gap-4 mb-4">
        <div>
          <label className="text-xs text-slate-500 mb-1 block">A (0-15)</label>
          <input type="number" min={0} max={15} value={a}
            onChange={e => { setA(Number(e.target.value) & 0xF); reset() }}
            className="w-16 px-2 py-1 border rounded font-mono" />
        </div>
        <div>
          <label className="text-xs text-slate-500 mb-1 block">B (0-15)</label>
          <input type="number" min={0} max={15} value={b}
            onChange={e => { setB(Number(e.target.value) & 0xF); reset() }}
            className="w-16 px-2 py-1 border rounded font-mono" />
        </div>
        <div className="flex items-end gap-2">
          <button onClick={startAnim} className="px-3 py-1 bg-green-500 text-white rounded text-sm">开始</button>
          <button onClick={reset} className="px-3 py-1 bg-slate-300 rounded text-sm"><RotateCcw className="w-4 h-4" /></button>
        </div>
      </div>

      <div className="bg-slate-50 dark:bg-slate-900 rounded-lg p-4 overflow-x-auto">
        <div className="flex items-center gap-1 min-w-[500px]">
          <div className="w-12" />
          {Array.from({ length: bits }, (_, i) => {
            const pos = bits - 1 - i
            const s = computeStep(pos)
            const active = step === pos
            const done = step > pos
            return (
              <motion.div key={pos} className="flex-1 flex flex-col items-center"
                animate={{ opacity: active || done ? 1 : 0.4 }}>
                <div className="text-[10px] text-slate-400 mb-1">位 {pos}</div>
                <div className={`px-2 py-1 rounded text-xs font-mono ${active ? 'bg-yellow-200 dark:bg-yellow-800' : done ? 'bg-green-100 dark:bg-green-900' : 'bg-slate-100 dark:bg-slate-800'}`}>
                  A={aBits[i]}
                </div>
                <div className={`px-2 py-1 rounded text-xs font-mono mt-0.5 ${active ? 'bg-yellow-200 dark:bg-yellow-800' : done ? 'bg-green-100 dark:bg-green-900' : 'bg-slate-100 dark:bg-slate-800'}`}>
                  B={bBits[i]}
                </div>
                <motion.div className="my-1 w-10 h-10 border-2 rounded-lg flex items-center justify-center text-xs font-bold font-mono"
                  animate={{ borderColor: active ? '#eab308' : done ? '#22c55e' : '#94a3b8', scale: active ? 1.1 : 1 }}>
                  FA
                </motion.div>
                <div className={`px-2 py-0.5 rounded text-xs font-mono ${active ? 'bg-blue-200 dark:bg-blue-800' : 'bg-slate-100 dark:bg-slate-800'}`}>
                  S={done || active ? s.sum : '?'}
                </div>
              </motion.div>
            )
          })}
          <div className="flex flex-col items-center">
            <div className="text-[10px] text-slate-400 mb-1">Cout</div>
            <motion.div className="w-10 h-10 rounded-full border-2 flex items-center justify-center text-xs font-bold font-mono"
              animate={{ borderColor: step >= 0 ? '#22c55e' : '#94a3b8' }}>
              {step >= 0 ? getCarryUpTo(0) : '?'}
            </motion.div>
          </div>
        </div>

        <div className="mt-3 flex items-center gap-1 min-w-[500px]">
          <span className="w-12 text-xs text-slate-500">进位→</span>
          {Array.from({ length: bits }, (_, i) => {
            const pos = bits - 1 - i
            const c = step >= pos ? getCarryUpTo(pos) : '?'
            return (
              <motion.div key={pos} className="flex-1 text-center"
                animate={{ color: step >= pos ? '#f59e0b' : '#94a3b8' }}>
                <span className="text-xs font-mono">C{pos}={c}</span>
              </motion.div>
            )
          })}
          <div className="w-12" />
        </div>
      </div>

      <motion.div className="mt-4 p-3 bg-green-50 dark:bg-green-950 rounded-lg text-center"
        animate={{ opacity: step >= bits - 1 ? 1 : 0.3 }}>
        <span className="text-sm text-green-700 font-medium">
          {a.toString(2).padStart(4, '0')} + {b.toString(2).padStart(4, '0')} = {finalResult.toString(2).padStart(5, '0')} ({a} + {b} = {finalResult})
        </span>
      </motion.div>
    </div>
  )
}
