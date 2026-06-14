'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { RefreshCw } from 'lucide-react'

export function ShiftAddMultiplier() {
  const [a, setA] = useState(13)
  const [b, setB] = useState(11)
  const [cycle, setCycle] = useState(-1)
  const [playing, setPlaying] = useState(false)
  const bits = 4

  const toBin = (v: number) => v.toString(2).padStart(bits, '0')

  const steps: { acc: number; Q: number; action: string }[] = []
  let acc = 0
  let q = b
  steps.push({ acc, Q: q, action: '初始值' })

  for (let i = 0; i < bits; i++) {
    if (q & 1) {
      acc += a
      steps.push({ acc, Q: q, action: `Q₀=1: ACC=ACC+A=${acc}` })
    } else {
      steps.push({ acc, Q: q, action: `Q₀=0: 无操作` })
    }
    const oldAcc = acc
    acc = acc >> 1
    q = ((oldAcc & 1) << (bits - 1)) | (q >> 1)
    steps.push({ acc, Q: q, action: `右移: ACC=${acc}, Q=${q}` })
  }

  useEffect(() => {
    if (playing && cycle < steps.length - 1) {
      const t = setTimeout(() => setCycle(c => c + 1), 900)
      return () => clearTimeout(t)
    }
    setPlaying(false)
  }, [playing, cycle, steps.length])

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <div className="flex items-center gap-2 mb-4">
        <RefreshCw className="w-5 h-5 text-teal-500" />
        <h3 className="text-lg font-bold">移位加乘法器</h3>
      </div>

      <div className="flex flex-wrap gap-4 mb-4">
        <div>
          <label className="text-xs text-slate-500 block">被乘数 A</label>
          <input type="number" min={0} max={15} value={a}
            onChange={e => { setA(Number(e.target.value) & 0xF); setCycle(-1) }}
            className="w-16 px-2 py-1 border rounded font-mono" />
        </div>
        <div>
          <label className="text-xs text-slate-500 block">乘数 B</label>
          <input type="number" min={0} max={15} value={b}
            onChange={e => { setB(Number(e.target.value) & 0xF); setCycle(-1) }}
            className="w-16 px-2 py-1 border rounded font-mono" />
        </div>
        <button onClick={() => { setCycle(-1); setPlaying(true) }}
          className="self-end px-3 py-1 bg-teal-500 text-white rounded text-sm">开始</button>
      </div>

      <div className="text-sm font-mono mb-2">A={toBin(a)}({a}), B={toBin(b)}({b})</div>

      <div className="bg-slate-50 dark:bg-slate-900 rounded-lg p-3">
        <div className="grid grid-cols-4 gap-2 text-xs font-mono text-center mb-2 font-bold text-slate-500">
          <span>步骤</span><span>ACC</span><span>Q</span><span>操作</span>
        </div>
        {steps.map((s, i) => (
          <motion.div key={i}
            className={`grid grid-cols-4 gap-2 text-xs font-mono text-center py-1 rounded ${cycle === i ? 'bg-teal-100 dark:bg-teal-900 font-bold' : cycle > i ? '' : 'opacity-20'}`}
            animate={{ opacity: cycle >= i ? 1 : 0.2 }}>
            <span className="text-slate-400">{i}</span>
            <span className="text-blue-600">{s.acc.toString(2).padStart(bits, '0')}</span>
            <span className="text-green-600">{s.Q.toString(2).padStart(bits, '0')}</span>
            <span className="text-slate-600 text-left truncate">{s.action}</span>
          </motion.div>
        ))}
      </div>

      <motion.div className="mt-3 p-2 bg-green-50 dark:bg-green-950 rounded text-center text-sm"
        animate={{ opacity: cycle >= steps.length - 1 ? 1 : 0.3 }}>
        结果: {a} × {b} = {a * b}
      </motion.div>
    </div>
  )
}
