'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Shuffle } from 'lucide-react'

export function CarrySelectAdder() {
  const [a, setA] = useState(0b10110110)
  const [b, setB] = useState(0b01101001)
  const [step, setStep] = useState(-1)
  const [playing, setPlaying] = useState(false)

  const bits = 8
  const groupSize = 4
  const groups = bits / groupSize

  const toBin = (v: number, n: number) => v.toString(2).padStart(n, '0').slice(-n)

  const addGroup = (aG: number, bG: number, cin: number) => {
    const sum = aG + bG + cin
    return { sum: sum & ((1 << groupSize) - 1), cout: sum >> groupSize }
  }

  const aGroups = Array.from({ length: groups }, (_, i) => (a >> (i * groupSize)) & ((1 << groupSize) - 1))
  const bGroups = Array.from({ length: groups }, (_, i) => (b >> (i * groupSize)) & ((1 << groupSize) - 1))

  const results = aGroups.map((ag, i) => ({
    cin0: addGroup(ag, bGroups[i], 0),
    cin1: addGroup(ag, bGroups[i], 1),
  }))

  const finalCarries = [0]
  for (let i = 0; i < groups; i++) {
    const cin = finalCarries[i]
    finalCarries.push(cin ? results[i].cin1.cout : results[i].cin0.cout)
  }

  const finalSum = aGroups.map((ag, i) => {
    const cin = finalCarries[i]
    return cin ? results[i].cin1.sum : results[i].cin0.sum
  })

  useEffect(() => {
    if (playing && step < groups + 1) {
      const t = setTimeout(() => setStep(s => s + 1), 1200)
      return () => clearTimeout(t)
    }
    setPlaying(false)
  }, [playing, step, groups])

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <div className="flex items-center gap-2 mb-4">
        <Shuffle className="w-5 h-5 text-teal-500" />
        <h3 className="text-lg font-bold">选择进位加法器 (Carry Select)</h3>
      </div>
      <p className="text-sm text-slate-500 mb-4">并行计算进位0和进位1的结果，再选择正确的</p>

      <div className="flex flex-wrap gap-3 mb-4">
        <div>
          <label className="text-xs text-slate-500 block">A (8-bit)</label>
          <input type="number" min={0} max={255} value={a}
            onChange={e => { setA(Number(e.target.value) & 0xFF); setStep(-1) }}
            className="w-20 px-2 py-1 border rounded font-mono" />
        </div>
        <div>
          <label className="text-xs text-slate-500 block">B</label>
          <input type="number" min={0} max={255} value={b}
            onChange={e => { setB(Number(e.target.value) & 0xFF); setStep(-1) }}
            className="w-20 px-2 py-1 border rounded font-mono" />
        </div>
        <button onClick={() => { setStep(-1); setPlaying(true) }}
          className="self-end px-3 py-1 bg-teal-500 text-white rounded text-sm">开始</button>
      </div>

      <div className="space-y-3">
        {Array.from({ length: groups }, (_, i) => {
          const gIdx = groups - 1 - i
          const active = step >= gIdx + 1
          const selected = step >= gIdx + 1
          return (
            <motion.div key={gIdx}
              className={`p-3 rounded-lg border ${active ? 'border-teal-300 bg-teal-50 dark:bg-teal-950' : 'border-slate-200 opacity-30'}`}
              animate={{ opacity: active ? 1 : 0.3 }}>
              <div className="flex items-center gap-3">
                <div className="text-xs font-bold text-slate-600 w-16">组 {gIdx}</div>
                <div className="flex-1 grid grid-cols-2 gap-2">
                  <div className="text-xs font-mono p-1 bg-blue-50 dark:bg-blue-950 rounded text-center">
                    Cin=0: {toBin(results[gIdx].cin0.sum, groupSize)} (C={results[gIdx].cin0.cout})
                  </div>
                  <div className="text-xs font-mono p-1 bg-purple-50 dark:bg-purple-950 rounded text-center">
                    Cin=1: {toBin(results[gIdx].cin1.sum, groupSize)} (C={results[gIdx].cin1.cout})
                  </div>
                </div>
                {selected && (
                  <motion.div className="text-xs font-bold text-teal-600"
                    initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                    选: Cin={finalCarries[gIdx]} → {toBin(finalSum[gIdx], groupSize)}
                  </motion.div>
                )}
              </div>
            </motion.div>
          )
        })}
      </div>

      <motion.div className="mt-3 p-2 bg-green-50 dark:bg-green-950 rounded text-center text-sm font-mono"
        animate={{ opacity: step >= groups + 1 ? 1 : 0.3 }}>
        结果: {finalSum.slice().reverse().map(g => toBin(g, groupSize)).join('')} = {a + b}
        <div className="text-xs text-slate-500 mt-1">每组并行计算两种进位结果，选择延迟仅 O(√n)</div>
      </motion.div>
    </div>
  )
}
