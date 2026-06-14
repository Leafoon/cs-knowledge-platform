'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { Cpu } from 'lucide-react'

export function FullAdderCircuit() {
  const [a, setA] = useState(0)
  const [b, setB] = useState(0)
  const [cin, setCin] = useState(0)

  const sum = a ^ b ^ cin
  const cout = (a & b) | (cin & (a ^ b))

  const gates = [
    { label: 'XOR₁', inputs: `A=${a}, B=${b}`, output: a ^ b, x: 1, y: 0 },
    { label: 'AND₁', inputs: `A=${a}, B=${b}`, output: a & b, x: 1, y: 1 },
    { label: 'XOR₂', inputs: `${a ^ b}, Cin=${cin}`, output: sum, x: 2, y: 0 },
    { label: 'AND₂', inputs: `${a ^ b}, Cin=${cin}`, output: (a ^ b) & cin, x: 2, y: 1 },
    { label: 'OR', inputs: `${a & b}, ${(a ^ b) & cin}`, output: cout, x: 3, y: 1 },
  ]

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <div className="flex items-center gap-2 mb-4">
        <Cpu className="w-5 h-5 text-blue-500" />
        <h3 className="text-lg font-bold">全加器电路演示</h3>
      </div>
      <p className="text-sm text-slate-500 mb-4">输入 A、B、Cin，观察门电路的信号传播</p>

      <div className="flex gap-4 mb-6">
        {[
          { label: 'A', value: a, set: setA },
          { label: 'B', value: b, set: setB },
          { label: 'Cin', value: cin, set: setCin },
        ].map(({ label, value, set }) => (
          <div key={label} className="flex items-center gap-2">
            <span className="text-sm font-medium w-10">{label}:</span>
            <button
              onClick={() => set(value === 0 ? 1 : 0)}
              className={`w-12 h-10 rounded font-mono font-bold text-lg transition-colors ${
                value ? 'bg-green-500 text-white' : 'bg-slate-200 text-slate-600'
              }`}
            >
              {value}
            </button>
          </div>
        ))}
      </div>

      <div className="relative bg-slate-50 dark:bg-slate-900 rounded-lg p-4 min-h-[200px]">
        <svg viewBox="0 0 400 180" className="w-full h-40">
          <line x1="20" y1="50" x2="80" y2="50" stroke={a ? '#22c55e' : '#94a3b8'} strokeWidth="2" />
          <line x1="20" y1="130" x2="80" y2="130" stroke={b ? '#22c55e' : '#94a3b8'} strokeWidth="2" />
          <line x1="20" y1="90" x2="80" y2="90" stroke={cin ? '#22c55e' : '#94a3b8'} strokeWidth="2" />

          <text x="10" y="54" className="text-xs font-mono" fill={a ? '#22c55e' : '#64748b'}>{a}</text>
          <text x="10" y="134" className="text-xs font-mono" fill={b ? '#22c55e' : '#64748b'}>{b}</text>
          <text x="10" y="94" className="text-xs font-mono" fill={cin ? '#22c55e' : '#64748b'}>{cin}</text>

          {gates.map((g, i) => {
            const gx = 80 + g.x * 80
            const gy = 30 + g.y * 80
            const color = g.output ? '#22c55e' : '#94a3b8'
            return (
              <motion.g
                key={i}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: i * 0.15 }}
              >
                <rect x={gx} y={gy - 15} width="50" height="30" rx="4"
                  fill={g.output ? '#dcfce7' : '#f1f5f9'} stroke={color} strokeWidth="1.5" />
                <text x={gx + 25} y={gy + 4} textAnchor="middle" className="text-[10px] font-mono font-bold" fill={color}>
                  {g.label}
                </text>
              </motion.g>
            )
          })}

          <motion.line x1="130" y1="50" x2="160" y2="50" stroke={a ^ b ? '#22c55e' : '#94a3b8'} strokeWidth="2"
            initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} />
          <motion.line x1="240" y1="50" x2="280" y2="50" stroke={sum ? '#22c55e' : '#94a3b8'} strokeWidth="2" />

          <motion.rect x="340" y="35" width="40" height="30" rx="4"
            fill={sum ? '#dcfce7' : '#f1f5f9'} stroke={sum ? '#22c55e' : '#94a3b8'} strokeWidth="2"
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.8 }} />
          <text x="360" y="54" textAnchor="middle" className="text-xs font-bold font-mono" fill={sum ? '#22c55e' : '#64748b'}>S={sum}</text>

          <motion.rect x="340" y="115" width="40" height="30" rx="4"
            fill={cout ? '#dcfce7' : '#f1f5f9'} stroke={cout ? '#22c55e' : '#94a3b8'} strokeWidth="2"
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1 }} />
          <text x="360" y="134" textAnchor="middle" className="text-xs font-bold font-mono" fill={cout ? '#22c55e' : '#64748b'}>Cout={cout}</text>
        </svg>
      </div>

      <div className="mt-4 grid grid-cols-2 gap-4">
        <div className="p-3 bg-green-50 dark:bg-green-950 rounded-lg text-center">
          <div className="text-xs text-green-600 mb-1">Sum (和)</div>
          <motion.div className="text-3xl font-bold font-mono text-green-700" key={sum}
            initial={{ scale: 1.5 }} animate={{ scale: 1 }}>{sum}</motion.div>
        </div>
        <div className="p-3 bg-blue-50 dark:bg-blue-950 rounded-lg text-center">
          <div className="text-xs text-blue-600 mb-1">Cout (进位)</div>
          <motion.div className="text-3xl font-bold font-mono text-blue-700" key={cout}
            initial={{ scale: 1.5 }} animate={{ scale: 1 }}>{cout}</motion.div>
        </div>
      </div>

      <div className="mt-4 text-xs text-slate-500 font-mono">
        Sum = A ⊕ B ⊕ Cin = {a} ⊕ {b} ⊕ {cin} = {sum} &nbsp;|&nbsp;
        Cout = (A·B) + (Cin·(A⊕B)) = {a & b} + {(a ^ b) & cin} = {cout}
      </div>
    </div>
  )
}
