'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { Box } from 'lucide-react'

export function ALUBlockDiagram() {
  const [operation, setOperation] = useState(0)
  const [a, setA] = useState(0b1010)
  const [b, setB] = useState(0b0110)

  const ops = [
    { name: 'ADD', fn: () => (a + b) & 0xF, icon: '+' },
    { name: 'SUB', fn: () => (a - b + 16) & 0xF, icon: '-' },
    { name: 'AND', fn: () => a & b, icon: '&' },
    { name: 'OR', fn: () => a | b, icon: '|' },
    { name: 'XOR', fn: () => a ^ b, icon: '^' },
    { name: 'NOT', fn: () => (~a + 16) & 0xF, icon: '~' },
    { name: 'SLT', fn: () => ((a & 7) < (b & 7) ? 1 : 0), icon: '<' },
    { name: 'SLL', fn: () => (a << 1) & 0xF, icon: '<<' },
  ]

  const current = ops[operation]
  const result = current.fn()
  const zero = result === 0 ? 1 : 0
  const neg = (result >> 3) & 1
  const carry = (a + b) > 15 ? 1 : (operation === 1 && a < b ? 1 : 0)
  const overflow = 0

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <div className="flex items-center gap-2 mb-4">
        <Box className="w-5 h-5 text-violet-500" />
        <h3 className="text-lg font-bold">ALU 框图</h3>
      </div>

      <div className="flex flex-wrap gap-3 mb-4">
        <div>
          <label className="text-xs text-slate-500 block">A (0-15)</label>
          <input type="number" min={0} max={15} value={a}
            onChange={e => setA(Number(e.target.value) & 0xF)}
            className="w-16 px-2 py-1 border rounded font-mono" />
        </div>
        <div>
          <label className="text-xs text-slate-500 block">B (0-15)</label>
          <input type="number" min={0} max={15} value={b}
            onChange={e => setB(Number(e.target.value) & 0xF)}
            className="w-16 px-2 py-1 border rounded font-mono" />
        </div>
      </div>

      <div className="flex flex-wrap gap-1 mb-4">
        {ops.map((o, i) => (
          <button key={i} onClick={() => setOperation(i)}
            className={`px-2 py-1 text-xs rounded font-mono ${operation === i ? 'bg-violet-500 text-white' : 'bg-slate-200 hover:bg-slate-300'}`}>
            {o.name}
          </button>
        ))}
      </div>

      <div className="bg-slate-50 dark:bg-slate-900 rounded-lg p-4">
        <svg viewBox="0 0 400 200" className="w-full h-40">
          <rect x="130" y="30" width="140" height="140" rx="10" fill="#8b5cf6" fillOpacity="0.15" stroke="#8b5cf6" strokeWidth="2" />
          <text x="200" y="55" textAnchor="middle" className="text-sm font-bold" fill="#7c3aed">ALU</text>
          <text x="200" y="75" textAnchor="middle" className="text-xs" fill="#7c3aed">{current.name} ({current.icon})</text>

          <line x1="50" y1="80" x2="130" y2="80" stroke="#3b82f6" strokeWidth="2" markerEnd="url(#arrow)" />
          <text x="40" y="78" textAnchor="end" className="text-xs font-mono" fill="#3b82f6">A={a.toString(2).padStart(4, '0')}</text>

          <line x1="50" y1="130" x2="130" y2="130" stroke="#a855f7" strokeWidth="2" markerEnd="url(#arrow)" />
          <text x="40" y="128" textAnchor="end" className="text-xs font-mono" fill="#a855f7">B={b.toString(2).padStart(4, '0')}</text>

          <line x1="200" y1="20" x2="200" y2="30" stroke="#f59e0b" strokeWidth="2" markerEnd="url(#arrow)" />
          <text x="200" y="15" textAnchor="middle" className="text-[10px] font-mono" fill="#f59e0b">OP={operation.toString(2).padStart(3, '0')}</text>

          <motion.line x1="270" y1="100" x2="360" y2="100" stroke="#22c55e" strokeWidth="2" markerEnd="url(#arrow)"
            animate={{ opacity: [0.5, 1, 0.5] }} transition={{ duration: 1, repeat: Infinity }} />
          <text x="370" y="98" className="text-xs font-bold font-mono" fill="#22c55e">F={result.toString(2).padStart(4, '0')}</text>

          <motion.g animate={{ opacity: [0.5, 1, 0.5] }} transition={{ duration: 1.5, repeat: Infinity }}>
            <rect x="340" y="140" width="50" height="50" rx="4" fill={zero ? '#22c55e' : '#e2e8f0'} stroke="#94a3b8" />
            <text x="365" y="155" textAnchor="middle" className="text-[9px] font-bold" fill={zero ? 'white' : '#94a3b8'}>ZF</text>
            <text x="365" y="172" textAnchor="middle" className="text-sm font-mono font-bold" fill={zero ? 'white' : '#94a3b8'}>{zero}</text>
          </motion.g>

          <rect x="340" y="80" width="25" height="25" rx="3" fill={neg ? '#ef4444' : '#e2e8f0'} stroke="#94a3b8" />
          <text x="352" y="97" textAnchor="middle" className="text-[9px] font-bold" fill={neg ? 'white' : '#94a3b8'}>N</text>

          <rect x="370" y="80" width="25" height="25" rx="3" fill={carry ? '#f59e0b' : '#e2e8f0'} stroke="#94a3b8" />
          <text x="382" y="97" textAnchor="middle" className="text-[9px] font-bold" fill={carry ? 'white' : '#94a3b8'}>C</text>

          <defs>
            <marker id="arrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
              <path d="M0,0 L6,3 L0,6 Z" fill="#94a3b8" />
            </marker>
          </defs>
        </svg>
      </div>

      <div className="mt-3 grid grid-cols-4 gap-2 text-xs text-center">
        {[
          { name: '结果', val: result.toString(2).padStart(4, '0'), color: 'green' },
          { name: 'ZF(零)', val: zero, color: zero ? 'green' : 'slate' },
          { name: 'SF(负)', val: neg, color: neg ? 'red' : 'slate' },
          { name: 'CF(进位)', val: carry, color: carry ? 'amber' : 'slate' },
        ].map(f => (
          <div key={f.name} className="p-1 bg-slate-100 dark:bg-slate-800 rounded">
            <div className="text-slate-500">{f.name}</div>
            <div className="font-bold font-mono">{f.val}</div>
          </div>
        ))}
      </div>
    </div>
  )
}
