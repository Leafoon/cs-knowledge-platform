'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { Activity } from 'lucide-react'

export function ALUOperationDemo() {
  const [a, setA] = useState(0b1100)
  const [b, setB] = useState(0b1010)
  const [op, setOp] = useState(0)

  const operations = [
    { name: '加法 ADD', op: '+', compute: (x: number, y: number) => x + y, detail: (x: number, y: number) => `${x}+${y}=${x + y}` },
    { name: '减法 SUB', op: '-', compute: (x: number, y: number) => x - y, detail: (x: number, y: number) => `${x}-${y}=${x - y}` },
    { name: '与 AND', op: '&', compute: (x: number, y: number) => x & y, detail: (x: number, y: number) => `${x}&${y}=${x & y}` },
    { name: '或 OR', op: '|', compute: (x: number, y: number) => x | y, detail: (x: number, y: number) => `${x}|${y}=${x | y}` },
    { name: '异或 XOR', op: '^', compute: (x: number, y: number) => x ^ y, detail: (x: number, y: number) => `${x}^${y}=${x ^ y}` },
    { name: '左移 SLL', op: '<<', compute: (x: number) => (x << 1) & 0xF, detail: (x: number) => `${x}<<1=${(x << 1) & 0xF}` },
    { name: '逻辑右移 SRL', op: '>>', compute: (x: number) => x >> 1, detail: (x: number) => `${x}>>1=${x >> 1}` },
  ]

  const current = operations[op]
  const result = current.compute(a, b) & 0xF
  const aBin = a.toString(2).padStart(4, '0')
  const bBin = b.toString(2).padStart(4, '0')
  const rBin = result.toString(2).padStart(4, '0')

  const bitOp = (idx: number) => {
    const ab = (a >> idx) & 1
    const bb = (b >> idx) & 1
    if (op <= 1) return ((op === 0 ? ab + bb : ab - bb) & 1).toString()
    if (op === 2) return (ab & bb).toString()
    if (op === 3) return (ab | bb).toString()
    if (op === 4) return (ab ^ bb).toString()
    return '?'
  }

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <div className="flex items-center gap-2 mb-4">
        <Activity className="w-5 h-5 text-orange-500" />
        <h3 className="text-lg font-bold">ALU 操作演示</h3>
      </div>

      <div className="flex flex-wrap gap-3 mb-4">
        <div>
          <label className="text-xs text-slate-500 block">A</label>
          <input type="number" min={0} max={15} value={a}
            onChange={e => setA(Number(e.target.value) & 0xF)}
            className="w-16 px-2 py-1 border rounded font-mono" />
        </div>
        <div>
          <label className="text-xs text-slate-500 block">B</label>
          <input type="number" min={0} max={15} value={b}
            onChange={e => setB(Number(e.target.value) & 0xF)}
            className="w-16 px-2 py-1 border rounded font-mono" />
        </div>
      </div>

      <div className="flex flex-wrap gap-1 mb-4">
        {operations.map((o, i) => (
          <button key={i} onClick={() => setOp(i)}
            className={`px-2 py-1 text-xs rounded ${op === i ? 'bg-orange-500 text-white' : 'bg-slate-200 hover:bg-slate-300'}`}>
            {o.name}
          </button>
        ))}
      </div>

      <div className="bg-slate-50 dark:bg-slate-900 rounded-lg p-4 mb-4">
        <div className="text-center text-sm font-mono mb-3">
          {aBin} ({a}) <span className="text-orange-500 font-bold text-lg mx-2">{current.op}</span> {bBin} ({b}) = <span className="text-green-600 font-bold">{rBin} ({result})</span>
        </div>

        <div className="flex justify-center gap-1">
          {[3, 2, 1, 0].map(i => (
            <motion.div key={i} className="text-center"
              initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.1 }}>
              <div className="text-[10px] text-slate-400 mb-1">位{i}</div>
              <div className="w-10 h-6 flex items-center justify-center bg-blue-100 dark:bg-blue-900 rounded text-xs font-mono">{(a >> i) & 1}</div>
              <div className="text-orange-500 text-xs my-0.5">{current.op}</div>
              <div className="w-10 h-6 flex items-center justify-center bg-purple-100 dark:bg-purple-900 rounded text-xs font-mono">{(b >> i) & 1}</div>
              <div className="border-t border-slate-300 my-0.5" />
              <motion.div className="w-10 h-6 flex items-center justify-center bg-green-100 dark:bg-green-900 rounded text-xs font-mono font-bold"
                animate={{ scale: [1, 1.1, 1] }} transition={{ delay: 0.5 + i * 0.1 }}>
                {bitOp(i)}
              </motion.div>
            </motion.div>
          ))}
        </div>
      </div>

      <div className="text-sm text-center">
        {current.detail(a, b)}
      </div>
    </div>
  )
}
