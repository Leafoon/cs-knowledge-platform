'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { Microchip } from 'lucide-react'

export function SN74181Explorer() {
  const [s, setS] = useState(0)
  const [a, setA] = useState(0b1010)
  const [b, setB] = useState(0b0110)
  const [m, setM] = useState(0)

  const sBits = Array.from({ length: 4 }, (_, i) => (s >> i) & 1)

  const logicOps: Record<number, (a: number, b: number) => number> = {
    0: (a) => (~a + 16) & 0xF,
    1: (a, b) => (~(a | b) + 16) & 0xF,
    2: (a, b) => ((~a + 16) & b) & 0xF,
    3: () => 0,
    4: (a, b) => (~(a & b) + 16) & 0xF,
    5: (a) => (~a + 16) & 0xF,
    6: (a, b) => (a ^ b) & 0xF,
    7: (a, b) => (a & (~b + 16)) & 0xF,
    8: (a, b) => ((~a + 16) | b) & 0xF,
    9: (a, b) => (~(a ^ b) + 16) & 0xF,
    10: (a) => a,
    11: (a, b) => (a | b) & 0xF,
    12: () => 0xF,
    13: (a, b) => (a | (~b + 16)) & 0xF,
    14: (a, b) => (a & b) & 0xF,
    15: (a) => a,
  }

  const arithOps: Record<number, (a: number, b: number) => { result: number; op: string }> = {
    0: (a) => ({ result: (a - 1 + 16) & 0xF, op: 'A减1' }),
    1: (a, b) => ({ result: ((a & b) - 1 + 16) & 0xF, op: 'AB减1' }),
    2: (a, b) => ({ result: ((a & (~b + 16)) - 1 + 16) & 0xF, op: 'AB\'减1' }),
    3: () => ({ result: 0xF, op: '-1 (全1)' }),
    4: (a, b) => ({ result: (a + (a | (~b + 16))) & 0xF, op: 'A加(A+B\')' }),
    5: (a, b) => ({ result: ((a & b) + (a | (~b + 16))) & 0xF, op: 'AB加(A+B\')' }),
    6: (a, b) => ({ result: (a - b - 1 + 32) & 0xF, op: 'A减B减1' }),
    7: (a, b) => ({ result: (a | (~b + 16)) & 0xF, op: 'A+B\'' }),
    8: (a, b) => ({ result: (a + (a | b)) & 0xF, op: 'A加(A+B)' }),
    9: (a, b) => ({ result: (a + b) & 0xF, op: 'A加B' }),
    10: (a, b) => ({ result: ((a & (~b + 16)) + (a | b)) & 0xF, op: 'AB\'+(A+B)' }),
    11: (a, b) => ({ result: (a | b) & 0xF, op: 'A+B' }),
    12: (a) => ({ result: (a + a) & 0xF, op: 'A加A' }),
    13: (a, b) => ({ result: ((a & b) + a) & 0xF, op: 'AB加A' }),
    14: (a, b) => ({ result: ((a & (~b + 16)) + a) & 0xF, op: 'AB\'+A' }),
    15: (a) => ({ result: a & 0xF, op: 'A' }),
  }

  const result = m ? logicOps[s](a, b) : arithOps[s](a, b).result
  const opDesc = m ? '逻辑运算' : arithOps[s](a, b).op

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <div className="flex items-center gap-2 mb-4">
        <Microchip className="w-5 h-5 text-emerald-500" />
        <h3 className="text-lg font-bold">74181 ALU 芯片探索器</h3>
      </div>
      <p className="text-sm text-slate-500 mb-4">经典4位ALU芯片，16种功能选择</p>

      <div className="flex flex-wrap gap-3 mb-4">
        <div>
          <label className="text-xs text-slate-500 block">A (0-15)</label>
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
        <div>
          <label className="text-xs text-slate-500 block">M (模式)</label>
          <button onClick={() => setM(m === 0 ? 1 : 0)}
            className={`px-3 py-1 rounded text-sm ${m ? 'bg-red-500 text-white' : 'bg-blue-500 text-white'}`}>
            {m ? '逻辑' : '算术'}
          </button>
        </div>
      </div>

      <div className="mb-4">
        <label className="text-xs text-slate-500 block mb-1">S (功能选择: {sBits.join('')} = S{s})</label>
        <input type="range" min={0} max={15} value={s} onChange={e => setS(Number(e.target.value))}
          className="w-full" />
        <div className="flex justify-between text-[10px] text-slate-400 font-mono">
          {Array.from({ length: 16 }, (_, i) => (
            <span key={i} className={s === i ? 'text-emerald-600 font-bold' : ''}>{i.toString(16).toUpperCase()}</span>
          ))}
        </div>
      </div>

      <motion.div className="bg-slate-50 dark:bg-slate-900 rounded-lg p-4 mb-4"
        key={`${s}-${m}`} initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
        <div className="text-center">
          <div className="text-sm font-bold mb-1">S={s} (S3S2S1S0={sBits.join('')}), M={m} ({m ? '逻辑' : '算术'})</div>
          <div className="font-mono text-lg">
            {a.toString(2).padStart(4, '0')} ({a}) 和 {b.toString(2).padStart(4, '0')} ({b})
          </div>
          <div className="text-sm text-slate-500 my-1">{opDesc}</div>
          <motion.div className="text-3xl font-bold text-emerald-600 font-mono"
            initial={{ scale: 0.5 }} animate={{ scale: 1 }}>
            = {result.toString(2).padStart(4, '0')} ({result})
          </motion.div>
        </div>
      </motion.div>

      <div className="bg-slate-50 dark:bg-slate-900 rounded-lg p-3">
        <div className="text-xs font-bold text-slate-600 mb-2">功能表 ({m ? '逻辑' : '算术'}模式)</div>
        <div className="grid grid-cols-4 gap-1 text-[10px] font-mono">
          {Array.from({ length: 16 }, (_, i) => {
            const r = m ? logicOps[i](a, b) : arithOps[i](a, b).result
            return (
              <motion.button key={i}
                className={`p-1 rounded text-center ${s === i ? 'bg-emerald-200 dark:bg-emerald-800 font-bold' : 'bg-slate-100 dark:bg-slate-800'}`}
                onClick={() => setS(i)}
                whileHover={{ scale: 1.05 }}>
                S{i}: {r}
              </motion.button>
            )
          })}
        </div>
      </div>
    </div>
  )
}
