'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { Flag } from 'lucide-react'

export function FlagRegisterViz() {
  const [a, setA] = useState(120)
  const [b, setB] = useState(50)
  const [bits, setBits] = useState(8)

  const maxVal = (1 << (bits - 1)) - 1
  const minVal = -(1 << (bits - 1))
  const uMax = (1 << bits) - 1

  const toTwos = (v: number) => {
    const u = v < 0 ? (1 << bits) + v : v
    return u.toString(2).padStart(bits, '0')
  }

  const aBin = toTwos(a)
  const bBin = toTwos(b)
  const sumRaw = a + b
  const sumBin = toTwos(sumRaw & uMax)
  const sumSigned = sumRaw > maxVal ? sumRaw - (1 << bits) : sumRaw < minVal ? sumRaw + (1 << bits) : sumRaw

  const ZF = (sumRaw & uMax) === 0 ? 1 : 0
  const SF = (sumRaw >> (bits - 1)) & 1
  const CF = (sumRaw > uMax || sumRaw < 0) ? 1 : 0
  const OF = (a > 0 && b > 0 && sumSigned < 0) || (a < 0 && b < 0 && sumSigned > 0) ? 1 : 0

  const flags = [
    { name: 'ZF', full: '零标志 (Zero Flag)', value: ZF, color: 'blue',
      desc: ZF ? '结果为零' : '结果非零', logic: `结果=${sumRaw & uMax}, 全零则ZF=1` },
    { name: 'SF', full: '符号标志 (Sign Flag)', value: SF, color: 'purple',
      desc: SF ? '结果为负' : '结果非负', logic: `取结果最高位=${SF}` },
    { name: 'CF', full: '进位标志 (Carry Flag)', value: CF, color: 'amber',
      desc: CF ? '无符号溢出' : '无进位', logic: `无符号运算超出${uMax}则CF=1` },
    { name: 'OF', full: '溢出标志 (Overflow Flag)', value: OF, color: 'red',
      desc: OF ? '有符号溢出!' : '无溢出', logic: `同号相加结果异号则OF=1` },
  ]

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <div className="flex items-center gap-2 mb-4">
        <Flag className="w-5 h-5 text-red-500" />
        <h3 className="text-lg font-bold">标志位可视化</h3>
      </div>

      <div className="flex flex-wrap gap-3 mb-4">
        <div>
          <label className="text-xs text-slate-500 block">位宽</label>
          <div className="flex gap-1 mt-1">
            {[4, 8].map(w => (
              <button key={w} onClick={() => { setBits(w); setA(0); setB(0) }}
                className={`px-3 py-1 text-sm rounded ${bits === w ? 'bg-red-500 text-white' : 'bg-slate-200'}`}>{w}位</button>
            ))}
          </div>
        </div>
        <div>
          <label className="text-xs text-slate-500 block">A ({minVal}~{maxVal})</label>
          <input type="number" min={minVal} max={maxVal} value={a}
            onChange={e => setA(Math.max(minVal, Math.min(maxVal, Number(e.target.value))))}
            className="w-20 px-2 py-1 border rounded font-mono" />
        </div>
        <div>
          <label className="text-xs text-slate-500 block">B</label>
          <input type="number" min={minVal} max={maxVal} value={b}
            onChange={e => setB(Math.max(minVal, Math.min(maxVal, Number(e.target.value))))}
            className="w-20 px-2 py-1 border rounded font-mono" />
        </div>
      </div>

      <div className="grid grid-cols-3 gap-3 mb-4 text-xs font-mono text-center">
        <div className="bg-blue-50 dark:bg-blue-950 rounded p-2">
          <div>A = {a}</div><div>{aBin}</div>
        </div>
        <div className="bg-purple-50 dark:bg-purple-950 rounded p-2">
          <div>B = {b}</div><div>{bBin}</div>
        </div>
        <div className={`rounded p-2 ${OF ? 'bg-red-100 dark:bg-red-950' : 'bg-green-50 dark:bg-green-950'}`}>
          <div>A+B = {sumRaw}</div><div>{sumBin}</div>
        </div>
      </div>

      <div className="grid grid-cols-4 gap-3">
        {flags.map(f => (
          <motion.div key={f.name}
            className={`p-3 rounded-lg border-2 text-center ${f.value ? `border-${f.color}-400 bg-${f.color}-50 dark:bg-${f.color}-950` : 'border-slate-200 bg-slate-50 dark:bg-slate-800'}`}
            animate={{ scale: f.value ? [1, 1.05, 1] : 1 }}>
            <div className="text-xs text-slate-500 mb-1">{f.full}</div>
            <motion.div className={`text-3xl font-bold font-mono ${f.value ? `text-${f.color}-600` : 'text-slate-300'}`}
              key={`${f.name}-${f.value}`}
              initial={{ scale: 0 }} animate={{ scale: 1 }}>
              {f.value}
            </motion.div>
            <div className={`text-xs mt-1 ${f.value ? `text-${f.color}-700` : 'text-slate-400'}`}>
              {f.desc}
            </div>
            <div className="text-[10px] text-slate-400 mt-1">{f.logic}</div>
          </motion.div>
        ))}
      </div>

      {OF ? (
        <div className="mt-3 p-2 bg-red-50 dark:bg-red-950 rounded text-xs text-red-700 font-bold text-center">
          ⚠️ 溢出! {a}+{b}={sumRaw} 超出 [{minVal}, {maxVal}] 范围
        </div>
      ) : null}
    </div>
  )
}
