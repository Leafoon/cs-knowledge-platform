'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { GitCompare } from 'lucide-react'

export function Radix4BoothDemo() {
  const [multiplier, setMultiplier] = useState(0b01101001)
  const [mode, setMode] = useState<'radix2' | 'radix4'>('radix4')
  const bits = 8

  const mBits = Array.from({ length: bits }, (_, i) => (multiplier >> i) & 1)

  const radix2Pairs = mBits.map((b, i) => {
    const prev = i > 0 ? mBits[i - 1] : 0
    const pair = `${b}${prev}`
    let op = '无操作'
    if (pair === '01') op = '+M'
    else if (pair === '10') op = '-M'
    return { bit: i, pair, op }
  })

  const radix4Groups: { group: string; bits: string; op: string; shift: number }[] = []
  const extBits = [...mBits, 0]
  for (let i = 0; i < bits; i += 2) {
    const b0 = extBits[i] || 0
    const b1 = extBits[i + 1] || 0
    const b2 = extBits[i + 2] || 0
    const val = b0 + b1 * 2 - b2 * 4
    const group = `${b2}${b1}${b0}`
    let op = '0'
    if (val === 1) op = '+M'
    else if (val === 2) op = '+2M'
    else if (val === -1) op = '-M'
    else if (val === -2) op = '-2M'
    radix4Groups.push({ group, bits: `位${i}-${i + 1}`, op, shift: i })
  }

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <div className="flex items-center gap-2 mb-4">
        <GitCompare className="w-5 h-5 text-cyan-500" />
        <h3 className="text-lg font-bold">基4 Booth 算法 (Radix-4)</h3>
      </div>
      <p className="text-sm text-slate-500 mb-4">对比基2和基4的编码方式，基4将步数减半</p>

      <div className="flex gap-4 mb-4">
        <div>
          <label className="text-xs text-slate-500 block">乘数 (8-bit)</label>
          <input type="number" min={0} max={255} value={multiplier}
            onChange={e => setMultiplier(Number(e.target.value) & 0xFF)}
            className="w-20 px-2 py-1 border rounded font-mono" />
        </div>
        <div className="flex items-end gap-1">
          {(['radix2', 'radix4'] as const).map(m => (
            <button key={m} onClick={() => setMode(m)}
              className={`px-3 py-1 text-sm rounded ${mode === m ? 'bg-cyan-500 text-white' : 'bg-slate-200'}`}>
              {m === 'radix2' ? '基2' : '基4'}
            </button>
          ))}
        </div>
      </div>

      <div className="mb-4 flex gap-1">
        {mBits.slice().reverse().map((b, i) => (
          <span key={i} className="w-7 h-7 flex items-center justify-center bg-blue-100 dark:bg-blue-900 rounded text-xs font-mono">{b}</span>
        ))}
      </div>

      {mode === 'radix2' ? (
        <div className="space-y-1">
          <div className="text-xs font-medium text-slate-600 mb-2">基2 Booth 编码 ({bits} 步)</div>
          {radix2Pairs.map((p, i) => (
            <motion.div key={i} className="flex items-center gap-3 text-xs font-mono px-2 py-1 rounded bg-slate-50 dark:bg-slate-800"
              initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.05 }}>
              <span className="w-12 text-slate-400">位{p.bit}</span>
              <span className="w-16">{p.pair}</span>
              <span className={`font-bold ${p.op.includes('+') ? 'text-green-600' : p.op.includes('-') ? 'text-red-600' : 'text-slate-400'}`}>{p.op}</span>
            </motion.div>
          ))}
        </div>
      ) : (
        <div className="space-y-1">
          <div className="text-xs font-medium text-slate-600 mb-2">基4 Booth 编码 ({bits / 2} 步，步数减半!)</div>
          {radix4Groups.map((g, i) => (
            <motion.div key={i} className="flex items-center gap-3 text-xs font-mono px-2 py-2 rounded bg-cyan-50 dark:bg-cyan-950 border border-cyan-200 dark:border-cyan-800"
              initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.1 }}>
              <span className="w-16 text-slate-400">{g.bits}</span>
              <span className="w-12 bg-cyan-100 dark:bg-cyan-900 rounded px-1 text-center">{g.group}</span>
              <span className={`font-bold ${g.op.includes('+') ? 'text-green-600' : g.op.includes('-') ? 'text-red-600' : 'text-slate-400'}`}>{g.op}</span>
              <span className="text-slate-400">左移{g.shift}位</span>
            </motion.div>
          ))}
        </div>
      )}

      <div className="mt-4 grid grid-cols-2 gap-3 text-xs">
        <div className="p-2 bg-slate-50 dark:bg-slate-800 rounded">
          <div className="font-bold mb-1">基2 Booth</div>
          <div>步数: {bits} 步</div>
          <div>每次移1位</div>
        </div>
        <div className="p-2 bg-cyan-50 dark:bg-cyan-950 rounded">
          <div className="font-bold mb-1 text-cyan-700">基4 Booth</div>
          <div>步数: {bits / 2} 步 ✓</div>
          <div>每次移2位，速度翻倍</div>
        </div>
      </div>
    </div>
  )
}
