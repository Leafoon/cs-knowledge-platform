'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { Grid3X3 } from 'lucide-react'

export function ArrayMultiplierViz() {
  const [a, setA] = useState(0b1101)
  const [b, setB] = useState(0b1011)
  const [highlightRow, setHighlightRow] = useState(-1)
  const bits = 4

  const aBits = Array.from({ length: bits }, (_, i) => (a >> i) & 1)
  const bBits = Array.from({ length: bits }, (_, i) => (b >> i) & 1)

  const partialProducts = bBits.map((bj, j) =>
    aBits.map((ai, i) => ai & bj)
  )

  const result = a * b
  const resultBin = result.toString(2).padStart(bits * 2, '0')

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <div className="flex items-center gap-2 mb-4">
        <Grid3X3 className="w-5 h-5 text-violet-500" />
        <h3 className="text-lg font-bold">阵列乘法器</h3>
      </div>
      <p className="text-sm text-slate-500 mb-4">网格展示部分积生成与累加</p>

      <div className="flex gap-4 mb-4">
        <div>
          <label className="text-xs text-slate-500 block">A</label>
          <input type="number" min={0} max={15} value={a}
            onChange={e => { setA(Number(e.target.value) & 0xF); setHighlightRow(-1) }}
            className="w-16 px-2 py-1 border rounded font-mono" />
        </div>
        <div>
          <label className="text-xs text-slate-500 block">B</label>
          <input type="number" min={0} max={15} value={b}
            onChange={e => { setB(Number(e.target.value) & 0xF); setHighlightRow(-1) }}
            className="w-16 px-2 py-1 border rounded font-mono" />
        </div>
      </div>

      <div className="text-sm font-mono mb-3">
        A = {aBits.slice().reverse().join('')} ({a}), B = {bBits.slice().reverse().join('')} ({b})
      </div>

      <div className="bg-slate-50 dark:bg-slate-900 rounded-lg p-4 overflow-x-auto">
        <table className="font-mono text-sm mx-auto">
          <thead>
            <tr>
              <td className="px-2 text-slate-400" />
              {aBits.slice().reverse().map((_, i) => (
                <th key={i} className="px-2 text-xs text-slate-400">A{bits - 1 - i}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {partialProducts.map((pp, j) => (
              <motion.tr key={j}
                className="cursor-pointer"
                onClick={() => setHighlightRow(j === highlightRow ? -1 : j)}
                animate={{ backgroundColor: highlightRow === j ? 'rgba(139,92,246,0.1)' : 'transparent' }}>
                <td className="px-2 text-xs text-slate-500">B{j}={bBits[j]}</td>
                {Array.from({ length: j }, (_, k) => (
                  <td key={`pad${k}`} className="px-1" />
                ))}
                {pp.map((v, i) => (
                  <motion.td key={i}
                    className={`px-2 py-1 text-center rounded ${highlightRow === j ? 'bg-violet-200 dark:bg-violet-800 font-bold' : v ? 'bg-green-100 dark:bg-green-900' : 'bg-slate-100 dark:bg-slate-800'}`}
                    animate={{ scale: highlightRow === j ? 1.1 : 1 }}>
                    {v}
                  </motion.td>
                ))}
              </motion.tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="mt-4 flex items-center gap-2">
        <span className="text-sm text-slate-500">部分积逐行右移后累加:</span>
      </div>

      <div className="mt-3 flex gap-1 items-center justify-center font-mono">
        {resultBin.split('').map((b, i) => (
          <motion.span key={i}
            className={`w-8 h-8 flex items-center justify-center rounded text-sm font-bold ${b === '1' ? 'bg-green-500 text-white' : 'bg-slate-200 text-slate-600'}`}
            initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.08 }}>
            {b}
          </motion.span>
        ))}
      </div>

      <div className="mt-2 text-center text-sm">
        {a} × {b} = {result} ({resultBin})
      </div>

      {highlightRow >= 0 && (
        <div className="mt-3 p-2 bg-violet-50 dark:bg-violet-950 rounded text-xs">
          部分积 {highlightRow}: B{highlightRow}={bBits[highlightRow]} × A = {partialProducts[highlightRow].slice().reverse().join('')}，
          左移 {highlightRow} 位
        </div>
      )}
    </div>
  )
}
