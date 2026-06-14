'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { TreePine } from 'lucide-react'

export function WallaceTreeViz() {
  const [a, setA] = useState(0b1101)
  const [b, setB] = useState(0b1011)
  const bits = 4

  const aBits = Array.from({ length: bits }, (_, i) => (a >> i) & 1)
  const bBits = Array.from({ length: bits }, (_, i) => (b >> i) & 1)

  const pp = bBits.map((bj, j) => {
    const row = Array(j).fill(0)
    aBits.forEach(ai => row.push(ai & bj))
    return row
  })

  const levels: string[][] = []
  let current = pp.map(r => r.join(''))
  levels.push([...current])

  while (current.length > 2) {
    const next: string[] = []
    for (let i = 0; i < current.length; i += 3) {
      if (i + 2 < current.length) {
        const maxLen = Math.max(current[i].length, current[i + 1].length, current[i + 2].length)
        const sum = Array(maxLen).fill(0)
        const carry = Array(maxLen + 1).fill(0)
        for (let k = 0; k < maxLen; k++) {
          const a = parseInt(current[i][current[i].length - 1 - k] || '0')
          const bb = parseInt(current[i + 1][current[i + 1].length - 1 - k] || '0')
          const c = parseInt(current[i + 2][current[i + 2].length - 1 - k] || '0')
          const total = a + bb + c
          sum[k] = total % 2
          carry[k + 1] = Math.floor(total / 2)
        }
        next.push(sum.reverse().join(''))
        if (carry.some(c => c)) next.push(carry.reverse().join(''))
      } else {
        next.push(current[i])
      }
    }
    current = next
    levels.push([...current])
  }

  const result = a * b
  const colors = ['bg-blue-100 dark:bg-blue-900', 'bg-green-100 dark:bg-green-900', 'bg-orange-100 dark:bg-orange-900', 'bg-purple-100 dark:bg-purple-900']

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <div className="flex items-center gap-2 mb-4">
        <TreePine className="w-5 h-5 text-emerald-500" />
        <h3 className="text-lg font-bold">Wallace 树压缩</h3>
      </div>
      <p className="text-sm text-slate-500 mb-4">部分积通过 3→2 压缩器逐层减少</p>

      <div className="flex gap-4 mb-4">
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

      <div className="space-y-4">
        {levels.map((level, li) => (
          <motion.div key={li}
            initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }}
            transition={{ delay: li * 0.3 }}>
            <div className="text-xs font-medium text-slate-500 mb-1">
              {li === 0 ? '部分积' : `第${li}层压缩 (${levels[li - 1].length}→${level.length})`}
            </div>
            <div className="space-y-1">
              {level.map((row, ri) => (
                <div key={ri} className="flex gap-0.5 font-mono text-xs">
                  {row.split('').map((bit, bi) => (
                    <motion.span key={bi}
                      className={`w-6 h-6 flex items-center justify-center rounded ${bit === '1' ? colors[li % colors.length] + ' font-bold' : 'bg-slate-100 dark:bg-slate-800'}`}
                      initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ delay: li * 0.3 + bi * 0.02 }}>
                      {bit}
                    </motion.span>
                  ))}
                </div>
              ))}
            </div>
            {li < levels.length - 1 && (
              <div className="text-slate-300 text-lg text-center my-1">↓</div>
            )}
          </motion.div>
        ))}
      </div>

      {levels.length > 0 && (
        <div className="mt-3 flex gap-1 items-center justify-center">
          <span className="text-xs text-slate-400">最终加法器:</span>
          {levels[levels.length - 1].map((row, i) => (
            <span key={i} className="font-mono text-xs bg-green-100 dark:bg-green-900 px-2 py-1 rounded">{row}={parseInt(row, 2)}</span>
          ))}
        </div>
      )}

      <div className="mt-3 p-2 bg-green-50 dark:bg-green-950 rounded text-center text-sm">
        {a} × {b} = {result} &nbsp;|&nbsp; Wallace树深度 = O(log n)，比串行快得多
      </div>
    </div>
  )
}
