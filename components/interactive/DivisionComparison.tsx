'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { BarChart3 } from 'lucide-react'

export function DivisionComparison() {
  const [dividend, setDividend] = useState(13)
  const [divisor, setDivisor] = useState(3)
  const bits = 4

  const restoringSteps = (() => {
    let count = 0
    let A = 0, Q = dividend
    for (let i = 0; i < bits; i++) {
      A = ((A << 1) | (Q >> (bits - 1))) & 0x1F
      Q = (Q << 1) & 0xF
      count++ // shift
      A = A - divisor
      count++ // subtract
      if (A < 0) {
        A = A + divisor
        count++ // restore
      } else {
        Q = Q | 1
      }
      count++ // set Q0
    }
    return count
  })()

  const nonRestoringSteps = (() => {
    let count = 0
    let A = 0, Q = dividend
    for (let i = 0; i < bits; i++) {
      A = ((A << 1) | (Q >> (bits - 1))) & 0x1F
      Q = (Q << 1) & 0xF
      count++
      if (A >= 0) { A = A - divisor } else { A = A + divisor }
      count++
      Q = Q | (A >= 0 ? 1 : 0)
      count++
    }
    if (A < 0) { A = A + divisor; count++ }
    return count
  })()

  const comparison = [
    { name: '恢复余数法', steps: restoringSteps, color: 'amber', pros: '直观易理解', cons: '可能需恢复步骤' },
    { name: '不恢复余数法', steps: nonRestoringSteps, color: 'cyan', pros: '无恢复步骤', cons: '逻辑稍复杂' },
  ]

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <div className="flex items-center gap-2 mb-4">
        <BarChart3 className="w-5 h-5 text-indigo-500" />
        <h3 className="text-lg font-bold">除法算法对比</h3>
      </div>

      <div className="flex flex-wrap gap-4 mb-4">
        <div>
          <label className="text-xs text-slate-500 block">被除数</label>
          <input type="number" min={0} max={15} value={dividend}
            onChange={e => setDividend(Number(e.target.value) & 0xF)}
            className="w-16 px-2 py-1 border rounded font-mono" />
        </div>
        <div>
          <label className="text-xs text-slate-500 block">除数</label>
          <input type="number" min={1} max={15} value={divisor}
            onChange={e => setDivisor(Math.max(1, Number(e.target.value) & 0xF))}
            className="w-16 px-2 py-1 border rounded font-mono" />
        </div>
      </div>

      <div className="text-sm mb-4">{dividend} ÷ {divisor} = {Math.floor(dividend / divisor)} 余 {dividend % divisor}</div>

      <div className="grid grid-cols-2 gap-4">
        {comparison.map(c => (
          <motion.div key={c.name}
            className={`p-4 rounded-lg border-2 border-${c.color}-300 bg-${c.color}-50 dark:bg-${c.color}-950`}
            initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
            <div className="font-bold text-sm mb-2">{c.name}</div>
            <div className="text-3xl font-bold text-center my-3">{c.steps}</div>
            <div className="text-xs text-center text-slate-500">操作次数</div>
            <div className="mt-3 text-xs space-y-1">
              <div className="text-green-600">✓ {c.pros}</div>
              <div className="text-red-600">✗ {c.cons}</div>
            </div>
          </motion.div>
        ))}
      </div>

      <div className="mt-4 bg-slate-50 dark:bg-slate-900 rounded-lg p-3">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b"><th className="py-1 text-left">特性</th><th>恢复余数法</th><th>不恢复余数法</th></tr>
          </thead>
          <tbody className="font-mono">
            <tr className="border-b"><td className="py-1">余数为负时</td><td>恢复再试减</td><td>下一步加除数</td></tr>
            <tr className="border-b"><td className="py-1">每步操作</td><td>减→判断→(恢复)</td><td>加/减→判断</td></tr>
            <tr className="border-b"><td className="py-1">最大步骤</td><td>3n次加减</td><td>n+1次加减</td></tr>
            <tr><td className="py-1">硬件复杂度</td><td>需恢复逻辑</td><td>更简单</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  )
}
