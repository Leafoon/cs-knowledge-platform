'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { ListOrdered, Play, RotateCcw, CheckCircle } from 'lucide-react'

interface ROBEntry {
  id: number
  instr: string
  dest: string
  value: string
  state: 'Issue' | 'Execute' | 'Write' | 'Commit'
}

const program = [
  { instr: 'LD F6, 34(R2)', dest: 'F6', value: 'M[R2+34]' },
  { instr: 'MULTD F0, F2, F4', dest: 'F0', value: 'F2×F4' },
  { instr: 'SUBD F8, F6, F2', dest: 'F8', value: 'F6-F2' },
  { instr: 'DIVD F10, F0, F6', dest: 'F10', value: 'F0/F6' },
  { instr: 'ADDD F6, F8, F2', dest: 'F6', value: 'F8+F2' },
]

export function ReorderBufferDemo() {
  const [cycle, setCycle] = useState(0)
  const maxCycle = 16

  const getStates = (cycle: number): ROBEntry['state'][] => {
    const states: ROBEntry['state'][] = []
    const issueCycles = [1, 2, 3, 4, 5]
    const execCycles = [2, 4, 5, 6, 7]
    const writeCycles = [3, 8, 6, 12, 8]
    const commitCycles = [4, 9, 7, 13, 9]

    for (let i = 0; i < program.length; i++) {
      if (cycle < issueCycles[i]) states.push('Issue')
      else if (cycle < writeCycles[i]) states.push('Execute')
      else if (cycle < commitCycles[i]) states.push('Write')
      else states.push('Commit')
    }
    return states
  }

  const states = getStates(cycle)
  const stateColors = {
    Issue: 'bg-slate-100 dark:bg-slate-800 border-slate-300 dark:border-slate-600',
    Execute: 'bg-blue-100 dark:bg-blue-900 border-blue-300 dark:border-blue-700',
    Write: 'bg-orange-100 dark:bg-orange-900 border-orange-300 dark:border-orange-700',
    Commit: 'bg-green-100 dark:bg-green-900 border-green-300 dark:border-green-700',
  }

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-lg font-bold mb-4">重排序缓冲 (Reorder Buffer)</h3>

      <div className="flex items-center gap-3 mb-4">
        <span className="text-sm font-medium">周期: {cycle}</span>
        <button onClick={() => setCycle(c => Math.min(c + 1, maxCycle))}
          className="px-3 py-1.5 text-sm rounded bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300">单步</button>
        <button onClick={() => setCycle(0)}
          className="flex items-center gap-1 px-3 py-1.5 text-sm rounded bg-slate-200 dark:bg-slate-700">
          <RotateCcw className="w-3 h-3" /> 重置
        </button>
      </div>

      {/* ROB Table */}
      <div className="overflow-x-auto mb-4">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-200 dark:border-slate-700">
              <th className="text-left p-2 font-medium text-slate-500">ROB#</th>
              <th className="text-left p-2 font-medium text-slate-500">指令</th>
              <th className="text-left p-2 font-medium text-slate-500">目标</th>
              <th className="text-left p-2 font-medium text-slate-500">值</th>
              <th className="text-center p-2 font-medium text-slate-500">状态</th>
            </tr>
          </thead>
          <tbody>
            {program.map((p, i) => (
              <motion.tr key={i}
                className={`border-b border-slate-100 dark:border-slate-800 ${stateColors[states[i]]}`}
                animate={{ x: states[i] === 'Commit' ? [0, 2, 0] : 0 }}
                transition={{ duration: 0.3 }}
              >
                <td className="p-2 font-mono text-xs">{i + 1}</td>
                <td className="p-2 font-mono text-xs">{p.instr}</td>
                <td className="p-2 font-mono text-xs">{p.dest}</td>
                <td className="p-2 font-mono text-xs">{states[i] === 'Commit' ? p.value : '...'}</td>
                <td className="p-2 text-center">
                  <span className="text-xs px-2 py-0.5 rounded-full bg-white/50 dark:bg-black/20">
                    {states[i] === 'Commit' && <CheckCircle className="w-3 h-3 inline mr-1 text-green-600" />}
                    {states[i]}
                  </span>
                </td>
              </motion.tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Key insight */}
      <div className="p-3 bg-purple-50 dark:bg-purple-950/20 rounded-lg border border-purple-200 dark:border-purple-800 text-xs text-purple-700 dark:text-purple-300">
        <p className="font-semibold mb-1">乱序执行 → 顺序提交</p>
        <p>ROB保证指令按程序序提交(commit)，即使执行完成顺序不同。DIVD最后执行完成，但通过ROB确保不会违反程序语义。</p>
      </div>
    </div>
  )
}
