'use client'

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Layers, Play, RotateCcw } from 'lucide-react'

const issueWidths = [1, 2, 4]
const instructions = [
  'ADD R1, R2, R3', 'SUB R4, R5, R6', 'MUL R7, R8, R9', 'ADD R10, R11, R12',
  'LW R13, 0(R14)', 'SW R15, 4(R16)', 'AND R17, R18, R19', 'OR R20, R21, R22'
]
const stages = ['IF', 'ID', 'EX', 'MEM', 'WB']

export function SuperscalarViz() {
  const [width, setWidth] = useState(2)
  const [cycle, setCycle] = useState(0)
  const [isRunning, setIsRunning] = useState(false)

  const totalCycles = Math.ceil(instructions.length / width) + stages.length - 1

  const step = () => {
    setCycle(c => (c + 1) % (totalCycles + 2))
  }

  const run = () => {
    setIsRunning(true)
    setCycle(0)
    let c = 0
    const timer = setInterval(() => {
      c++
      if (c > totalCycles) { clearInterval(timer); setIsRunning(false); return }
      setCycle(c)
    }, 600)
  }

  const getInstrsInCycle = (c: number) => {
    const result: { instr: string; stage: string; slot: number }[] = []
    for (let i = 0; i < instructions.length; i++) {
      const issueGroup = Math.floor(i / width)
      const startCycle = issueGroup
      const stageIdx = c - startCycle
      if (stageIdx >= 0 && stageIdx < stages.length) {
        result.push({ instr: instructions[i], stage: stages[stageIdx], slot: i % width })
      }
    }
    return result
  }

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-lg font-bold mb-4">超标量处理器 (Superscalar Processor)</h3>

      <div className="flex items-center gap-4 mb-4">
        <div>
          <label className="block text-sm font-medium mb-1">发射宽度: {width}</label>
          <div className="flex gap-1">
            {issueWidths.map(w => (
              <button key={w} onClick={() => { setWidth(w); setCycle(0) }}
                className={`px-3 py-1 text-sm rounded ${width === w ? 'bg-blue-600 text-white' : 'bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-300'}`}>
                {w}-wide
              </button>
            ))}
          </div>
        </div>
        <div className="flex gap-2 ml-auto">
          <button onClick={step} disabled={isRunning}
            className="px-3 py-1.5 text-sm rounded bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300 disabled:opacity-50">
            单步
          </button>
          <button onClick={run} disabled={isRunning}
            className="flex items-center gap-1 px-3 py-1.5 text-sm rounded bg-green-500 text-white disabled:opacity-50">
            <Play className="w-3 h-3" /> 运行
          </button>
          <button onClick={() => setCycle(0)}
            className="flex items-center gap-1 px-3 py-1.5 text-sm rounded bg-slate-200 dark:bg-slate-700">
            <RotateCcw className="w-3 h-3" />
          </button>
        </div>
      </div>

      {/* Pipeline diagram */}
      <div className="p-3 bg-white dark:bg-slate-900 rounded-lg border border-slate-200 dark:border-slate-700 mb-4 overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr>
              <th className="text-left p-1 font-medium text-slate-500">指令</th>
              {Array.from({ length: totalCycles + 1 }, (_, c) => (
                <th key={c} className={`p-1 font-mono text-center min-w-[32px] ${c === cycle ? 'bg-yellow-200 dark:bg-yellow-800 rounded' : ''}`}>
                  {c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {instructions.map((instr, i) => {
              const issueGroup = Math.floor(i / width)
              return (
                <tr key={i}>
                  <td className="p-1 font-mono text-slate-600 dark:text-slate-400 whitespace-nowrap text-[10px]">{instr}</td>
                  {Array.from({ length: totalCycles + 1 }, (_, c) => {
                    const stageIdx = c - issueGroup
                    const active = stageIdx >= 0 && stageIdx < stages.length
                    const colors = ['bg-blue-200 dark:bg-blue-800', 'bg-green-200 dark:bg-green-800', 'bg-orange-200 dark:bg-orange-800', 'bg-purple-200 dark:bg-purple-800', 'bg-red-200 dark:bg-red-800']
                    return (
                      <td key={c} className={`p-1 text-center rounded ${active ? colors[stageIdx] : ''} ${c === cycle && active ? 'ring-2 ring-yellow-400' : ''}`}>
                        {active && <span className="text-[10px] font-bold">{stages[stageIdx]}</span>}
                      </td>
                    )
                  })}
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      <div className="grid grid-cols-3 gap-3">
        <div className="p-3 bg-slate-50 dark:bg-slate-800/50 rounded-lg text-center">
          <p className="text-xs text-slate-500 mb-1">发射宽度</p>
          <p className="text-xl font-bold text-blue-600">{width}</p>
        </div>
        <div className="p-3 bg-slate-50 dark:bg-slate-800/50 rounded-lg text-center">
          <p className="text-xs text-slate-500 mb-1">IPC (理论峰值)</p>
          <p className="text-xl font-bold text-green-600">{width}</p>
        </div>
        <div className="p-3 bg-slate-50 dark:bg-slate-800/50 rounded-lg text-center">
          <p className="text-xs text-slate-500 mb-1">总周期</p>
          <p className="text-xl font-bold text-purple-600">{totalCycles}</p>
        </div>
      </div>
    </div>
  )
}
