'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { Users, RotateCcw } from 'lucide-react'

const threadA = ['A1: LD R1,0(R2)', 'A2: ADD R3,R1,R4', 'A3: SUB R5,R3,R6', 'A4: SW R5,8(R2)']
const threadB = ['B1: LD R7,0(R8)', 'B2: MUL R9,R7,R10', 'B3: ADD R11,R9,R12', 'B4: SW R11,4(R8)']
const stages = ['IF', 'ID', 'EX', 'MEM', 'WB']

export function SMTVisualizer() {
  const [cycle, setCycle] = useState(0)
  const [mode, setMode] = useState<'single' | 'smt'>('smt')
  const maxCycle = mode === 'smt' ? 12 : 20

  const getInstrStage = (threadIdx: number, instrIdx: number, cycle: number) => {
    if (mode === 'single' && threadIdx === 1) return -1
    const startCycle = mode === 'smt'
      ? Math.floor(instrIdx / 2) + (threadIdx === 1 ? 1 : 0)
      : threadIdx * 4 + instrIdx
    const stageIdx = cycle - startCycle
    return stageIdx >= 0 && stageIdx < 5 ? stageIdx : -1
  }

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-lg font-bold mb-4">同时多线程 SMT (Simultaneous Multithreading)</h3>

      <div className="flex items-center gap-3 mb-4">
        <div className="flex gap-1">
          {(['single', 'smt'] as const).map(m => (
            <button key={m} onClick={() => { setMode(m); setCycle(0) }}
              className={`px-3 py-1.5 text-sm rounded ${mode === m ? 'bg-indigo-600 text-white' : 'bg-slate-200 dark:bg-slate-700'}`}>
              {m === 'single' ? '单线程' : 'SMT (2线程)'}
            </button>
          ))}
        </div>
        <button onClick={() => setCycle(c => Math.min(c + 1, maxCycle))}
          className="px-3 py-1.5 text-sm rounded bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300">单步</button>
        <button onClick={() => setCycle(0)}
          className="flex items-center gap-1 px-3 py-1.5 text-sm rounded bg-slate-200 dark:bg-slate-700">
          <RotateCcw className="w-3 h-3" />
        </button>
        <span className="ml-auto text-sm">周期: {cycle}</span>
      </div>

      {/* Thread A */}
      <div className="mb-3">
        <div className="flex items-center gap-2 mb-1">
          <Users className="w-4 h-4 text-blue-500" />
          <span className="text-sm font-medium text-blue-600 dark:text-blue-400">线程 A</span>
        </div>
        <div className="overflow-x-auto">
          <table className="text-xs">
            <tbody>
              {threadA.map((instr, i) => (
                <tr key={i}>
                  <td className="pr-3 py-1 font-mono text-slate-600 dark:text-slate-400 whitespace-nowrap">{instr}</td>
                  {Array.from({ length: maxCycle + 1 }, (_, c) => {
                    const si = getInstrStage(0, i, c)
                    const colors = ['bg-blue-300 dark:bg-blue-700', 'bg-green-300 dark:bg-green-700', 'bg-orange-300 dark:bg-orange-700', 'bg-purple-300 dark:bg-purple-700', 'bg-red-300 dark:bg-red-700']
                    return (
                      <td key={c} className={`w-6 h-5 text-center rounded-sm ${si >= 0 ? colors[si] : ''} ${c === cycle && si >= 0 ? 'ring-1 ring-yellow-400' : ''}`}>
                        {si >= 0 && <span className="font-bold">{stages[si][0]}</span>}
                      </td>
                    )
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Thread B */}
      {mode === 'smt' && (
        <div>
          <div className="flex items-center gap-2 mb-1">
            <Users className="w-4 h-4 text-green-500" />
            <span className="text-sm font-medium text-green-600 dark:text-green-400">线程 B</span>
          </div>
          <div className="overflow-x-auto">
            <table className="text-xs">
              <tbody>
                {threadB.map((instr, i) => (
                  <tr key={i}>
                    <td className="pr-3 py-1 font-mono text-slate-600 dark:text-slate-400 whitespace-nowrap">{instr}</td>
                    {Array.from({ length: maxCycle + 1 }, (_, c) => {
                      const si = getInstrStage(1, i, c)
                      const colors = ['bg-blue-300 dark:bg-blue-700', 'bg-green-300 dark:bg-green-700', 'bg-orange-300 dark:bg-orange-700', 'bg-purple-300 dark:bg-purple-700', 'bg-red-300 dark:bg-red-700']
                      return (
                        <td key={c} className={`w-6 h-5 text-center rounded-sm ${si >= 0 ? colors[si] : ''} ${c === cycle && si >= 0 ? 'ring-1 ring-yellow-400' : ''}`}>
                          {si >= 0 && <span className="font-bold">{stages[si][0]}</span>}
                        </td>
                      )
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      <div className="grid grid-cols-3 gap-3 mt-4">
        <div className="p-3 bg-slate-50 dark:bg-slate-800/50 rounded-lg text-center">
          <p className="text-xs text-slate-500 mb-1">模式</p>
          <p className="text-sm font-bold">{mode === 'smt' ? 'SMT' : '单线程'}</p>
        </div>
        <div className="p-3 bg-slate-50 dark:bg-slate-800/50 rounded-lg text-center">
          <p className="text-xs text-slate-500 mb-1">IPC 提升</p>
          <p className="text-sm font-bold text-green-600">{mode === 'smt' ? '~1.5-2x' : '1x'}</p>
        </div>
        <div className="p-3 bg-slate-50 dark:bg-slate-800/50 rounded-lg text-center">
          <p className="text-xs text-slate-500 mb-1">资源共享</p>
          <p className="text-sm font-bold text-blue-600">{mode === 'smt' ? 'ALU/Cache/ROB' : '独占'}</p>
        </div>
      </div>
    </div>
  )
}
