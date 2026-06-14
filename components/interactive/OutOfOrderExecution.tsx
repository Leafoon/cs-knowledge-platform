'use client'

import { useState, useMemo } from 'react'
import { motion } from 'framer-motion'
import { Shuffle, RotateCcw, Play } from 'lucide-react'

interface OOOInstr {
  id: number
  op: string
  dest: string
  src1: string
  src2: string
  latency: number
  issueCycle: number
  startExec: number
  endExec: number
  writeback: number
}

const instructions = [
  { op: 'LD', dest: 'R1', src1: '0(R2)', src2: '', latency: 2 },
  { op: 'MULT', dest: 'R3', src1: 'R1', src2: 'R4', latency: 4 },
  { op: 'LD', dest: 'R5', src1: '8(R2)', src2: '', latency: 2 },
  { op: 'ADD', dest: 'R6', src1: 'R5', src2: 'R7', latency: 1 },
  { op: 'SUB', dest: 'R8', src1: 'R3', src2: 'R6', latency: 1 },
  { op: 'SD', dest: '16(R2)', src1: 'R8', src2: '', latency: 1 },
]

export function OutOfOrderExecution() {
  const [cycle, setCycle] = useState(0)
  const maxCycle = 18

  const schedule = useMemo(() => {
    const result: OOOInstr[] = []
    let issue = 1
    const readyAt: Record<string, number> = {}

    for (let i = 0; i < instructions.length; i++) {
      const instr = instructions[i]
      const src1Ready = instr.src1.match(/R\d+/) ? (readyAt[instr.src1.match(/R\d+/)![0]] || 0) : 0
      const src2Ready = instr.src2 ? (readyAt[instr.src2] || 0) : 0
      const dataReady = Math.max(src1Ready, src2Ready)
      const startExec = Math.max(issue + 1, dataReady + 1)
      const endExec = startExec + instr.latency - 1
      const writeback = endExec + 1

      readyAt[instr.dest] = endExec
      result.push({ ...instr, id: i, issueCycle: issue, startExec, endExec, writeback })
      issue++
    }
    return result
  }, [])

  const getInstrState = (instr: OOOInstr) => {
    if (cycle < instr.issueCycle) return '等待'
    if (cycle <= instr.endExec) return '执行'
    if (cycle <= instr.writeback) return '写回'
    return '完成'
  }

  const stateColor = (s: string) => {
    switch (s) {
      case '等待': return 'bg-slate-100 dark:bg-slate-800 text-slate-400'
      case '执行': return 'bg-orange-100 dark:bg-orange-900 text-orange-700 dark:text-orange-300'
      case '写回': return 'bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300'
      default: return 'bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-300'
    }
  }

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-lg font-bold mb-4">乱序执行流程 (Out-of-Order Execution)</h3>

      <div className="flex items-center gap-3 mb-4">
        <span className="text-sm">周期: {cycle}</span>
        <button onClick={() => setCycle(c => Math.min(c + 1, maxCycle))}
          className="px-3 py-1.5 text-sm rounded bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300">单步</button>
        <button onClick={() => { setCycle(0) }}
          className="flex items-center gap-1 px-3 py-1.5 text-sm rounded bg-slate-200 dark:bg-slate-700">
          <RotateCcw className="w-3 h-3" />
        </button>
      </div>

      {/* Timeline */}
      <div className="overflow-x-auto mb-4">
        <table className="text-xs">
          <thead>
            <tr>
              <th className="text-left p-1 font-medium text-slate-500 min-w-[140px]">指令</th>
              {Array.from({ length: maxCycle + 1 }, (_, c) => (
                <th key={c} className={`p-1 font-mono text-center w-6 ${c === cycle ? 'bg-yellow-200 dark:bg-yellow-800 rounded' : ''}`}>{c}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {schedule.map((instr, i) => (
              <tr key={i}>
                <td className="p-1 font-mono text-slate-600 dark:text-slate-400 whitespace-nowrap">
                  {instr.op} {instr.dest}, {instr.src1}{instr.src2 ? ', ' + instr.src2 : ''}
                </td>
                {Array.from({ length: maxCycle + 1 }, (_, c) => {
                  const isIssue = c === instr.issueCycle
                  const isExec = c >= instr.startExec && c <= instr.endExec
                  const isWB = c === instr.writeback
                  let bg = ''
                  if (isIssue) bg = 'bg-blue-300 dark:bg-blue-700'
                  else if (isExec) bg = 'bg-orange-300 dark:bg-orange-700'
                  else if (isWB) bg = 'bg-green-300 dark:bg-green-700'
                  return (
                    <td key={c} className={`w-6 h-5 text-center rounded-sm ${bg} ${c === cycle && bg ? 'ring-1 ring-yellow-400' : ''}`}>
                      {isIssue && <span className="font-bold text-[9px]">I</span>}
                      {isExec && !isIssue && <span className="font-bold text-[9px]">E</span>}
                      {isWB && !isExec && <span className="font-bold text-[9px]">W</span>}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Legend + insight */}
      <div className="flex flex-wrap gap-4 text-xs">
        <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-blue-300 dark:bg-blue-700" /> 发射</span>
        <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-orange-300 dark:bg-orange-700" /> 执行</span>
        <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-green-300 dark:bg-green-700" /> 写回</span>
      </div>

      <div className="mt-3 p-3 bg-amber-50 dark:bg-amber-950/20 rounded-lg border border-amber-200 dark:border-amber-800 text-xs text-amber-700 dark:text-amber-300">
        <p className="font-semibold">乱序关键: ADD(I4) 在 MULT(I2) 之前完成执行</p>
        <p>ADD 依赖 LD(I3)，MULT 依赖 LD(I1)。由于 LD(I3)比LD(I1)晚发射但延迟相同，ADD可以先于MULT完成。</p>
      </div>
    </div>
  )
}
