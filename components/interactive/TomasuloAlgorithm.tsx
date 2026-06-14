'use client'

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Cpu, Play, RotateCcw, ArrowRight } from 'lucide-react'

interface Instruction {
  op: string
  dest: string
  src1: string
  src2: string
  issueCycle: number
  execCycle: number
  writeCycle: number
}

const program: Instruction[] = [
  { op: 'LD', dest: 'F6', src1: '34(R2)', src2: '', issueCycle: 1, execCycle: 2, writeCycle: 3 },
  { op: 'LD', dest: 'F2', src1: '45(R3)', src2: '', issueCycle: 2, execCycle: 3, writeCycle: 4 },
  { op: 'MULTD', dest: 'F0', src1: 'F2', src2: 'F4', issueCycle: 3, execCycle: 5, writeCycle: 9 },
  { op: 'SUBD', dest: 'F8', src1: 'F6', src2: 'F2', issueCycle: 4, execCycle: 5, writeCycle: 6 },
  { op: 'DIVD', dest: 'F10', src1: 'F0', src2: 'F6', issueCycle: 5, execCycle: 10, writeCycle: 16 },
  { op: 'ADDD', dest: 'F6', src1: 'F8', src2: 'F2', issueCycle: 6, execCycle: 7, writeCycle: 8 },
]

export function TomasuloAlgorithm() {
  const [cycle, setCycle] = useState(0)
  const maxCycle = 20

  const step = () => setCycle(c => Math.min(c + 1, maxCycle))
  const reset = () => setCycle(0)

  const getRsStatus = (instr: Instruction) => {
    if (cycle < instr.issueCycle) return '空闲'
    if (cycle < instr.execCycle) return '已发射'
    if (cycle < instr.writeCycle) return '执行中'
    return '已写回'
  }

  const statusColor = (s: string) => {
    if (s === '空闲') return 'bg-slate-100 dark:bg-slate-800 text-slate-400'
    if (s === '已发射') return 'bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300'
    if (s === '执行中') return 'bg-orange-100 dark:bg-orange-900 text-orange-700 dark:text-orange-300'
    return 'bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-300'
  }

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-lg font-bold mb-4">Tomasulo 算法演示</h3>

      <div className="flex items-center gap-3 mb-4">
        <span className="text-sm font-medium">周期: {cycle}</span>
        <button onClick={step} className="px-3 py-1.5 text-sm rounded bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300">单步</button>
        <button onClick={reset} className="flex items-center gap-1 px-3 py-1.5 text-sm rounded bg-slate-200 dark:bg-slate-700">
          <RotateCcw className="w-3 h-3" /> 重置
        </button>
        <div className="ml-auto flex gap-3 text-xs">
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-slate-100 dark:bg-slate-800 border" /> 空闲</span>
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-blue-100 dark:bg-blue-900" /> 已发射</span>
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-orange-100 dark:bg-orange-900" /> 执行中</span>
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-green-100 dark:bg-green-900" /> 已写回</span>
        </div>
      </div>

      {/* Program + Status */}
      <div className="space-y-2 mb-4">
        {program.map((instr, i) => {
          const status = getRsStatus(instr)
          return (
            <motion.div key={i}
              className={`p-3 rounded-lg border border-slate-200 dark:border-slate-700 ${statusColor(status)}`}
              animate={{ scale: cycle === instr.issueCycle || cycle === instr.execCycle || cycle === instr.writeCycle ? 1.02 : 1 }}
            >
              <div className="flex items-center gap-3 text-sm">
                <span className="w-6 text-center font-mono text-xs text-slate-400">I{i + 1}</span>
                <span className="font-mono font-medium">{instr.op} {instr.dest}, {instr.src1}{instr.src2 ? ', ' + instr.src2 : ''}</span>
                <span className="ml-auto text-xs px-2 py-0.5 rounded bg-white/50 dark:bg-black/20">{status}</span>
              </div>
              <div className="flex gap-4 mt-1 text-xs text-slate-500 pl-9">
                <span>发射: T{instr.issueCycle}</span>
                <span>执行: T{instr.execCycle}{instr.op === 'MULTD' || instr.op === 'DIVD' ? `-${instr.writeCycle - 1}` : ''}</span>
                <span>写回: T{instr.writeCycle}</span>
              </div>
            </motion.div>
          )
        })}
      </div>

      {/* CDB */}
      <div className="p-3 bg-yellow-50 dark:bg-yellow-950/20 rounded-lg border border-yellow-200 dark:border-yellow-800">
        <div className="flex items-center gap-2 text-sm">
          <ArrowRight className="w-4 h-4 text-yellow-600" />
          <span className="font-medium text-yellow-700 dark:text-yellow-300">公共数据总线 (CDB)</span>
          <span className="text-xs text-yellow-600 dark:text-yellow-400 ml-2">
            {cycle > 0 ? `当前广播: ${program.find(p => p.writeCycle === cycle)?.dest || '无'}` : '等待开始'}
          </span>
        </div>
      </div>
    </div>
  )
}
