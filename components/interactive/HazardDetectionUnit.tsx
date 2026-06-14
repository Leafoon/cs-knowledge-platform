'use client'

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Shield, AlertTriangle, CheckCircle, XCircle } from 'lucide-react'

interface HazardCase {
  id: string
  type: 'RAW' | 'Load-Use' | 'Control'
  instr1: string
  instr2: string
  detected: boolean
  solution: string
}

const hazardCases: HazardCase[] = [
  { id: '1', type: 'RAW', instr1: 'ADD R1, R2, R3', instr2: 'SUB R4, R1, R5', detected: true, solution: '数据前递 (Forwarding)' },
  { id: '2', type: 'Load-Use', instr1: 'LW R1, 0(R2)', instr2: 'ADD R3, R1, R4', detected: true, solution: '插入1个气泡 (Stall)' },
  { id: '3', type: 'Control', instr1: 'BEQ R1, R2, label', instr2: 'ADD R3, R4, R5', detected: true, solution: '分支预测或延迟槽' },
  { id: '4', type: 'RAW', instr1: 'MUL R1, R2, R3', instr2: 'ADD R4, R1, R5', detected: true, solution: '多周期前递' },
]

export function HazardDetectionUnit() {
  const [selectedCase, setSelectedCase] = useState<string>('1')
  const [showLogic, setShowLogic] = useState(false)

  const current = hazardCases.find(c => c.id === selectedCase)!

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-lg font-bold mb-4">冒险检测单元 (Hazard Detection Unit)</h3>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-4">
        {hazardCases.map(c => (
          <button
            key={c.id}
            onClick={() => setSelectedCase(c.id)}
            className={`px-3 py-2 rounded text-xs font-medium transition-colors ${
              selectedCase === c.id
                ? 'bg-red-100 dark:bg-red-900 text-red-700 dark:text-red-300 border border-red-300 dark:border-red-700'
                : 'bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 border border-slate-200 dark:border-slate-700'
            }`}
          >
            {c.type} 冒险
          </button>
        ))}
      </div>

      <AnimatePresence mode="wait">
        <motion.div
          key={selectedCase}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
          className="p-4 bg-white dark:bg-slate-900 rounded-lg border border-slate-200 dark:border-slate-700 mb-4"
        >
          <div className="flex items-center gap-2 mb-3">
            <AlertTriangle className="w-5 h-5 text-amber-500" />
            <span className="font-semibold text-sm">{current.type} 冒险</span>
            <span className={`ml-auto px-2 py-0.5 rounded text-xs ${current.detected ? 'bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300' : 'bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300'}`}>
              {current.detected ? '检测到冒险' : '无冒险'}
            </span>
          </div>

          <div className="grid grid-cols-2 gap-4 mb-3">
            <div className="p-2 bg-blue-50 dark:bg-blue-950/30 rounded">
              <p className="text-xs text-slate-500 mb-1">生产者指令</p>
              <p className="font-mono text-sm">{current.instr1}</p>
            </div>
            <div className="p-2 bg-orange-50 dark:bg-orange-950/30 rounded">
              <p className="text-xs text-slate-500 mb-1">消费者指令</p>
              <p className="font-mono text-sm">{current.instr2}</p>
            </div>
          </div>

          <div className="p-2 bg-green-50 dark:bg-green-950/30 rounded border border-green-200 dark:border-green-800">
            <div className="flex items-center gap-2">
              <CheckCircle className="w-4 h-4 text-green-600" />
              <span className="text-sm font-medium text-green-700 dark:text-green-300">解决方案: {current.solution}</span>
            </div>
          </div>
        </motion.div>
      </AnimatePresence>

      <button
        onClick={() => setShowLogic(!showLogic)}
        className="flex items-center gap-2 px-3 py-1.5 text-sm rounded bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors mb-3"
      >
        <Shield className="w-4 h-4" />
        {showLogic ? '隐藏检测逻辑' : '显示检测逻辑'}
      </button>

      <AnimatePresence>
        {showLogic && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="overflow-hidden"
          >
            <div className="p-3 bg-slate-50 dark:bg-slate-800/50 rounded-lg text-xs font-mono space-y-1">
              <p className="text-slate-600 dark:text-slate-300 font-sans font-semibold mb-2">冒险检测逻辑电路:</p>
              <p className="text-blue-600 dark:text-blue-400">if (ID/EX.MemRead == 1</p>
              <p className="text-blue-600 dark:text-blue-400 pl-4">and (ID/EX.Rd == IF/ID.Rs1 or ID/EX.Rd == IF/ID.Rs2))</p>
              <p className="text-red-600 dark:text-red-400 pl-4">→ Stall the pipeline</p>
              <p className="text-blue-600 dark:text-blue-400 mt-2">if (EX/MEM.RegWrite and EX/MEM.Rd ≠ 0</p>
              <p className="text-blue-600 dark:text-blue-400 pl-4">and (EX/MEM.Rd == ID/EX.Rs1 or EX/MEM.Rd == ID/EX.Rs2))</p>
              <p className="text-green-600 dark:text-green-400 pl-4">→ Forward from EX/MEM</p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
