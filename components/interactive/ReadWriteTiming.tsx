'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { Clock, ArrowRight } from 'lucide-react'

type Operation = 'read' | 'write'

const readPhases = [
  { name: '送地址', duration: 1, desc: 'CPU将地址送到地址总线' },
  { name: '译码', duration: 1, desc: '地址译码器选中存储单元' },
  { name: '读数据', duration: 2, desc: '存储单元数据送到数据总线' },
  { name: '数据稳定', duration: 1, desc: 'CPU在T3下降沿采样数据' },
]

const writePhases = [
  { name: '送地址', duration: 1, desc: 'CPU将地址送到地址总线' },
  { name: '译码', duration: 1, desc: '地址译码器选中存储单元' },
  { name: '送数据', duration: 1, desc: 'CPU将数据送到数据总线' },
  { name: '写入', duration: 2, desc: 'WE有效，数据写入存储单元' },
]

export function ReadWriteTiming() {
  const [operation, setOperation] = useState<Operation>('read')
  const [cycle, setCycle] = useState(0)

  const phases = operation === 'read' ? readPhases : writePhases
  const totalDuration = phases.reduce((s, p) => s + p.duration, 0)
  const maxCycle = totalDuration + 2

  const getPhaseAtCycle = (c: number) => {
    let acc = 0
    for (const p of phases) {
      if (c < acc + p.duration) return p
      acc += p.duration
    }
    return null
  }

  const currentPhase = getPhaseAtCycle(cycle)

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-lg font-bold mb-4">读写时序图 (Read/Write Timing)</h3>

      <div className="flex items-center gap-3 mb-4">
        {(['read', 'write'] as Operation[]).map(op => (
          <button key={op} onClick={() => { setOperation(op); setCycle(0) }}
            className={`px-3 py-1.5 text-sm rounded ${operation === op ? 'bg-blue-600 text-white' : 'bg-slate-200 dark:bg-slate-700'}`}>
            {op === 'read' ? '读时序' : '写时序'}
          </button>
        ))}
        <button onClick={() => setCycle(c => Math.min(c + 1, maxCycle))}
          className="px-3 py-1.5 text-sm rounded bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300">单步</button>
        <button onClick={() => setCycle(0)}
          className="px-3 py-1.5 text-sm rounded bg-slate-200 dark:bg-slate-700">重置</button>
        <span className="ml-auto text-sm">T = {cycle}</span>
      </div>

      {/* Timing diagram */}
      <div className="p-4 bg-white dark:bg-slate-900 rounded-lg border border-slate-200 dark:border-slate-700 mb-4">
        {/* Clock */}
        <div className="mb-3">
          <p className="text-[10px] text-slate-400 mb-1">CLK</p>
          <svg viewBox="0 0 400 30" className="w-full h-8">
            {Array.from({ length: maxCycle + 1 }, (_, i) => (
              <g key={i}>
                <line x1={i * 40} y1={i % 2 === 0 ? 5 : 25} x2={i * 40 + 20} y2={i % 2 === 0 ? 5 : 25} stroke="#6366f1" strokeWidth="2" />
                <line x1={i * 40 + 20} y1={i % 2 === 0 ? 25 : 5} x2={(i + 1) * 40} y2={i % 2 === 0 ? 25 : 5} stroke="#6366f1" strokeWidth="2" />
                <text x={i * 40 + 20} y={30} textAnchor="middle" className="text-[8px] fill-slate-400">T{i}</text>
              </g>
            ))}
          </svg>
        </div>

        {/* Address */}
        <div className="mb-2">
          <p className="text-[10px] text-slate-400 mb-1">地址总线 A</p>
          <div className="flex h-6">
            {Array.from({ length: maxCycle }, (_, c) => {
              const isActive = c < 2
              return (
                <div key={c} className={`flex-1 border-r border-slate-100 dark:border-slate-800 flex items-center justify-center ${
                  isActive ? 'bg-blue-100 dark:bg-blue-900' : ''
                } ${c === cycle ? 'ring-1 ring-yellow-400' : ''}`}>
                  {isActive && <span className="text-[9px] font-mono text-blue-600 dark:text-blue-400">ADDR</span>}
                </div>
              )
            })}
          </div>
        </div>

        {/* Data */}
        <div className="mb-2">
          <p className="text-[10px] text-slate-400 mb-1">数据总线 D</p>
          <div className="flex h-6">
            {Array.from({ length: maxCycle }, (_, c) => {
              const isActive = operation === 'read' ? (c >= 2 && c < 4) : (c >= 2 && c < 4)
              return (
                <div key={c} className={`flex-1 border-r border-slate-100 dark:border-slate-800 flex items-center justify-center ${
                  isActive ? 'bg-green-100 dark:bg-green-900' : ''
                } ${c === cycle ? 'ring-1 ring-yellow-400' : ''}`}>
                  {isActive && <span className="text-[9px] font-mono text-green-600 dark:text-green-400">DATA</span>}
                </div>
              )
            })}
          </div>
        </div>

        {/* Control */}
        <div>
          <p className="text-[10px] text-slate-400 mb-1">{operation === 'read' ? 'OE (输出使能)' : 'WE (写使能)'}</p>
          <div className="flex h-6">
            {Array.from({ length: maxCycle }, (_, c) => {
              const isActive = operation === 'read' ? (c >= 2 && c < 4) : (c >= 3 && c < 5)
              return (
                <div key={c} className={`flex-1 border-r border-slate-100 dark:border-slate-800 flex items-center justify-center ${
                  isActive ? 'bg-red-100 dark:bg-red-900' : 'bg-slate-50 dark:bg-slate-800'
                } ${c === cycle ? 'ring-1 ring-yellow-400' : ''}`}>
                  <span className="text-[9px] font-mono">{isActive ? '低' : '高'}</span>
                </div>
              )
            })}
          </div>
        </div>
      </div>

      {/* Current phase */}
      {currentPhase && (
        <motion.div key={cycle}
          initial={{ opacity: 0, x: -10 }}
          animate={{ opacity: 1, x: 0 }}
          className="p-3 bg-yellow-50 dark:bg-yellow-950/20 rounded-lg border border-yellow-200 dark:border-yellow-800"
        >
          <div className="flex items-center gap-2">
            <Clock className="w-4 h-4 text-yellow-600" />
            <span className="font-medium text-sm text-yellow-700 dark:text-yellow-300">{currentPhase.name}</span>
            <ArrowRight className="w-3 h-3 text-slate-400" />
            <span className="text-xs text-yellow-600 dark:text-yellow-400">{currentPhase.desc}</span>
          </div>
        </motion.div>
      )}
    </div>
  )
}
