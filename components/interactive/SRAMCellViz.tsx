'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { Cpu, Zap } from 'lucide-react'

export function SRAMCellViz() {
  const [state, setState] = useState<0 | 1>(1)
  const [mode, setMode] = useState<'idle' | 'read' | 'write'>('idle')
  const [writeValue, setWriteValue] = useState<0 | 1>(0)

  const doRead = () => {
    setMode('read')
    setTimeout(() => setMode('idle'), 1500)
  }

  const doWrite = () => {
    setState(writeValue)
    setMode('write')
    setTimeout(() => setMode('idle'), 1500)
  }

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-lg font-bold mb-4">SRAM 存储单元 (6-Transistor Cell)</h3>

      {/* Circuit visualization */}
      <div className="flex justify-center mb-4">
        <div className="relative p-6 bg-white dark:bg-slate-900 rounded-lg border border-slate-200 dark:border-slate-700 w-full max-w-md">
          <svg viewBox="0 0 300 200" className="w-full">
            {/* Cross-coupled inverters */}
            <rect x="80" y="40" width="60" height="40" rx="4" fill="none" stroke="#6366f1" strokeWidth="2" />
            <text x="110" y="65" textAnchor="middle" className="text-[10px] fill-indigo-600 dark:fill-indigo-400 font-bold">INV1</text>
            <rect x="160" y="40" width="60" height="40" rx="4" fill="none" stroke="#6366f1" strokeWidth="2" />
            <text x="190" y="65" textAnchor="middle" className="text-[10px] fill-indigo-600 dark:fill-indigo-400 font-bold">INV2</text>

            {/* Cross connections */}
            <line x1="140" y1="50" x2="160" y2="50" stroke={state === 1 ? '#22c55e' : '#ef4444'} strokeWidth="2" />
            <line x1="140" y1="70" x2="160" y2="70" stroke={state === 1 ? '#ef4444' : '#22c55e'} strokeWidth="2" />

            {/* State labels */}
            <text x="150" y="45" textAnchor="middle" className="text-[9px] fill-slate-500">Q</text>
            <text x="150" y="80" textAnchor="middle" className="text-[9px] fill-slate-500">Q̄</text>

            {/* Access transistors */}
            <rect x="60" y="110" width="40" height="30" rx="4" fill="none" stroke="#f59e0b" strokeWidth="2" />
            <text x="80" y="130" textAnchor="middle" className="text-[9px] fill-amber-600">M5</text>
            <rect x="200" y="110" width="40" height="30" rx="4" fill="none" stroke="#f59e0b" strokeWidth="2" />
            <text x="220" y="130" textAnchor="middle" className="text-[9px] fill-amber-600">M6</text>

            {/* Bit lines */}
            <line x1="80" y1="125" x2="30" y2="125" stroke="#3b82f6" strokeWidth="2" />
            <text x="20" y="122" textAnchor="end" className="text-[9px] fill-blue-600">BL</text>
            <line x1="220" y1="125" x2="270" y2="125" stroke="#3b82f6" strokeWidth="2" />
            <text x="280" y="122" className="text-[9px] fill-blue-600">BL̄</text>

            {/* Word line */}
            <line x1="80" y1="110" x2="220" y2="110" stroke="#ef4444" strokeWidth="2" />
            <text x="150" y="105" textAnchor="middle" className="text-[9px] fill-red-600">WL (字线)</text>

            {/* VCC */}
            <line x1="150" y1="20" x2="150" y2="40" stroke="#22c55e" strokeWidth="2" />
            <text x="150" y="15" textAnchor="middle" className="text-[9px] fill-green-600">VCC</text>

            {/* GND */}
            <line x1="150" y1="80" x2="150" y2="95" stroke="#64748b" strokeWidth="2" />
            <text x="150" y="195" textAnchor="middle" className="text-[9px] fill-slate-500">GND</text>

            {/* State indicator */}
            <motion.circle cx="250" cy="60" r="12"
              fill={state === 1 ? '#22c55e' : '#ef4444'}
              animate={{ scale: mode !== 'idle' ? [1, 1.3, 1] : 1 }}
              transition={{ duration: 0.5 }}
            />
            <text x="250" y="65" textAnchor="middle" className="text-[10px] fill-white font-bold">{state}</text>
          </svg>
        </div>
      </div>

      {/* Controls */}
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="p-3 bg-blue-50 dark:bg-blue-950/20 rounded-lg border border-blue-200 dark:border-blue-800">
          <p className="text-xs font-semibold text-blue-700 dark:text-blue-300 mb-2">读操作</p>
          <p className="text-xs text-blue-600 dark:text-blue-400 mb-2">激活字线WL，通过BL/BL̄差分读出存储值</p>
          <button onClick={doRead} disabled={mode !== 'idle'}
            className="w-full px-3 py-1.5 text-sm rounded bg-blue-500 text-white disabled:opacity-50">
            {mode === 'read' ? '读取中...' : '读取'}
          </button>
        </div>
        <div className="p-3 bg-green-50 dark:bg-green-950/20 rounded-lg border border-green-200 dark:border-green-800">
          <p className="text-xs font-semibold text-green-700 dark:text-green-300 mb-2">写操作</p>
          <p className="text-xs text-green-600 dark:text-green-400 mb-2">驱动BL=值, BL̄=¬值，激活WL写入</p>
          <div className="flex gap-2 mb-2">
            <button onClick={() => setWriteValue(0)}
              className={`flex-1 px-2 py-1 text-xs rounded ${writeValue === 0 ? 'bg-red-500 text-white' : 'bg-slate-200 dark:bg-slate-700'}`}>写0</button>
            <button onClick={() => setWriteValue(1)}
              className={`flex-1 px-2 py-1 text-xs rounded ${writeValue === 1 ? 'bg-green-500 text-white' : 'bg-slate-200 dark:bg-slate-700'}`}>写1</button>
          </div>
          <button onClick={doWrite} disabled={mode !== 'idle'}
            className="w-full px-3 py-1.5 text-sm rounded bg-green-500 text-white disabled:opacity-50">
            {mode === 'write' ? '写入中...' : '写入'}
          </button>
        </div>
      </div>

      <div className="p-3 bg-slate-50 dark:bg-slate-800/50 rounded-lg text-xs text-slate-600 dark:text-slate-400">
        <p className="font-semibold mb-1">6管SRAM特点:</p>
        <p>双稳态触发器存储，无需刷新，访问速度快(~1ns)，但面积大、功耗高，用于Cache。</p>
      </div>
    </div>
  )
}
