'use client'

import { useState, useMemo } from 'react'
import { motion } from 'framer-motion'
import { Grid3x3, RotateCcw, Play } from 'lucide-react'

export function InterleavedMemory() {
  const [numBanks, setNumBanks] = useState(4)
  const [cycle, setCycle] = useState(0)
  const [isRunning, setIsRunning] = useState(false)

  const accessPattern = useMemo(() => {
    const pattern: { bank: number; addr: number }[] = []
    for (let i = 0; i < 16; i++) {
      pattern.push({ bank: i % numBanks, addr: i })
    }
    return pattern
  }, [numBanks])

  const maxCycle = accessPattern.length + numBanks

  const getBankState = (bankIdx: number, cycle: number) => {
    for (let i = 0; i < accessPattern.length; i++) {
      const accessCycle = i + (accessPattern[i].bank === bankIdx ? 0 : 0)
      if (accessPattern[i].bank === bankIdx) {
        const startCycle = i
        if (cycle === startCycle) return 'access'
        if (cycle > startCycle && cycle < startCycle + numBanks) return 'busy'
      }
    }
    return 'idle'
  }

  const run = () => {
    setIsRunning(true)
    setCycle(0)
    let c = 0
    const timer = setInterval(() => {
      c++
      if (c > maxCycle) { clearInterval(timer); setIsRunning(false); return }
      setCycle(c)
    }, 400)
  }

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-lg font-bold mb-4">交叉存储器 (Interleaved Memory)</h3>

      <div className="flex items-center gap-3 mb-4">
        <div>
          <label className="block text-sm font-medium mb-1">存储体数: {numBanks}</label>
          <input type="range" min={2} max={8} step={2} value={numBanks}
            onChange={e => { setNumBanks(+e.target.value); setCycle(0) }} className="w-full accent-blue-500" />
        </div>
        <button onClick={() => setCycle(c => Math.min(c + 1, maxCycle))} disabled={isRunning}
          className="px-3 py-1.5 text-sm rounded bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300 disabled:opacity-50">单步</button>
        <button onClick={run} disabled={isRunning}
          className="flex items-center gap-1 px-3 py-1.5 text-sm rounded bg-green-500 text-white disabled:opacity-50">
          <Play className="w-3 h-3" /> 流水访问
        </button>
        <button onClick={() => setCycle(0)}
          className="flex items-center gap-1 px-3 py-1.5 text-sm rounded bg-slate-200 dark:bg-slate-700">
          <RotateCcw className="w-3 h-3" />
        </button>
        <span className="ml-auto text-sm">周期: {cycle}</span>
      </div>

      {/* Banks visualization */}
      <div className="grid grid-cols-4 gap-3 mb-4" style={{ gridTemplateColumns: `repeat(${Math.min(numBanks, 4)}, 1fr)` }}>
        {Array.from({ length: numBanks }, (_, bankIdx) => {
          const state = getBankState(bankIdx, cycle)
          return (
            <motion.div key={bankIdx}
              className={`p-3 rounded-lg border text-center ${
                state === 'access' ? 'border-green-400 bg-green-100 dark:bg-green-900' :
                state === 'busy' ? 'border-amber-400 bg-amber-50 dark:bg-amber-950/20' :
                'border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900'
              }`}
              animate={{ scale: state === 'access' ? 1.05 : 1 }}
            >
              <p className="text-xs font-bold mb-1">Bank {bankIdx}</p>
              <p className="text-[10px] text-slate-500">地址 = {bankIdx} mod {numBanks}</p>
              <div className={`mt-1 text-xs px-2 py-0.5 rounded ${
                state === 'access' ? 'bg-green-500 text-white' :
                state === 'busy' ? 'bg-amber-400 text-white' :
                'bg-slate-100 dark:bg-slate-800 text-slate-400'
              }`}>
                {state === 'access' ? '访问中' : state === 'busy' ? '忙碌' : '空闲'}
              </div>
            </motion.div>
          )
        })}
      </div>

      {/* Access timeline */}
      <div className="p-3 bg-white dark:bg-slate-900 rounded-lg border border-slate-200 dark:border-slate-700 mb-4">
        <p className="text-xs text-slate-500 mb-2">访问时序</p>
        <div className="flex gap-0.5 overflow-x-auto">
          {Array.from({ length: Math.min(maxCycle, 20) }, (_, c) => {
            const access = accessPattern.find((a, i) => i === c)
            return (
              <div key={c} className={`w-8 h-8 flex items-center justify-center text-[10px] rounded ${
                c === cycle ? 'bg-yellow-300 dark:bg-yellow-700 text-yellow-800 dark:text-yellow-200 ring-2 ring-yellow-400' :
                c < cycle ? 'bg-slate-100 dark:bg-slate-800 text-slate-400' :
                'bg-white dark:bg-slate-900 text-slate-300'
              }`}>
                {access ? `B${access.bank}` : ''}
              </div>
            )
          })}
        </div>
      </div>

      {/* Performance comparison */}
      <div className="grid grid-cols-2 gap-3">
        <div className="p-3 bg-red-50 dark:bg-red-950/20 rounded-lg border border-red-200 dark:border-red-800">
          <p className="text-xs font-semibold text-red-700 dark:text-red-300 mb-1">顺序存储</p>
          <p className="text-xs text-red-600 dark:text-red-400">连续访问需等上一个完成</p>
          <p className="text-lg font-bold text-red-600">{16 * numBanks} T</p>
        </div>
        <div className="p-3 bg-green-50 dark:bg-green-950/20 rounded-lg border border-green-200 dark:border-green-800">
          <p className="text-xs font-semibold text-green-700 dark:text-green-300 mb-1">低位交叉</p>
          <p className="text-xs text-green-600 dark:text-green-400">各体可并行工作</p>
          <p className="text-lg font-bold text-green-600">{16 + numBanks - 1} T</p>
        </div>
      </div>
    </div>
  )
}
