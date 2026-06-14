'use client'

import { useState, useMemo } from 'react'
import { motion } from 'framer-motion'
import { RefreshCw, Play, RotateCcw } from 'lucide-react'

type RefreshMode = 'burst' | 'distributed' | 'async'

export function DRAMRefreshDemo() {
  const [mode, setMode] = useState<RefreshMode>('distributed')
  const [cycle, setCycle] = useState(0)
  const totalRows = 8
  const refreshInterval = 4

  const getCycleState = (row: number, cycle: number): 'refresh' | 'access' | 'idle' => {
    if (mode === 'burst') {
      const burstStart = Math.floor(cycle / (refreshInterval * totalRows)) * (refreshInterval * totalRows)
      const burstCycle = cycle - burstStart
      if (burstCycle < totalRows && row === burstCycle) return 'refresh'
    } else if (mode === 'distributed') {
      if (cycle % refreshInterval === 0) {
        const refreshRow = (Math.floor(cycle / refreshInterval)) % totalRows
        if (row === refreshRow) return 'refresh'
      }
    }
    return cycle % 3 === row % 3 ? 'access' : 'idle'
  }

  const modeNames: Record<RefreshMode, string> = {
    burst: '集中刷新',
    distributed: '分散刷新',
    async: '异步刷新'
  }

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-lg font-bold mb-4">DRAM 刷新演示 (DRAM Refresh)</h3>

      <div className="flex items-center gap-3 mb-4">
        {(['burst', 'distributed', 'async'] as RefreshMode[]).map(m => (
          <button key={m} onClick={() => { setMode(m); setCycle(0) }}
            className={`px-3 py-1.5 text-sm rounded ${mode === m ? 'bg-cyan-600 text-white' : 'bg-slate-200 dark:bg-slate-700'}`}>
            {modeNames[m]}
          </button>
        ))}
        <button onClick={() => setCycle(c => c + 1)}
          className="px-3 py-1.5 text-sm rounded bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300">单步</button>
        <button onClick={() => setCycle(0)}
          className="flex items-center gap-1 px-3 py-1.5 text-sm rounded bg-slate-200 dark:bg-slate-700">
          <RotateCcw className="w-3 h-3" />
        </button>
        <span className="ml-auto text-sm">周期: {cycle}</span>
      </div>

      {/* Memory grid */}
      <div className="p-3 bg-white dark:bg-slate-900 rounded-lg border border-slate-200 dark:border-slate-700 mb-4">
        <div className="grid gap-1" style={{ gridTemplateColumns: `repeat(${totalRows}, 1fr)` }}>
          {Array.from({ length: totalRows }, (_, row) => (
            <div key={row} className="text-center">
              <p className="text-[10px] text-slate-400 mb-1">R{row}</p>
              {Array.from({ length: 16 }, (_, c) => {
                const state = c < cycle % 16 ? 'idle' : (getCycleState(row, cycle) === 'refresh' && c === cycle % 16 ? 'refresh' : 'idle')
                const isRefresh = getCycleState(row, cycle) === 'refresh' && c === cycle % 16
                const isAccess = getCycleState(row, cycle) === 'access' && c === cycle % 16
                return (
                  <motion.div
                    key={c}
                    className={`h-3 rounded-sm mb-0.5 ${
                      isRefresh ? 'bg-amber-400' :
                      isAccess ? 'bg-blue-400' :
                      'bg-slate-100 dark:bg-slate-800'
                    }`}
                    animate={{ opacity: isRefresh || isAccess ? 1 : 0.3 }}
                  />
                )
              })}
            </div>
          ))}
        </div>
        <div className="flex justify-center gap-4 mt-2 text-xs text-slate-500">
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-amber-400" /> 刷新</span>
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-blue-400" /> 正常访问</span>
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-slate-200 dark:bg-slate-700" /> 空闲</span>
        </div>
      </div>

      {/* Mode explanation */}
      <div className="grid grid-cols-3 gap-3">
        {(['burst', 'distributed', 'async'] as RefreshMode[]).map(m => (
          <div key={m} className={`p-3 rounded-lg border text-xs ${
            mode === m ? 'border-cyan-400 bg-cyan-50 dark:bg-cyan-950/20' : 'border-slate-200 dark:border-slate-700'
          }`}>
            <p className="font-semibold mb-1">{modeNames[m]}</p>
            {m === 'burst' && <p className="text-slate-600 dark:text-slate-400">集中在一个时间段刷新所有行，期间CPU不能访问</p>}
            {m === 'distributed' && <p className="text-slate-600 dark:text-slate-400">每个刷新周期刷新一行，分散到各时间段</p>}
            {m === 'async' && <p className="text-slate-600 dark:text-slate-400">不固定周期，由控制器在空闲时异步刷新</p>}
          </div>
        ))}
      </div>
    </div>
  )
}
