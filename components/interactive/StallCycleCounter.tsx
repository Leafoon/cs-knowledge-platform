'use client'

import { useState, useMemo } from 'react'
import { motion } from 'framer-motion'
import { Timer, AlertCircle, TrendingDown } from 'lucide-react'

interface StallConfig {
  numInstrs: number
  stallCycles: number
  pipelineDepth: number
}

export function StallCycleCounter() {
  const [config, setConfig] = useState<StallConfig>({
    numInstrs: 10,
    stallCycles: 1,
    pipelineDepth: 5,
  })

  const metrics = useMemo(() => {
    const { numInstrs: n, stallCycles: s, pipelineDepth: k } = config
    const idealCycles = k + n - 1
    const actualCycles = idealCycles + (n - 1) * s
    const speedup = (n * k) / actualCycles
    const idealSpeedup = (n * k) / idealCycles
    const efficiency = (n / (actualCycles * k)) * 100
    const penalty = ((actualCycles - idealCycles) / idealCycles * 100)
    return { idealCycles, actualCycles, speedup, idealSpeedup, efficiency, penalty }
  }, [config])

  const bubblePositions = useMemo(() => {
    const positions: number[][] = []
    for (let i = 1; i < config.numInstrs; i++) {
      const bubbles: number[] = []
      for (let b = 0; b < config.stallCycles; b++) {
        bubbles.push(config.pipelineDepth + (i - 1) * (1 + config.stallCycles) + b)
      }
      positions.push(bubbles)
    }
    return positions
  }, [config])

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-lg font-bold mb-4">停顿周期计数器 (Stall Cycle Counter)</h3>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-4">
        <div>
          <label className="block text-sm font-medium mb-1">指令数: {config.numInstrs}</label>
          <input type="range" min={4} max={20} value={config.numInstrs}
            onChange={e => setConfig(c => ({ ...c, numInstrs: +e.target.value }))}
            className="w-full accent-red-500" />
        </div>
        <div>
          <label className="block text-sm font-medium mb-1">每条冒险停顿: {config.stallCycles}</label>
          <input type="range" min={0} max={3} value={config.stallCycles}
            onChange={e => setConfig(c => ({ ...c, stallCycles: +e.target.value }))}
            className="w-full accent-red-500" />
        </div>
        <div>
          <label className="block text-sm font-medium mb-1">流水线深度: {config.pipelineDepth}</label>
          <input type="range" min={3} max={7} value={config.pipelineDepth}
            onChange={e => setConfig(c => ({ ...c, pipelineDepth: +e.target.value }))}
            className="w-full accent-red-500" />
        </div>
      </div>

      {/* Simplified timeline with bubbles */}
      <div className="p-3 bg-white dark:bg-slate-900 rounded-lg border border-slate-200 dark:border-slate-700 mb-4 overflow-x-auto">
        <div className="flex gap-0.5 min-w-max">
          {Array.from({ length: Math.min(metrics.actualCycles, 30) }, (_, c) => {
            const isBubble = bubblePositions.flat().includes(c)
            return (
              <motion.div
                key={c}
                className={`w-8 h-8 flex items-center justify-center text-[10px] font-mono rounded ${
                  isBubble
                    ? 'bg-red-200 dark:bg-red-800 text-red-700 dark:text-red-300'
                    : 'bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300'
                }`}
                initial={{ opacity: 0, scale: 0.5 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: c * 0.02 }}
              >
                {isBubble ? 'B' : `T${c + 1}`}
              </motion.div>
            )
          })}
        </div>
        <div className="flex items-center gap-4 mt-2 text-xs text-slate-500">
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-blue-100 dark:bg-blue-900" /> 正常</span>
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-red-200 dark:bg-red-800" /> 气泡 (Bubble)</span>
        </div>
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        {[
          { label: '理想周期', value: `${metrics.idealCycles}T`, icon: <Timer className="w-4 h-4" />, color: 'text-green-600' },
          { label: '实际周期', value: `${metrics.actualCycles}T`, icon: <AlertCircle className="w-4 h-4" />, color: 'text-red-600' },
          { label: '性能损失', value: `${metrics.penalty.toFixed(1)}%`, icon: <TrendingDown className="w-4 h-4" />, color: 'text-amber-600' },
          { label: '加速比', value: `${metrics.speedup.toFixed(2)}x`, icon: <Timer className="w-4 h-4" />, color: 'text-blue-600' },
        ].map((m, i) => (
          <div key={i} className="p-3 bg-slate-50 dark:bg-slate-800/50 rounded-lg border border-slate-200 dark:border-slate-700">
            <div className="flex items-center gap-1 text-xs text-slate-500 mb-1">{m.icon} {m.label}</div>
            <p className={`text-xl font-bold ${m.color}`}>{m.value}</p>
          </div>
        ))}
      </div>

      <div className="mt-4 p-3 bg-amber-50 dark:bg-amber-950/20 rounded-lg border border-amber-200 dark:border-amber-800 text-xs text-amber-700 dark:text-amber-300">
        <p>每个数据冒险插入 {config.stallCycles} 个气泡，共 {(config.numInstrs - 1) * config.stallCycles} 个气泡周期，效率从 {(config.numInstrs / metrics.idealCycles * 100).toFixed(1)}% 降至 {metrics.efficiency.toFixed(1)}%</p>
      </div>
    </div>
  )
}
