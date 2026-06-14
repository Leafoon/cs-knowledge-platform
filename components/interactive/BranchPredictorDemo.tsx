'use client'

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { GitBranch, Play, RotateCcw } from 'lucide-react'

type PredictorBit = 0 | 1
type State2Bit = 'SN' | 'WN' | 'WT' | 'ST'

const stateLabels: Record<State2Bit, string> = {
  SN: '强不跳转', WN: '弱不跳转', WT: '弱跳转', ST: '强跳转'
}

const stateColors: Record<State2Bit, string> = {
  SN: 'bg-red-200 dark:bg-red-800 text-red-800 dark:text-red-200',
  WN: 'bg-orange-200 dark:bg-orange-800 text-orange-800 dark:text-orange-200',
  WT: 'bg-blue-200 dark:bg-blue-800 text-blue-800 dark:text-blue-200',
  ST: 'bg-green-200 dark:bg-green-800 text-green-800 dark:text-green-200',
}

export function BranchPredictorDemo() {
  const [mode, setMode] = useState<'1bit' | '2bit'>('2bit')
  const [state1bit, setState1bit] = useState<PredictorBit>(0)
  const [state2bit, setState2bit] = useState<State2Bit>('WN')
  const [history, setHistory] = useState<string[]>([])

  const step = (taken: boolean) => {
    if (mode === '1bit') {
      const prediction = state1bit === 1
      const correct = prediction === taken
      setState1bit(taken ? 1 : 0)
      setHistory(prev => [...prev, `预测:${prediction?'跳转':'不跳'} 实际:${taken?'跳转':'不跳'} ${correct?'✓':'✗'}`])
    } else {
      const isJump = state2bit === 'WT' || state2bit === 'ST'
      const prediction = isJump
      const correct = prediction === taken
      const transitions: Record<State2Bit, [State2Bit, State2Bit]> = {
        SN: ['SN', 'WN'],
        WN: ['SN', 'WT'],
        WT: ['WN', 'ST'],
        ST: ['WT', 'ST'],
      }
      setState2bit(taken ? transitions[state2bit][1] : transitions[state2bit][0])
      setHistory(prev => [...prev, `预测:${prediction?'跳转':'不跳'} 实际:${taken?'跳转':'不跳'} ${correct?'✓':'✗'}`])
    }
  }

  const reset = () => {
    setState1bit(0)
    setState2bit('WN')
    setHistory([])
  }

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-lg font-bold mb-4">分支预测器 (Branch Predictor)</h3>

      <div className="flex gap-2 mb-4">
        {(['1bit', '2bit'] as const).map(m => (
          <button
            key={m}
            onClick={() => { setMode(m); reset() }}
            className={`px-3 py-1.5 text-sm rounded transition-colors ${
              mode === m ? 'bg-purple-600 text-white' : 'bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-300'
            }`}
          >
            {m === '1bit' ? '1位预测器' : '2位饱和计数器'}
          </button>
        ))}
        <button onClick={reset} className="ml-auto flex items-center gap-1 px-3 py-1.5 text-sm rounded bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-300">
          <RotateCcw className="w-3 h-3" /> 重置
        </button>
      </div>

      {/* State visualization */}
      <div className="p-4 bg-white dark:bg-slate-900 rounded-lg border border-slate-200 dark:border-slate-700 mb-4">
        {mode === '1bit' ? (
          <div className="flex items-center justify-center gap-8">
            <motion.div
              className={`w-20 h-20 rounded-full flex items-center justify-center text-2xl font-bold border-4 ${
                state1bit === 0 ? 'border-red-400 bg-red-100 dark:bg-red-900 text-red-700 dark:text-red-300' : 'border-green-400 bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-300'
              }`}
              animate={{ scale: [1, 1.1, 1] }}
              transition={{ duration: 0.3 }}
            >
              {state1bit === 0 ? 'NT' : 'T'}
            </motion.div>
            <div className="text-sm text-slate-600 dark:text-slate-400">
              当前状态: <span className="font-bold">{state1bit === 0 ? '不跳转 (Not Taken)' : '跳转 (Taken)'}</span>
            </div>
          </div>
        ) : (
          <div className="flex items-center justify-center gap-3">
            {(['SN', 'WN', 'WT', 'ST'] as State2Bit[]).map(s => (
              <motion.div
                key={s}
                className={`w-16 h-16 rounded-lg flex flex-col items-center justify-center text-xs font-bold border-2 ${
                  state2bit === s ? 'border-yellow-400 shadow-lg' : 'border-slate-300 dark:border-slate-600'
                } ${stateColors[s]}`}
                animate={{ scale: state2bit === s ? 1.1 : 1 }}
              >
                <span>{s}</span>
                <span className="text-[9px] mt-0.5">{stateLabels[s]}</span>
              </motion.div>
            ))}
          </div>
        )}
      </div>

      {/* Controls */}
      <div className="flex gap-3 mb-4">
        <motion.button
          onClick={() => step(true)}
          className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg bg-green-500 hover:bg-green-600 text-white font-medium text-sm"
          whileTap={{ scale: 0.95 }}
        >
          <GitBranch className="w-4 h-4" /> 跳转 (Taken)
        </motion.button>
        <motion.button
          onClick={() => step(false)}
          className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg bg-red-500 hover:bg-red-600 text-white font-medium text-sm"
          whileTap={{ scale: 0.95 }}
        >
          <GitBranch className="w-4 h-4" /> 不跳转 (Not Taken)
        </motion.button>
      </div>

      {/* History */}
      {history.length > 0 && (
        <div className="p-3 bg-slate-50 dark:bg-slate-800/50 rounded-lg max-h-32 overflow-y-auto">
          <p className="text-xs font-semibold text-slate-500 mb-1">执行历史:</p>
          {history.slice(-10).map((h, i) => (
            <p key={i} className={`text-xs font-mono ${h.includes('✓') ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
              {history.length - 10 + i >= 0 ? history.length - 10 + i + 1 : i + 1}. {h}
            </p>
          ))}
        </div>
      )}
    </div>
  )
}
