'use client'

import { useState, useMemo } from 'react'
import { motion } from 'framer-motion'
import { ToggleLeft } from 'lucide-react'

export function ChipSelectDecoder() {
  const [inputs, setInputs] = useState([0, 0, 0])
  const [enable, setEnable] = useState(true)

  const toggleInput = (i: number) => {
    setInputs(prev => prev.map((v, idx) => idx === i ? (v === 0 ? 1 : 0) : v))
  }

  const output = useMemo(() => {
    if (!enable) return Array(8).fill(1)
    const idx = inputs[0] * 4 + inputs[1] * 2 + inputs[2]
    return Array.from({ length: 8 }, (_, i) => i === idx ? 0 : 1)
  }, [inputs, enable])

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-lg font-bold mb-4">片选信号译码器 74LS138</h3>

      {/* Inputs */}
      <div className="flex items-center gap-4 mb-4">
        <span className="text-sm font-medium">使能:</span>
        {['G1', 'G2A̅', 'G2B̅'].map((name, i) => (
          <button key={name} onClick={() => i === 0 ? setEnable(!enable) : setEnable(enable)}
            className={`px-2 py-1 text-xs rounded border ${
              i === 0 ? (enable ? 'bg-green-400 text-white border-green-500' : 'bg-slate-200 dark:bg-slate-700 border-slate-300 dark:border-slate-600')
              : (enable ? 'bg-red-100 dark:bg-red-900 border-red-300 dark:border-red-700' : 'bg-slate-200 dark:bg-slate-700 border-slate-300 dark:border-slate-600')
            }`}>
            {name}={i === 0 ? (enable ? 1 : 0) : (enable ? 0 : 1)}
          </button>
        ))}
        <span className="text-xs text-slate-500 ml-2">G1=1, G2A̅=0, G2B̅=0 → 使能</span>
      </div>

      <div className="flex items-center gap-4 mb-4">
        <span className="text-sm font-medium">地址输入:</span>
        {[['C', 0], ['B', 1], ['A', 2]].map(([name, idx]) => (
          <motion.button key={name as string} onClick={() => toggleInput(idx as number)}
            className={`w-12 h-12 rounded-lg text-lg font-bold border-2 ${
              inputs[idx as number] ? 'bg-blue-500 text-white border-blue-600' : 'bg-slate-100 dark:bg-slate-800 border-slate-300 dark:border-slate-600'
            }`}
            whileTap={{ scale: 0.9 }}
          >
            {inputs[idx as number]}
          </motion.button>
        ))}
        <span className="text-xs text-slate-500 ml-2">
          {inputs.join('')} (二进制) = {inputs[0] * 4 + inputs[1] * 2 + inputs[2]} (十进制)
        </span>
      </div>

      {/* Decoder chip visualization */}
      <div className="p-4 bg-white dark:bg-slate-900 rounded-lg border border-slate-200 dark:border-slate-700 mb-4">
        <div className="flex items-center justify-between">
          {/* Inputs left */}
          <div className="space-y-2">
            {['G1', 'C', 'B', 'A', 'G2A̅', 'G2B̅'].map((name, i) => (
              <div key={name} className="flex items-center gap-2">
                <span className="text-xs font-mono w-8 text-right">{name}</span>
                <div className={`w-3 h-3 rounded-full ${
                  i < 4 ? (i === 0 ? (enable ? 'bg-green-400' : 'bg-slate-400') : (inputs[i - 1] ? 'bg-blue-400' : 'bg-slate-300 dark:bg-slate-600'))
                  : (enable ? 'bg-red-300' : 'bg-slate-400')
                }`} />
              </div>
            ))}
          </div>

          {/* Chip body */}
          <div className="flex-1 mx-6 p-4 border-2 border-slate-300 dark:border-slate-600 rounded-lg bg-slate-50 dark:bg-slate-800">
            <p className="text-center text-xs font-bold text-slate-600 dark:text-slate-300 mb-2">74LS138</p>
            <p className="text-center text-[10px] text-slate-400">3-to-8 Decoder</p>
          </div>

          {/* Outputs right */}
          <div className="space-y-1">
            {output.map((val, i) => (
              <motion.div key={i}
                className={`flex items-center gap-2 px-2 py-0.5 rounded text-xs font-mono ${
                  val === 0 ? 'bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-300' : 'text-slate-400'
                }`}
                animate={{ scale: val === 0 ? 1.1 : 1 }}
              >
                <div className={`w-3 h-3 rounded-full ${val === 0 ? 'bg-green-500' : 'bg-slate-300 dark:bg-slate-600'}`} />
                Y{i}̅ = {val}
              </motion.div>
            ))}
          </div>
        </div>
      </div>

      {/* Truth table */}
      <div className="overflow-x-auto">
        <table className="text-xs w-full">
          <thead>
            <tr className="border-b border-slate-200 dark:border-slate-700">
              <th className="p-1 font-medium text-slate-500">C</th>
              <th className="p-1 font-medium text-slate-500">B</th>
              <th className="p-1 font-medium text-slate-500">A</th>
              {Array.from({ length: 8 }, (_, i) => (
                <th key={i} className="p-1 font-medium text-slate-500">Y{i}̅</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {Array.from({ length: 8 }, (_, idx) => {
              const c = (idx >> 2) & 1, b = (idx >> 1) & 1, a = idx & 1
              const isActive = inputs[0] === c && inputs[1] === b && inputs[2] === a
              return (
                <tr key={idx} className={isActive ? 'bg-green-50 dark:bg-green-950/20' : ''}>
                  <td className="p-1 text-center">{c}</td>
                  <td className="p-1 text-center">{b}</td>
                  <td className="p-1 text-center">{a}</td>
                  {Array.from({ length: 8 }, (_, i) => (
                    <td key={i} className={`p-1 text-center font-bold ${i === idx ? 'text-green-600' : 'text-slate-400'}`}>
                      {i === idx ? '0' : '1'}
                    </td>
                  ))}
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
