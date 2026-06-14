'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { ArrowRight, Zap, RotateCcw } from 'lucide-react'

const forwardingPaths = [
  { from: 'EX/MEM', to: 'EX', label: 'EX冒险前递', color: 'text-blue-600', desc: 'ALU结果直接从EX/MEM寄存器前递到ALU输入' },
  { from: 'MEM/WB', to: 'EX', label: 'MEM冒险前递', color: 'text-green-600', desc: 'MEM结果从MEM/WB寄存器前递到ALU输入' },
]

export function ForwardingNetworkViz() {
  const [activePath, setActivePath] = useState<number | null>(null)
  const [showAll, setShowAll] = useState(false)

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-lg font-bold mb-4">前递/旁路网络 (Forwarding Network)</h3>

      <div className="flex gap-3 mb-4">
        <button
          onClick={() => setShowAll(!showAll)}
          className="flex items-center gap-1 px-3 py-1.5 text-sm rounded bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300 hover:bg-blue-200 dark:hover:bg-blue-800 transition-colors"
        >
          <Zap className="w-4 h-4" /> {showAll ? '隐藏全部' : '显示全部路径'}
        </button>
        <button
          onClick={() => setActivePath(null)}
          className="flex items-center gap-1 px-3 py-1.5 text-sm rounded bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors"
        >
          <RotateCcw className="w-4 h-4" /> 重置
        </button>
      </div>

      {/* Pipeline diagram with forwarding paths */}
      <div className="relative p-4 bg-white dark:bg-slate-900 rounded-lg border border-slate-200 dark:border-slate-700 mb-4">
        <div className="flex items-center justify-between mb-6">
          {['IF', 'ID', 'EX', 'MEM', 'WB'].map((stage, i) => (
            <motion.div
              key={stage}
              className={`px-4 py-2 rounded font-bold text-sm ${
                stage === 'EX' ? 'bg-orange-200 dark:bg-orange-800 text-orange-800 dark:text-orange-200'
                : stage === 'MEM' ? 'bg-purple-200 dark:bg-purple-800 text-purple-800 dark:text-purple-200'
                : stage === 'WB' ? 'bg-red-200 dark:bg-red-800 text-red-800 dark:text-red-200'
                : 'bg-blue-200 dark:bg-blue-800 text-blue-800 dark:text-blue-200'
              }`}
              animate={{ scale: stage === 'EX' && activePath !== null ? 1.1 : 1 }}
            >
              {stage}
            </motion.div>
          ))}
        </div>

        {/* Register banks */}
        <div className="flex justify-between px-12 mb-2">
          {['ID/EX', 'EX/MEM', 'MEM/WB'].map((reg, i) => (
            <div key={reg} className="text-center">
              <div className={`px-3 py-1 rounded text-xs font-mono border ${
                (activePath === 0 && reg === 'EX/MEM') || (activePath === 1 && reg === 'MEM/WB')
                  ? 'border-yellow-500 bg-yellow-100 dark:bg-yellow-900 text-yellow-800 dark:text-yellow-200'
                  : 'border-slate-300 dark:border-slate-600 bg-slate-50 dark:bg-slate-800 text-slate-600 dark:text-slate-400'
              }`}>
                {reg}
              </div>
            </div>
          ))}
        </div>

        {/* Forwarding arrows */}
        {(showAll || activePath === 0) && (
          <motion.div
            initial={{ opacity: 0, pathLength: 0 }}
            animate={{ opacity: 1 }}
            className="flex items-center gap-2 mt-3 p-2 rounded bg-blue-50 dark:bg-blue-950/30 border border-blue-200 dark:border-blue-800"
          >
            <ArrowRight className="w-4 h-4 text-blue-600" />
            <span className="text-sm text-blue-700 dark:text-blue-300">
              EX/MEM → EX: ALU结果直接旁路到ALU输入端
            </span>
          </motion.div>
        )}
        {(showAll || activePath === 1) && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex items-center gap-2 mt-2 p-2 rounded bg-green-50 dark:bg-green-950/30 border border-green-200 dark:border-green-800"
          >
            <ArrowRight className="w-4 h-4 text-green-600" />
            <span className="text-sm text-green-700 dark:text-green-300">
              MEM/WB → EX: MEM阶段结果旁路到ALU输入端
            </span>
          </motion.div>
        )}
      </div>

      {/* Path selector */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {forwardingPaths.map((path, i) => (
          <motion.button
            key={i}
            onClick={() => setActivePath(activePath === i ? null : i)}
            className={`p-3 rounded-lg border text-left transition-colors ${
              activePath === i
                ? 'border-yellow-400 bg-yellow-50 dark:bg-yellow-950/30'
                : 'border-slate-200 dark:border-slate-700 hover:border-slate-300 dark:hover:border-slate-600'
            }`}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            <p className={`font-semibold text-sm ${path.color}`}>{path.label}</p>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">{path.from} → {path.to}</p>
            <p className="text-xs text-slate-600 dark:text-slate-300 mt-1">{path.desc}</p>
          </motion.button>
        ))}
      </div>

      <div className="mt-4 p-3 bg-slate-50 dark:bg-slate-800/50 rounded-lg text-xs text-slate-600 dark:text-slate-400">
        <p className="font-semibold mb-1">前递条件:</p>
        <p>EX/MEM前递: if (EX/MEM.RegWrite and EX/MEM.Rd ≠ 0 and EX/MEM.Rd = ID/EX.Rs1/Rs2)</p>
        <p>MEM/WB前递: if (MEM/WB.RegWrite and MEM/WB.Rd ≠ 0 and MEM/WB.Rd = ID/EX.Rs1/Rs2)</p>
      </div>
    </div>
  )
}
