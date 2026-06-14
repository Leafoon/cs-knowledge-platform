'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { Scale, RotateCcw } from 'lucide-react'

interface ArbiterInput {
  id: number
  name: string
  priority: number
  active: boolean
  granted: boolean
}

export function PriorityArbiter() {
  const [inputs, setInputs] = useState<ArbiterInput[]>([
    { id: 0, name: 'IRQ0 时钟', priority: 0, active: false, granted: false },
    { id: 1, name: 'IRQ1 键盘', priority: 1, active: false, granted: false },
    { id: 2, name: 'IRQ2 级联', priority: 2, active: false, granted: false },
    { id: 3, name: 'IRQ3 串口', priority: 3, active: false, granted: false },
    { id: 4, name: 'IRQ4 串口', priority: 4, active: false, granted: false },
    { id: 5, name: 'IRQ5 并口', priority: 5, active: false, granted: false },
    { id: 6, name: 'IRQ6 软盘', priority: 6, active: false, granted: false },
    { id: 7, name: 'IRQ7 打印机', priority: 7, active: false, granted: false },
  ])

  const toggleActive = (id: number) => {
    setInputs(prev => prev.map(inp => inp.id === id ? { ...inp, active: !inp.active, granted: false } : inp))
  }

  const arbitrate = () => {
    const activeInputs = inputs.filter(i => i.active).sort((a, b) => a.priority - b.priority)
    setInputs(prev => prev.map(inp => ({
      ...inp,
      granted: activeInputs.length > 0 && inp.id === activeInputs[0].id
    })))
  }

  const reset = () => {
    setInputs(prev => prev.map(inp => ({ ...inp, active: false, granted: false })))
  }

  const winner = inputs.find(i => i.granted)

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-lg font-bold mb-4">优先级仲裁器 (Priority Arbiter)</h3>

      <div className="flex gap-2 mb-4">
        <button onClick={arbitrate}
          className="flex items-center gap-1 px-3 py-1.5 text-sm rounded bg-purple-500 text-white hover:bg-purple-600 transition-colors">
          <Scale className="w-4 h-4" /> 仲裁
        </button>
        <button onClick={reset}
          className="flex items-center gap-1 px-3 py-1.5 text-sm rounded bg-slate-200 dark:bg-slate-700">
          <RotateCcw className="w-3 h-3" /> 重置
        </button>
      </div>

      {/* Priority visualization */}
      <div className="space-y-2 mb-4">
        {inputs.map((inp, i) => (
          <motion.div
            key={inp.id}
            className={`flex items-center gap-3 p-2 rounded-lg border transition-colors cursor-pointer ${
              inp.granted ? 'border-green-400 bg-green-50 dark:bg-green-950/30' :
              inp.active ? 'border-red-300 bg-red-50 dark:bg-red-950/20' :
              'border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900'
            }`}
            onClick={() => toggleActive(inp.id)}
            whileHover={{ scale: 1.01 }}
            whileTap={{ scale: 0.99 }}
          >
            <span className="w-8 text-center text-xs font-bold text-slate-400">P{inp.priority}</span>
            <div className={`w-4 h-4 rounded-full ${inp.active ? 'bg-red-500' : 'bg-slate-300 dark:bg-slate-600'}`} />
            <span className="text-sm font-medium flex-1">{inp.name}</span>
            {inp.granted && (
              <motion.span
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                className="px-2 py-0.5 rounded bg-green-500 text-white text-xs font-bold"
              >
                已授权
              </motion.span>
            )}
            {inp.active && !inp.granted && (
              <span className="px-2 py-0.5 rounded bg-red-100 dark:bg-red-900 text-red-600 dark:text-red-300 text-xs">
                请求中
              </span>
            )}
          </motion.div>
        ))}
      </div>

      {/* Circuit diagram */}
      <div className="p-3 bg-slate-50 dark:bg-slate-800/50 rounded-lg text-xs text-slate-600 dark:text-slate-400">
        <p className="font-semibold mb-1">硬件判优逻辑:</p>
        <p>链式优先级编码器: 高优先级输入通过 grant 信号屏蔽低优先级</p>
        <p className="font-mono mt-1">Grant[i] = Request[i] AND NOT(Grant[i-1] OR Grant[i-2] OR ... OR Grant[0])</p>
        {winner && (
          <p className="mt-2 text-green-600 dark:text-green-400 font-semibold">
            当前授权: {winner.name} (最高优先级活跃请求)
          </p>
        )}
      </div>
    </div>
  )
}
