'use client'

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Layers, Play, RotateCcw } from 'lucide-react'

interface NestedEvent {
  cycle: number
  level: number
  action: string
  detail: string
}

const scenarios: Record<string, NestedEvent[]> = {
  'basic': [
    { cycle: 1, level: 0, action: '主程序执行', detail: 'CPU正常执行用户程序' },
    { cycle: 2, level: 1, action: 'IRQ3 (串口)', detail: '低优先级中断到来' },
    { cycle: 3, level: 1, action: '保存现场L1', detail: '保存PSW, PC到栈' },
    { cycle: 4, level: 2, action: 'IRQ1 (键盘)', detail: '高优先级中断到来，打断L1' },
    { cycle: 5, level: 2, action: '保存现场L2', detail: '再次保存PSW, PC' },
    { cycle: 6, level: 2, action: '处理IRQ1', detail: '执行键盘中断服务程序' },
    { cycle: 7, level: 1, action: '返回L1继续', detail: '恢复L1现场，继续处理IRQ3' },
    { cycle: 8, level: 0, action: '返回主程序', detail: '恢复主程序现场' },
  ],
  'masking': [
    { cycle: 1, level: 0, action: '主程序执行', detail: 'CPU正常执行' },
    { cycle: 2, level: 1, action: 'IRQ5 (并口)', detail: '中断到来' },
    { cycle: 3, level: 1, action: '设置屏蔽字', detail: '屏蔽 IRQ5及以下' },
    { cycle: 4, level: 1, action: '处理中...', detail: 'IRQ7到来但被屏蔽，不响应' },
    { cycle: 5, level: 1, action: '处理完成', detail: '清除屏蔽字' },
    { cycle: 6, level: 0, action: '返回主程序', detail: '恢复现场' },
  ]
}

export function NestedInterruptDemo() {
  const [scenario, setScenario] = useState<'basic' | 'masking'>('basic')
  const [currentStep, setCurrentStep] = useState(-1)

  const events = scenarios[scenario]

  const step = () => setCurrentStep(s => Math.min(s + 1, events.length - 1))
  const reset = () => setCurrentStep(-1)

  const maxLevel = Math.max(...events.map(e => e.level))

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-lg font-bold mb-4">嵌套中断演示 (Nested Interrupt)</h3>

      <div className="flex items-center gap-3 mb-4">
        <div className="flex gap-1">
          {(['basic', 'masking'] as const).map(s => (
            <button key={s} onClick={() => { setScenario(s); reset() }}
              className={`px-3 py-1.5 text-sm rounded ${scenario === s ? 'bg-indigo-600 text-white' : 'bg-slate-200 dark:bg-slate-700'}`}>
              {s === 'basic' ? '基本嵌套' : '屏蔽字处理'}
            </button>
          ))}
        </div>
        <button onClick={step}
          className="px-3 py-1.5 text-sm rounded bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300">
          下一步
        </button>
        <button onClick={reset}
          className="flex items-center gap-1 px-3 py-1.5 text-sm rounded bg-slate-200 dark:bg-slate-700">
          <RotateCcw className="w-3 h-3" /> 重置
        </button>
      </div>

      {/* Stack visualization */}
      <div className="p-4 bg-white dark:bg-slate-900 rounded-lg border border-slate-200 dark:border-slate-700 mb-4">
        <p className="text-xs font-semibold text-slate-500 mb-3">中断嵌套栈 (从上到下)</p>
        <div className="flex flex-col items-center gap-1">
          {events.filter((_, i) => i <= currentStep).map((evt, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              className={`w-full max-w-md p-2 rounded border text-sm flex items-center gap-3 ${
                i === currentStep ? 'border-yellow-400 bg-yellow-50 dark:bg-yellow-950/30' : 'border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50'
              }`}
              style={{ marginLeft: `${evt.level * 24}px` }}
            >
              <span className="w-6 text-center text-xs font-mono text-slate-400">L{evt.level}</span>
              <span className="font-medium">{evt.action}</span>
              <span className="text-xs text-slate-500 ml-auto">{evt.detail}</span>
            </motion.div>
          ))}
        </div>
        {currentStep === -1 && (
          <p className="text-sm text-slate-400 text-center py-8">点击"下一步"开始演示</p>
        )}
      </div>

      {/* Level indicator */}
      <div className="flex items-center gap-2 mb-2">
        {Array.from({ length: maxLevel + 1 }, (_, l) => (
          <div key={l} className={`flex items-center gap-1 px-2 py-1 rounded text-xs ${
            events[currentStep]?.level === l ? 'bg-yellow-200 dark:bg-yellow-800 text-yellow-800 dark:text-yellow-200' : 'bg-slate-100 dark:bg-slate-800 text-slate-500'
          }`}>
            <Layers className="w-3 h-3" /> L{l}
          </div>
        ))}
      </div>

      <div className="p-3 bg-blue-50 dark:bg-blue-950/20 rounded-lg border border-blue-200 dark:border-blue-800 text-xs text-blue-700 dark:text-blue-300">
        <p className="font-semibold mb-1">关键规则:</p>
        <p>高优先级中断可打断低优先级服务程序。同级或低级中断被屏蔽字阻断。中断返回使用栈式恢复(LIFO)。</p>
      </div>
    </div>
  )
}
