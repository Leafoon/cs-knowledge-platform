'use client'

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Cpu, RotateCcw } from 'lucide-react'

interface RegisterState {
  irr: number[]
  imr: number[]
  isr: number[]
}

export function InterruptControllerViz() {
  const [pending, setPending] = useState<number[]>([])
  const [mask, setMask] = useState<number[]>([7])
  const [inService, setInService] = useState<number[]>([])
  const [step, setStep] = useState(0)

  const irqs = [0, 1, 2, 3, 4, 5, 6, 7]

  const togglePending = (irq: number) => {
    setPending(prev => prev.includes(irq) ? prev.filter(x => x !== irq) : [...prev, irq].sort())
  }

  const toggleMask = (irq: number) => {
    setMask(prev => prev.includes(irq) ? prev.filter(x => x !== irq) : [...prev, irq].sort())
  }

  const handleInterrupt = () => {
    const available = pending.filter(p => !mask.includes(p) && !inService.includes(p))
    if (available.length > 0) {
      const highest = available[0]
      setPending(prev => prev.filter(x => x !== highest))
      setInService(prev => [...prev, highest])
      setStep(s => s + 1)
    }
  }

  const clearISR = (irq: number) => {
    setInService(prev => prev.filter(x => x !== irq))
    setStep(s => s + 1)
  }

  const reset = () => {
    setPending([])
    setMask([7])
    setInService([])
    setStep(0)
  }

  const isPending = (irq: number) => pending.includes(irq)
  const isMasked = (irq: number) => mask.includes(irq)
  const isInService = (irq: number) => inService.includes(irq)
  const highestUnmasked = pending.filter(p => !mask.includes(p) && !inService.includes(p))[0]

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-lg font-bold mb-4">中断控制器 8259A (Interrupt Controller)</h3>

      <div className="flex gap-2 mb-4">
        <button onClick={handleInterrupt}
          className="px-3 py-1.5 text-sm rounded bg-red-500 text-white hover:bg-red-600 transition-colors">
          响应中断
        </button>
        <button onClick={() => inService.forEach(i => clearISR(i))}
          className="px-3 py-1.5 text-sm rounded bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300">
          EOI (清除ISR)
        </button>
        <button onClick={reset}
          className="flex items-center gap-1 px-3 py-1.5 text-sm rounded bg-slate-200 dark:bg-slate-700">
          <RotateCcw className="w-3 h-3" /> 重置
        </button>
      </div>

      {/* Register display */}
      <div className="grid grid-cols-3 gap-4 mb-4">
        {/* IRR */}
        <div className="p-3 bg-white dark:bg-slate-900 rounded-lg border border-slate-200 dark:border-slate-700">
          <p className="text-xs font-semibold text-slate-500 mb-2">IRR (中断请求寄存器)</p>
          <div className="flex gap-1">
            {irqs.map(irq => (
              <motion.button key={irq} onClick={() => togglePending(irq)}
                className={`w-8 h-8 rounded text-xs font-bold transition-colors ${
                  isPending(irq) ? 'bg-red-400 text-white' : 'bg-slate-100 dark:bg-slate-800 text-slate-400'
                }`}
                whileTap={{ scale: 0.9 }}
              >
                {irq}
              </motion.button>
            ))}
          </div>
          <p className="text-[10px] text-slate-400 mt-1">点击设置中断请求</p>
        </div>

        {/* IMR */}
        <div className="p-3 bg-white dark:bg-slate-900 rounded-lg border border-slate-200 dark:border-slate-700">
          <p className="text-xs font-semibold text-slate-500 mb-2">IMR (中断屏蔽寄存器)</p>
          <div className="flex gap-1">
            {irqs.map(irq => (
              <motion.button key={irq} onClick={() => toggleMask(irq)}
                className={`w-8 h-8 rounded text-xs font-bold transition-colors ${
                  isMasked(irq) ? 'bg-amber-400 text-white' : 'bg-slate-100 dark:bg-slate-800 text-slate-400'
                }`}
                whileTap={{ scale: 0.9 }}
              >
                {irq}
              </motion.button>
            ))}
          </div>
          <p className="text-[10px] text-slate-400 mt-1">点击切换屏蔽</p>
        </div>

        {/* ISR */}
        <div className="p-3 bg-white dark:bg-slate-900 rounded-lg border border-slate-200 dark:border-slate-700">
          <p className="text-xs font-semibold text-slate-500 mb-2">ISR (服务中寄存器)</p>
          <div className="flex gap-1">
            {irqs.map(irq => (
              <motion.div key={irq}
                className={`w-8 h-8 rounded text-xs font-bold flex items-center justify-center ${
                  isInService(irq) ? 'bg-green-400 text-white' : 'bg-slate-100 dark:bg-slate-800 text-slate-400'
                }`}
              >
                {irq}
              </motion.div>
            ))}
          </div>
          <p className="text-[10px] text-slate-400 mt-1">当前服务中</p>
        </div>
      </div>

      {/* Priority & current action */}
      <div className="p-3 bg-slate-50 dark:bg-slate-800/50 rounded-lg text-xs">
        <p className="font-semibold text-slate-600 dark:text-slate-300 mb-1">优先级: IRQ0 (最高) → IRQ7 (最低)</p>
        <p className="text-slate-500">
          {highestUnmasked !== undefined
            ? `下一个待响应: IRQ${highestUnmasked}`
            : pending.length > 0 ? '所有请求被屏蔽或正在服务' : '无待处理中断'}
        </p>
      </div>
    </div>
  )
}
