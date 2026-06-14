'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { CircuitBoard } from 'lucide-react'

export function DivisionCircuitViz() {
  const [highlight, setHighlight] = useState<string | null>(null)

  const blocks = [
    { id: 'alu', label: 'ALU', desc: 'A ± D (余数±除数)', x: 50, y: 60, w: 100, h: 60, color: 'blue' },
    { id: 'regA', label: '寄存器 A', desc: '余数/被除数', x: 200, y: 30, w: 90, h: 50, color: 'green' },
    { id: 'regQ', label: '寄存器 Q', desc: '商', x: 200, y: 100, w: 90, h: 50, color: 'orange' },
    { id: 'regD', label: '寄存器 D', desc: '除数', x: 10, y: 140, w: 80, h: 50, color: 'purple' },
    { id: 'ctrl', label: '控制器', desc: '计数/状态', x: 320, y: 60, w: 80, h: 60, color: 'red' },
  ]

  const connections = [
    { from: 'regA', to: 'alu', label: 'A输出' },
    { from: 'regD', to: 'alu', label: 'D输入' },
    { from: 'alu', to: 'regA', label: '结果写回' },
    { from: 'ctrl', to: 'alu', label: '加/减控制' },
  ]

  const colorMap: Record<string, string> = {
    blue: 'border-blue-400 bg-blue-50 dark:bg-blue-950',
    green: 'border-green-400 bg-green-50 dark:bg-green-950',
    orange: 'border-orange-400 bg-orange-50 dark:bg-orange-950',
    purple: 'border-purple-400 bg-purple-50 dark:bg-purple-950',
    red: 'border-red-400 bg-red-50 dark:bg-red-950',
  }

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <div className="flex items-center gap-2 mb-4">
        <CircuitBoard className="w-5 h-5 text-indigo-500" />
        <h3 className="text-lg font-bold">除法器硬件电路</h3>
      </div>
      <p className="text-sm text-slate-500 mb-4">点击各部件查看功能说明</p>

      <div className="bg-slate-50 dark:bg-slate-900 rounded-lg p-4">
        <svg viewBox="0 0 420 200" className="w-full h-48">
          {connections.map((c, i) => {
            const from = blocks.find(b => b.id === c.from)!
            const to = blocks.find(b => b.id === c.to)!
            const x1 = from.x + from.w / 2
            const y1 = from.y + from.h / 2
            const x2 = to.x + to.w / 2
            const y2 = to.y + to.h / 2
            return (
              <motion.line key={i} x1={x1} y1={y1} x2={x2} y2={y2}
                stroke="#94a3b8" strokeWidth="1.5" strokeDasharray="4"
                initial={{ pathLength: 0 }} animate={{ pathLength: 1 }}
                transition={{ delay: 0.5 + i * 0.2 }} />
            )
          })}

          {blocks.map((b) => (
            <motion.g key={b.id} className="cursor-pointer"
              onClick={() => setHighlight(highlight === b.id ? null : b.id)}
              whileHover={{ scale: 1.05 }}>
              <motion.rect x={b.x} y={b.y} width={b.w} height={b.h} rx="6"
                className={`${colorMap[b.color]} ${highlight === b.id ? 'ring-2 ring-yellow-400' : ''}`}
                fill="currentColor" stroke="currentColor" strokeWidth="2"
                initial={{ opacity: 0, scale: 0.8 }} animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.1 }} />
              <text x={b.x + b.w / 2} y={b.y + b.h / 2 - 4} textAnchor="middle"
                className="text-[10px] font-bold" fill="#1e293b">{b.label}</text>
              <text x={b.x + b.w / 2} y={b.y + b.h / 2 + 10} textAnchor="middle"
                className="text-[8px]" fill="#64748b">{b.desc}</text>
            </motion.g>
          ))}

          <text x="150" y="195" textAnchor="middle" className="text-[9px]" fill="#94a3b8">
            AQ 联合移位寄存器: 每次左移，Q的MSB移入A的LSB
          </text>
        </svg>
      </div>

      {highlight && (
        <motion.div className="mt-3 p-3 bg-yellow-50 dark:bg-yellow-950 rounded-lg border border-yellow-300"
          initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}>
          <div className="text-sm font-bold">{blocks.find(b => b.id === highlight)?.label}</div>
          <div className="text-xs text-slate-600 mt-1">
            {highlight === 'alu' && 'ALU执行A-D（试减）或A+D（恢复/加交替）。比较结果符号决定商位。'}
            {highlight === 'regA' && '存储当前余数。与Q联合左移，MSB送入ALU，ALU结果写回。'}
            {highlight === 'regQ' && '存储商。每步从LSB开始填入商位。A≥0则Q₀=1，否则Q₀=0。'}
            {highlight === 'regD' && '存储除数，不变。每次送入ALU与A做加减。'}
            {highlight === 'ctrl' && '控制迭代次数（n步），生成加/减信号，管理移位时序。'}
          </div>
        </motion.div>
      )}

      <div className="mt-4 grid grid-cols-5 gap-1 text-xs text-center">
        {blocks.map(b => (
          <button key={b.id} onClick={() => setHighlight(b.id)}
            className={`p-1 rounded border ${highlight === b.id ? 'border-yellow-400 bg-yellow-50' : 'border-slate-200'}`}>
            {b.label}
          </button>
        ))}
      </div>
    </div>
  )
}
