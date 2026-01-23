'use client'

import React, { useState } from 'react'
import { motion } from 'framer-motion'

export default function FloatFormatBitLayout() {
  const [value, setValue] = useState(3.14)

  const toBinary = (num: number, format: 'fp16' | 'bf16') => {
    const buffer = new ArrayBuffer(4)
    const view = new DataView(buffer)
    view.setFloat32(0, num)
    const bits = view.getUint32(0).toString(2).padStart(32, '0')
    
    if (format === 'fp16') {
      return {
        sign: bits[0],
        exp: bits.slice(1, 6),
        mantissa: bits.slice(6, 16)
      }
    } else {
      return {
        sign: bits[0],
        exp: bits.slice(1, 9),
        mantissa: bits.slice(9, 16)
      }
    }
  }

  const fp16 = toBinary(value, 'fp16')
  const bf16 = toBinary(value, 'bf16')

  return (
    <div className="my-8 p-6 bg-gradient-to-br from-purple-50 to-pink-50 dark:from-slate-900 dark:to-purple-950 rounded-xl border border-slate-200 dark:border-slate-700">
      <div className="mb-6">
        <h3 className="text-xl font-bold text-slate-900 dark:text-white">
          FP16 vs BF16 位布局
        </h3>
        <input
          type="number"
          value={value}
          onChange={(e) => setValue(parseFloat(e.target.value) || 0)}
          className="mt-3 px-3 py-2 border rounded"
          step="0.01"
        />
      </div>

      <div className="space-y-4">
        <div className="p-4 bg-white dark:bg-slate-800 rounded-lg border">
          <div className="text-sm font-bold mb-2">FP16 (Half Precision)</div>
          <div className="flex gap-1">
            <div className="px-2 py-1 bg-red-500 text-white text-xs font-mono">{fp16.sign}</div>
            <div className="px-2 py-1 bg-green-500 text-white text-xs font-mono">{fp16.exp}</div>
            <div className="px-2 py-1 bg-blue-500 text-white text-xs font-mono">{fp16.mantissa}</div>
          </div>
          <div className="text-xs text-slate-500 mt-2">指数5位 → 范围小，易溢出</div>
        </div>

        <div className="p-4 bg-white dark:bg-slate-800 rounded-lg border">
          <div className="text-sm font-bold mb-2">BF16 (Brain Float)</div>
          <div className="flex gap-1">
            <div className="px-2 py-1 bg-red-500 text-white text-xs font-mono">{bf16.sign}</div>
            <div className="px-2 py-1 bg-green-500 text-white text-xs font-mono">{bf16.exp}</div>
            <div className="px-2 py-1 bg-blue-500 text-white text-xs font-mono">{bf16.mantissa}</div>
          </div>
          <div className="text-xs text-slate-500 mt-2">指数8位 → 范围大，不易溢出</div>
        </div>
      </div>
    </div>
  )
}
