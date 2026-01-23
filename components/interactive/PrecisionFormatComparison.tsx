'use client'

import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { Zap } from 'lucide-react'

export default function PrecisionFormatComparison() {
  const [selectedFormat, setSelectedFormat] = useState('fp32')

  const formats: Record<string, any> = {
    fp32: { name: 'FP32', bits: 32, exp: 8, mantissa: 23, range: '±3.4×10³⁸', precision: '~7位', color: 'blue' },
    fp16: { name: 'FP16', bits: 16, exp: 5, mantissa: 10, range: '±6.5×10⁴', precision: '~3位', color: 'purple' },
    bf16: { name: 'BF16', bits: 16, exp: 8, mantissa: 7, range: '±3.4×10³⁸', precision: '~2位', color: 'green' },
    tf32: { name: 'TF32', bits: 19, exp: 8, mantissa: 10, range: '±3.4×10³⁸', precision: '~3位', color: 'orange' }
  }

  const selected = formats[selectedFormat]

  return (
    <div className="my-8 p-6 bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-slate-900 dark:to-blue-950 rounded-xl border border-slate-200 dark:border-slate-700">
      <div className="mb-6">
        <h3 className="text-xl font-bold text-slate-900 dark:text-white flex items-center gap-2">
          <Zap className="w-5 h-5 text-blue-500" />
          精度格式对比
        </h3>
      </div>

      <div className="grid grid-cols-4 gap-3 mb-6">
        {Object.keys(formats).map((key) => (
          <button
            key={key}
            onClick={() => setSelectedFormat(key)}
            className={`p-3 rounded-lg border-2 transition-all ${
              selectedFormat === key ? `border-${formats[key].color}-500 bg-${formats[key].color}-50 dark:bg-${formats[key].color}-900/20` : 'border-slate-200'
            }`}
          >
            <div className="text-sm font-bold">{formats[key].name}</div>
            <div className="text-xs text-slate-500">{formats[key].bits} bit</div>
          </button>
        ))}
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        <div className="p-4 bg-white dark:bg-slate-800 rounded-lg border">
          <div className="text-sm font-semibold mb-3">位分配</div>
          <div className="flex h-12">
            <div className="flex-shrink-0 w-8 bg-red-500 flex items-center justify-center text-white text-xs">S</div>
            <div style={{width: `${(selected.exp / selected.bits) * 100}%`}} className="bg-green-500 flex items-center justify-center text-white text-xs">
              指数 ({selected.exp})
            </div>
            <div style={{width: `${(selected.mantissa / selected.bits) * 100}%`}} className="bg-blue-500 flex items-center justify-center text-white text-xs">
              尾数 ({selected.mantissa})
            </div>
          </div>
        </div>

        <div className="space-y-3">
          <div className="p-3 bg-white dark:bg-slate-800 rounded-lg border">
            <div className="text-xs text-slate-600 dark:text-slate-400">数值范围</div>
            <div className="text-lg font-bold text-blue-600">{selected.range}</div>
          </div>
          <div className="p-3 bg-white dark:bg-slate-800 rounded-lg border">
            <div className="text-xs text-slate-600 dark:text-slate-400">精度</div>
            <div className="text-lg font-bold text-green-600">{selected.precision}小数</div>
          </div>
        </div>
      </div>
    </div>
  )
}
