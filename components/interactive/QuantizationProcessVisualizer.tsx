'use client'

import React from 'react'
import { motion } from 'framer-motion'

export default function QuantizationProcessVisualizer() {
  const fpValues = [0.87, -0.43, 1.23, -1.56, 0.12]
  const scale = 1.56 / 127
  const quantized = fpValues.map(v => Math.round(v / scale))

  return (
    <div className="my-8 p-6 bg-gradient-to-br from-green-50 to-emerald-50 dark:from-slate-900 dark:to-green-950 rounded-xl border border-slate-200 dark:border-slate-700">
      <h3 className="text-xl font-bold mb-6">量化过程可视化</h3>
      <div className="grid md:grid-cols-3 gap-4">
        <div className="p-4 bg-white dark:bg-slate-800 rounded-lg">
          <div className="text-sm font-bold mb-2">原始FP32</div>
          {fpValues.map((v, i) => (
            <div key={i} className="text-lg font-mono text-blue-600">{v.toFixed(2)}</div>
          ))}
        </div>
        <div className="flex items-center justify-center">
          <div className="text-2xl">→</div>
        </div>
        <div className="p-4 bg-white dark:bg-slate-800 rounded-lg">
          <div className="text-sm font-bold mb-2">量化INT8</div>
          {quantized.map((v, i) => (
            <div key={i} className="text-lg font-mono text-green-600">{v}</div>
          ))}
        </div>
      </div>
      <div className="mt-4 p-3 bg-slate-900 rounded text-green-400 font-mono text-sm">
        scale = max(abs(x)) / 127 = {scale.toFixed(4)}<br/>
        q = round(x / scale)
      </div>
    </div>
  )
}
