'use client'

import React from 'react'

export default function GradientAccumulationVisualizer() {
  return (
    <div className="my-8 p-6 bg-gradient-to-br from-amber-50 to-yellow-50 dark:from-slate-900 dark:to-amber-950 rounded-xl border border-slate-200 dark:border-slate-700">
      <h3 className="text-xl font-bold mb-6">梯度累积可视化</h3>
      <div className="grid md:grid-cols-2 gap-6">
        <div className="p-4 bg-white dark:bg-slate-800 rounded-lg border">
          <div className="text-sm font-bold mb-3">传统训练 (batch=32)</div>
          <div className="space-y-2">
            {[1].map(i => (
              <div key={i} className="h-16 bg-blue-500 rounded flex items-center justify-center text-white">
                Batch 32 → 更新
              </div>
            ))}
          </div>
          <div className="text-xs text-slate-500 mt-2">显存占用: 高</div>
        </div>
        <div className="p-4 bg-white dark:bg-slate-800 rounded-lg border">
          <div className="text-sm font-bold mb-3">梯度累积 (batch=8 × 4)</div>
          <div className="space-y-2">
            {[1,2,3,4].map(i => (
              <div key={i} className={`h-4 rounded flex items-center justify-center text-white text-xs ${i === 4 ? 'bg-green-500' : 'bg-purple-400'}`}>
                Batch 8 {i === 4 && '→ 更新'}
              </div>
            ))}
          </div>
          <div className="text-xs text-slate-500 mt-2">显存占用: 低</div>
        </div>
      </div>
      <div className="mt-4 p-3 bg-slate-900 rounded text-green-400 font-mono text-sm">
        gradient_accumulation_steps=4<br/>
        effective_batch_size = 8 × 4 = 32
      </div>
    </div>
  )
}
