'use client'

import React from 'react'
import { motion } from 'framer-motion'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'

const performanceData = [
  { gpus: 1, ideal: 1, actual: 1, efficiency: 100 },
  { gpus: 2, ideal: 2, actual: 1.9, efficiency: 95 },
  { gpus: 4, ideal: 4, actual: 3.6, efficiency: 90 },
  { gpus: 8, ideal: 8, actual: 6.8, efficiency: 85 },
  { gpus: 16, ideal: 16, actual: 12.8, efficiency: 80 },
  { gpus: 32, ideal: 32, actual: 24, efficiency: 75 }
]

export default function FSDPScalingChart() {
  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-blue-50 to-cyan-50 rounded-xl border border-blue-200">
      <h3 className="text-2xl font-bold text-center mb-6 text-slate-800">
        📈 FSDP 扩展性能分析
      </h3>

      <ResponsiveContainer width="100%" height={350}>
        <LineChart data={performanceData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="gpus" label={{ value: 'GPU 数量', position: 'insideBottom', offset: -5 }} />
          <YAxis label={{ value: '加速倍数', angle: -90, position: 'insideLeft' }} />
          <Tooltip />
          <Legend />
          <Line type="monotone" dataKey="ideal" stroke="#94a3b8" strokeWidth={2} strokeDasharray="5 5" name="理想线性加速" />
          <Line type="monotone" dataKey="actual" stroke="#3b82f6" strokeWidth={3} name="FSDP 实际性能" />
        </LineChart>
      </ResponsiveContainer>

      <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-4">
        {performanceData.slice(1, 5).map((item, idx) => (
          <motion.div
            key={idx}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: idx * 0.1 }}
            className="bg-white rounded-lg p-4 border border-blue-200"
          >
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">{item.gpus} GPUs</div>
              <div className="text-sm text-slate-600 mt-1">效率: {item.efficiency}%</div>
              <div className="text-xs text-slate-500 mt-1">{item.actual}x 加速</div>
            </div>
          </motion.div>
        ))}
      </div>

      <div className="mt-6 bg-blue-50 rounded-lg p-4 border border-blue-200">
        <h4 className="font-bold text-blue-800 mb-2 text-sm">💡 扩展性分析</h4>
        <ul className="text-xs text-blue-700 space-y-1">
          <li><strong>通信开销</strong>: GPU 数量增加导致 all-gather 和 reduce-scatter 开销上升</li>
          <li><strong>效率下降</strong>: 从 95% (2卡) 逐渐降至 75% (32卡)</li>
          <li><strong>最佳配置</strong>: 4-8 卡通常是性价比最优选择</li>
          <li><strong>超大规模</strong>: 32+ 卡需结合 Pipeline Parallelism</li>
        </ul>
      </div>
    </div>
  )
}
