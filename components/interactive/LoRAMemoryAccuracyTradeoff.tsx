'use client'

import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ZAxis } from 'recharts'

const rankData = [
  { rank: 1, accuracy: 0.82, memory: 2.1, label: 'r=1' },
  { rank: 2, accuracy: 0.86, memory: 2.3, label: 'r=2' },
  { rank: 4, accuracy: 0.91, memory: 2.8, label: 'r=4' },
  { rank: 8, accuracy: 0.95, memory: 3.8, label: 'r=8' },
  { rank: 16, accuracy: 0.98, memory: 5.8, label: 'r=16' },
  { rank: 32, accuracy: 0.99, memory: 9.8, label: 'r=32' },
  { rank: 64, accuracy: 0.995, memory: 17.8, label: 'r=64' },
  { rank: 0, accuracy: 1.0, memory: 40, label: '全参数' }
]

export default function LoRAMemoryAccuracyTradeoff() {
  const [selectedRank, setSelectedRank] = useState(8)

  const selectedData = rankData.find(d => d.rank === selectedRank)

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-purple-50 to-pink-50 rounded-xl border border-purple-200">
      <h3 className="text-2xl font-bold text-center mb-6 text-slate-800">
        ⚖️ LoRA 秩（r）：显存-准确率权衡
      </h3>

      <ResponsiveContainer width="100%" height={350}>
        <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            type="number"
            dataKey="memory"
            name="显存占用 (GB)"
            label={{ value: '显存占用 (GB)', position: 'insideBottom', offset: -10 }}
          />
          <YAxis
            type="number"
            dataKey="accuracy"
            name="准确率"
            domain={[0.8, 1.0]}
            label={{ value: '准确率', angle: -90, position: 'insideLeft' }}
          />
          <ZAxis range={[100, 400]} />
          <Tooltip
            cursor={{ strokeDasharray: '3 3' }}
            content={({ payload }) => {
              if (!payload || !payload[0]) return null
              const data = payload[0].payload
              return (
                <div className="bg-white p-3 rounded-lg shadow-lg border border-purple-200">
                  <div className="font-bold text-purple-800">{data.label}</div>
                  <div className="text-sm text-slate-600">准确率: {(data.accuracy * 100).toFixed(1)}%</div>
                  <div className="text-sm text-slate-600">显存: {data.memory} GB</div>
                </div>
              )
            }}
          />
          <Scatter
            name="LoRA配置"
            data={rankData}
            fill="#a855f7"
            onClick={(data) => setSelectedRank(data.rank)}
          />
        </ScatterChart>
      </ResponsiveContainer>

      {/* 秩选择器 */}
      <div className="mt-6 bg-white rounded-xl p-6 border border-purple-200">
        <h4 className="font-bold text-slate-800 mb-4">选择 LoRA 秩 (r)</h4>
        <div className="flex gap-2 flex-wrap">
          {[1, 2, 4, 8, 16, 32, 64].map((r) => (
            <button
              key={r}
              onClick={() => setSelectedRank(r)}
              className={`px-4 py-2 rounded-lg font-mono text-sm transition-all ${
                selectedRank === r
                  ? 'bg-gradient-to-r from-purple-500 to-pink-500 text-white shadow-lg'
                  : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
              }`}
            >
              r={r}
            </button>
          ))}
        </div>

        {selectedData && (
          <motion.div
            key={selectedRank}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-6 grid grid-cols-3 gap-4"
          >
            <div className="bg-purple-50 rounded-lg p-4">
              <div className="text-xs text-purple-600 font-bold mb-1">准确率</div>
              <div className="text-2xl font-bold text-purple-800">
                {(selectedData.accuracy * 100).toFixed(1)}%
              </div>
            </div>
            <div className="bg-pink-50 rounded-lg p-4">
              <div className="text-xs text-pink-600 font-bold mb-1">显存占用</div>
              <div className="text-2xl font-bold text-pink-800">
                {selectedData.memory} GB
              </div>
            </div>
            <div className="bg-indigo-50 rounded-lg p-4">
              <div className="text-xs text-indigo-600 font-bold mb-1">可训练参数</div>
              <div className="text-2xl font-bold text-indigo-800">
                {selectedRank === 0 ? '100%' : `${(selectedRank / 10).toFixed(1)}%`}
              </div>
            </div>
          </motion.div>
        )}

        <div className="mt-6 bg-purple-50 rounded-lg p-4 border border-purple-200">
          <h5 className="font-bold text-purple-800 mb-2 text-sm">💡 选择建议</h5>
          <ul className="text-xs text-purple-700 space-y-1">
            <li><strong>r=4-8</strong>: 通用推荐，性价比最高</li>
            <li><strong>r=16-32</strong>: 复杂任务，需要更强表达能力</li>
            <li><strong>r=64+</strong>: 接近全参数微调效果，但显存占用高</li>
            <li><strong>r=1-2</strong>: 极端显存受限，准确率显著下降</li>
          </ul>
        </div>
      </div>
    </div>
  )
}
