'use client'

import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { BarChart3, TrendingDown } from 'lucide-react'

export default function NF4EncodingVisualizer() {
  const [inputValue, setInputValue] = useState(0.45)

  // NF4 的 16 个量化级别
  const NF4_LEVELS = [
    -1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848,
    -0.0911, 0.0, 0.0796, 0.1609, 0.2461, 0.3379,
    0.4407, 0.5626, 0.7230, 1.0
  ]

  // 找到最接近的 NF4 级别
  const findClosestNF4 = (value: number) => {
    let closest = 0
    let minDiff = Math.abs(value - NF4_LEVELS[0])
    
    NF4_LEVELS.forEach((level, idx) => {
      const diff = Math.abs(value - level)
      if (diff < minDiff) {
        minDiff = diff
        closest = idx
      }
    })
    
    return closest
  }

  const quantizedIdx = findClosestNF4(inputValue)
  const quantizedValue = NF4_LEVELS[quantizedIdx]
  const error = Math.abs(inputValue - quantizedValue)

  // 正态分布数据（用于背景）
  const generateNormalDist = () => {
    const points = []
    for (let i = -3; i <= 3; i += 0.1) {
      const y = (1 / Math.sqrt(2 * Math.PI)) * Math.exp(-(i * i) / 2)
      points.push({ x: i / 3, y })  // 归一化到 [-1, 1]
    }
    return points
  }

  const normalDist = generateNormalDist()

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-purple-50 rounded-xl shadow-lg">
      <div className="flex items-center gap-3 mb-6">
        <TrendingDown className="w-8 h-8 text-purple-600" />
        <h3 className="text-2xl font-bold text-slate-800">NF4 编码过程可视化</h3>
      </div>

      {/* 输入控制 */}
      <div className="mb-6 p-4 bg-white rounded-lg shadow">
        <label className="block text-sm font-medium text-slate-700 mb-2">
          输入权重值: {inputValue.toFixed(4)}
        </label>
        <input
          type="range"
          min="-1"
          max="1"
          step="0.01"
          value={inputValue}
          onChange={(e) => setInputValue(Number(e.target.value))}
          className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer"
        />
        <div className="flex justify-between text-xs text-slate-500 mt-1">
          <span>-1.0</span>
          <span>0.0</span>
          <span>1.0</span>
        </div>
      </div>

      {/* 可视化 */}
      <div className="grid grid-cols-2 gap-6 mb-6">
        {/* 左侧：正态分布 + NF4 Bins */}
        <div className="bg-white p-5 rounded-lg shadow">
          <h4 className="font-bold text-slate-800 mb-3 flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-purple-600" />
            NF4 分位数分布
          </h4>
          
          {/* 正态分布曲线 */}
          <div className="relative h-48 bg-slate-50 rounded border border-slate-200 overflow-hidden mb-4">
            {/* Y轴 */}
            <div className="absolute left-2 top-2 text-xs text-slate-500">密度</div>
            
            {/* 正态分布曲线 */}
            <svg className="absolute inset-0" viewBox="-1.2 -0.5 2.4 1.5">
              <path
                d={normalDist.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${1 - p.y}`).join(' ')}
                fill="none"
                stroke="#9333ea"
                strokeWidth="0.02"
              />
              
              {/* NF4 边界线 */}
              {NF4_LEVELS.map((level, idx) => (
                <line
                  key={idx}
                  x1={level}
                  y1="0"
                  x2={level}
                  y2="1"
                  stroke={idx === quantizedIdx ? '#dc2626' : '#cbd5e1'}
                  strokeWidth={idx === quantizedIdx ? '0.03' : '0.01'}
                  strokeDasharray={idx === quantizedIdx ? '0' : '0.02'}
                />
              ))}
              
              {/* 当前输入值标记 */}
              <circle
                cx={inputValue}
                cy="0.5"
                r="0.04"
                fill="#3b82f6"
                stroke="white"
                strokeWidth="0.01"
              />
            </svg>
            
            {/* X轴标签 */}
            <div className="absolute bottom-2 left-0 right-0 flex justify-between px-4 text-xs text-slate-500">
              <span>-1</span>
              <span>0</span>
              <span>1</span>
            </div>
          </div>

          <div className="text-xs text-slate-600 bg-purple-50 p-3 rounded border border-purple-200">
            <strong>分位数量化</strong>: NF4 将权重按照标准正态分布的分位点划分为 16 个 bins，
            每个 bin 包含相等数量的权重（而非相等的数值范围）
          </div>
        </div>

        {/* 右侧：量化过程 */}
        <div className="bg-white p-5 rounded-lg shadow">
          <h4 className="font-bold text-slate-800 mb-3">量化步骤</h4>
          
          <div className="space-y-3">
            {/* 步骤1 */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              className="p-3 bg-blue-50 rounded border border-blue-200"
            >
              <div className="flex items-center gap-2 mb-1">
                <div className="w-6 h-6 bg-blue-500 rounded-full text-white flex items-center justify-center text-xs font-bold">
                  1
                </div>
                <div className="text-sm font-bold text-blue-800">输入权重</div>
              </div>
              <div className="ml-8 font-mono text-lg text-blue-600">
                {inputValue.toFixed(4)}
              </div>
            </motion.div>

            {/* 步骤2 */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
              className="p-3 bg-purple-50 rounded border border-purple-200"
            >
              <div className="flex items-center gap-2 mb-1">
                <div className="w-6 h-6 bg-purple-500 rounded-full text-white flex items-center justify-center text-xs font-bold">
                  2
                </div>
                <div className="text-sm font-bold text-purple-800">查找最近的 NF4 级别</div>
              </div>
              <div className="ml-8 space-y-1">
                <div className="text-xs text-slate-600">索引: {quantizedIdx}</div>
                <div className="font-mono text-lg text-purple-600">
                  {quantizedValue.toFixed(4)}
                </div>
              </div>
            </motion.div>

            {/* 步骤3 */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.4 }}
              className="p-3 bg-green-50 rounded border border-green-200"
            >
              <div className="flex items-center gap-2 mb-1">
                <div className="w-6 h-6 bg-green-500 rounded-full text-white flex items-center justify-center text-xs font-bold">
                  3
                </div>
                <div className="text-sm font-bold text-green-800">存储 4-bit 索引</div>
              </div>
              <div className="ml-8">
                <div className="font-mono text-sm bg-white p-2 rounded border">
                  {quantizedIdx.toString(2).padStart(4, '0')} <span className="text-slate-400">(二进制)</span>
                </div>
              </div>
            </motion.div>

            {/* 量化误差 */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.6 }}
              className="p-3 bg-red-50 rounded border border-red-200"
            >
              <div className="text-sm font-bold text-red-800 mb-1">量化误差</div>
              <div className="ml-2">
                <div className="text-xs text-slate-600">绝对误差</div>
                <div className="font-mono text-lg text-red-600">
                  {error.toFixed(6)}
                </div>
                <div className="text-xs text-slate-500 mt-1">
                  相对误差: {inputValue !== 0 ? ((error / Math.abs(inputValue)) * 100).toFixed(2) : 0}%
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </div>

      {/* NF4 量化表 */}
      <div className="bg-white p-6 rounded-lg shadow">
        <h4 className="font-bold text-slate-800 mb-4">NF4 量化表（16 级别）</h4>
        <div className="grid grid-cols-8 gap-2">
          {NF4_LEVELS.map((level, idx) => (
            <div
              key={idx}
              className={`p-2 rounded border-2 transition-all ${
                idx === quantizedIdx
                  ? 'border-red-500 bg-red-50 shadow-md scale-105'
                  : 'border-slate-200 bg-slate-50'
              }`}
            >
              <div className="text-xs text-slate-600 text-center mb-1">
                {idx.toString(2).padStart(4, '0')}
              </div>
              <div className={`text-sm font-mono text-center ${
                idx === quantizedIdx ? 'text-red-700 font-bold' : 'text-slate-700'
              }`}>
                {level.toFixed(4)}
              </div>
            </div>
          ))}
        </div>

        <div className="mt-4 p-3 bg-purple-50 border border-purple-200 rounded text-sm">
          <strong>关键优势</strong>: NF4 针对神经网络权重的正态分布特性优化，
          在 0 附近密集分配量化级别（大部分权重聚集处），远端稀疏（极端值罕见），
          相比 INT4 均匀分布，<strong>量化误差降低 ~30%</strong>。
        </div>
      </div>
    </div>
  )
}
