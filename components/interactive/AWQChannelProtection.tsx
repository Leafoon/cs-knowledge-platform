'use client'

import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { Eye, Zap, Shield } from 'lucide-react'

export default function AWQChannelProtection() {
  const [highlightSalient, setHighlightSalient] = useState(false)
  const [showScaling, setShowScaling] = useState(false)

  // 模拟权重矩阵和激活值
  const channels = Array.from({ length: 8 }, (_, i) => {
    const isSalient = [0, 2, 7].includes(i) // 1%, 3%, 8% 是重要通道
    const weightMagnitude = isSalient ? Math.random() * 2 + 1 : Math.random() * 0.5
    const activationMagnitude = isSalient ? Math.random() * 10 + 5 : Math.random() * 2
    const importance = weightMagnitude * activationMagnitude
    
    return {
      id: i,
      isSalient,
      weight: weightMagnitude,
      activation: activationMagnitude,
      importance,
      alpha: isSalient ? 1.5 : 1.0, // 缩放因子
    }
  })

  const maxImportance = Math.max(...channels.map(c => c.importance))

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-amber-50 rounded-xl border border-slate-200">
      <h3 className="text-2xl font-bold text-center mb-6 text-slate-800">
        AWQ 通道保护策略
      </h3>

      {/* 控制按钮 */}
      <div className="flex gap-3 mb-6">
        <motion.button
          onClick={() => setHighlightSalient(!highlightSalient)}
          className={`flex-1 px-4 py-3 rounded-lg border-2 transition-all ${
            highlightSalient
              ? 'border-amber-500 bg-amber-50 text-amber-700'
              : 'border-slate-300 bg-white text-slate-600'
          }`}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          <Eye className="w-5 h-5 mx-auto mb-1" />
          <div className="text-sm font-bold">识别重要通道</div>
        </motion.button>

        <motion.button
          onClick={() => setShowScaling(!showScaling)}
          className={`flex-1 px-4 py-3 rounded-lg border-2 transition-all ${
            showScaling
              ? 'border-green-500 bg-green-50 text-green-700'
              : 'border-slate-300 bg-white text-slate-600'
          }`}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          <Shield className="w-5 h-5 mx-auto mb-1" />
          <div className="text-sm font-bold">应用缩放保护</div>
        </motion.button>
      </div>

      {/* 通道重要性可视化 */}
      <div className="bg-white p-6 rounded-xl border border-slate-200 mb-6">
        <h4 className="font-bold text-slate-800 mb-4 flex items-center gap-2">
          <Zap className="w-5 h-5 text-amber-500" />
          通道重要性分析
        </h4>
        
        <div className="space-y-3">
          {channels.map((channel, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: idx * 0.05 }}
              className="relative"
            >
              <div className="flex items-center gap-3 mb-1">
                <div className="w-20 text-sm font-medium text-slate-600">
                  Channel {channel.id}
                </div>
                <div className="flex-1 h-10 bg-slate-100 rounded-lg overflow-hidden relative">
                  <motion.div
                    className={`h-full rounded-lg ${
                      highlightSalient && channel.isSalient
                        ? 'bg-gradient-to-r from-amber-400 to-orange-500'
                        : 'bg-gradient-to-r from-blue-400 to-blue-600'
                    }`}
                    initial={{ width: 0 }}
                    animate={{ width: `${(channel.importance / maxImportance) * 100}%` }}
                    transition={{ duration: 0.5, delay: idx * 0.05 }}
                  />
                  <div className="absolute inset-0 flex items-center justify-between px-3">
                    <span className="text-xs text-white font-bold">
                      重要性: {channel.importance.toFixed(2)}
                    </span>
                    {highlightSalient && channel.isSalient && (
                      <span className="px-2 py-0.5 bg-white/20 rounded text-xs text-white font-bold">
                        重要通道
                      </span>
                    )}
                  </div>
                </div>
                
                {showScaling && (
                  <motion.div
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    className={`w-16 h-10 rounded-lg flex items-center justify-center text-white font-bold text-sm ${
                      channel.isSalient
                        ? 'bg-gradient-to-br from-green-500 to-green-600'
                        : 'bg-slate-400'
                    }`}
                  >
                    α={channel.alpha}
                  </motion.div>
                )}
              </div>
              
              {/* 详细指标 */}
              <div className="ml-24 flex gap-4 text-xs text-slate-500">
                <span>权重: {channel.weight.toFixed(2)}</span>
                <span>激活: {channel.activation.toFixed(2)}</span>
                {showScaling && (
                  <span className="text-green-600 font-medium">
                    缩放后权重: {(channel.weight * channel.alpha).toFixed(2)}
                  </span>
                )}
              </div>
            </motion.div>
          ))}
        </div>
      </div>

      {/* 统计信息 */}
      <div className="grid md:grid-cols-3 gap-4 mb-6">
        <div className="bg-gradient-to-br from-amber-50 to-orange-50 p-4 rounded-lg border border-amber-200">
          <div className="text-sm font-medium text-amber-700 mb-1">重要通道占比</div>
          <div className="text-2xl font-bold text-amber-800">
            {((channels.filter(c => c.isSalient).length / channels.length) * 100).toFixed(0)}%
          </div>
          <div className="text-xs text-amber-600 mt-1">
            {channels.filter(c => c.isSalient).length} / {channels.length} 通道
          </div>
        </div>

        <div className="bg-gradient-to-br from-blue-50 to-cyan-50 p-4 rounded-lg border border-blue-200">
          <div className="text-sm font-medium text-blue-700 mb-1">输出贡献度</div>
          <div className="text-2xl font-bold text-blue-800">~80%</div>
          <div className="text-xs text-blue-600 mt-1">
            来自 {((channels.filter(c => c.isSalient).length / channels.length) * 100).toFixed(0)}% 的通道
          </div>
        </div>

        <div className="bg-gradient-to-br from-green-50 to-emerald-50 p-4 rounded-lg border border-green-200">
          <div className="text-sm font-medium text-green-700 mb-1">量化误差降低</div>
          <div className="text-2xl font-bold text-green-800">~50%</div>
          <div className="text-xs text-green-600 mt-1">
            通过保护重要通道实现
          </div>
        </div>
      </div>

      {/* 数学原理 */}
      <div className="bg-gradient-to-br from-purple-50 to-indigo-50 p-6 rounded-xl border border-purple-200 mb-6">
        <h4 className="font-bold text-purple-800 mb-4">AWQ 数学原理</h4>
        
        <div className="space-y-4">
          <div>
            <div className="text-sm font-medium text-purple-700 mb-2">1. 识别重要通道</div>
            <div className="bg-white p-3 rounded-lg font-mono text-sm overflow-x-auto">
              {'$'}s_i = \\frac{'{'}1{'}'}{'{'} N {'}'} \\sum_{'{'}j=1{'}'}^{'{'} N {'}'} | \\mathbf{'{'}X{'}'}_{'{'}ij{'}'} \\cdot \\mathbf{'{'}W{'}'}_{'{'}i{'}'} |{'$'}
            </div>
          </div>

          <div>
            <div className="text-sm font-medium text-purple-700 mb-2">2. Per-Channel 缩放</div>
            <div className="bg-white p-3 rounded-lg font-mono text-sm overflow-x-auto">
              {'$'}\\mathbf{'{'}W{'}'}_{'{'}i{'}'}{'}'} = \\alpha_i \\cdot \\mathbf{'{'}W{'}'}_{'{'}i{'}'}, \\quad \\mathbf{'{'}X{'}'}_{'{'}i{'}'}{'}'} = \\frac{'{'}\\mathbf{'{'}X{'}'}_{'{'}i{'}'}{'}'}{'{'} \\alpha_i {'}'}{'$'}
            </div>
          </div>

          <div>
            <div className="text-sm font-medium text-purple-700 mb-2">3. 缩放因子计算</div>
            <div className="bg-white p-3 rounded-lg font-mono text-sm overflow-x-auto">
              $\alpha_i = \max(|\mathbf{'{'}X{'}'}_{'{'}i{'}'}|)^\alpha / \max(|\mathbf{'{'}W{'}'}_{'{'}i{'}'}|)^{'{'}1-\alpha{'}'}$
            </div>
          </div>
        </div>

        <div className="mt-4 p-4 bg-white/50 rounded-lg text-sm text-purple-700">
          <strong>核心思想：</strong>通过等价变换 $\mathbf{'{'}W{'}'} \mathbf{'{'}X{'}'} = (\alpha \mathbf{'{'}W{'}'}) \cdot (\mathbf{'{'}X{'}'}/\alpha)$，
          将量化误差从重要权重转移到不重要的激活值上。
        </div>
      </div>

      {/* 对比 GPTQ */}
      <div className="bg-white p-6 rounded-xl border border-slate-200">
        <h4 className="font-bold text-slate-800 mb-4">AWQ vs GPTQ</h4>
        <div className="grid md:grid-cols-2 gap-4">
          <div className="p-4 bg-amber-50 rounded-lg border border-amber-200">
            <h5 className="font-bold text-amber-800 mb-2">AWQ（激活值感知）</h5>
            <ul className="text-sm text-amber-700 space-y-1">
              <li>• 基于激活值统计（启发式）</li>
              <li>• 量化时间: 3-5 分钟</li>
              <li>• 推理速度: ⭐⭐⭐⭐⭐</li>
              <li>• 精度: ⭐⭐⭐⭐</li>
            </ul>
          </div>

          <div className="p-4 bg-blue-50 rounded-lg border border-blue-200">
            <h5 className="font-bold text-blue-800 mb-2">GPTQ（二阶优化）</h5>
            <ul className="text-sm text-blue-700 space-y-1">
              <li>• 基于 Hessian 矩阵（全局优化）</li>
              <li>• 量化时间: 5-10 分钟</li>
              <li>• 推理速度: ⭐⭐⭐⭐</li>
              <li>• 精度: ⭐⭐⭐⭐⭐</li>
            </ul>
          </div>
        </div>
      </div>

      <div className="mt-4 text-xs text-slate-500 text-center">
        🎯 幂律分布：1% 的通道贡献 80% 的输出幅度
      </div>
    </div>
  )
}
