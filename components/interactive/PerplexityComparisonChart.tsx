'use client'

import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { BarChart3, TrendingDown, Info } from 'lucide-react'

export default function PerplexityComparisonChart() {
  const [selectedModel, setSelectedModel] = useState<string | null>(null)

  // LLaMA-7B 在 WikiText-2 上的实测困惑度
  const data = [
    {
      config: 'FP16 (baseline)',
      ppl: 5.68,
      color: 'from-slate-400 to-slate-600',
      degradation: 0,
      memory: '14 GB',
      speed: '18 tokens/s',
    },
    {
      config: 'INT8 (EETQ)',
      ppl: 5.74,
      color: 'from-green-400 to-green-600',
      degradation: 1.1,
      memory: '7 GB',
      speed: '32 tokens/s',
    },
    {
      config: 'GPTQ 4-bit',
      ppl: 6.12,
      color: 'from-blue-400 to-blue-600',
      degradation: 7.7,
      memory: '4.5 GB',
      speed: '35 tokens/s',
    },
    {
      config: 'AWQ 4-bit',
      ppl: 6.18,
      color: 'from-amber-400 to-amber-600',
      degradation: 8.8,
      memory: '4.2 GB',
      speed: '38 tokens/s',
    },
    {
      config: 'bitsandbytes 4-bit',
      ppl: 6.28,
      color: 'from-purple-400 to-purple-600',
      degradation: 10.6,
      memory: '4.8 GB',
      speed: '28 tokens/s',
    },
    {
      config: 'GPTQ 3-bit',
      ppl: 7.45,
      color: 'from-red-400 to-red-600',
      degradation: 31.2,
      memory: '3.2 GB',
      speed: '42 tokens/s',
    },
  ]

  const maxPpl = Math.max(...data.map(d => d.ppl))
  const baselinePpl = data[0].ppl

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-blue-50 rounded-xl border border-slate-200">
      <h3 className="text-2xl font-bold text-center mb-6 text-slate-800">
        困惑度对比分析
      </h3>

      {/* 说明 */}
      <div className="bg-blue-50 p-4 rounded-lg border border-blue-200 mb-6">
        <div className="flex items-start gap-3">
          <Info className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
          <div className="text-sm text-blue-800">
            <strong>困惑度 (Perplexity)</strong> 衡量语言模型的预测能力，越低越好。
            基准模型 (FP16) PPL = {baselinePpl}，量化后 PPL 增加表示精度下降。
          </div>
        </div>
      </div>

      {/* 柱状图 */}
      <div className="bg-white p-6 rounded-xl border border-slate-200 mb-6">
        <div className="flex items-center justify-between mb-4">
          <h4 className="font-bold text-slate-800 flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-blue-500" />
            困惑度对比 (LLaMA-7B on WikiText-2)
          </h4>
          <div className="text-sm text-slate-600">
            基准: {baselinePpl} PPL
          </div>
        </div>

        <div className="space-y-3">
          {data.map((item, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: idx * 0.1 }}
              onMouseEnter={() => setSelectedModel(item.config)}
              onMouseLeave={() => setSelectedModel(null)}
              className={`cursor-pointer transition-all ${
                selectedModel === item.config ? 'transform scale-105' : ''
              }`}
            >
              <div className="flex items-center gap-4">
                {/* 配置名称 */}
                <div className="w-48 text-sm font-medium text-slate-700">
                  {item.config}
                </div>

                {/* 柱状图 */}
                <div className="flex-1 relative">
                  <div className="h-12 bg-slate-100 rounded-lg overflow-hidden">
                    <motion.div
                      className={`h-full bg-gradient-to-r ${item.color} flex items-center justify-between px-4`}
                      initial={{ width: 0 }}
                      animate={{ width: `${(item.ppl / maxPpl) * 100}%` }}
                      transition={{ duration: 0.8, delay: idx * 0.1 }}
                    >
                      <span className="text-white font-bold text-sm">{item.ppl.toFixed(2)}</span>
                      {item.degradation > 0 && (
                        <span className="px-2 py-0.5 bg-white/20 rounded text-xs text-white font-medium">
                          +{item.degradation.toFixed(1)}%
                        </span>
                      )}
                    </motion.div>
                  </div>
                  
                  {/* 基准线 */}
                  {idx === 0 && (
                    <div
                      className="absolute top-0 bottom-0 w-0.5 bg-red-500 z-10"
                      style={{ left: `${(baselinePpl / maxPpl) * 100}%` }}
                    >
                      <div className="absolute -top-6 left-1/2 -translate-x-1/2 text-xs text-red-600 font-bold whitespace-nowrap">
                        baseline
                      </div>
                    </div>
                  )}
                </div>

                {/* 指标 */}
                <div className="flex gap-2 text-xs">
                  <div className="px-2 py-1 bg-slate-100 rounded text-slate-600">
                    {item.memory}
                  </div>
                  <div className="px-2 py-1 bg-green-100 rounded text-green-700">
                    {item.speed}
                  </div>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>

      {/* 详细信息 */}
      {selectedModel && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-gradient-to-br from-blue-50 to-indigo-50 p-6 rounded-xl border border-blue-200 mb-6"
        >
          {(() => {
            const item = data.find(d => d.config === selectedModel)!
            return (
              <div>
                <h4 className="font-bold text-lg text-slate-800 mb-3">{item.config}</h4>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div>
                    <div className="text-xs text-slate-600 mb-1">困惑度</div>
                    <div className="text-xl font-bold text-blue-600">{item.ppl.toFixed(2)}</div>
                  </div>
                  <div>
                    <div className="text-xs text-slate-600 mb-1">精度损失</div>
                    <div className={`text-xl font-bold ${
                      item.degradation < 5 ? 'text-green-600' : 
                      item.degradation < 15 ? 'text-amber-600' : 'text-red-600'
                    }`}>
                      +{item.degradation.toFixed(1)}%
                    </div>
                  </div>
                  <div>
                    <div className="text-xs text-slate-600 mb-1">显存占用</div>
                    <div className="text-xl font-bold text-purple-600">{item.memory}</div>
                  </div>
                  <div>
                    <div className="text-xs text-slate-600 mb-1">推理速度</div>
                    <div className="text-xl font-bold text-green-600">{item.speed}</div>
                  </div>
                </div>
              </div>
            )
          })()}
        </motion.div>
      )}

      {/* 趋势分析 */}
      <div className="bg-white p-6 rounded-xl border border-slate-200">
        <h4 className="font-bold text-slate-800 mb-4 flex items-center gap-2">
          <TrendingDown className="w-5 h-5 text-green-500" />
          量化策略建议
        </h4>

        <div className="grid md:grid-cols-3 gap-4">
          <div className="p-4 bg-green-50 rounded-lg border border-green-200">
            <div className="font-bold text-green-800 mb-2">🎯 高精度场景</div>
            <div className="text-sm text-green-700 space-y-1">
              <div>• 选择: <strong>INT8 (EETQ)</strong></div>
              <div>• PPL 增加: &lt;2%</div>
              <div>• 显存节省: 50%</div>
              <div>• 适用: 精度敏感任务</div>
            </div>
          </div>

          <div className="p-4 bg-blue-50 rounded-lg border border-blue-200">
            <div className="font-bold text-blue-800 mb-2">⚖️ 平衡场景</div>
            <div className="text-sm text-blue-700 space-y-1">
              <div>• 选择: <strong>GPTQ/AWQ 4-bit</strong></div>
              <div>• PPL 增加: 7-9%</div>
              <div>• 显存节省: 70%</div>
              <div>• 适用: 大多数应用</div>
            </div>
          </div>

          <div className="p-4 bg-amber-50 rounded-lg border border-amber-200">
            <div className="font-bold text-amber-800 mb-2">🚀 极限压缩</div>
            <div className="text-sm text-amber-700 space-y-1">
              <div>• 选择: <strong>GPTQ 3-bit</strong></div>
              <div>• PPL 增加: 30%+</div>
              <div>• 显存节省: 77%</div>
              <div>• 适用: 资源极度受限</div>
            </div>
          </div>
        </div>

        <div className="mt-4 p-4 bg-gradient-to-r from-amber-50 to-orange-50 rounded-lg border border-amber-200">
          <div className="font-bold text-amber-800 mb-2">⚠️ 重要提示</div>
          <ul className="text-sm text-amber-700 space-y-1">
            <li>• 困惑度增加 &lt;10% 通常可接受（下游任务影响 &lt;2%）</li>
            <li>• 3-bit 量化精度损失较大，需在实际任务上验证</li>
            <li>• 不同模型、数据集上的表现可能差异较大</li>
          </ul>
        </div>
      </div>

      <div className="mt-4 text-xs text-slate-500 text-center">
        💡 悬停柱状图查看详细指标 | PPL 越低表示模型预测能力越强
      </div>
    </div>
  )
}
