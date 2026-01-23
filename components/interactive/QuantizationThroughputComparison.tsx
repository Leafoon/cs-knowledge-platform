'use client'

import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { Activity, Zap, TrendingUp } from 'lucide-react'

export default function QuantizationThroughputComparison() {
  const [batchSize, setBatchSize] = useState<1 | 8 | 32>(1)
  const [selectedMethod, setSelectedMethod] = useState<string | null>(null)

  // 不同 batch size 下的吞吐量数据 (LLaMA-7B on RTX 4090)
  const throughputData = {
    1: [
      { method: 'FP16', throughput: 18, latency: 55, color: 'from-slate-400 to-slate-600' },
      { method: 'INT8 (EETQ)', throughput: 32, latency: 31, color: 'from-green-400 to-green-600' },
      { method: 'GPTQ 4-bit', throughput: 35, latency: 28, color: 'from-blue-400 to-blue-600' },
      { method: 'AWQ 4-bit', throughput: 38, latency: 26, color: 'from-amber-400 to-amber-600' },
      { method: 'BNB 4-bit', throughput: 28, latency: 35, color: 'from-purple-400 to-purple-600' },
    ],
    8: [
      { method: 'FP16', throughput: 125, latency: 64, color: 'from-slate-400 to-slate-600' },
      { method: 'INT8 (EETQ)', throughput: 245, latency: 33, color: 'from-green-400 to-green-600' },
      { method: 'GPTQ 4-bit', throughput: 280, latency: 29, color: 'from-blue-400 to-blue-600' },
      { method: 'AWQ 4-bit', throughput: 312, latency: 26, color: 'from-amber-400 to-amber-600' },
      { method: 'BNB 4-bit', throughput: 220, latency: 36, color: 'from-purple-400 to-purple-600' },
    ],
    32: [
      { method: 'FP16', throughput: 420, latency: 76, color: 'from-slate-400 to-slate-600' },
      { method: 'INT8 (EETQ)', throughput: 850, latency: 38, color: 'from-green-400 to-green-600' },
      { method: 'GPTQ 4-bit', throughput: 980, latency: 33, color: 'from-blue-400 to-blue-600' },
      { method: 'AWQ 4-bit', throughput: 1120, latency: 29, color: 'from-amber-400 to-amber-600' },
      { method: 'BNB 4-bit', throughput: 780, latency: 41, color: 'from-purple-400 to-purple-600' },
    ],
  }

  const currentData = throughputData[batchSize]
  const maxThroughput = Math.max(...currentData.map(d => d.throughput))
  const baselineThroughput = currentData[0].throughput

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-green-50 rounded-xl border border-slate-200">
      <h3 className="text-2xl font-bold text-center mb-6 text-slate-800">
        推理吞吐量对比
      </h3>

      {/* Batch Size 选择 */}
      <div className="flex gap-3 mb-6">
        {([1, 8, 32] as const).map((size) => (
          <motion.button
            key={size}
            onClick={() => setBatchSize(size)}
            className={`flex-1 px-4 py-3 rounded-lg border-2 transition-all ${
              batchSize === size
                ? 'border-green-500 bg-green-50 text-green-700'
                : 'border-slate-300 bg-white text-slate-600'
            }`}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            <div className="text-sm font-bold">Batch Size = {size}</div>
            <div className="text-xs opacity-75 mt-1">
              {size === 1 ? '在线推理' : size === 8 ? '小批量' : '大批量'}
            </div>
          </motion.button>
        ))}
      </div>

      {/* 吞吐量柱状图 */}
      <div className="bg-white p-6 rounded-xl border border-slate-200 mb-6">
        <div className="flex items-center justify-between mb-4">
          <h4 className="font-bold text-slate-800 flex items-center gap-2">
            <Activity className="w-5 h-5 text-green-500" />
            吞吐量对比 (tokens/s)
          </h4>
          <div className="text-sm text-slate-600">
            Batch Size: {batchSize} | Baseline: {baselineThroughput} tokens/s
          </div>
        </div>

        <div className="space-y-3">
          {currentData.map((item, idx) => {
            const speedup = item.throughput / baselineThroughput
            return (
              <motion.div
                key={item.method}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: idx * 0.1 }}
                onMouseEnter={() => setSelectedMethod(item.method)}
                onMouseLeave={() => setSelectedMethod(null)}
                className={`cursor-pointer transition-all ${
                  selectedMethod === item.method ? 'transform scale-105' : ''
                }`}
              >
                <div className="flex items-center gap-4">
                  {/* 方法名称 */}
                  <div className="w-40 text-sm font-medium text-slate-700">
                    {item.method}
                  </div>

                  {/* 柱状图 */}
                  <div className="flex-1 relative">
                    <div className="h-14 bg-slate-100 rounded-lg overflow-hidden">
                      <motion.div
                        className={`h-full bg-gradient-to-r ${item.color} flex items-center justify-between px-4`}
                        initial={{ width: 0 }}
                        animate={{ width: `${(item.throughput / maxThroughput) * 100}%` }}
                        transition={{ duration: 0.8, delay: idx * 0.1 }}
                      >
                        <span className="text-white font-bold">
                          {item.throughput} tokens/s
                        </span>
                        {speedup > 1 && (
                          <span className="px-2 py-0.5 bg-white/20 rounded text-sm text-white font-medium">
                            {speedup.toFixed(2)}x
                          </span>
                        )}
                      </motion.div>
                    </div>
                  </div>

                  {/* 延迟 */}
                  <div className="w-24 text-right">
                    <div className="text-xs text-slate-500">延迟</div>
                    <div className="text-sm font-bold text-slate-700">
                      {item.latency} ms/token
                    </div>
                  </div>
                </div>
              </motion.div>
            )
          })}
        </div>
      </div>

      {/* 加速比对比 */}
      <div className="grid md:grid-cols-3 gap-4 mb-6">
        {[
          { label: 'Batch=1 最快', method: 'AWQ 4-bit', speedup: currentData.find(d => d.method === 'AWQ 4-bit')!.throughput / baselineThroughput },
          { label: 'Batch=8 最快', method: 'AWQ 4-bit', speedup: throughputData[8].find(d => d.method === 'AWQ 4-bit')!.throughput / throughputData[8][0].throughput },
          { label: 'Batch=32 最快', method: 'AWQ 4-bit', speedup: throughputData[32].find(d => d.method === 'AWQ 4-bit')!.throughput / throughputData[32][0].throughput },
        ].map((item, idx) => (
          <motion.div
            key={idx}
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: idx * 0.1 }}
            className="bg-gradient-to-br from-green-50 to-emerald-50 p-4 rounded-lg border border-green-200"
          >
            <div className="text-sm font-medium text-green-700 mb-1">{item.label}</div>
            <div className="text-2xl font-bold text-green-800 flex items-center gap-2">
              <Zap className="w-6 h-6" />
              {item.speedup.toFixed(2)}x
            </div>
            <div className="text-xs text-green-600 mt-1">{item.method}</div>
          </motion.div>
        ))}
      </div>

      {/* 详细分析 */}
      {selectedMethod && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-gradient-to-br from-blue-50 to-indigo-50 p-6 rounded-xl border border-blue-200 mb-6"
        >
          {(() => {
            const item = currentData.find(d => d.method === selectedMethod)!
            const speedup = item.throughput / baselineThroughput
            return (
              <div>
                <h4 className="font-bold text-lg text-slate-800 mb-4">{item.method} 性能分析</h4>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div>
                    <div className="text-xs text-slate-600 mb-1">吞吐量</div>
                    <div className="text-xl font-bold text-blue-600">
                      {item.throughput} <span className="text-sm">tokens/s</span>
                    </div>
                  </div>
                  <div>
                    <div className="text-xs text-slate-600 mb-1">加速比</div>
                    <div className="text-xl font-bold text-green-600">
                      {speedup.toFixed(2)}x
                    </div>
                  </div>
                  <div>
                    <div className="text-xs text-slate-600 mb-1">延迟</div>
                    <div className="text-xl font-bold text-amber-600">
                      {item.latency} ms
                    </div>
                  </div>
                  <div>
                    <div className="text-xs text-slate-600 mb-1">Batch Size</div>
                    <div className="text-xl font-bold text-purple-600">{batchSize}</div>
                  </div>
                </div>
              </div>
            )
          })()}
        </motion.div>
      )}

      {/* 性能建议 */}
      <div className="bg-white p-6 rounded-xl border border-slate-200">
        <h4 className="font-bold text-slate-800 mb-4 flex items-center gap-2">
          <TrendingUp className="w-5 h-5 text-blue-500" />
          性能优化建议
        </h4>

        <div className="grid md:grid-cols-2 gap-4">
          <div className="p-4 bg-amber-50 rounded-lg border border-amber-200">
            <div className="font-bold text-amber-800 mb-2">🚀 在线推理 (Batch=1)</div>
            <ul className="text-sm text-amber-700 space-y-1">
              <li>• 首选: <strong>AWQ 4-bit</strong> (最低延迟 26ms)</li>
              <li>• 次选: GPTQ 4-bit (28ms)</li>
              <li>• 加速比: 2.0-2.1x vs FP16</li>
              <li>• 适用: ChatBot、API 服务</li>
            </ul>
          </div>

          <div className="p-4 bg-blue-50 rounded-lg border border-blue-200">
            <div className="font-bold text-blue-800 mb-2">📊 批处理 (Batch≥8)</div>
            <ul className="text-sm text-blue-700 space-y-1">
              <li>• 首选: <strong>AWQ 4-bit</strong> (最高吞吐)</li>
              <li>• 次选: GPTQ 4-bit</li>
              <li>• 加速比: 2.5-2.7x vs FP16</li>
              <li>• 适用: 批量翻译、数据标注</li>
            </ul>
          </div>
        </div>

        <div className="mt-4 p-4 bg-gradient-to-r from-green-50 to-emerald-50 rounded-lg border border-green-200">
          <div className="font-bold text-green-800 mb-2">💡 关键洞察</div>
          <ul className="text-sm text-green-700 space-y-1">
            <li>• AWQ 在所有 batch size 下都是最快的量化方法</li>
            <li>• INT8 精度更高，但速度略慢于 4-bit</li>
            <li>• bitsandbytes 速度最慢，但支持 QLoRA 微调</li>
            <li>• Batch size 越大，量化带来的加速比越明显</li>
          </ul>
        </div>
      </div>

      <div className="mt-4 text-xs text-slate-500 text-center">
        💡 悬停柱状图查看详细指标 | 测试环境: LLaMA-7B on RTX 4090
      </div>
    </div>
  )
}
