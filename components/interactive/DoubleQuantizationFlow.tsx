'use client'

import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { Layers, ArrowDown, Database } from 'lucide-react'

export default function DoubleQuantizationFlow() {
  const [step, setStep] = useState(0)

  const blockSize = 64
  const numBlocks = 4

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-orange-50 rounded-xl shadow-lg">
      <div className="flex items-center gap-3 mb-6">
        <Layers className="w-8 h-8 text-orange-600" />
        <h3 className="text-2xl font-bold text-slate-800">双重量化 (Double Quantization)</h3>
      </div>

      {/* 步骤控制 */}
      <div className="flex gap-2 mb-6">
        {['标准量化', '提取常数', '量化常数', '最终存储'].map((label, idx) => (
          <button
            key={idx}
            onClick={() => setStep(idx)}
            className={`flex-1 p-3 rounded-lg border-2 transition-all ${
              step === idx
                ? 'border-orange-600 bg-orange-50 shadow-md'
                : 'border-slate-200 bg-white hover:border-orange-300'
            }`}
          >
            <div className={`text-sm font-bold ${step === idx ? 'text-orange-900' : 'text-slate-700'}`}>
              Step {idx + 1}
            </div>
            <div className="text-xs text-slate-600 mt-1">{label}</div>
          </button>
        ))}
      </div>

      {/* 可视化 */}
      <div className="space-y-6">
        {/* Step 0: 标准量化 */}
        {step >= 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-white p-5 rounded-lg shadow-lg border-2 border-blue-200"
          >
            <h4 className="font-bold text-blue-800 mb-3 flex items-center gap-2">
              <Database className="w-5 h-5" />
              Step 1: 标准 NF4 量化
            </h4>

            <div className="grid grid-cols-4 gap-3">
              {Array.from({ length: numBlocks }).map((_, blockIdx) => (
                <div key={blockIdx} className="p-3 bg-blue-50 rounded border border-blue-300">
                  <div className="text-xs text-slate-600 mb-2 font-bold">
                    Block {blockIdx + 1} ({blockSize} params)
                  </div>
                  
                  {/* 权重 */}
                  <div className="mb-2">
                    <div className="text-xs text-slate-500 mb-1">4-bit 权重</div>
                    <div className="grid grid-cols-8 gap-0.5">
                      {Array.from({ length: 8 }).map((_, i) => (
                        <div key={i} className="h-4 bg-blue-400 rounded-sm" />
                      ))}
                    </div>
                    <div className="text-xs text-slate-500 mt-1">
                      {blockSize} × 4 bit = {blockSize * 4} bits
                    </div>
                  </div>

                  {/* absmax (FP32) */}
                  <div className="p-2 bg-red-100 border border-red-300 rounded">
                    <div className="text-xs text-slate-600">absmax (FP32)</div>
                    <div className="font-mono text-xs text-red-700">
                      {(0.5 + blockIdx * 0.1).toFixed(4)}
                    </div>
                    <div className="text-xs text-red-600 font-bold mt-1">32 bits</div>
                  </div>
                </div>
              ))}
            </div>

            <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded">
              <div className="font-bold text-red-800 mb-1">问题</div>
              <div className="text-sm text-slate-700">
                每个 block 存储 1 个 FP32 absmax，额外开销 = 32 / {blockSize} = <strong>0.5 bits/param</strong>
                <br />
                LLaMA-65B (65B params) 额外需要: 65 × 10⁹ × 0.5 / 8 ≈ <strong>4 GB</strong>
              </div>
            </div>
          </motion.div>
        )}

        {/* Step 1: 提取所有 absmax */}
        {step >= 1 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="bg-white p-5 rounded-lg shadow-lg border-2 border-purple-200"
          >
            <div className="flex items-center justify-center mb-4">
              <ArrowDown className="w-8 h-8 text-purple-600" />
            </div>

            <h4 className="font-bold text-purple-800 mb-3">
              Step 2: 提取所有量化常数
            </h4>

            <div className="flex items-center justify-center gap-4">
              {Array.from({ length: numBlocks }).map((_, idx) => (
                <motion.div
                  key={idx}
                  initial={{ opacity: 0, scale: 0.5 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 0.3 + idx * 0.1 }}
                  className="p-3 bg-purple-100 border border-purple-300 rounded"
                >
                  <div className="text-xs text-slate-600 mb-1">absmax_{idx + 1}</div>
                  <div className="font-mono text-sm text-purple-700 font-bold">
                    {(0.5 + idx * 0.1).toFixed(4)}
                  </div>
                  <div className="text-xs text-purple-600 mt-1">FP32</div>
                </motion.div>
              ))}
            </div>

            <div className="mt-4 text-center text-sm text-slate-700">
              收集所有 {numBlocks} 个 blocks 的 absmax 常数
            </div>
          </motion.div>
        )}

        {/* Step 2: 量化 absmax */}
        {step >= 2 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="bg-white p-5 rounded-lg shadow-lg border-2 border-green-200"
          >
            <div className="flex items-center justify-center mb-4">
              <ArrowDown className="w-8 h-8 text-green-600" />
            </div>

            <h4 className="font-bold text-green-800 mb-3">
              Step 3: 量化这些常数本身（FP32 → FP8）
            </h4>

            <div className="grid grid-cols-2 gap-6">
              {/* 量化前 */}
              <div>
                <div className="text-sm text-slate-600 mb-2">量化前 (FP32)</div>
                <div className="space-y-2">
                  {Array.from({ length: numBlocks }).map((_, idx) => (
                    <div key={idx} className="p-2 bg-purple-50 rounded border border-purple-200">
                      <div className="flex justify-between">
                        <span className="font-mono text-sm">{(0.5 + idx * 0.1).toFixed(4)}</span>
                        <span className="text-xs text-purple-600">32 bits</span>
                      </div>
                    </div>
                  ))}
                </div>
                <div className="mt-2 p-2 bg-red-50 rounded text-xs text-red-700 font-bold">
                  总计: {numBlocks} × 32 = {numBlocks * 32} bits
                </div>
              </div>

              {/* 量化后 */}
              <div>
                <div className="text-sm text-slate-600 mb-2">量化后 (FP8)</div>
                <div className="space-y-2">
                  {Array.from({ length: numBlocks }).map((_, idx) => (
                    <div key={idx} className="p-2 bg-green-50 rounded border border-green-200">
                      <div className="flex justify-between">
                        <span className="font-mono text-sm">{(0.5 + idx * 0.1).toFixed(2)}</span>
                        <span className="text-xs text-green-600">8 bits</span>
                      </div>
                    </div>
                  ))}
                </div>
                <div className="mt-2 p-2 bg-green-100 rounded text-xs text-green-700 font-bold">
                  总计: {numBlocks} × 8 = {numBlocks * 8} bits
                </div>
              </div>
            </div>

            <div className="mt-4 p-3 bg-green-50 border border-green-200 rounded">
              <div className="font-bold text-green-800 mb-1">✓ 优势</div>
              <div className="text-sm text-slate-700">
                节省: ({numBlocks * 32} - {numBlocks * 8}) bits = <strong>{numBlocks * 24} bits</strong>
                <br />
                但需要额外存储 1 个 global_absmax (FP32): +32 bits
                <br />
                净节省: {numBlocks * 24 - 32} bits
              </div>
            </div>
          </motion.div>
        )}

        {/* Step 3: 最终存储 */}
        {step >= 3 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="bg-white p-5 rounded-lg shadow-lg border-2 border-indigo-200"
          >
            <h4 className="font-bold text-indigo-800 mb-3">
              Step 4: 最终存储结构
            </h4>

            <div className="space-y-4">
              {/* 每个 block */}
              {Array.from({ length: numBlocks }).map((_, blockIdx) => (
                <div key={blockIdx} className="p-4 bg-indigo-50 rounded border border-indigo-300">
                  <div className="font-bold text-indigo-800 mb-2">
                    Block {blockIdx + 1}
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <div className="p-2 bg-white rounded border">
                      <div className="text-xs text-slate-600 mb-1">4-bit 权重</div>
                      <div className="h-6 bg-indigo-400 rounded" />
                      <div className="text-xs text-indigo-600 mt-1 font-bold">
                        {blockSize * 4} bits
                      </div>
                    </div>
                    <div className="p-2 bg-white rounded border">
                      <div className="text-xs text-slate-600 mb-1">量化后的 absmax (FP8)</div>
                      <div className="h-6 bg-green-400 rounded" />
                      <div className="text-xs text-green-600 mt-1 font-bold">8 bits</div>
                    </div>
                  </div>
                </div>
              ))}

              {/* Global absmax */}
              <div className="p-4 bg-purple-50 rounded border border-purple-300">
                <div className="font-bold text-purple-800 mb-2">
                  Global absmax (FP32)
                </div>
                <div className="p-2 bg-white rounded border">
                  <div className="text-xs text-slate-600 mb-1">用于反量化 FP8 absmax</div>
                  <div className="h-6 bg-purple-400 rounded" />
                  <div className="text-xs text-purple-600 mt-1 font-bold">32 bits</div>
                </div>
              </div>
            </div>

            <div className="mt-4 grid grid-cols-2 gap-4">
              <div className="p-3 bg-red-50 border border-red-200 rounded">
                <div className="font-bold text-red-800 mb-1">标准量化</div>
                <div className="text-sm text-slate-700">
                  {blockSize * 4} + 32 = <strong>{blockSize * 4 + 32} bits/block</strong>
                  <br />
                  = <strong>{((blockSize * 4 + 32) / blockSize).toFixed(2)} bits/param</strong>
                </div>
              </div>
              <div className="p-3 bg-green-50 border border-green-200 rounded">
                <div className="font-bold text-green-800 mb-1">双重量化</div>
                <div className="text-sm text-slate-700">
                  {blockSize * 4} + 8 + (32/{numBlocks}) = <strong>{(blockSize * 4 + 8 + 32 / numBlocks).toFixed(0)} bits/block</strong>
                  <br />
                  = <strong>{((blockSize * 4 + 8 + 32 / numBlocks) / blockSize).toFixed(2)} bits/param</strong>
                </div>
              </div>
            </div>

            <div className="mt-4 p-4 bg-gradient-to-r from-green-50 to-blue-50 border border-green-300 rounded">
              <div className="font-bold text-green-800 mb-2">🎯 总节省</div>
              <div className="text-sm text-slate-700">
                每个参数节省: {((32 / blockSize) - (8 / blockSize + 32 / numBlocks / blockSize)).toFixed(3)} bits
                <br />
                <strong>LLaMA-65B</strong> (256 个 blocks 时): 65B × {((32 / blockSize) - (8 / blockSize + 32 / 256 / blockSize)).toFixed(3)} / 8 ≈ <strong>~3 GB</strong> 节省
              </div>
            </div>
          </motion.div>
        )}
      </div>
    </div>
  )
}
