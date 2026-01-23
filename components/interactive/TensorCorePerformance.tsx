'use client'

import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { Cpu, Zap, TrendingUp } from 'lucide-react'

export default function TensorCorePerformance() {
  const [selectedGPU, setSelectedGPU] = useState<'V100' | 'A100' | 'H100'>('A100')

  const gpuData = {
    V100: {
      name: 'NVIDIA V100 (Volta)',
      year: 2017,
      arch: 'Volta',
      fp32: 15.7,
      fp16: 125,
      bf16: 0,  // 不支持
      fp8: 0,   // 不支持
      memory: '32GB HBM2',
      bandwidth: '900 GB/s',
      speedup: {
        fp16: 8.0,
        bf16: 0,
        fp8: 0,
      },
    },
    A100: {
      name: 'NVIDIA A100 (Ampere)',
      year: 2020,
      arch: 'Ampere',
      fp32: 19.5,
      fp16: 312,
      bf16: 312,
      fp8: 0,  // A100 不支持 FP8
      memory: '80GB HBM2e',
      bandwidth: '2039 GB/s',
      speedup: {
        fp16: 16.0,
        bf16: 16.0,
        fp8: 0,
      },
    },
    H100: {
      name: 'NVIDIA H100 (Hopper)',
      year: 2022,
      arch: 'Hopper',
      fp32: 67,
      fp16: 1979,
      bf16: 1979,
      fp8: 3958,
      memory: '80GB HBM3',
      bandwidth: '3350 GB/s',
      speedup: {
        fp16: 29.5,
        bf16: 29.5,
        fp8: 59.1,
      },
    },
  }

  const current = gpuData[selectedGPU]

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-cyan-50 rounded-xl shadow-lg">
      <div className="flex items-center gap-3 mb-6">
        <Cpu className="w-8 h-8 text-cyan-600" />
        <h3 className="text-2xl font-bold text-slate-800">Tensor Core 性能对比</h3>
      </div>

      {/* GPU 选择 */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        {Object.entries(gpuData).map(([key, gpu]) => (
          <button
            key={key}
            onClick={() => setSelectedGPU(key as any)}
            className={`p-4 rounded-lg border-2 transition-all ${
              selectedGPU === key
                ? 'border-cyan-600 bg-cyan-50 shadow-lg'
                : 'border-slate-200 bg-white hover:border-cyan-300'
            }`}
          >
            <div className={`font-bold ${
              selectedGPU === key ? 'text-cyan-900' : 'text-slate-700'
            }`}>
              {gpu.arch}
            </div>
            <div className="text-xs text-slate-600 mt-1">{gpu.year}</div>
          </button>
        ))}
      </div>

      {/* GPU 详情 */}
      <motion.div
        key={selectedGPU}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="space-y-6"
      >
        {/* 基本信息 */}
        <div className="bg-white p-6 rounded-lg shadow-lg">
          <h4 className="text-xl font-bold text-cyan-900 mb-4">{current.name}</h4>
          <div className="grid grid-cols-3 gap-4">
            <div>
              <div className="text-sm text-slate-600">架构</div>
              <div className="text-lg font-bold text-cyan-600">{current.arch}</div>
            </div>
            <div>
              <div className="text-sm text-slate-600">显存</div>
              <div className="text-lg font-bold text-blue-600">{current.memory}</div>
            </div>
            <div>
              <div className="text-sm text-slate-600">带宽</div>
              <div className="text-lg font-bold text-green-600">{current.bandwidth}</div>
            </div>
          </div>
        </div>

        {/* 性能对比柱状图 */}
        <div className="bg-white p-6 rounded-lg shadow-lg">
          <h4 className="font-bold text-slate-800 mb-4">计算性能（TFLOPS）</h4>
          
          <div className="space-y-4">
            {/* FP32 */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-semibold text-slate-700">FP32 (Float32)</span>
                <span className="font-mono text-lg font-bold text-blue-600">
                  {current.fp32} TFLOPS
                </span>
              </div>
              <div className="h-8 bg-slate-100 rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${(current.fp32 / 4000) * 100}%` }}
                  className="h-full bg-blue-500 flex items-center justify-end pr-2"
                >
                  <span className="text-white text-xs font-bold">基准</span>
                </motion.div>
              </div>
            </div>

            {/* FP16 */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-semibold text-slate-700">FP16 (Half)</span>
                <span className="font-mono text-lg font-bold text-orange-600">
                  {current.fp16} TFLOPS
                  <span className="text-sm text-green-600 ml-2">
                    ({current.speedup.fp16.toFixed(1)}x)
                  </span>
                </span>
              </div>
              <div className="h-8 bg-slate-100 rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${(current.fp16 / 4000) * 100}%` }}
                  className="h-full bg-orange-500 flex items-center justify-end pr-2"
                >
                  <Zap className="w-4 h-4 text-white" />
                </motion.div>
              </div>
            </div>

            {/* BF16 */}
            {current.bf16 > 0 && (
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-semibold text-slate-700">BF16 (BFloat16)</span>
                  <span className="font-mono text-lg font-bold text-green-600">
                    {current.bf16} TFLOPS
                    <span className="text-sm text-green-600 ml-2">
                      ({current.speedup.bf16.toFixed(1)}x)
                    </span>
                  </span>
                </div>
                <div className="h-8 bg-slate-100 rounded-full overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${(current.bf16 / 4000) * 100}%` }}
                    className="h-full bg-green-500 flex items-center justify-end pr-2"
                  >
                    <Zap className="w-4 h-4 text-white" />
                  </motion.div>
                </div>
              </div>
            )}

            {/* FP8 */}
            {current.fp8 > 0 && (
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-semibold text-slate-700">FP8 (新一代)</span>
                  <span className="font-mono text-lg font-bold text-purple-600">
                    {current.fp8} TFLOPS
                    <span className="text-sm text-purple-600 ml-2">
                      ({current.speedup.fp8.toFixed(1)}x)
                    </span>
                  </span>
                </div>
                <div className="h-8 bg-slate-100 rounded-full overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${(current.fp8 / 4000) * 100}%` }}
                    className="h-full bg-purple-500 flex items-center justify-end pr-2"
                  >
                    <TrendingUp className="w-4 h-4 text-white" />
                  </motion.div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* 实际训练性能 */}
        <div className="bg-white p-6 rounded-lg shadow-lg">
          <h4 className="font-bold text-slate-800 mb-4">
            实际训练性能（BERT-Large，batch=32）
          </h4>
          <div className="grid grid-cols-2 gap-4">
            <div className="p-4 bg-blue-50 rounded border border-blue-200">
              <div className="text-sm text-slate-600 mb-1">FP32 速度</div>
              <div className="text-2xl font-bold text-blue-600">
                {selectedGPU === 'V100' ? '45' : selectedGPU === 'A100' ? '120' : '280'} samples/s
              </div>
              <div className="text-xs text-slate-500 mt-1">基准性能</div>
            </div>

            <div className="p-4 bg-green-50 rounded border border-green-200">
              <div className="text-sm text-slate-600 mb-1">
                {current.bf16 > 0 ? 'BF16' : 'FP16'} 速度
              </div>
              <div className="text-2xl font-bold text-green-600">
                {selectedGPU === 'V100' ? '95' : selectedGPU === 'A100' ? '280' : '650'} samples/s
              </div>
              <div className="text-xs text-green-600 mt-1 font-semibold">
                {selectedGPU === 'V100' ? '2.1x' : selectedGPU === 'A100' ? '2.3x' : '2.3x'} 加速
              </div>
            </div>
          </div>
        </div>

        {/* 架构特性 */}
        <div className="bg-gradient-to-r from-cyan-50 to-blue-50 p-6 rounded-lg border-2 border-cyan-300">
          <h4 className="font-bold text-cyan-900 mb-3">
            {current.arch} 架构亮点
          </h4>
          <ul className="text-sm text-slate-700 space-y-2">
            {selectedGPU === 'V100' && (
              <>
                <li>✓ 首个支持 Tensor Core 的架构</li>
                <li>✓ FP16 Tensor Core 提供 125 TFLOPS</li>
                <li>⚠️ 不支持 BF16（需手动 loss scaling）</li>
                <li>📅 适用于旧项目/预算有限场景</li>
              </>
            )}
            {selectedGPU === 'A100' && (
              <>
                <li>✓ 引入 BF16 支持（与 FP32 同范围）</li>
                <li>✓ 第三代 Tensor Core（312 TFLOPS）</li>
                <li>✓ 80GB HBM2e 大显存（训练 LLaMA-65B）</li>
                <li>🏆 当前深度学习主力 GPU（2024）</li>
              </>
            )}
            {selectedGPU === 'H100' && (
              <>
                <li>✓ 第四代 Tensor Core（1979 TFLOPS BF16）</li>
                <li>✓ 首个支持 FP8（3958 TFLOPS！）</li>
                <li>✓ HBM3 超高带宽（3.35 TB/s）</li>
                <li>🚀 超大模型（GPT-4 规模）训练首选</li>
              </>
            )}
          </ul>
        </div>
      </motion.div>

      {/* 总结 */}
      <div className="mt-6 p-5 bg-yellow-50 border border-yellow-300 rounded-lg">
        <h5 className="font-bold text-yellow-800 mb-2">性能要点</h5>
        <div className="text-sm text-slate-700 space-y-1">
          <p>• <strong>Tensor Core</strong> 是混合精度加速的硬件基础</p>
          <p>• FP16/BF16 在现代 GPU 上可获得 <strong>2-16倍</strong> 加速</p>
          <p>• BF16 在 Ampere/Hopper 上与 FP16 性能相同，但<strong>更稳定</strong></p>
          <p>• 显存带宽同样重要：H100 比 V100 快 <strong>3.7倍</strong></p>
        </div>
      </div>
    </div>
  )
}
