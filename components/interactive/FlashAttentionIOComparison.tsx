'use client'

import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { Cpu, HardDrive, Zap, TrendingDown } from 'lucide-react'

export default function FlashAttentionIOComparison() {
  const [seqLen, setSeqLen] = useState(2048)

  // 计算IO次数（简化）
  const standardIO = (seqLen * seqLen * 3) / 1000 // QK^T, Softmax, @V (单位: K operations)
  const flashIO = (seqLen * 2) / 1000 // 只读写一次 (单位: K operations)
  const speedup = standardIO / flashIO

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-orange-50 rounded-xl shadow-lg">
      <div className="flex items-center gap-3 mb-6">
        <Zap className="w-8 h-8 text-orange-600" />
        <h3 className="text-2xl font-bold text-slate-800">Flash Attention IO 优化原理</h3>
      </div>

      {/* 序列长度滑块 */}
      <div className="mb-6 p-4 bg-white rounded-lg shadow">
        <label className="block text-sm font-medium text-slate-700 mb-2">
          序列长度: {seqLen}
        </label>
        <input
          type="range"
          min="512"
          max="8192"
          step="512"
          value={seqLen}
          onChange={(e) => setSeqLen(Number(e.target.value))}
          className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer"
        />
        <div className="flex justify-between text-xs text-slate-500 mt-1">
          <span>512</span>
          <span>2048</span>
          <span>4096</span>
          <span>8192</span>
        </div>
      </div>

      {/* 对比可视化 */}
      <div className="grid grid-cols-2 gap-6 mb-6">
        {/* 标准Attention */}
        <div className="bg-white p-6 rounded-lg shadow-lg border-2 border-red-200">
          <h4 className="font-bold text-red-800 mb-4 flex items-center gap-2">
            <HardDrive className="w-5 h-5" />
            标准 Attention (慢)
          </h4>

          <div className="space-y-3">
            {/* 步骤1 */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.1 }}
              className="p-3 bg-red-50 rounded border border-red-300"
            >
              <div className="flex items-center gap-2 mb-2">
                <div className="w-6 h-6 bg-red-500 rounded-full text-white flex items-center justify-center text-xs font-bold">
                  1
                </div>
                <div className="text-sm font-bold text-red-800">计算 QK^T</div>
              </div>
              <div className="text-xs text-slate-600 mb-1">从 SRAM 读取 Q, K</div>
              <div className="flex gap-1">
                <div className="flex-1 h-2 bg-blue-300 rounded" />
                <div className="flex-1 h-2 bg-green-300 rounded" />
              </div>
              <div className="text-xs text-red-700 mt-1 font-bold">
                → 写入 HBM ({(seqLen * seqLen / 1000).toFixed(1)}K 元素)
              </div>
            </motion.div>

            {/* 步骤2 */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.3 }}
              className="p-3 bg-orange-50 rounded border border-orange-300"
            >
              <div className="flex items-center gap-2 mb-2">
                <div className="w-6 h-6 bg-orange-500 rounded-full text-white flex items-center justify-center text-xs font-bold">
                  2
                </div>
                <div className="text-sm font-bold text-orange-800">计算 Softmax</div>
              </div>
              <div className="text-xs text-slate-600 mb-1">从 HBM 读取 QK^T (慢!)</div>
              <div className="h-2 bg-yellow-300 rounded" />
              <div className="text-xs text-orange-700 mt-1 font-bold">
                → 写回 HBM ({(seqLen * seqLen / 1000).toFixed(1)}K 元素)
              </div>
            </motion.div>

            {/* 步骤3 */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.5 }}
              className="p-3 bg-red-50 rounded border border-red-300"
            >
              <div className="flex items-center gap-2 mb-2">
                <div className="w-6 h-6 bg-red-500 rounded-full text-white flex items-center justify-center text-xs font-bold">
                  3
                </div>
                <div className="text-sm font-bold text-red-800">计算 Softmax @ V</div>
              </div>
              <div className="text-xs text-slate-600 mb-1">从 HBM 读取 Softmax 和 V (慢!)</div>
              <div className="flex gap-1">
                <div className="flex-1 h-2 bg-yellow-300 rounded" />
                <div className="flex-1 h-2 bg-purple-300 rounded" />
              </div>
              <div className="text-xs text-red-700 mt-1 font-bold">
                → 输出
              </div>
            </motion.div>
          </div>

          <div className="mt-4 p-3 bg-red-100 rounded">
            <div className="text-sm font-bold text-red-800 mb-1">
              总 IO 操作: {standardIO.toFixed(1)}K
            </div>
            <div className="text-xs text-slate-600">
              多次 HBM 读写 (慢，带宽瓶颈)
            </div>
          </div>
        </div>

        {/* Flash Attention */}
        <div className="bg-white p-6 rounded-lg shadow-lg border-2 border-green-200">
          <h4 className="font-bold text-green-800 mb-4 flex items-center gap-2">
            <Cpu className="w-5 h-5" />
            Flash Attention (快)
          </h4>

          <div className="space-y-3">
            {/* 分块处理 */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
              className="p-3 bg-green-50 rounded border border-green-300"
            >
              <div className="flex items-center gap-2 mb-2">
                <div className="w-6 h-6 bg-green-500 rounded-full text-white flex items-center justify-center text-xs font-bold">
                  1
                </div>
                <div className="text-sm font-bold text-green-800">分块加载到 SRAM</div>
              </div>
              <div className="text-xs text-slate-600 mb-1">Q, K, V 分块 (Tiling)</div>
              <div className="grid grid-cols-4 gap-1">
                {Array.from({ length: 4 }).map((_, i) => (
                  <div key={i} className="h-8 bg-green-300 rounded flex items-center justify-center text-xs">
                    Block {i + 1}
                  </div>
                ))}
              </div>
            </motion.div>

            {/* 融合计算 */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.4 }}
              className="p-3 bg-blue-50 rounded border border-blue-300"
            >
              <div className="flex items-center gap-2 mb-2">
                <div className="w-6 h-6 bg-blue-500 rounded-full text-white flex items-center justify-center text-xs font-bold">
                  2
                </div>
                <div className="text-sm font-bold text-blue-800">SRAM 中融合计算</div>
              </div>
              <div className="text-xs text-slate-600 mb-2">QK^T + Softmax + @V 一次完成</div>
              <div className="p-2 bg-blue-100 rounded font-mono text-xs">
                <div>for block in blocks:</div>
                <div className="ml-4">S = Q @ K^T  # SRAM</div>
                <div className="ml-4">P = softmax(S)  # SRAM</div>
                <div className="ml-4">O += P @ V  # SRAM</div>
              </div>
              <div className="text-xs text-green-700 mt-1 font-bold">
                ✓ 中间结果不写 HBM
              </div>
            </motion.div>

            {/* 写回 */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.6 }}
              className="p-3 bg-green-50 rounded border border-green-300"
            >
              <div className="flex items-center gap-2 mb-2">
                <div className="w-6 h-6 bg-green-500 rounded-full text-white flex items-center justify-center text-xs font-bold">
                  3
                </div>
                <div className="text-sm font-bold text-green-800">写回最终结果</div>
              </div>
              <div className="h-2 bg-green-400 rounded" />
              <div className="text-xs text-green-700 mt-1">
                只写一次 HBM ({seqLen / 1000}K 元素)
              </div>
            </motion.div>
          </div>

          <div className="mt-4 p-3 bg-green-100 rounded">
            <div className="text-sm font-bold text-green-800 mb-1">
              总 IO 操作: {flashIO.toFixed(1)}K
            </div>
            <div className="text-xs text-slate-600">
              最小化 HBM 访问 (快，计算密集)
            </div>
          </div>
        </div>
      </div>

      {/* 性能对比 */}
      <div className="bg-white p-6 rounded-lg shadow">
        <h4 className="font-bold text-slate-800 mb-4">性能提升</h4>
        
        <div className="grid grid-cols-3 gap-4 mb-4">
          <div className="p-4 bg-red-50 rounded">
            <div className="text-sm text-slate-600 mb-1">标准 Attention IO</div>
            <div className="text-2xl font-bold text-red-600">{standardIO.toFixed(1)}K</div>
          </div>
          <div className="p-4 bg-green-50 rounded">
            <div className="text-sm text-slate-600 mb-1">Flash Attention IO</div>
            <div className="text-2xl font-bold text-green-600">{flashIO.toFixed(1)}K</div>
          </div>
          <div className="p-4 bg-blue-50 rounded">
            <div className="text-sm text-slate-600 mb-1">加速比</div>
            <div className="text-2xl font-bold text-blue-600">{speedup.toFixed(1)}x</div>
          </div>
        </div>

        {/* IO复杂度对比 */}
        <div className="p-4 bg-slate-50 rounded">
          <div className="text-sm font-bold text-slate-800 mb-2">IO 复杂度</div>
          <div className="flex items-center gap-4">
            <div className="flex-1">
              <div className="text-xs text-slate-600 mb-1">标准 Attention</div>
              <div className="font-mono text-sm text-red-600">O(N²)</div>
            </div>
            <TrendingDown className="w-5 h-5 text-green-600" />
            <div className="flex-1">
              <div className="text-xs text-slate-600 mb-1">Flash Attention</div>
              <div className="font-mono text-sm text-green-600">O(N)</div>
            </div>
          </div>
        </div>

        <div className="mt-4 p-3 bg-orange-50 border border-orange-200 rounded text-sm">
          <div className="font-bold text-orange-800 mb-1">核心创新</div>
          <ul className="space-y-1 text-slate-700">
            <li>• <strong>Tiling</strong>: 分块处理，适配 SRAM 大小</li>
            <li>• <strong>Kernel Fusion</strong>: 多步操作融合，减少 HBM 访问</li>
            <li>• <strong>Recomputation</strong>: 反向传播时重算 Softmax（换空间）</li>
          </ul>
        </div>
      </div>
    </div>
  )
}
