'use client'

import React, { useState } from 'react'
import { motion } from 'framer-motion'

type MaskType = 'none' | 'padding' | 'causal' | 'combined'

export default function MaskBuilder() {
  const [seqLen, setSeqLen] = useState(8)
  const [maskType, setMaskType] = useState<MaskType>('none')
  const [paddingPositions, setPaddingPositions] = useState<Set<number>>(new Set([6, 7]))
  const [customMask, setCustomMask] = useState<boolean[][]>([])
  const [isDrawing, setIsDrawing] = useState(false)

  // 生成 mask
  const generateMask = (): boolean[][] => {
    const mask: boolean[][] = Array(seqLen).fill(null).map(() => Array(seqLen).fill(true))

    switch (maskType) {
      case 'none':
        return mask

      case 'padding':
        // Padding Mask: 屏蔽 padding 位置的列
        for (let i = 0; i < seqLen; i++) {
          for (let j = 0; j < seqLen; j++) {
            if (paddingPositions.has(j)) {
              mask[i][j] = false
            }
          }
        }
        break

      case 'causal':
        // Causal Mask: 下三角矩阵
        for (let i = 0; i < seqLen; i++) {
          for (let j = i + 1; j < seqLen; j++) {
            mask[i][j] = false
          }
        }
        break

      case 'combined':
        // Combined: Causal + Padding
        for (let i = 0; i < seqLen; i++) {
          for (let j = 0; j < seqLen; j++) {
            // Causal
            if (j > i) {
              mask[i][j] = false
            }
            // Padding
            if (paddingPositions.has(j)) {
              mask[i][j] = false
            }
          }
        }
        break
    }

    return mask
  }

  const mask = generateMask()

  // 生成 PyTorch 代码
  const generateCode = () => {
    let code = ""

    if (maskType === 'padding') {
      code = `# Padding Mask
def create_padding_mask(seq):
    """
    seq: [batch_size, seq_len]
    返回: [batch_size, 1, 1, seq_len]
    """
    # PAD token ID = 0
    mask = (seq != 0).unsqueeze(1).unsqueeze(2)
    return mask

# 示例
seq = torch.tensor([[1, 2, 3, 0, 0]])
mask = create_padding_mask(seq)
# mask[0, 0, 0] = [True, True, True, False, False]`
    } else if (maskType === 'causal') {
      code = `# Causal Mask (下三角矩阵)
def create_causal_mask(seq_len):
    """
    返回: [1, 1, seq_len, seq_len]
    """
    mask = torch.tril(torch.ones(seq_len, seq_len))
    mask = mask.unsqueeze(0).unsqueeze(0)
    return mask

# 示例
mask = create_causal_mask(${seqLen})
# 形状: [1, 1, ${seqLen}, ${seqLen}]`
    } else if (maskType === 'combined') {
      code = `# Combined Mask (Causal + Padding)
def create_combined_mask(tgt_seq):
    """
    tgt_seq: [batch_size, tgt_len]
    """
    tgt_len = tgt_seq.size(1)
    
    # 1. Causal Mask
    causal = torch.tril(torch.ones(tgt_len, tgt_len))
    causal = causal.unsqueeze(0).unsqueeze(1)
    
    # 2. Padding Mask
    padding = (tgt_seq != 0).unsqueeze(1).unsqueeze(2)
    
    # 3. 组合 (逻辑与)
    combined = causal & padding
    
    return combined

# 示例
tgt_seq = torch.tensor([[1, 2, 3, 0, 0]])
mask = create_combined_mask(tgt_seq)`
    } else {
      code = `# 无 Mask
# 所有位置都可以互相关注`
    }

    return code
  }

  const togglePadding = (pos: number) => {
    const newPadding = new Set(paddingPositions)
    if (newPadding.has(pos)) {
      newPadding.delete(pos)
    } else {
      newPadding.add(pos)
    }
    setPaddingPositions(newPadding)
  }

  const presets = [
    { name: '编码器 Mask', type: 'padding' as MaskType, padding: new Set<number>([6, 7]) },
    { name: '解码器 Mask', type: 'causal' as MaskType, padding: new Set<number>() },
    { name: '组合 Mask', type: 'combined' as MaskType, padding: new Set<number>([6, 7]) },
    { name: 'Prefix LM', type: 'combined' as MaskType, padding: new Set<number>([5, 6, 7]) },
  ]

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-indigo-50 rounded-xl border border-slate-200">
      {/* 标题 */}
      <div className="text-center mb-6">
        <h3 className="text-2xl font-bold text-slate-800 mb-2">
          🎨 Attention Mask 构建器
        </h3>
        <p className="text-slate-600">
          交互式构建和可视化不同类型的 Attention Mask
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* 左侧：配置 */}
        <div className="space-y-4">
          {/* Mask 类型选择 */}
          <div className="bg-white rounded-lg border border-slate-200 p-5">
            <h4 className="text-lg font-semibold text-slate-800 mb-4">
              🔧 Mask 类型
            </h4>
            <div className="grid grid-cols-2 gap-3">
              {[
                { type: 'none' as MaskType, name: '无 Mask', icon: '⬜', desc: '全连接' },
                { type: 'padding' as MaskType, name: 'Padding', icon: '🚫', desc: '屏蔽 PAD' },
                { type: 'causal' as MaskType, name: 'Causal', icon: '◣', desc: '下三角' },
                { type: 'combined' as MaskType, name: 'Combined', icon: '🔀', desc: 'Causal+Padding' },
              ].map((item) => (
                <button
                  key={item.type}
                  onClick={() => setMaskType(item.type)}
                  className={`p-4 rounded-lg border-2 transition-all ${
                    maskType === item.type
                      ? 'border-blue-600 bg-blue-50'
                      : 'border-slate-200 bg-white hover:bg-slate-50'
                  }`}
                >
                  <div className="text-3xl mb-1">{item.icon}</div>
                  <div className="font-semibold text-slate-800">{item.name}</div>
                  <div className="text-xs text-slate-500 mt-1">{item.desc}</div>
                </button>
              ))}
            </div>
          </div>

          {/* 序列长度 */}
          <div className="bg-white rounded-lg border border-slate-200 p-5">
            <h4 className="text-lg font-semibold text-slate-800 mb-4">
              📏 序列长度：{seqLen}
            </h4>
            <input
              type="range"
              min="4"
              max="16"
              value={seqLen}
              onChange={(e) => {
                const newLen = Number(e.target.value)
                setSeqLen(newLen)
                // 调整 padding 位置
                setPaddingPositions(new Set(
                  Array.from(paddingPositions).filter(p => p < newLen)
                ))
              }}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-slate-500 mt-2">
              <span>4</span>
              <span>8</span>
              <span>12</span>
              <span>16</span>
            </div>
          </div>

          {/* Padding 位置配置 */}
          {(maskType === 'padding' || maskType === 'combined') && (
            <div className="bg-white rounded-lg border border-slate-200 p-5">
              <h4 className="text-lg font-semibold text-slate-800 mb-4">
                🚫 选择 Padding 位置
              </h4>
              <div className="flex flex-wrap gap-2">
                {Array.from({ length: seqLen }, (_, i) => (
                  <button
                    key={i}
                    onClick={() => togglePadding(i)}
                    className={`w-12 h-12 rounded-lg font-bold transition-all ${
                      paddingPositions.has(i)
                        ? 'bg-red-600 text-white shadow-lg scale-105'
                        : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
                    }`}
                  >
                    {i}
                  </button>
                ))}
              </div>
              <div className="mt-3 text-sm text-slate-600">
                已选择: {paddingPositions.size > 0 
                  ? Array.from(paddingPositions).sort((a, b) => a - b).join(', ')
                  : '无'}
              </div>
            </div>
          )}

          {/* 预设模板 */}
          <div className="bg-white rounded-lg border border-slate-200 p-5">
            <h4 className="text-lg font-semibold text-slate-800 mb-4">
              ⚡ 快速预设
            </h4>
            <div className="grid grid-cols-2 gap-2">
              {presets.map((preset, i) => (
                <button
                  key={i}
                  onClick={() => {
                    setMaskType(preset.type)
                    setPaddingPositions(new Set(preset.padding))
                  }}
                  className="px-3 py-2 text-sm bg-gradient-to-r from-blue-50 to-purple-50 border border-blue-200 rounded-lg hover:from-blue-100 hover:to-purple-100 transition-colors"
                >
                  {preset.name}
                </button>
              ))}
            </div>
          </div>

          {/* 说明 */}
          <div className="bg-amber-50 rounded-lg border border-amber-200 p-5">
            <h4 className="text-sm font-semibold text-amber-900 mb-2">
              💡 Mask 说明
            </h4>
            <ul className="text-xs text-amber-800 space-y-1">
              {maskType === 'none' && (
                <li>• 无 Mask：所有位置都可以互相关注（全连接）</li>
              )}
              {maskType === 'padding' && (
                <>
                  <li>• <strong>Padding Mask</strong>：用于编码器</li>
                  <li>• 屏蔽 [PAD] token，防止模型关注填充内容</li>
                  <li>• 实现：将 padding 位置的列全部屏蔽</li>
                </>
              )}
              {maskType === 'causal' && (
                <>
                  <li>• <strong>Causal Mask</strong>：用于解码器</li>
                  <li>• 下三角矩阵，防止看到未来信息</li>
                  <li>• 位置 i 只能关注位置 0 到 i（自回归）</li>
                </>
              )}
              {maskType === 'combined' && (
                <>
                  <li>• <strong>Combined Mask</strong>：Causal + Padding</li>
                  <li>• 同时满足自回归和屏蔽 padding</li>
                  <li>• 用于解码器处理变长序列</li>
                </>
              )}
            </ul>
          </div>
        </div>

        {/* 右侧：可视化 + 代码 */}
        <div className="space-y-4">
          {/* Mask 可视化 */}
          <div className="bg-white rounded-lg border border-slate-200 p-5">
            <h4 className="text-lg font-semibold text-slate-800 mb-4">
              👁️ Mask 矩阵可视化
            </h4>
            
            <div className="overflow-x-auto">
              <div className="inline-block min-w-full">
                {/* 列标题 */}
                <div className="flex mb-1">
                  <div className="w-12 h-8" />
                  {Array.from({ length: seqLen }, (_, i) => (
                    <div
                      key={i}
                      className={`w-12 h-8 flex items-center justify-center text-xs font-medium ${
                        paddingPositions.has(i) ? 'text-red-600' : 'text-slate-600'
                      }`}
                    >
                      {i}
                    </div>
                  ))}
                </div>

                {/* Mask 矩阵 */}
                {mask.map((row, i) => (
                  <div key={i} className="flex">
                    {/* 行标题 */}
                    <div className="w-12 h-12 flex items-center justify-center text-xs font-medium text-slate-600">
                      {i}
                    </div>
                    
                    {/* Mask cells */}
                    {row.map((isAttended, j) => (
                      <motion.div
                        key={j}
                        className={`w-12 h-12 border border-slate-300 flex items-center justify-center text-xs font-bold ${
                          isAttended
                            ? 'bg-green-100 text-green-700'
                            : 'bg-red-100 text-red-700'
                        }`}
                        initial={{ opacity: 0, scale: 0.8 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ delay: (i + j) * 0.01 }}
                      >
                        {isAttended ? '✓' : '✗'}
                      </motion.div>
                    ))}
                  </div>
                ))}
              </div>
            </div>

            {/* 图例 */}
            <div className="mt-4 flex items-center gap-4 text-sm">
              <div className="flex items-center gap-2">
                <div className="w-6 h-6 bg-green-100 border border-slate-300 rounded flex items-center justify-center text-green-700 font-bold">
                  ✓
                </div>
                <span className="text-slate-700">可关注</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-6 h-6 bg-red-100 border border-slate-300 rounded flex items-center justify-center text-red-700 font-bold">
                  ✗
                </div>
                <span className="text-slate-700">已屏蔽</span>
              </div>
            </div>
          </div>

          {/* 关注模式 */}
          <div className="bg-blue-50 rounded-lg border border-blue-200 p-5">
            <h4 className="text-lg font-semibold text-blue-900 mb-3">
              📍 各位置可关注范围
            </h4>
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {mask.map((row, i) => {
                const attendedPositions = row
                  .map((isAttended, j) => isAttended ? j : -1)
                  .filter(j => j !== -1)
                
                return (
                  <div key={i} className="text-sm">
                    <span className="font-semibold text-blue-900">位置 {i}:</span>
                    <span className="text-blue-700 ml-2">
                      [{attendedPositions.join(', ')}]
                    </span>
                    <span className="text-blue-600 ml-2 text-xs">
                      ({attendedPositions.length} 个位置)
                    </span>
                  </div>
                )
              })}
            </div>
          </div>

          {/* 代码生成 */}
          <div className="bg-slate-900 rounded-lg p-5 text-white">
            <h4 className="text-lg font-semibold mb-3">
              💻 PyTorch 代码
            </h4>
            <pre className="text-xs overflow-x-auto">
              <code className="text-green-400">{generateCode()}</code>
            </pre>
          </div>

          {/* 统计信息 */}
          <div className="bg-white rounded-lg border border-slate-200 p-5">
            <h4 className="text-lg font-semibold text-slate-800 mb-3">
              📊 Mask 统计
            </h4>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <div className="text-sm text-slate-600">总元素</div>
                <div className="text-2xl font-bold text-slate-800">
                  {seqLen * seqLen}
                </div>
              </div>
              <div>
                <div className="text-sm text-slate-600">可关注</div>
                <div className="text-2xl font-bold text-green-600">
                  {mask.flat().filter(x => x).length}
                </div>
              </div>
              <div>
                <div className="text-sm text-slate-600">已屏蔽</div>
                <div className="text-2xl font-bold text-red-600">
                  {mask.flat().filter(x => !x).length}
                </div>
              </div>
              <div>
                <div className="text-sm text-slate-600">屏蔽比例</div>
                <div className="text-2xl font-bold text-purple-600">
                  {((mask.flat().filter(x => !x).length / (seqLen * seqLen)) * 100).toFixed(1)}%
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
