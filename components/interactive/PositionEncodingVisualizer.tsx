'use client'

import React, { useState, useEffect } from 'react'
import { motion } from 'framer-motion'

type EncodingType = 'sinusoidal' | 'learned' | 'rope' | 'alibi' | 't5'

export default function PositionEncodingVisualizer() {
  const [encodingType, setEncodingType] = useState<EncodingType>('sinusoidal')
  const [maxLen, setMaxLen] = useState(50)
  const [dModel, setDModel] = useState(64)
  const [testLen, setTestLen] = useState(50)
  const [heatmapData, setHeatmapData] = useState<number[][]>([])

  // 生成 Sinusoidal Position Encoding
  const generateSinusoidal = (maxLen: number, dModel: number): number[][] => {
    const pe: number[][] = []
    
    for (let pos = 0; pos < maxLen; pos++) {
      const row: number[] = []
      for (let i = 0; i < dModel; i++) {
        const div_term = Math.exp((i - (i % 2)) * (-Math.log(10000.0) / dModel))
        const value = i % 2 === 0
          ? Math.sin(pos * div_term)
          : Math.cos(pos * div_term)
        row.push(value)
      }
      pe.push(row)
    }
    
    return pe
  }

  // 生成 Learned Position Embedding (随机模拟)
  const generateLearned = (maxLen: number, dModel: number): number[][] => {
    return Array(maxLen).fill(null).map(() =>
      Array(dModel).fill(null).map(() => (Math.random() - 0.5) * 2)
    )
  }

  // 生成 RoPE (简化可视化)
  const generateRoPE = (maxLen: number, dModel: number): number[][] => {
    const pe: number[][] = []
    
    for (let pos = 0; pos < maxLen; pos++) {
      const row: number[] = []
      for (let i = 0; i < dModel; i += 2) {
        const theta = Math.pow(10000, -i / dModel)
        const angle = pos * theta
        row.push(Math.cos(angle))
        if (i + 1 < dModel) {
          row.push(Math.sin(angle))
        }
      }
      pe.push(row)
    }
    
    return pe
  }

  // 生成 ALiBi (线性偏置)
  const generateALiBi = (maxLen: number, dModel: number): number[][] => {
    const pe: number[][] = []
    const slope = -0.1  // 简化：单一斜率
    
    for (let pos = 0; pos < maxLen; pos++) {
      const row: number[] = []
      for (let i = 0; i < dModel; i++) {
        // ALiBi 是位置偏置，不是 embedding
        const bias = slope * pos
        row.push(bias)
      }
      pe.push(row)
    }
    
    return pe
  }

  // 生成 T5 Relative Position Bias (简化)
  const generateT5 = (maxLen: number, dModel: number): number[][] => {
    const pe: number[][] = []
    
    for (let pos = 0; pos < maxLen; pos++) {
      const row: number[] = []
      for (let i = 0; i < dModel; i++) {
        // T5 使用相对位置，这里简化为基于距离的偏置
        const relativePos = pos - (maxLen / 2)
        const bucket = Math.floor(Math.log2(Math.abs(relativePos) + 1))
        const value = Math.tanh(bucket / 5)
        row.push(value)
      }
      pe.push(row)
    }
    
    return pe
  }

  // 更新热力图数据
  useEffect(() => {
    let data: number[][] = []
    
    switch (encodingType) {
      case 'sinusoidal':
        data = generateSinusoidal(maxLen, dModel)
        break
      case 'learned':
        data = generateLearned(maxLen, dModel)
        break
      case 'rope':
        data = generateRoPE(maxLen, dModel)
        break
      case 'alibi':
        data = generateALiBi(maxLen, dModel)
        break
      case 't5':
        data = generateT5(maxLen, dModel)
        break
    }
    
    setHeatmapData(data)
  }, [encodingType, maxLen, dModel])

  const getColor = (value: number) => {
    // 归一化到 [0, 1]
    const normalized = (value + 1) / 2
    const r = Math.floor((1 - normalized) * 255)
    const b = Math.floor(normalized * 255)
    return `rgb(${r}, 100, ${b})`
  }

  const encodingInfo = {
    sinusoidal: {
      name: 'Sinusoidal',
      formula: 'PE(pos, 2i) = sin(pos / 10000^(2i/d))',
      pros: ['无需学习参数', '可外推到更长序列', '相对位置信息'],
      cons: ['固定模式', '可能不够灵活'],
      used: ['Transformer 原始论文', 'GPT-3'],
    },
    learned: {
      name: 'Learned Embedding',
      formula: 'PE = Embedding(position_id)',
      pros: ['灵活，可学习任意模式', '通常性能更好', '简单实现'],
      cons: ['无法外推 > max_len', '增加参数量'],
      used: ['BERT', 'GPT-2'],
    },
    rope: {
      name: 'RoPE (旋转位置编码)',
      formula: 'q\' = R(θ, m) · q, k\' = R(θ, m) · k',
      pros: ['相对位置信息', '可外推', '不增加参数', '高效计算'],
      cons: ['实现稍复杂'],
      used: ['LLaMA', 'Mistral', 'Qwen'],
    },
    alibi: {
      name: 'ALiBi',
      formula: 'softmax(q·k^T + m·[-i, ..., -1, 0])',
      pros: ['极简实现', '优秀外推性', '训练高效', '无参数'],
      cons: ['仅位置偏置，不是 embedding'],
      used: ['BLOOM', 'MPT'],
    },
    't5': {
      name: 'T5 Relative Position',
      formula: 'Bias(i, j) = learned_bias[bucket(i-j)]',
      pros: ['相对位置', '分桶减少参数', '双向可用'],
      cons: ['需要学习', '实现复杂'],
      used: ['T5', 'DeBERTa'],
    },
  }

  const currentInfo = encodingInfo[encodingType]

  return (
    <div className="w-full max-w-7xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-cyan-50 rounded-xl border border-slate-200">
      {/* 标题 */}
      <div className="text-center mb-6">
        <h3 className="text-2xl font-bold text-slate-800 mb-2">
          📐 Position Encoding 对比器
        </h3>
        <p className="text-slate-600">
          对比不同位置编码方法的特性与可视化
        </p>
      </div>

      {/* 编码类型选择 */}
      <div className="mb-6">
        <h4 className="text-sm font-medium text-slate-700 mb-3">
          选择编码方法
        </h4>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
          {(Object.keys(encodingInfo) as EncodingType[]).map((type) => (
            <button
              key={type}
              onClick={() => setEncodingType(type)}
              className={`p-4 rounded-lg border-2 transition-all text-left ${
                encodingType === type
                  ? 'border-blue-600 bg-blue-50 shadow-lg scale-105'
                  : 'border-slate-200 bg-white hover:bg-slate-50'
              }`}
            >
              <div className="font-bold text-slate-800 mb-1">
                {encodingInfo[type].name}
              </div>
              <div className="text-xs text-slate-500">
                {type === 'sinusoidal' && '🌊 正弦波'}
                {type === 'learned' && '🎓 可学习'}
                {type === 'rope' && '🔄 旋转'}
                {type === 'alibi' && '📏 线性'}
                {type === 't5' && '📊 相对'}
              </div>
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* 左侧：配置 */}
        <div className="space-y-4">
          {/* 参数配置 */}
          <div className="bg-white rounded-lg border border-slate-200 p-5">
            <h4 className="text-lg font-semibold text-slate-800 mb-4">
              ⚙️ 参数配置
            </h4>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-2">
                  训练序列长度：{maxLen}
                </label>
                <input
                  type="range"
                  min="20"
                  max="100"
                  step="10"
                  value={maxLen}
                  onChange={(e) => setMaxLen(Number(e.target.value))}
                  className="w-full"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-700 mb-2">
                  模型维度：{dModel}
                </label>
                <input
                  type="range"
                  min="32"
                  max="128"
                  step="16"
                  value={dModel}
                  onChange={(e) => setDModel(Number(e.target.value))}
                  className="w-full"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-700 mb-2">
                  测试序列长度：{testLen}
                </label>
                <input
                  type="range"
                  min="20"
                  max="150"
                  step="10"
                  value={testLen}
                  onChange={(e) => setTestLen(Number(e.target.value))}
                  className="w-full"
                />
                {testLen > maxLen && (
                  <div className="text-xs text-amber-600 mt-1">
                    ⚠️ 超出训练长度（测试外推能力）
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* 方法信息 */}
          <div className="bg-gradient-to-br from-blue-50 to-purple-50 rounded-lg border border-blue-200 p-5">
            <h4 className="text-lg font-semibold text-blue-900 mb-3">
              📝 {currentInfo.name}
            </h4>
            
            <div className="space-y-3">
              <div>
                <div className="text-sm font-medium text-blue-800 mb-1">
                  公式
                </div>
                <div className="text-xs font-mono bg-white p-2 rounded border border-blue-200 overflow-x-auto">
                  {currentInfo.formula}
                </div>
              </div>

              <div>
                <div className="text-sm font-medium text-green-800 mb-1">
                  ✅ 优点
                </div>
                <ul className="text-xs text-green-700 space-y-0.5">
                  {currentInfo.pros.map((pro, i) => (
                    <li key={i}>• {pro}</li>
                  ))}
                </ul>
              </div>

              <div>
                <div className="text-sm font-medium text-red-800 mb-1">
                  ❌ 缺点
                </div>
                <ul className="text-xs text-red-700 space-y-0.5">
                  {currentInfo.cons.map((con, i) => (
                    <li key={i}>• {con}</li>
                  ))}
                </ul>
              </div>

              <div>
                <div className="text-sm font-medium text-purple-800 mb-1">
                  🏆 代表模型
                </div>
                <div className="flex flex-wrap gap-1 mt-1">
                  {currentInfo.used.map((model, i) => (
                    <span
                      key={i}
                      className="text-xs px-2 py-1 bg-purple-100 text-purple-700 rounded-full"
                    >
                      {model}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* 外推能力测试 */}
          <div className="bg-white rounded-lg border border-slate-200 p-5">
            <h4 className="text-lg font-semibold text-slate-800 mb-3">
              🎯 外推能力
            </h4>
            
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm text-slate-700">训练长度</span>
                <span className="text-lg font-bold text-slate-800">{maxLen}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-slate-700">测试长度</span>
                <span className={`text-lg font-bold ${
                  testLen > maxLen ? 'text-amber-600' : 'text-green-600'
                }`}>
                  {testLen}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-slate-700">外推倍数</span>
                <span className="text-lg font-bold text-blue-600">
                  {(testLen / maxLen).toFixed(2)}x
                </span>
              </div>

              {testLen > maxLen && (
                <div className="mt-3 p-3 bg-amber-50 border border-amber-200 rounded">
                  <div className="text-xs font-medium text-amber-900 mb-1">
                    外推性能预估
                  </div>
                  <div className="text-xs text-amber-700">
                    {encodingType === 'sinusoidal' && '✅ 优秀：Sinusoidal 支持无限外推'}
                    {encodingType === 'learned' && '❌ 较差：Learned 无法外推'}
                    {encodingType === 'rope' && '✅ 优秀：RoPE 支持外推（可能需要插值）'}
                    {encodingType === 'alibi' && '✅ 极佳：ALiBi 外推性能最好'}
                    {encodingType === 't5' && '⚠️ 中等：T5 需要调整分桶策略'}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* 中间+右侧：可视化 */}
        <div className="lg:col-span-2 space-y-4">
          {/* 热力图 */}
          <div className="bg-white rounded-lg border border-slate-200 p-5">
            <h4 className="text-lg font-semibold text-slate-800 mb-4">
              🌈 编码热力图 ({maxLen} × {dModel})
            </h4>
            
            <div className="overflow-auto" style={{ maxHeight: '500px' }}>
              <div className="inline-block">
                {heatmapData.slice(0, Math.min(maxLen, 80)).map((row, i) => (
                  <div key={i} className="flex">
                    {row.slice(0, Math.min(dModel, 64)).map((value, j) => (
                      <div
                        key={j}
                        className="w-3 h-3 border border-slate-100"
                        style={{ backgroundColor: getColor(value) }}
                        title={`Pos ${i}, Dim ${j}: ${value.toFixed(3)}`}
                      />
                    ))}
                  </div>
                ))}
              </div>
            </div>

            {/* 图例 */}
            <div className="mt-4 flex items-center gap-2">
              <span className="text-xs text-slate-600">数值范围：</span>
              <div className="flex items-center gap-1">
                <div className="w-20 h-4 rounded" style={{
                  background: 'linear-gradient(to right, rgb(255,100,0), rgb(128,100,128), rgb(0,100,255))'
                }} />
                <span className="text-xs text-slate-500 ml-2">-1.0 → 0 → 1.0</span>
              </div>
            </div>
          </div>

          {/* 位置特征曲线 */}
          <div className="bg-white rounded-lg border border-slate-200 p-5">
            <h4 className="text-lg font-semibold text-slate-800 mb-4">
              📈 位置特征曲线（前 4 个维度）
            </h4>
            
            <div className="space-y-3">
              {[0, 1, 2, 3].map((dimIdx) => (
                <div key={dimIdx}>
                  <div className="text-xs font-medium text-slate-700 mb-1">
                    维度 {dimIdx}
                  </div>
                  <svg width="100%" height="50" className="bg-slate-50 rounded">
                    <polyline
                      points={heatmapData.map((row, i) => {
                        const x = (i / maxLen) * 700
                        const y = 25 - (row[dimIdx] || 0) * 20
                        return `${x},${y}`
                      }).join(' ')}
                      fill="none"
                      stroke={['#3b82f6', '#10b981', '#f59e0b', '#ef4444'][dimIdx]}
                      strokeWidth="2"
                    />
                    <line x1="0" y1="25" x2="700" y2="25" stroke="#cbd5e1" strokeWidth="1" strokeDasharray="4" />
                  </svg>
                </div>
              ))}
            </div>
          </div>

          {/* 对比表格 */}
          <div className="bg-white rounded-lg border border-slate-200 p-5 overflow-x-auto">
            <h4 className="text-lg font-semibold text-slate-800 mb-4">
              📊 全方位对比
            </h4>
            
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-slate-200">
                  <th className="text-left p-2 font-semibold">特性</th>
                  <th className="text-center p-2">Sinusoidal</th>
                  <th className="text-center p-2">Learned</th>
                  <th className="text-center p-2">RoPE</th>
                  <th className="text-center p-2">ALiBi</th>
                  <th className="text-center p-2">T5</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b border-slate-100">
                  <td className="p-2 font-medium">参数量</td>
                  <td className="text-center p-2 text-green-600">0</td>
                  <td className="text-center p-2 text-red-600">L×D</td>
                  <td className="text-center p-2 text-green-600">0</td>
                  <td className="text-center p-2 text-green-600">0</td>
                  <td className="text-center p-2 text-amber-600">少量</td>
                </tr>
                <tr className="border-b border-slate-100">
                  <td className="p-2 font-medium">外推能力</td>
                  <td className="text-center p-2">✅ 优</td>
                  <td className="text-center p-2">❌ 差</td>
                  <td className="text-center p-2">✅ 优</td>
                  <td className="text-center p-2">✅✅ 极佳</td>
                  <td className="text-center p-2">⚠️ 中</td>
                </tr>
                <tr className="border-b border-slate-100">
                  <td className="p-2 font-medium">相对位置</td>
                  <td className="text-center p-2">⚠️ 间接</td>
                  <td className="text-center p-2">❌ 绝对</td>
                  <td className="text-center p-2">✅ 直接</td>
                  <td className="text-center p-2">✅ 直接</td>
                  <td className="text-center p-2">✅ 直接</td>
                </tr>
                <tr className="border-b border-slate-100">
                  <td className="p-2 font-medium">计算效率</td>
                  <td className="text-center p-2">高</td>
                  <td className="text-center p-2">高</td>
                  <td className="text-center p-2">中</td>
                  <td className="text-center p-2">极高</td>
                  <td className="text-center p-2">中</td>
                </tr>
                <tr>
                  <td className="p-2 font-medium">实现难度</td>
                  <td className="text-center p-2">简单</td>
                  <td className="text-center p-2">极简</td>
                  <td className="text-center p-2">中等</td>
                  <td className="text-center p-2">简单</td>
                  <td className="text-center p-2">复杂</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* 总结 */}
      <div className="mt-6 p-4 bg-gradient-to-r from-blue-50 to-purple-50 border border-blue-200 rounded-lg">
        <div className="text-sm font-medium text-blue-900 mb-2">
          💡 选择建议
        </div>
        <ul className="text-sm text-blue-800 space-y-1">
          <li>• <strong>通用场景</strong>：Sinusoidal（无参数，稳定）</li>
          <li>• <strong>固定长度</strong>：Learned（性能最优）</li>
          <li>• <strong>长文本 LLM</strong>：RoPE 或 ALiBi（外推性能好）</li>
          <li>• <strong>极致外推</strong>：ALiBi（BLOOM 2048→11k 无损）</li>
          <li>• <strong>编码器-解码器</strong>：T5 Relative（双向友好）</li>
        </ul>
      </div>
    </div>
  )
}
