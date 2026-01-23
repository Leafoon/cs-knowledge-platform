'use client'

import React, { useState, useEffect } from 'react'
import { motion } from 'framer-motion'

export default function AttentionWeightHeatmap() {
  const [sentence, setSentence] = useState("The cat sat on the mat")
  const [tokens, setTokens] = useState<string[]>([])
  const [selectedToken, setSelectedToken] = useState<number | null>(null)
  const [attentionWeights, setAttentionWeights] = useState<number[][]>([])
  const [showSteps, setShowSteps] = useState(false)
  const [currentStep, setCurrentStep] = useState(0)

  const steps = [
    { name: "1. 输入 Embedding", description: "将 tokens 转换为向量" },
    { name: "2. 计算 Q, K, V", description: "通过线性变换得到 Query, Key, Value" },
    { name: "3. 计算 QK^T", description: "Query 和 Key 的点积" },
    { name: "4. 缩放 /√d_k", description: "除以维度的平方根" },
    { name: "5. Softmax", description: "归一化为概率分布" },
    { name: "6. 乘以 V", description: "加权求和得到输出" },
  ]

  // 简单分词
  useEffect(() => {
    const words = sentence.trim().split(/\s+/).filter(w => w.length > 0)
    setTokens(words)
    
    // 生成模拟的注意力权重
    const n = words.length
    const weights: number[][] = []
    
    for (let i = 0; i < n; i++) {
      const row: number[] = []
      let sum = 0
      
      for (let j = 0; j < n; j++) {
        // 模拟注意力模式：
        // 1. 自注意力较高
        // 2. 相邻词有一定关注
        // 3. 远距离词关注度降低
        let weight = 0
        
        if (i === j) {
          weight = 0.3 + Math.random() * 0.2  // 自注意力
        } else {
          const distance = Math.abs(i - j)
          weight = Math.max(0.05, 0.3 / (distance + 1) + Math.random() * 0.1)
        }
        
        row.push(weight)
        sum += weight
      }
      
      // 归一化
      weights.push(row.map(w => w / sum))
    }
    
    setAttentionWeights(weights)
    setSelectedToken(null)
  }, [sentence])

  const getHeatmapColor = (value: number) => {
    // 从白色到深蓝色的渐变
    const intensity = Math.floor(value * 255)
    const r = 255 - intensity
    const g = 255 - intensity
    const b = 255
    return `rgb(${r}, ${g}, ${b})`
  }

  const exampleSentences = [
    "The cat sat on the mat",
    "I love natural language processing",
    "Attention is all you need",
    "The quick brown fox jumps",
  ]

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-purple-50 rounded-xl border border-slate-200">
      {/* 标题 */}
      <div className="text-center mb-6">
        <h3 className="text-2xl font-bold text-slate-800 mb-2">
          🔥 Attention 权重热力图
        </h3>
        <p className="text-slate-600">
          实时计算并可视化 Self-Attention 权重矩阵
        </p>
      </div>

      {/* 输入区域 */}
      <div className="mb-6">
        <label className="block text-sm font-medium text-slate-700 mb-2">
          输入句子
        </label>
        <input
          type="text"
          value={sentence}
          onChange={(e) => setSentence(e.target.value)}
          className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          placeholder="输入英文句子..."
        />
        
        {/* 示例句子 */}
        <div className="mt-2 flex flex-wrap gap-2">
          {exampleSentences.map((ex, i) => (
            <button
              key={i}
              onClick={() => setSentence(ex)}
              className="text-xs px-3 py-1 bg-white border border-slate-300 rounded-full hover:bg-blue-50 hover:border-blue-400 transition-colors"
            >
              {ex}
            </button>
          ))}
        </div>
      </div>

      {/* 步骤切换 */}
      <div className="mb-6 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <label className="relative inline-flex items-center cursor-pointer">
            <input
              type="checkbox"
              checked={showSteps}
              onChange={(e) => {
                setShowSteps(e.target.checked)
                if (!e.target.checked) setCurrentStep(0)
              }}
              className="sr-only peer"
            />
            <div className="w-11 h-6 bg-slate-300 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-slate-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
          </label>
          <span className="text-sm font-medium text-slate-700">
            显示计算步骤
          </span>
        </div>

        {showSteps && (
          <div className="flex gap-2">
            <button
              onClick={() => setCurrentStep(Math.max(0, currentStep - 1))}
              disabled={currentStep === 0}
              className="px-3 py-1 text-sm bg-white border border-slate-300 rounded-lg hover:bg-slate-50 disabled:opacity-50"
            >
              ← 上一步
            </button>
            <button
              onClick={() => setCurrentStep(Math.min(steps.length - 1, currentStep + 1))}
              disabled={currentStep === steps.length - 1}
              className="px-3 py-1 text-sm bg-white border border-slate-300 rounded-lg hover:bg-slate-50 disabled:opacity-50"
            >
              下一步 →
            </button>
          </div>
        )}
      </div>

      {/* 步骤进度 */}
      {showSteps && (
        <div className="mb-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <div className="flex items-center gap-2 mb-3">
            {steps.map((step, i) => (
              <React.Fragment key={i}>
                <div
                  className={`flex items-center justify-center w-8 h-8 rounded-full text-sm font-bold ${
                    i <= currentStep
                      ? 'bg-blue-600 text-white'
                      : 'bg-white text-slate-400 border border-slate-300'
                  }`}
                >
                  {i + 1}
                </div>
                {i < steps.length - 1 && (
                  <div
                    className={`flex-1 h-1 rounded ${
                      i < currentStep ? 'bg-blue-600' : 'bg-slate-300'
                    }`}
                  />
                )}
              </React.Fragment>
            ))}
          </div>
          <div className="text-sm">
            <div className="font-semibold text-blue-900">
              {steps[currentStep].name}
            </div>
            <div className="text-blue-700 mt-1">
              {steps[currentStep].description}
            </div>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* 左侧：热力图 */}
        <div className="bg-white rounded-lg border border-slate-200 p-5">
          <h4 className="text-lg font-semibold text-slate-800 mb-4">
            📊 注意力权重矩阵
          </h4>
          
          {tokens.length > 0 && (
            <div className="overflow-x-auto">
              <table className="w-full border-collapse">
                <thead>
                  <tr>
                    <th className="p-2 text-xs font-medium text-slate-600 border border-slate-200 bg-slate-50">
                      Query ↓ Key →
                    </th>
                    {tokens.map((token, j) => (
                      <th
                        key={j}
                        className={`p-2 text-xs font-medium border border-slate-200 ${
                          selectedToken === j ? 'bg-yellow-100' : 'bg-slate-50'
                        }`}
                      >
                        {token}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {tokens.map((token, i) => (
                    <tr key={i}>
                      <td
                        className={`p-2 text-xs font-medium border border-slate-200 ${
                          selectedToken === i ? 'bg-yellow-100' : 'bg-slate-50'
                        }`}
                      >
                        {token}
                      </td>
                      {attentionWeights[i]?.map((weight, j) => (
                        <motion.td
                          key={j}
                          className="p-2 text-center text-xs font-mono border border-slate-200 cursor-pointer"
                          style={{ backgroundColor: getHeatmapColor(weight) }}
                          onMouseEnter={() => setSelectedToken(i)}
                          onMouseLeave={() => setSelectedToken(null)}
                          whileHover={{ scale: 1.1, zIndex: 10 }}
                        >
                          {weight.toFixed(3)}
                        </motion.td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* 图例 */}
          <div className="mt-4 flex items-center gap-2">
            <span className="text-xs text-slate-600">权重值：</span>
            <div className="flex items-center gap-1">
              {[0, 0.25, 0.5, 0.75, 1.0].map((v) => (
                <div key={v} className="flex flex-col items-center">
                  <div
                    className="w-8 h-4 border border-slate-300"
                    style={{ backgroundColor: getHeatmapColor(v) }}
                  />
                  <span className="text-xs text-slate-500 mt-1">
                    {v.toFixed(2)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* 右侧：详细信息 */}
        <div className="space-y-4">
          {/* Token 选择 */}
          <div className="bg-white rounded-lg border border-slate-200 p-5">
            <h4 className="text-lg font-semibold text-slate-800 mb-3">
              🎯 选择 Token
            </h4>
            <div className="flex flex-wrap gap-2">
              {tokens.map((token, i) => (
                <button
                  key={i}
                  onClick={() => setSelectedToken(selectedToken === i ? null : i)}
                  className={`px-4 py-2 rounded-lg font-medium transition-all ${
                    selectedToken === i
                      ? 'bg-blue-600 text-white shadow-lg scale-105'
                      : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
                  }`}
                >
                  {token}
                </button>
              ))}
            </div>
          </div>

          {/* 注意力分布 */}
          {selectedToken !== null && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-white rounded-lg border border-slate-200 p-5"
            >
              <h4 className="text-lg font-semibold text-slate-800 mb-3">
                📈 &quot;{tokens[selectedToken]}&quot; 的注意力分布
              </h4>
              <div className="space-y-2">
                {tokens.map((token, j) => {
                  const weight = attentionWeights[selectedToken]?.[j] || 0
                  return (
                    <div key={j}>
                      <div className="flex justify-between text-sm mb-1">
                        <span className="font-medium text-slate-700">
                          {token}
                        </span>
                        <span className="text-blue-600 font-mono">
                          {weight.toFixed(4)}
                        </span>
                      </div>
                      <div className="w-full bg-slate-200 rounded-full h-3 overflow-hidden">
                        <motion.div
                          className="h-full bg-gradient-to-r from-blue-400 to-blue-600"
                          initial={{ width: 0 }}
                          animate={{ width: `${weight * 100}%` }}
                          transition={{ duration: 0.5 }}
                        />
                      </div>
                    </div>
                  )
                })}
              </div>
            </motion.div>
          )}

          {/* 公式说明 */}
          <div className="bg-purple-50 rounded-lg border border-purple-200 p-5">
            <h4 className="text-lg font-semibold text-purple-900 mb-3">
              📐 Scaled Dot-Product Attention
            </h4>
            <div className="text-sm text-purple-800 space-y-2">
              <div className="font-mono bg-white p-3 rounded border border-purple-200 overflow-x-auto">
                Attention(Q, K, V) = softmax(QK^T / √d_k) V
              </div>
              <ul className="space-y-1 text-xs">
                <li>• <strong>QK^T</strong>: 计算相似度（点积）</li>
                <li>• <strong>/√d_k</strong>: 缩放因子（d_k=64 时除以8）</li>
                <li>• <strong>softmax</strong>: 归一化为概率分布</li>
                <li>• <strong>×V</strong>: 加权求和得到输出</li>
              </ul>
            </div>
          </div>

          {/* 统计信息 */}
          <div className="bg-green-50 rounded-lg border border-green-200 p-5">
            <h4 className="text-lg font-semibold text-green-900 mb-3">
              📊 矩阵统计
            </h4>
            <div className="grid grid-cols-2 gap-3 text-sm">
              <div>
                <div className="text-green-700 font-medium">序列长度</div>
                <div className="text-2xl font-bold text-green-600">
                  {tokens.length}
                </div>
              </div>
              <div>
                <div className="text-green-700 font-medium">矩阵大小</div>
                <div className="text-2xl font-bold text-green-600">
                  {tokens.length}×{tokens.length}
                </div>
              </div>
              {selectedToken !== null && (
                <>
                  <div>
                    <div className="text-green-700 font-medium">最大注意力</div>
                    <div className="text-xl font-bold text-green-600">
                      {Math.max(...(attentionWeights[selectedToken] || [])).toFixed(3)}
                    </div>
                  </div>
                  <div>
                    <div className="text-green-700 font-medium">熵值</div>
                    <div className="text-xl font-bold text-green-600">
                      {(() => {
                        const weights = attentionWeights[selectedToken] || []
                        const entropy = -weights.reduce((sum, w) => 
                          sum + (w > 0 ? w * Math.log2(w) : 0), 0
                        )
                        return entropy.toFixed(2)
                      })()}
                    </div>
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* 说明 */}
      <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
        <div className="text-sm font-medium text-blue-900 mb-2">
          💡 如何阅读热力图
        </div>
        <ul className="text-sm text-blue-800 space-y-1">
          <li>• <strong>行（Query）</strong>：当前 token 在查询什么</li>
          <li>• <strong>列（Key）</strong>：每个 token 提供的信息</li>
          <li>• <strong>颜色深度</strong>：注意力权重大小（深蓝=高关注）</li>
          <li>• <strong>对角线</strong>：Self-Attention（通常较高）</li>
          <li>• <strong>每行和</strong>：必定等于 1.0（softmax 归一化）</li>
        </ul>
      </div>
    </div>
  )
}
