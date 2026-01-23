'use client'

import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

interface Token {
  id: number
  text: string
  position: { x: number; y: number }
  assignedExperts: number[]
  weights: number[]
}

export default function MoERouting() {
  const [numExperts] = useState(8)
  const [topK, setTopK] = useState(2)
  const [isAnimating, setIsAnimating] = useState(false)
  const [currentStep, setCurrentStep] = useState(0)
  const [tokens, setTokens] = useState<Token[]>([])
  const [expertLoad, setExpertLoad] = useState<number[]>(Array(8).fill(0))

  const exampleTokens = ['The', 'cat', 'sat', 'on', 'the', 'mat', 'because', 'it']

  // 初始化 tokens
  useEffect(() => {
    const initialTokens: Token[] = exampleTokens.map((text, idx) => ({
      id: idx,
      text,
      position: { x: 50, y: 100 + idx * 60 },
      assignedExperts: [],
      weights: []
    }))
    setTokens(initialTokens)
  }, [exampleTokens])

  // 模拟路由决策
  const simulateRouting = (tokenId: number) => {
    // 生成随机路由概率
    const probs = Array(numExperts).fill(0).map(() => Math.random())
    const sum = probs.reduce((a, b) => a + b, 0)
    const normalized = probs.map(p => p / sum)

    // Top-K 选择
    const indexed = normalized.map((prob, idx) => ({ prob, idx }))
    indexed.sort((a, b) => b.prob - a.prob)
    const topKExperts = indexed.slice(0, topK)

    // 重新归一化
    const topKSum = topKExperts.reduce((sum, e) => sum + e.prob, 0)
    const topKWeights = topKExperts.map(e => e.prob / topKSum)

    return {
      experts: topKExperts.map(e => e.idx),
      weights: topKWeights
    }
  }

  // 开始动画
  const startAnimation = () => {
    setIsAnimating(true)
    setCurrentStep(0)
    setExpertLoad(Array(numExperts).fill(0))

    const newTokens = [...tokens]
    let step = 0

    const interval = setInterval(() => {
      if (step >= newTokens.length) {
        clearInterval(interval)
        setIsAnimating(false)
        return
      }

      const routing = simulateRouting(step)
      newTokens[step] = {
        ...newTokens[step],
        assignedExperts: routing.experts,
        weights: routing.weights
      }

      // 更新专家负载
      setExpertLoad(prev => {
        const updated = [...prev]
        routing.experts.forEach((expertId, idx) => {
          updated[expertId] += routing.weights[idx]
        })
        return updated
      })

      setTokens(newTokens)
      setCurrentStep(step + 1)
      step++
    }, 1000)
  }

  // 重置
  const reset = () => {
    setIsAnimating(false)
    setCurrentStep(0)
    setExpertLoad(Array(numExperts).fill(0))
    const resetTokens = tokens.map(t => ({
      ...t,
      assignedExperts: [],
      weights: []
    }))
    setTokens(resetTokens)
  }

  // 计算专家颜色强度
  const getExpertColor = (expertIdx: number, load: number) => {
    const intensity = Math.min(load / 2, 1) // 归一化到 0-1
    return `rgba(168, 85, 247, ${0.2 + intensity * 0.6})`
  }

  return (
    <div className="w-full space-y-6 my-8">
      {/* 标题 */}
      <div className="text-center">
        <h3 className="text-2xl font-bold mb-2">MoE 路由可视化</h3>
        <p className="text-gray-600">观察 Top-K 路由如何将 tokens 分配给不同的专家网络</p>
      </div>

      {/* 控制面板 */}
      <div className="bg-white rounded-xl shadow-lg border border-gray-200 p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="space-y-2">
            <div className="flex items-center gap-4">
              <label className="text-sm font-semibold">Top-K:</label>
              <div className="flex gap-2">
                {[1, 2, 4].map(k => (
                  <button
                    key={k}
                    onClick={() => !isAnimating && setTopK(k)}
                    disabled={isAnimating}
                    className={`px-4 py-2 rounded-lg transition-all ${
                      topK === k
                        ? 'bg-purple-500 text-white'
                        : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                    } ${isAnimating ? 'opacity-50 cursor-not-allowed' : ''}`}
                  >
                    K={k}
                  </button>
                ))}
              </div>
            </div>

            <div className="text-xs text-gray-500">
              每个 token 将被路由到 {topK} 个专家
            </div>
          </div>

          <div className="flex gap-2">
            <button
              onClick={startAnimation}
              disabled={isAnimating}
              className={`px-6 py-3 rounded-lg font-semibold transition-all ${
                isAnimating
                  ? 'bg-gray-300 cursor-not-allowed'
                  : 'bg-gradient-to-r from-purple-500 to-pink-500 text-white hover:shadow-lg'
              }`}
            >
              {isAnimating ? `处理中 (${currentStep}/${tokens.length})` : '开始路由'}
            </button>
            <button
              onClick={reset}
              className="px-6 py-3 bg-gray-200 hover:bg-gray-300 rounded-lg font-semibold transition-all"
            >
              重置
            </button>
          </div>
        </div>

        {/* 进度条 */}
        {isAnimating && (
          <div className="w-full bg-gray-200 rounded-full h-2">
            <motion.div
              className="bg-gradient-to-r from-purple-500 to-pink-500 h-2 rounded-full"
              initial={{ width: 0 }}
              animate={{ width: `${(currentStep / tokens.length) * 100}%` }}
              transition={{ duration: 0.3 }}
            />
          </div>
        )}
      </div>

      {/* 主可视化区域 */}
      <div className="bg-gradient-to-br from-purple-50 to-pink-50 rounded-xl p-8 min-h-[600px]">
        <div className="grid grid-cols-12 gap-4 h-full">
          {/* 左侧：Tokens */}
          <div className="col-span-3 space-y-3">
            <div className="text-sm font-bold text-gray-700 mb-4">Input Tokens</div>
            {tokens.map((token, idx) => (
              <motion.div
                key={token.id}
                className={`bg-white rounded-lg p-3 border-2 transition-all ${
                  idx < currentStep
                    ? 'border-purple-400 shadow-md'
                    : 'border-gray-200'
                }`}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: idx * 0.1 }}
              >
                <div className="font-mono font-bold text-sm">{token.text}</div>
                {token.assignedExperts.length > 0 && (
                  <div className="mt-2 space-y-1">
                    {token.assignedExperts.map((expertId, i) => (
                      <div key={expertId} className="flex items-center gap-2 text-xs">
                        <div className={`w-2 h-2 rounded-full bg-purple-500`} />
                        <span>Expert {expertId}: {(token.weights[i] * 100).toFixed(0)}%</span>
                      </div>
                    ))}
                  </div>
                )}
              </motion.div>
            ))}
          </div>

          {/* 中间：路由可视化 */}
          <div className="col-span-6 relative">
            <div className="absolute inset-0">
              <svg className="w-full h-full">
                {/* 绘制连接线 */}
                {tokens.map(token => {
                  if (token.assignedExperts.length === 0) return null
                  
                  return token.assignedExperts.map((expertId, idx) => {
                    const tokenY = 50 + token.id * 60
                    const expertX = 400
                    const expertY = 50 + expertId * 60
                    const weight = token.weights[idx]

                    return (
                      <motion.path
                        key={`${token.id}-${expertId}`}
                        d={`M 100 ${tokenY} Q 250 ${tokenY} ${expertX} ${expertY}`}
                        stroke="rgba(168, 85, 247, 0.6)"
                        strokeWidth={weight * 8}
                        fill="none"
                        initial={{ pathLength: 0, opacity: 0 }}
                        animate={{ pathLength: 1, opacity: 0.6 }}
                        transition={{ duration: 0.5 }}
                      />
                    )
                  })
                })}
              </svg>

              {/* 中间标签 */}
              <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
                <div className="bg-white rounded-lg shadow-lg p-4 border-2 border-purple-300">
                  <div className="text-xs font-semibold text-purple-700 mb-1">Router Gate</div>
                  <div className="text-2xl">🔀</div>
                  <div className="text-xs text-gray-500 mt-1">Top-{topK} Selection</div>
                </div>
              </div>
            </div>
          </div>

          {/* 右侧：Experts */}
          <div className="col-span-3 space-y-3">
            <div className="text-sm font-bold text-gray-700 mb-4">Expert Networks</div>
            {Array(numExperts).fill(0).map((_, idx) => {
              const load = expertLoad[idx]
              const isActive = load > 0
              
              return (
                <motion.div
                  key={idx}
                  className={`rounded-lg p-3 border-2 transition-all ${
                    isActive ? 'border-purple-400 shadow-md' : 'border-gray-200'
                  }`}
                  style={{
                    backgroundColor: getExpertColor(idx, load)
                  }}
                  whileHover={{ scale: 1.05 }}
                >
                  <div className="flex items-center justify-between">
                    <div className="font-bold text-sm">Expert {idx}</div>
                    {isActive && (
                      <div className="text-xs bg-white px-2 py-1 rounded-full">
                        {load.toFixed(2)}
                      </div>
                    )}
                  </div>
                  {isActive && (
                    <div className="mt-2">
                      <div className="w-full bg-white/50 rounded-full h-2">
                        <motion.div
                          className="bg-purple-600 h-2 rounded-full"
                          initial={{ width: 0 }}
                          animate={{ width: `${Math.min(load / 4 * 100, 100)}%` }}
                          transition={{ duration: 0.3 }}
                        />
                      </div>
                    </div>
                  )}
                </motion.div>
              )
            })}
          </div>
        </div>
      </div>

      {/* 统计信息 */}
      <div className="grid grid-cols-3 gap-4">
        <div className="bg-white rounded-lg shadow border p-4">
          <div className="text-xs text-gray-500 mb-1">总 Tokens 处理</div>
          <div className="text-3xl font-bold text-purple-600">{currentStep}</div>
          <div className="text-xs text-gray-400 mt-1">/ {tokens.length}</div>
        </div>

        <div className="bg-white rounded-lg shadow border p-4">
          <div className="text-xs text-gray-500 mb-1">激活专家数</div>
          <div className="text-3xl font-bold text-pink-600">
            {expertLoad.filter(l => l > 0).length}
          </div>
          <div className="text-xs text-gray-400 mt-1">/ {numExperts}</div>
        </div>

        <div className="bg-white rounded-lg shadow border p-4">
          <div className="text-xs text-gray-500 mb-1">负载均衡分数</div>
          <div className="text-3xl font-bold text-blue-600">
            {expertLoad.length > 0
              ? (1 - (Math.max(...expertLoad) - Math.min(...expertLoad)) / (Math.max(...expertLoad) || 1)).toFixed(2)
              : '0.00'}
          </div>
          <div className="text-xs text-gray-400 mt-1">0.0 (不均) - 1.0 (完美)</div>
        </div>
      </div>

      {/* 负载均衡损失说明 */}
      <div className="bg-gradient-to-r from-yellow-50 to-orange-50 border border-yellow-200 rounded-lg p-4">
        <h4 className="text-sm font-bold text-yellow-900 mb-2">⚖️ 负载均衡（Load Balancing）</h4>
        <div className="text-sm text-yellow-800 space-y-2">
          <p>
            MoE 模型需要确保专家被均匀使用，避免少数专家过载而其他专家闲置。
          </p>
          <div className="bg-white/60 rounded p-3 font-mono text-xs">
            <div>负载均衡损失:</div>
            <div className="mt-1">
              L_balance = α · ∑ (load_i - avg_load)²
            </div>
          </div>
          <p className="text-xs">
            通过在训练时添加此损失项，鼓励路由器均匀分配 tokens 给所有专家。
          </p>
        </div>
      </div>

      {/* 代码示例 */}
      <div className="bg-gray-900 rounded-lg p-4 overflow-x-auto">
        <div className="text-xs text-gray-300 mb-2">MoE 层实现（简化）</div>
        <pre className="text-sm text-gray-100">
          <code>{`class MoELayer(nn.Module):
    def __init__(self, hidden_size, num_experts=8, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # 路由器
        self.gate = nn.Linear(hidden_size, num_experts)
        
        # 专家网络
        self.experts = nn.ModuleList([
            FeedForward(hidden_size) for _ in range(num_experts)
        ])
    
    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
        router_logits = self.gate(x)  # (batch, seq_len, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Top-K 选择
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # 分发到专家并聚合
        output = torch.zeros_like(x)
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, :, i]
            weight = top_k_probs[:, :, i:i+1]
            # ... 专家计算 ...
        
        return output`}</code>
        </pre>
      </div>
    </div>
  )
}
