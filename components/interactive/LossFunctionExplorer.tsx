'use client'

import React, { useState, useMemo } from 'react'
import { motion } from 'framer-motion'
import { TrendingUp, Flame, Target, Layers, Code } from 'lucide-react'

type LossFunction = 'cross_entropy' | 'focal' | 'label_smoothing' | 'contrastive' | 'kl_divergence'

interface LossConfig {
  type: LossFunction
  name: string
  description: string
  formula: string
  color: string
  icon: JSX.Element
  params: { [key: string]: { value: number; min: number; max: number; step: number; label: string } }
}

const lossConfigs: LossConfig[] = [
  {
    type: 'cross_entropy',
    name: 'Cross Entropy',
    description: 'Standard classification loss',
    formula: 'L = -Σ y_i log(p_i)',
    color: 'from-blue-500 to-cyan-500',
    icon: <Target className="w-5 h-5" />,
    params: {}
  },
  {
    type: 'focal',
    name: 'Focal Loss',
    description: 'Addresses class imbalance by down-weighting easy examples',
    formula: 'L = -α(1-p_t)^γ log(p_t)',
    color: 'from-orange-500 to-red-500',
    icon: <Flame className="w-5 h-5" />,
    params: {
      alpha: { value: 0.25, min: 0.1, max: 0.9, step: 0.05, label: 'Alpha (α)' },
      gamma: { value: 2.0, min: 0.5, max: 5.0, step: 0.5, label: 'Gamma (γ)' }
    }
  },
  {
    type: 'label_smoothing',
    name: 'Label Smoothing',
    description: 'Prevents overconfident predictions',
    formula: 'L = (1-ε)L_CE + ε/K',
    color: 'from-green-500 to-emerald-500',
    icon: <TrendingUp className="w-5 h-5" />,
    params: {
      epsilon: { value: 0.1, min: 0.0, max: 0.3, step: 0.05, label: 'Epsilon (ε)' }
    }
  },
  {
    type: 'contrastive',
    name: 'Contrastive Loss (NT-Xent)',
    description: 'Pulls positive pairs together, pushes negatives apart',
    formula: 'L = -log(exp(sim(z_i,z_j)/τ) / Σ_k exp(sim(z_i,z_k)/τ))',
    color: 'from-purple-500 to-pink-500',
    icon: <Layers className="w-5 h-5" />,
    params: {
      temperature: { value: 0.07, min: 0.01, max: 1.0, step: 0.01, label: 'Temperature (τ)' }
    }
  },
  {
    type: 'kl_divergence',
    name: 'KL Divergence',
    description: 'Knowledge distillation loss',
    formula: 'L = KL(softmax(z_s/T) || softmax(z_t/T))',
    color: 'from-indigo-500 to-purple-500',
    icon: <Code className="w-5 h-5" />,
    params: {
      temperature: { value: 2.0, min: 1.0, max: 10.0, step: 0.5, label: 'Temperature (T)' },
      alpha: { value: 0.5, min: 0.0, max: 1.0, step: 0.1, label: 'Alpha (weight)' }
    }
  }
]

export default function LossFunctionExplorer() {
  const [selectedLoss, setSelectedLoss] = useState<LossFunction>('cross_entropy')
  const [params, setParams] = useState<{ [key: string]: { [key: string]: number } }>({
    focal: { alpha: 0.25, gamma: 2.0 },
    label_smoothing: { epsilon: 0.1 },
    contrastive: { temperature: 0.07 },
    kl_divergence: { temperature: 2.0, alpha: 0.5 }
  })

  const currentConfig = lossConfigs.find(c => c.type === selectedLoss)!

  // 更新参数
  const updateParam = (paramName: string, value: number) => {
    setParams(prev => ({
      ...prev,
      [selectedLoss]: {
        ...prev[selectedLoss],
        [paramName]: value
      }
    }))
  }

  // 生成损失曲线数据（模拟）
  const lossData = useMemo(() => {
    const points = 100
    const data: { probability: number; loss: number }[] = []

    for (let i = 0; i <= points; i++) {
      const p = i / points  // 预测概率 0 to 1

      let loss = 0
      switch (selectedLoss) {
        case 'cross_entropy':
          // L = -log(p)
          loss = -Math.log(Math.max(p, 1e-7))
          break
        
        case 'focal':
          const { alpha, gamma } = params.focal
          // L = -α(1-p)^γ log(p)
          loss = -alpha * Math.pow(1 - p, gamma) * Math.log(Math.max(p, 1e-7))
          break
        
        case 'label_smoothing':
          const { epsilon } = params.label_smoothing
          const K = 10  // 假设 10 类
          // L = (1-ε)(-log(p)) + ε/K
          loss = (1 - epsilon) * (-Math.log(Math.max(p, 1e-7))) + epsilon / K
          break
        
        case 'contrastive':
          // 简化版：展示温度对相似度的影响
          const { temperature } = params.contrastive
          // 假设相似度为 p，负样本相似度为 0.5
          const pos_sim = p
          const neg_sim = 0.3
          const exp_pos = Math.exp(pos_sim / temperature)
          const exp_neg = Math.exp(neg_sim / temperature)
          loss = -Math.log(exp_pos / (exp_pos + exp_neg))
          break
        
        case 'kl_divergence':
          const T = params.kl_divergence.temperature
          // 假设教师概率固定为 0.8
          const p_teacher = 0.8
          const p_student = p
          // KL(P||Q) = P log(P/Q)
          const kl = p_teacher * Math.log(Math.max(p_teacher / Math.max(p_student, 1e-7), 1e-7)) +
                     (1 - p_teacher) * Math.log(Math.max((1 - p_teacher) / Math.max(1 - p_student, 1e-7), 1e-7))
          loss = kl * T * T
          break
      }

      data.push({ probability: p, loss: Math.min(loss, 10) })  // Clamp at 10
    }

    return data
  }, [selectedLoss, params])

  // SVG 曲线路径
  const svgPath = useMemo(() => {
    const width = 400
    const height = 200
    const padding = 20

    const xScale = (p: number) => padding + (p * (width - 2 * padding))
    const yScale = (loss: number) => height - padding - ((loss / 10) * (height - 2 * padding))

    const path = lossData
      .map((d, i) => {
        const x = xScale(d.probability)
        const y = yScale(d.loss)
        return i === 0 ? `M ${x} ${y}` : `L ${x} ${y}`
      })
      .join(' ')

    return path
  }, [lossData])

  // 生成代码
  const generateCode = () => {
    let code = `import torch\nimport torch.nn.functional as F\n\n`

    switch (selectedLoss) {
      case 'cross_entropy':
        code += `# Standard Cross Entropy Loss\n`
        code += `loss = F.cross_entropy(logits, labels)\n`
        break
      
      case 'focal':
        const { alpha, gamma } = params.focal
        code += `# Focal Loss\n`
        code += `class FocalLoss(nn.Module):\n`
        code += `    def __init__(self, alpha=${alpha}, gamma=${gamma}):\n`
        code += `        super().__init__()\n`
        code += `        self.alpha = alpha\n`
        code += `        self.gamma = gamma\n\n`
        code += `    def forward(self, logits, targets):\n`
        code += `        ce_loss = F.cross_entropy(logits, targets, reduction='none')\n`
        code += `        p_t = torch.exp(-ce_loss)\n`
        code += `        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss\n`
        code += `        return focal_loss.mean()\n\n`
        code += `loss_fn = FocalLoss()\n`
        code += `loss = loss_fn(logits, labels)\n`
        break
      
      case 'label_smoothing':
        const { epsilon } = params.label_smoothing
        code += `# Label Smoothing\n`
        code += `loss = F.cross_entropy(\n`
        code += `    logits,\n`
        code += `    labels,\n`
        code += `    label_smoothing=${epsilon}\n`
        code += `)\n`
        break
      
      case 'contrastive':
        const { temperature } = params.contrastive
        code += `# Contrastive Loss (NT-Xent)\n`
        code += `# Assume z1, z2 are normalized embeddings\n`
        code += `z = torch.cat([z1, z2], dim=0)  # (2B, D)\n`
        code += `sim_matrix = torch.matmul(z, z.T) / ${temperature}\n\n`
        code += `# Mask out self-similarity\n`
        code += `mask = torch.eye(2 * batch_size, device=z.device).bool()\n`
        code += `sim_matrix.masked_fill_(mask, -1e9)\n\n`
        code += `# Positive pairs: z1[i] with z2[i]\n`
        code += `labels = torch.arange(batch_size, device=z.device)\n`
        code += `labels = torch.cat([labels + batch_size, labels])\n\n`
        code += `loss = F.cross_entropy(sim_matrix, labels)\n`
        break
      
      case 'kl_divergence':
        const T = params.kl_divergence.temperature
        const a = params.kl_divergence.alpha
        code += `# Knowledge Distillation (KL Divergence)\n`
        code += `# Student and teacher logits\n`
        code += `student_log_probs = F.log_softmax(student_logits / ${T}, dim=-1)\n`
        code += `teacher_probs = F.softmax(teacher_logits / ${T}, dim=-1)\n\n`
        code += `soft_loss = F.kl_div(\n`
        code += `    student_log_probs,\n`
        code += `    teacher_probs,\n`
        code += `    reduction='batchmean'\n`
        code += `) * (${T} ** 2)\n\n`
        code += `hard_loss = F.cross_entropy(student_logits, labels)\n`
        code += `loss = ${a} * soft_loss + ${1-a} * hard_loss\n`
        break
    }

    return code
  }

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-purple-50 rounded-xl border-2 border-purple-200">
      <div className="flex items-center gap-3 mb-6">
        <div className="p-2 bg-gradient-to-br from-purple-500 to-pink-600 rounded-lg">
          <TrendingUp className="w-6 h-6 text-white" />
        </div>
        <div>
          <h3 className="text-2xl font-bold text-gray-800">Loss Function Explorer</h3>
          <p className="text-sm text-gray-600">Compare and visualize different loss functions</p>
        </div>
      </div>

      {/* Loss Function Selection */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mb-6">
        {lossConfigs.map((config) => (
          <motion.button
            key={config.type}
            whileHover={{ scale: 1.03 }}
            whileTap={{ scale: 0.97 }}
            onClick={() => setSelectedLoss(config.type)}
            className={`p-3 rounded-lg border-2 transition-all ${
              selectedLoss === config.type
                ? 'border-purple-500 bg-gradient-to-r ' + config.color + ' text-white'
                : 'border-gray-200 bg-white text-gray-700 hover:border-gray-300'
            }`}
          >
            <div className="flex items-center justify-center mb-2">
              {config.icon}
            </div>
            <div className="font-semibold text-xs text-center">{config.name}</div>
          </motion.button>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left: Loss Curve */}
        <div className="space-y-4">
          <div className="bg-white rounded-lg p-4 border-2 border-gray-200">
            <h4 className="font-bold text-gray-700 mb-3">Loss Curve</h4>
            
            {/* SVG Plot */}
            <svg width="100%" height="220" viewBox="0 0 400 220" className="border border-gray-200 rounded bg-gray-50">
              {/* Grid lines */}
              {[0, 0.25, 0.5, 0.75, 1.0].map(v => (
                <line
                  key={`grid-x-${v}`}
                  x1={20 + v * 360}
                  y1={20}
                  x2={20 + v * 360}
                  y2={200}
                  stroke="#e5e7eb"
                  strokeWidth="1"
                  strokeDasharray="4"
                />
              ))}
              {[0, 2.5, 5, 7.5, 10].map(v => (
                <line
                  key={`grid-y-${v}`}
                  x1={20}
                  y1={200 - (v / 10) * 160}
                  x2={380}
                  y2={200 - (v / 10) * 160}
                  stroke="#e5e7eb"
                  strokeWidth="1"
                  strokeDasharray="4"
                />
              ))}

              {/* Curve */}
              <motion.path
                d={svgPath}
                fill="none"
                stroke="url(#gradient)"
                strokeWidth="3"
                strokeLinecap="round"
                initial={{ pathLength: 0 }}
                animate={{ pathLength: 1 }}
                transition={{ duration: 1, ease: 'easeInOut' }}
              />

              {/* Gradient definition */}
              <defs>
                <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="#8b5cf6" />
                  <stop offset="100%" stopColor="#ec4899" />
                </linearGradient>
              </defs>

              {/* Axes */}
              <line x1="20" y1="200" x2="380" y2="200" stroke="#374151" strokeWidth="2" />
              <line x1="20" y1="20" x2="20" y2="200" stroke="#374151" strokeWidth="2" />

              {/* X-axis labels */}
              <text x="20" y="215" fontSize="10" fill="#6b7280" textAnchor="middle">0.0</text>
              <text x="200" y="215" fontSize="10" fill="#6b7280" textAnchor="middle">0.5</text>
              <text x="380" y="215" fontSize="10" fill="#6b7280" textAnchor="middle">1.0</text>
              <text x="200" y="235" fontSize="12" fill="#374151" textAnchor="middle" fontWeight="bold">
                Predicted Probability (p)
              </text>

              {/* Y-axis labels */}
              <text x="10" y="204" fontSize="10" fill="#6b7280" textAnchor="end">0</text>
              <text x="10" y="120" fontSize="10" fill="#6b7280" textAnchor="end">5</text>
              <text x="10" y="24" fontSize="10" fill="#6b7280" textAnchor="end">10</text>
            </svg>
          </div>

          {/* Parameter Controls */}
          {Object.keys(currentConfig.params).length > 0 && (
            <div className="bg-white rounded-lg p-4 border-2 border-gray-200">
              <h4 className="font-bold text-gray-700 mb-3">Parameters</h4>
              <div className="space-y-3">
                {Object.entries(currentConfig.params).map(([paramName, config]) => (
                  <div key={paramName}>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      {config.label}: <span className="text-purple-600 font-bold">
                        {params[selectedLoss]?.[paramName]?.toFixed(2) ?? config.value.toFixed(2)}
                      </span>
                    </label>
                    <input
                      type="range"
                      min={config.min}
                      max={config.max}
                      step={config.step}
                      value={params[selectedLoss]?.[paramName] ?? config.value}
                      onChange={(e) => updateParam(paramName, Number(e.target.value))}
                      className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-purple-500"
                    />
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Right: Details & Code */}
        <div className="space-y-4">
          {/* Description */}
          <div className="bg-white rounded-lg p-4 border-2 border-gray-200">
            <h4 className="font-bold text-gray-700 mb-2">{currentConfig.name}</h4>
            <p className="text-sm text-gray-600 mb-3">{currentConfig.description}</p>
            
            <div className="bg-gray-50 rounded p-3 border border-gray-200">
              <div className="text-xs text-gray-500 mb-1">Formula:</div>
              <div className="font-mono text-sm text-gray-800">{currentConfig.formula}</div>
            </div>
          </div>

          {/* Use Cases */}
          <div className="bg-white rounded-lg p-4 border-2 border-gray-200">
            <h4 className="font-bold text-gray-700 mb-2">Use Cases</h4>
            <ul className="text-sm text-gray-700 space-y-1">
              {selectedLoss === 'cross_entropy' && (
                <>
                  <li>• Standard classification tasks</li>
                  <li>• Balanced datasets</li>
                  <li>• General-purpose loss</li>
                </>
              )}
              {selectedLoss === 'focal' && (
                <>
                  <li>• Imbalanced classification (1:100 ratio)</li>
                  <li>• Object detection (RetinaNet)</li>
                  <li>• Hard example mining</li>
                </>
              )}
              {selectedLoss === 'label_smoothing' && (
                <>
                  <li>• Prevent overconfidence</li>
                  <li>• Improve generalization</li>
                  <li>• Regularization technique</li>
                </>
              )}
              {selectedLoss === 'contrastive' && (
                <>
                  <li>• Self-supervised learning (SimCLR)</li>
                  <li>• Representation learning</li>
                  <li>• Image-text matching (CLIP)</li>
                </>
              )}
              {selectedLoss === 'kl_divergence' && (
                <>
                  <li>• Knowledge distillation</li>
                  <li>• Model compression</li>
                  <li>• Teacher-student training</li>
                </>
              )}
            </ul>
          </div>

          {/* Code */}
          <div className="bg-gray-900 rounded-lg p-4 border-2 border-gray-700 max-h-[280px] overflow-hidden flex flex-col">
            <h4 className="font-bold text-gray-200 mb-3 flex items-center gap-2">
              <Code className="w-5 h-5 text-green-400" />
              PyTorch Implementation
            </h4>
            <pre className="flex-1 overflow-y-auto text-xs text-green-400 font-mono bg-gray-950 rounded p-3 border border-gray-700">
              {generateCode()}
            </pre>
          </div>
        </div>
      </div>

      {/* Comparison Table */}
      <div className="mt-6 bg-white rounded-lg p-4 border-2 border-gray-200 overflow-x-auto">
        <h4 className="font-bold text-gray-700 mb-3">Quick Comparison</h4>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b-2 border-gray-200">
              <th className="text-left py-2 px-3">Loss Function</th>
              <th className="text-left py-2 px-3">Best For</th>
              <th className="text-left py-2 px-3">Key Parameters</th>
              <th className="text-left py-2 px-3">Complexity</th>
            </tr>
          </thead>
          <tbody className="text-gray-700">
            <tr className="border-b border-gray-100">
              <td className="py-2 px-3 font-semibold">Cross Entropy</td>
              <td className="py-2 px-3">Balanced data</td>
              <td className="py-2 px-3">None</td>
              <td className="py-2 px-3">O(N)</td>
            </tr>
            <tr className="border-b border-gray-100">
              <td className="py-2 px-3 font-semibold">Focal Loss</td>
              <td className="py-2 px-3">Imbalanced classes</td>
              <td className="py-2 px-3">α, γ</td>
              <td className="py-2 px-3">O(N)</td>
            </tr>
            <tr className="border-b border-gray-100">
              <td className="py-2 px-3 font-semibold">Label Smoothing</td>
              <td className="py-2 px-3">Prevent overfitting</td>
              <td className="py-2 px-3">ε</td>
              <td className="py-2 px-3">O(N)</td>
            </tr>
            <tr className="border-b border-gray-100">
              <td className="py-2 px-3 font-semibold">Contrastive</td>
              <td className="py-2 px-3">Self-supervised</td>
              <td className="py-2 px-3">τ (temperature)</td>
              <td className="py-2 px-3">O(N²)</td>
            </tr>
            <tr>
              <td className="py-2 px-3 font-semibold">KL Divergence</td>
              <td className="py-2 px-3">Distillation</td>
              <td className="py-2 px-3">T, α</td>
              <td className="py-2 px-3">O(N)</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  )
}
