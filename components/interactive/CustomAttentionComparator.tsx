'use client'

import React, { useState, useMemo } from 'react'
import { motion } from 'framer-motion'
import { Zap, Wind, Grid3x3, Sparkles, Eye, Code } from 'lucide-react'

type AttentionType = 'full' | 'local' | 'strided' | 'sparse'

interface AttentionPattern {
  type: AttentionType
  name: string
  description: string
  complexity: string
  color: string
  icon: JSX.Element
}

const attentionPatterns: AttentionPattern[] = [
  {
    type: 'full',
    name: 'Full Attention',
    description: 'Every token attends to all tokens',
    complexity: 'O(n²)',
    color: 'from-blue-500 to-cyan-500',
    icon: <Grid3x3 className="w-5 h-5" />
  },
  {
    type: 'local',
    name: 'Local Window',
    description: 'Attends within sliding window',
    complexity: 'O(n×w)',
    color: 'from-green-500 to-emerald-500',
    icon: <Wind className="w-5 h-5" />
  },
  {
    type: 'strided',
    name: 'Strided Attention',
    description: 'Attends every stride positions',
    complexity: 'O(n×n/s)',
    color: 'from-yellow-500 to-orange-500',
    icon: <Zap className="w-5 h-5" />
  },
  {
    type: 'sparse',
    name: 'Sparse Random',
    description: 'Random subset of positions',
    complexity: 'O(n×k)',
    color: 'from-purple-500 to-pink-500',
    icon: <Sparkles className="w-5 h-5" />
  }
]

export default function CustomAttentionComparator() {
  const [selectedPattern, setSelectedPattern] = useState<AttentionType>('full')
  const [seqLength, setSeqLength] = useState<number>(16)
  const [windowSize, setWindowSize] = useState<number>(4)
  const [stride, setStride] = useState<number>(4)
  const [sparseK, setSparseK] = useState<number>(4)
  const [hoveredCell, setHoveredCell] = useState<{ row: number; col: number } | null>(null)

  // 生成注意力矩阵
  const attentionMatrix = useMemo(() => {
    const matrix: boolean[][] = Array(seqLength)
      .fill(null)
      .map(() => Array(seqLength).fill(false))

    for (let i = 0; i < seqLength; i++) {
      for (let j = 0; j < seqLength; j++) {
        switch (selectedPattern) {
          case 'full':
            matrix[i][j] = true
            break
          
          case 'local':
            // 局部窗口：|i - j| <= windowSize
            if (Math.abs(i - j) <= windowSize) {
              matrix[i][j] = true
            }
            break
          
          case 'strided':
            // 跨步注意力：自己 + stride 的倍数位置
            if (i === j || j % stride === 0) {
              matrix[i][j] = true
            }
            break
          
          case 'sparse':
            // 稀疏随机：自己 + 随机 k 个位置（确定性随机）
            if (i === j) {
              matrix[i][j] = true
            } else {
              // 使用简单哈希生成确定性随机
              const hash = (i * 31 + j * 17) % seqLength
              if (hash < sparseK) {
                matrix[i][j] = true
              }
            }
            break
        }
      }
    }

    return matrix
  }, [selectedPattern, seqLength, windowSize, stride, sparseK])

  // 计算统计信息
  const stats = useMemo(() => {
    let totalConnections = 0
    let maxConnections = 0
    let minConnections = seqLength

    attentionMatrix.forEach(row => {
      const connections = row.filter(Boolean).length
      totalConnections += connections
      maxConnections = Math.max(maxConnections, connections)
      minConnections = Math.min(minConnections, connections)
    })

    const coverage = (totalConnections / (seqLength * seqLength)) * 100

    return {
      totalConnections,
      coverage: coverage.toFixed(1),
      avgConnections: (totalConnections / seqLength).toFixed(1),
      maxConnections,
      minConnections
    }
  }, [attentionMatrix, seqLength])

  // 生成 PyTorch 代码
  const generateCode = () => {
    const pattern = attentionPatterns.find(p => p.type === selectedPattern)!
    
    let code = `import torch\nimport torch.nn as nn\nimport math\n\n`
    
    switch (selectedPattern) {
      case 'full':
        code += `# Full Attention (Standard Multi-Head Attention)\n`
        code += `attention_scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(d_k)\n`
        code += `attention_probs = torch.softmax(attention_scores, dim=-1)\n`
        code += `output = torch.matmul(attention_probs, V)\n`
        break
      
      case 'local':
        code += `# Local Window Attention\n`
        code += `def create_local_mask(seq_len, window_size, device):\n`
        code += `    positions = torch.arange(seq_len, device=device)\n`
        code += `    distance = torch.abs(positions.unsqueeze(1) - positions.unsqueeze(0))\n`
        code += `    mask = (distance <= window_size).float()\n`
        code += `    # Convert to attention mask (0 -> -inf, 1 -> 0)\n`
        code += `    attention_mask = (1.0 - mask) * torch.finfo(torch.float32).min\n`
        code += `    return attention_mask\n\n`
        code += `# Apply mask\n`
        code += `local_mask = create_local_mask(seq_len, window_size=${windowSize}, device=Q.device)\n`
        code += `attention_scores = attention_scores + local_mask\n`
        code += `attention_probs = torch.softmax(attention_scores, dim=-1)\n`
        break
      
      case 'strided':
        code += `# Strided Attention\n`
        code += `def create_strided_mask(seq_len, stride, device):\n`
        code += `    positions = torch.arange(seq_len, device=device)\n`
        code += `    # Attend to: (1) self (2) positions that are multiples of stride\n`
        code += `    mask = torch.zeros(seq_len, seq_len, device=device)\n`
        code += `    strided_indices = positions % stride == 0\n`
        code += `    mask[:, strided_indices] = 1.0  # All attend to stride positions\n`
        code += `    mask.fill_diagonal_(1.0)  # Attend to self\n`
        code += `    attention_mask = (1.0 - mask) * torch.finfo(torch.float32).min\n`
        code += `    return attention_mask\n\n`
        code += `strided_mask = create_strided_mask(seq_len, stride=${stride}, device=Q.device)\n`
        code += `attention_scores = attention_scores + strided_mask\n`
        break
      
      case 'sparse':
        code += `# Sparse Random Attention\n`
        code += `def create_sparse_mask(seq_len, k, device):\n`
        code += `    # Randomly select k positions for each query\n`
        code += `    mask = torch.zeros(seq_len, seq_len, device=device)\n`
        code += `    for i in range(seq_len):\n`
        code += `        # Always attend to self\n`
        code += `        indices = [i]\n`
        code += `        # Randomly select k-1 other positions\n`
        code += `        other_indices = torch.randperm(seq_len, device=device)[:k-1]\n`
        code += `        indices.extend(other_indices.tolist())\n`
        code += `        mask[i, indices] = 1.0\n`
        code += `    attention_mask = (1.0 - mask) * torch.finfo(torch.float32).min\n`
        code += `    return attention_mask\n\n`
        code += `sparse_mask = create_sparse_mask(seq_len, k=${sparseK}, device=Q.device)\n`
        code += `attention_scores = attention_scores + sparse_mask\n`
        break
    }

    return code
  }

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-purple-50 rounded-xl border-2 border-purple-200">
      <div className="flex items-center gap-3 mb-6">
        <div className="p-2 bg-gradient-to-br from-purple-500 to-pink-600 rounded-lg">
          <Eye className="w-6 h-6 text-white" />
        </div>
        <div>
          <h3 className="text-2xl font-bold text-gray-800">Custom Attention Comparator</h3>
          <p className="text-sm text-gray-600">Compare different attention mechanisms</p>
        </div>
      </div>

      {/* Attention Pattern Selection */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
        {attentionPatterns.map((pattern) => (
          <motion.button
            key={pattern.type}
            whileHover={{ scale: 1.03 }}
            whileTap={{ scale: 0.97 }}
            onClick={() => setSelectedPattern(pattern.type)}
            className={`p-4 rounded-lg border-2 transition-all ${
              selectedPattern === pattern.type
                ? 'border-purple-500 bg-gradient-to-r ' + pattern.color + ' text-white'
                : 'border-gray-200 bg-white text-gray-700 hover:border-gray-300'
            }`}
          >
            <div className="flex items-center gap-2 mb-2">
              {pattern.icon}
              <span className="font-bold text-sm">{pattern.name}</span>
            </div>
            <p className={`text-xs ${selectedPattern === pattern.type ? 'text-white/90' : 'text-gray-600'}`}>
              {pattern.description}
            </p>
            <div className={`mt-2 text-xs font-mono ${selectedPattern === pattern.type ? 'text-white/80' : 'text-gray-500'}`}>
              {pattern.complexity}
            </div>
          </motion.button>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left: Configuration & Matrix */}
        <div className="space-y-4">
          {/* Configuration */}
          <div className="bg-white rounded-lg p-4 border-2 border-gray-200">
            <h4 className="font-bold text-gray-700 mb-3">Configuration</h4>
            
            {/* Sequence Length */}
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Sequence Length: <span className="text-blue-600 font-bold">{seqLength}</span>
              </label>
              <input
                type="range"
                min="8"
                max="32"
                step="4"
                value={seqLength}
                onChange={(e) => setSeqLength(Number(e.target.value))}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-500"
              />
            </div>

            {/* Pattern-specific parameters */}
            {selectedPattern === 'local' && (
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Window Size: <span className="text-green-600 font-bold">{windowSize}</span>
                </label>
                <input
                  type="range"
                  min="1"
                  max={Math.floor(seqLength / 2)}
                  value={windowSize}
                  onChange={(e) => setWindowSize(Number(e.target.value))}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-green-500"
                />
              </div>
            )}

            {selectedPattern === 'strided' && (
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Stride: <span className="text-yellow-600 font-bold">{stride}</span>
                </label>
                <input
                  type="range"
                  min="2"
                  max={Math.floor(seqLength / 2)}
                  value={stride}
                  onChange={(e) => setStride(Number(e.target.value))}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-yellow-500"
                />
              </div>
            )}

            {selectedPattern === 'sparse' && (
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Sparse K: <span className="text-purple-600 font-bold">{sparseK}</span>
                </label>
                <input
                  type="range"
                  min="2"
                  max={Math.min(8, seqLength)}
                  value={sparseK}
                  onChange={(e) => setSparseK(Number(e.target.value))}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-purple-500"
                />
              </div>
            )}
          </div>

          {/* Attention Matrix Visualization */}
          <div className="bg-white rounded-lg p-4 border-2 border-gray-200">
            <h4 className="font-bold text-gray-700 mb-3">Attention Matrix</h4>
            <div className="flex items-start gap-4">
              {/* Y-axis label */}
              <div className="flex flex-col justify-center items-end" style={{ height: `${seqLength * 20}px` }}>
                <span className="text-xs text-gray-500 transform -rotate-90 whitespace-nowrap">Query Position</span>
              </div>
              
              {/* Matrix */}
              <div>
                <div className="grid gap-0.5" style={{ gridTemplateColumns: `repeat(${seqLength}, 20px)` }}>
                  {attentionMatrix.map((row, i) =>
                    row.map((attended, j) => (
                      <motion.div
                        key={`${i}-${j}`}
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        transition={{ delay: (i * seqLength + j) * 0.005 }}
                        onMouseEnter={() => setHoveredCell({ row: i, col: j })}
                        onMouseLeave={() => setHoveredCell(null)}
                        className={`w-5 h-5 rounded-sm ${
                          attended
                            ? 'bg-gradient-to-br from-blue-400 to-blue-600'
                            : 'bg-gray-200'
                        } ${hoveredCell?.row === i && hoveredCell?.col === j ? 'ring-2 ring-yellow-400' : ''}`}
                        title={`Query ${i} → Key ${j}: ${attended ? 'Attended' : 'Masked'}`}
                      />
                    ))
                  )}
                </div>
                {/* X-axis label */}
                <div className="text-center mt-2">
                  <span className="text-xs text-gray-500">Key Position</span>
                </div>
              </div>
            </div>

            {/* Hover info */}
            {hoveredCell && (
              <div className="mt-3 p-2 bg-blue-50 rounded border border-blue-200 text-sm">
                <span className="font-semibold">Query {hoveredCell.row}</span> 
                {attentionMatrix[hoveredCell.row][hoveredCell.col] ? ' can attend to ' : ' cannot attend to '}
                <span className="font-semibold">Key {hoveredCell.col}</span>
              </div>
            )}
          </div>
        </div>

        {/* Right: Statistics & Code */}
        <div className="space-y-4">
          {/* Statistics */}
          <div className="bg-white rounded-lg p-4 border-2 border-gray-200">
            <h4 className="font-bold text-gray-700 mb-3">Statistics</h4>
            <div className="grid grid-cols-2 gap-3 text-sm">
              <div className="p-3 bg-blue-50 rounded-lg border border-blue-200">
                <div className="text-gray-600 mb-1">Coverage</div>
                <div className="text-2xl font-bold text-blue-600">{stats.coverage}%</div>
              </div>
              <div className="p-3 bg-green-50 rounded-lg border border-green-200">
                <div className="text-gray-600 mb-1">Avg Connections</div>
                <div className="text-2xl font-bold text-green-600">{stats.avgConnections}</div>
              </div>
              <div className="p-3 bg-purple-50 rounded-lg border border-purple-200">
                <div className="text-gray-600 mb-1">Max Connections</div>
                <div className="text-2xl font-bold text-purple-600">{stats.maxConnections}</div>
              </div>
              <div className="p-3 bg-orange-50 rounded-lg border border-orange-200">
                <div className="text-gray-600 mb-1">Min Connections</div>
                <div className="text-2xl font-bold text-orange-600">{stats.minConnections}</div>
              </div>
            </div>

            {/* Complexity comparison */}
            <div className="mt-4 p-3 bg-gray-50 rounded-lg border border-gray-200">
              <div className="text-xs text-gray-600 mb-2">Computational Complexity:</div>
              <div className="font-mono text-sm font-bold text-gray-800">
                {attentionPatterns.find(p => p.type === selectedPattern)?.complexity}
              </div>
              <div className="text-xs text-gray-500 mt-1">
                {selectedPattern === 'full' && 'Quadratic scaling - best for short sequences'}
                {selectedPattern === 'local' && `Linear with window=${windowSize} - efficient for local patterns`}
                {selectedPattern === 'strided' && `Reduced by stride=${stride} - balances local & global`}
                {selectedPattern === 'sparse' && `Linear with k=${sparseK} - most efficient`}
              </div>
            </div>
          </div>

          {/* Generated Code */}
          <div className="bg-gray-900 rounded-lg p-4 border-2 border-gray-700 max-h-[400px] overflow-hidden flex flex-col">
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
        <h4 className="font-bold text-gray-700 mb-3">Pattern Comparison</h4>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b-2 border-gray-200">
              <th className="text-left py-2 px-3">Pattern</th>
              <th className="text-left py-2 px-3">Complexity</th>
              <th className="text-left py-2 px-3">Pros</th>
              <th className="text-left py-2 px-3">Cons</th>
              <th className="text-left py-2 px-3">Use Cases</th>
            </tr>
          </thead>
          <tbody className="text-gray-700">
            <tr className="border-b border-gray-100">
              <td className="py-2 px-3 font-semibold">Full Attention</td>
              <td className="py-2 px-3 font-mono text-xs">O(n²)</td>
              <td className="py-2 px-3 text-green-600">✓ Global context</td>
              <td className="py-2 px-3 text-red-600">✗ Memory intensive</td>
              <td className="py-2 px-3">Seq ≤ 512 tokens</td>
            </tr>
            <tr className="border-b border-gray-100">
              <td className="py-2 px-3 font-semibold">Local Window</td>
              <td className="py-2 px-3 font-mono text-xs">O(n×w)</td>
              <td className="py-2 px-3 text-green-600">✓ Efficient, local patterns</td>
              <td className="py-2 px-3 text-red-600">✗ No long-range</td>
              <td className="py-2 px-3">Documents, Code</td>
            </tr>
            <tr className="border-b border-gray-100">
              <td className="py-2 px-3 font-semibold">Strided</td>
              <td className="py-2 px-3 font-mono text-xs">O(n×n/s)</td>
              <td className="py-2 px-3 text-green-600">✓ Balanced</td>
              <td className="py-2 px-3 text-red-600">✗ Complex impl</td>
              <td className="py-2 px-3">Longformer, BigBird</td>
            </tr>
            <tr>
              <td className="py-2 px-3 font-semibold">Sparse Random</td>
              <td className="py-2 px-3 font-mono text-xs">O(n×k)</td>
              <td className="py-2 px-3 text-green-600">✓ Extreme efficiency</td>
              <td className="py-2 px-3 text-red-600">✗ Unstable</td>
              <td className="py-2 px-3">Research exploration</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  )
}
