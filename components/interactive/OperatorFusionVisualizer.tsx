"use client"

import React, { useState } from 'react'
import { motion } from 'framer-motion'

type OptimizationPass = 'original' | 'constant-folding' | 'operator-fusion' | 'all'

interface GraphNode {
  id: string
  type: string
  label: string
  inputs: string[]
  outputs: string[]
}

interface OptimizationExample {
  name: string
  before: GraphNode[]
  after: GraphNode[]
  speedup: number
  description: string
}

const OperatorFusionVisualizer: React.FC = () => {
  const [selectedPass, setSelectedPass] = useState<OptimizationPass>('original')
  const [selectedExample, setSelectedExample] = useState(0)

  const examples: OptimizationExample[] = [
    {
      name: 'MatMul + Add → Gemm',
      speedup: 1.2,
      description: '矩阵乘法与加法融合为单个 Gemm 算子',
      before: [
        { id: 'input', type: 'Input', label: 'Input\n[B, 768]', inputs: [], outputs: ['mm'] },
        { id: 'weight', type: 'Constant', label: 'Weight\n[768, 3072]', inputs: [], outputs: ['mm'] },
        { id: 'mm', type: 'MatMul', label: 'MatMul', inputs: ['input', 'weight'], outputs: ['add'] },
        { id: 'bias', type: 'Constant', label: 'Bias\n[3072]', inputs: [], outputs: ['add'] },
        { id: 'add', type: 'Add', label: 'Add', inputs: ['mm', 'bias'], outputs: ['output'] },
        { id: 'output', type: 'Output', label: 'Output\n[B, 3072]', inputs: ['add'], outputs: [] },
      ],
      after: [
        { id: 'input', type: 'Input', label: 'Input\n[B, 768]', inputs: [], outputs: ['gemm'] },
        { id: 'weight', type: 'Constant', label: 'Weight\n[768, 3072]', inputs: [], outputs: ['gemm'] },
        { id: 'bias', type: 'Constant', label: 'Bias\n[3072]', inputs: [], outputs: ['gemm'] },
        { id: 'gemm', type: 'Gemm', label: 'Gemm\n(Fused)', inputs: ['input', 'weight', 'bias'], outputs: ['output'] },
        { id: 'output', type: 'Output', label: 'Output\n[B, 3072]', inputs: ['gemm'], outputs: [] },
      ],
    },
    {
      name: 'LayerNorm + Add 融合',
      speedup: 1.5,
      description: 'Transformer 中常见的 LayerNorm 与残差连接融合',
      before: [
        { id: 'x', type: 'Input', label: 'X', inputs: [], outputs: ['ln', 'add'] },
        { id: 'ln', type: 'LayerNorm', label: 'LayerNorm', inputs: ['x'], outputs: ['add'] },
        { id: 'add', type: 'Add', label: 'Add\n(Residual)', inputs: ['ln', 'x'], outputs: ['output'] },
        { id: 'output', type: 'Output', label: 'Output', inputs: ['add'], outputs: [] },
      ],
      after: [
        { id: 'x', type: 'Input', label: 'X', inputs: [], outputs: ['fused'] },
        { id: 'fused', type: 'FusedLayerNorm', label: 'FusedLayerNorm\n+ Residual', inputs: ['x'], outputs: ['output'] },
        { id: 'output', type: 'Output', label: 'Output', inputs: ['fused'], outputs: [] },
      ],
    },
    {
      name: 'GELU 近似优化',
      speedup: 2.1,
      description: '将精确 GELU 替换为 FastGELU 近似',
      before: [
        { id: 'x', type: 'Input', label: 'X', inputs: [], outputs: ['div'] },
        { id: 'div', type: 'Div', label: 'Div\n(x / √2)', inputs: ['x'], outputs: ['erf'] },
        { id: 'erf', type: 'Erf', label: 'Erf', inputs: ['div'], outputs: ['add1'] },
        { id: 'add1', type: 'Add', label: 'Add\n(+1)', inputs: ['erf'], outputs: ['mul1'] },
        { id: 'mul1', type: 'Mul', label: 'Mul\n(×0.5)', inputs: ['add1'], outputs: ['mul2'] },
        { id: 'mul2', type: 'Mul', label: 'Mul\n(×x)', inputs: ['mul1', 'x'], outputs: ['output'] },
        { id: 'output', type: 'Output', label: 'Output', inputs: ['mul2'], outputs: [] },
      ],
      after: [
        { id: 'x', type: 'Input', label: 'X', inputs: [], outputs: ['gelu'] },
        { id: 'gelu', type: 'FastGELU', label: 'FastGELU\n(Approximation)', inputs: ['x'], outputs: ['output'] },
        { id: 'output', type: 'Output', label: 'Output', inputs: ['gelu'], outputs: [] },
      ],
    },
    {
      name: 'Multi-Head Attention 融合',
      speedup: 3.2,
      description: '将多个注意力计算步骤融合为单个高效算子',
      before: [
        { id: 'x', type: 'Input', label: 'X', inputs: [], outputs: ['q', 'k', 'v'] },
        { id: 'q', type: 'Linear', label: 'Q Linear', inputs: ['x'], outputs: ['reshape1'] },
        { id: 'k', type: 'Linear', label: 'K Linear', inputs: ['x'], outputs: ['reshape2'] },
        { id: 'v', type: 'Linear', label: 'V Linear', inputs: ['x'], outputs: ['reshape3'] },
        { id: 'reshape1', type: 'Reshape', label: 'Reshape Q', inputs: ['q'], outputs: ['mm1'] },
        { id: 'reshape2', type: 'Reshape', label: 'Reshape K', inputs: ['k'], outputs: ['transpose'] },
        { id: 'transpose', type: 'Transpose', label: 'Transpose K', inputs: ['reshape2'], outputs: ['mm1'] },
        { id: 'mm1', type: 'MatMul', label: 'Q @ K^T', inputs: ['reshape1', 'transpose'], outputs: ['softmax'] },
        { id: 'softmax', type: 'Softmax', label: 'Softmax', inputs: ['mm1'], outputs: ['mm2'] },
        { id: 'reshape3', type: 'Reshape', label: 'Reshape V', inputs: ['v'], outputs: ['mm2'] },
        { id: 'mm2', type: 'MatMul', label: 'Attn @ V', inputs: ['softmax', 'reshape3'], outputs: ['output'] },
        { id: 'output', type: 'Output', label: 'Output', inputs: ['mm2'], outputs: [] },
      ],
      after: [
        { id: 'x', type: 'Input', label: 'X', inputs: [], outputs: ['attn'] },
        { id: 'q_w', type: 'Constant', label: 'Q Weight', inputs: [], outputs: ['attn'] },
        { id: 'k_w', type: 'Constant', label: 'K Weight', inputs: [], outputs: ['attn'] },
        { id: 'v_w', type: 'Constant', label: 'V Weight', inputs: [], outputs: ['attn'] },
        { id: 'attn', type: 'FusedAttention', label: 'FusedAttention\n(All-in-one)', inputs: ['x', 'q_w', 'k_w', 'v_w'], outputs: ['output'] },
        { id: 'output', type: 'Output', label: 'Output', inputs: ['attn'], outputs: [] },
      ],
    },
  ]

  const currentExample = examples[selectedExample]

  const getNodeColor = (type: string) => {
    const colors: Record<string, string> = {
      Input: 'bg-blue-500',
      Output: 'bg-green-500',
      Constant: 'bg-gray-400',
      MatMul: 'bg-purple-500',
      Add: 'bg-orange-500',
      Gemm: 'bg-pink-500',
      LayerNorm: 'bg-indigo-500',
      FusedLayerNorm: 'bg-pink-500',
      Erf: 'bg-yellow-500',
      Div: 'bg-orange-400',
      Mul: 'bg-orange-400',
      FastGELU: 'bg-pink-500',
      Linear: 'bg-purple-400',
      Reshape: 'bg-teal-400',
      Transpose: 'bg-teal-500',
      Softmax: 'bg-red-400',
      FusedAttention: 'bg-pink-500',
    }
    return colors[type] || 'bg-gray-500'
  }

  const optimizationPasses = [
    { id: 'original' as OptimizationPass, label: '原始图', icon: '📊' },
    { id: 'constant-folding' as OptimizationPass, label: '常量折叠', icon: '📁' },
    { id: 'operator-fusion' as OptimizationPass, label: '算子融合', icon: '🔀' },
    { id: 'all' as OptimizationPass, label: '全部优化', icon: '⚡' },
  ]

  const renderGraph = (nodes: GraphNode[], title: string) => {
    return (
      <div className="flex-1">
        <h4 className="font-semibold mb-3 text-center">{title}</h4>
        <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4 min-h-[300px] flex flex-col items-center justify-center gap-3">
          {nodes.map((node, idx) => (
            <motion.div
              key={node.id}
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: idx * 0.1 }}
              className="relative"
            >
              {/* 输入连线 */}
              {node.inputs.length > 0 && (
                <div className="absolute -top-3 left-1/2 -translate-x-1/2 w-0.5 h-3 bg-gray-400"></div>
              )}

              {/* 节点 */}
              <div
                className={`${getNodeColor(
                  node.type
                )} text-white px-4 py-2 rounded-lg shadow-md text-xs font-medium text-center min-w-[120px] whitespace-pre-line`}
              >
                {node.label}
              </div>

              {/* 输出连线 */}
              {node.outputs.length > 0 && (
                <div className="absolute -bottom-3 left-1/2 -translate-x-1/2 w-0.5 h-3 bg-gray-400"></div>
              )}
            </motion.div>
          ))}
        </div>
        <div className="mt-3 text-center">
          <span className="text-sm text-gray-600 dark:text-gray-400">
            节点数: <strong>{nodes.length}</strong>
          </span>
        </div>
      </div>
    )
  }

  return (
    <div className="w-full space-y-6 my-8">
      {/* 标题 */}
      <div className="text-center">
        <h3 className="text-2xl font-bold mb-2">ONNX 图优化：算子融合</h3>
        <p className="text-gray-600 dark:text-gray-400">
          通过融合多个算子提升推理性能
        </p>
      </div>

      {/* 示例选择 */}
      <div>
        <h4 className="font-semibold mb-3">选择优化示例：</h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
          {examples.map((example, idx) => (
            <button
              key={idx}
              onClick={() => setSelectedExample(idx)}
              className={`p-3 rounded-lg text-left transition-all ${
                selectedExample === idx
                  ? 'bg-indigo-500 text-white shadow-lg'
                  : 'bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700'
              }`}
            >
              <div className="text-xs font-medium">{example.name}</div>
              <div className="text-xs opacity-80 mt-1">
                {example.speedup.toFixed(1)}x 加速
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* 描述 */}
      <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
        <p className="text-sm">
          <strong>{currentExample.name}</strong>: {currentExample.description}
        </p>
      </div>

      {/* 计算图对比 */}
      <motion.div
        key={selectedExample}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg"
      >
        <div className="grid md:grid-cols-2 gap-6">
          {renderGraph(currentExample.before, '优化前')}
          <div className="flex items-center justify-center">
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ delay: 0.5 }}
              className="text-4xl"
            >
              →
            </motion.div>
          </div>
          {renderGraph(currentExample.after, '优化后')}
        </div>

        {/* 性能提升 */}
        <div className="mt-6 p-4 bg-gradient-to-r from-green-50 to-blue-50 dark:from-green-900/20 dark:to-blue-900/20 rounded-lg border-2 border-green-500">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-sm text-gray-600 dark:text-gray-400">
                性能提升
              </div>
              <div className="text-3xl font-bold text-green-600 dark:text-green-400">
                {currentExample.speedup.toFixed(1)}x
              </div>
            </div>
            <div>
              <div className="text-sm text-gray-600 dark:text-gray-400">
                节点减少
              </div>
              <div className="text-2xl font-bold">
                {currentExample.before.length} → {currentExample.after.length}
                <span className="text-sm ml-2 text-green-600 dark:text-green-400">
                  (-
                  {(
                    ((currentExample.before.length - currentExample.after.length) /
                      currentExample.before.length) *
                    100
                  ).toFixed(0)}
                  %)
                </span>
              </div>
            </div>
          </div>
        </div>
      </motion.div>

      {/* 优化技术说明 */}
      <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h4 className="text-lg font-bold mb-4">常见优化技术</h4>
        <div className="grid md:grid-cols-3 gap-4">
          <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
            <div className="text-2xl mb-2">📁</div>
            <h5 className="font-semibold mb-2">常量折叠</h5>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              在编译时计算常量表达式，减少运行时计算
            </p>
            <div className="mt-2 text-xs font-mono bg-gray-100 dark:bg-gray-900 p-2 rounded">
              x * 2 + 3 → x * 5
            </div>
          </div>

          <div className="p-4 bg-orange-50 dark:bg-orange-900/20 rounded-lg">
            <div className="text-2xl mb-2">🔀</div>
            <h5 className="font-semibold mb-2">算子融合</h5>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              将多个算子合并为单个高效算子
            </p>
            <div className="mt-2 text-xs font-mono bg-gray-100 dark:bg-gray-900 p-2 rounded">
              MatMul + Add → Gemm
            </div>
          </div>

          <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
            <div className="text-2xl mb-2">🗑️</div>
            <h5 className="font-semibold mb-2">死代码消除</h5>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              移除未使用的节点和参数
            </p>
            <div className="mt-2 text-xs font-mono bg-gray-100 dark:bg-gray-900 p-2 rounded">
              unused_node → (removed)
            </div>
          </div>
        </div>
      </div>

      {/* 实际应用 */}
      <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h4 className="text-lg font-bold mb-4">ONNX Runtime 优化示例</h4>
        <pre className="text-xs font-mono bg-gray-900 text-gray-100 p-4 rounded overflow-x-auto">
{`from optimum.onnxruntime import ORTModelForSequenceClassification
from optimum.onnxruntime.configuration import OptimizationConfig

# 加载模型
model = ORTModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    export=True
)

# 配置优化
optimization_config = OptimizationConfig(
    optimization_level=99,  # 最高优化级别
    optimize_for_gpu=True,
    fp16=True,
    enable_gelu_approximation=True,  # GELU 近似
    enable_transformers_specific_optimizations=True  # Transformer 优化
)

# 应用优化
model.optimize(optimization_config, save_dir="./bert_optimized")`}
        </pre>
      </div>

      {/* 性能对比表 */}
      <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg overflow-x-auto">
        <h4 className="text-lg font-bold mb-4">优化效果对比</h4>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b-2 border-gray-300 dark:border-gray-600">
              <th className="text-left py-2 px-4">优化类型</th>
              <th className="text-left py-2 px-4">融合前</th>
              <th className="text-left py-2 px-4">融合后</th>
              <th className="text-left py-2 px-4">加速比</th>
            </tr>
          </thead>
          <tbody>
            {examples.map((example, idx) => (
              <tr
                key={idx}
                className="border-b border-gray-200 dark:border-gray-700"
              >
                <td className="py-2 px-4 font-medium">{example.name}</td>
                <td className="py-2 px-4">{example.before.length} 节点</td>
                <td className="py-2 px-4">{example.after.length} 节点</td>
                <td className="py-2 px-4 text-green-600 dark:text-green-400 font-bold">
                  {example.speedup.toFixed(1)}x
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export default OperatorFusionVisualizer
