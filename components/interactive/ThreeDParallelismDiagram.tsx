'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'

type ParallelismType = 'data' | 'tensor' | 'pipeline' | '3d'

export default function ThreeDParallelismDiagram() {
  const [activeType, setActiveType] = useState<ParallelismType>('3d')

  const parallelismTypes = [
    {
      id: 'data' as ParallelismType,
      name: '数据并行 (DP)',
      description: '每个 GPU 持有完整模型副本，输入数据分片',
      color: 'blue',
      memoryPerGPU: '100%',
      communication: 'All-Reduce 梯度'
    },
    {
      id: 'tensor' as ParallelismType,
      name: '张量并行 (TP)',
      description: '层内矩阵分片，每个 GPU 计算部分',
      color: 'purple',
      memoryPerGPU: '~25%（4-way TP）',
      communication: 'All-Reduce 激活值'
    },
    {
      id: 'pipeline' as ParallelismType,
      name: '流水线并行 (PP)',
      description: '模型按层切分，形成流水线',
      color: 'green',
      memoryPerGPU: '~50%（2-stage）',
      communication: '点对点传输激活值'
    },
    {
      id: '3d' as ParallelismType,
      name: '3D 并行（组合）',
      description: 'DP + TP + PP 三维度并行',
      color: 'orange',
      memoryPerGPU: '~12.5%（DP=4, TP=8, PP=2）',
      communication: '混合通信模式'
    }
  ]

  const currentType = parallelismTypes.find(t => t.id === activeType)!

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-white dark:bg-gray-800 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold mb-6 text-center text-gray-100">
        3D 并行架构可视化
      </h3>

      {/* 并行类型选择 */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-8">
        {parallelismTypes.map(type => (
          <button
            key={type.id}
            onClick={() => setActiveType(type.id)}
            className={`p-4 rounded-lg border-2 transition-all ${
              activeType === type.id
                ? `border-${type.color}-500 bg-${type.color}-50 dark:bg-${type.color}-900/30`
                : 'border-gray-300 dark:border-gray-600 hover:border-gray-400'
            }`}
          >
            <div className={`font-bold mb-1 ${
              activeType === type.id 
                ? `text-${type.color}-700 dark:text-${type.color}-300` 
                : 'text-gray-100'
            }`}>
              {type.name}
            </div>
            <div className="text-xs text-gray-300">
              {type.memoryPerGPU}
            </div>
          </button>
        ))}
      </div>

      {/* 当前类型说明 */}
      <div className={`p-4 mb-6 rounded-lg bg-${currentType.color}-50 dark:bg-${currentType.color}-900/20 border-2 border-${currentType.color}-300 dark:border-${currentType.color}-700`}>
        <h4 className={`font-bold text-${currentType.color}-900 dark:text-${currentType.color}-300 mb-2`}>
          {currentType.name}
        </h4>
        <p className={`text-sm text-${currentType.color}-800 dark:text-${currentType.color}-200 mb-2`}>
          {currentType.description}
        </p>
        <div className="grid grid-cols-2 gap-4 text-xs">
          <div>
            <span className={`font-semibold text-${currentType.color}-900 dark:text-${currentType.color}-300`}>
              显存占用：
            </span>
            <span className={`text-${currentType.color}-700 dark:text-${currentType.color}-400 ml-1`}>
              {currentType.memoryPerGPU}
            </span>
          </div>
          <div>
            <span className={`font-semibold text-${currentType.color}-900 dark:text-${currentType.color}-300`}>
              通信模式：
            </span>
            <span className={`text-${currentType.color}-700 dark:text-${currentType.color}-400 ml-1`}>
              {currentType.communication}
            </span>
          </div>
        </div>
      </div>

      {/* 可视化图 */}
      <div className="bg-gray-50 dark:bg-gray-900 rounded-xl p-8 mb-6 overflow-x-auto">
        {activeType === 'data' && <DataParallelismViz />}
        {activeType === 'tensor' && <TensorParallelismViz />}
        {activeType === 'pipeline' && <PipelineParallelismViz />}
        {activeType === '3d' && <ThreeDParallelismViz />}
      </div>

      {/* 配置示例 */}
      <div className="p-4 bg-gray-900 dark:bg-black rounded-lg">
        <p className="text-xs text-gray-400 mb-2">
          DeepSpeed 配置示例（175B 模型，64 GPU）：
        </p>
        <pre className="text-sm text-green-400 overflow-x-auto">
{activeType === 'data' && `{
  "train_batch_size": 512,
  "train_micro_batch_size_per_gpu": 8,
  "zero_optimization": {"stage": 2}
  // 8 GPU × DDP，每 GPU 存储完整模型
}`}
{activeType === 'tensor' && `{
  "tensor_parallel": {"tp_size": 8},
  // 每层的 Q/K/V 矩阵分片到 8 GPU
  // 单层显存占用降至 1/8
}`}
{activeType === 'pipeline' && `{
  "pipeline": {
    "stages": 4,  // 模型切分为 4 个阶段
    "partition": "type:transformer",
    "micro_batch_size": 1
  }
  // 每个 stage 包含 1/4 的模型层
}`}
{activeType === '3d' && `{
  "train_batch_size": 512,
  "train_micro_batch_size_per_gpu": 1,
  "gradient_accumulation_steps": 128,
  
  // 数据并行维度
  "zero_optimization": {"stage": 3},
  
  // 张量并行维度
  "tensor_parallel": {"tp_size": 8},
  
  // 流水线并行维度
  "pipeline": {"stages": 2}
  
  // 总 GPU = DP(4) × TP(8) × PP(2) = 64
}`}
        </pre>
      </div>

      {/* 性能对比 */}
      <div className="mt-6 overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="bg-gray-100 dark:bg-gray-700">
            <tr>
              <th className="px-4 py-2 text-left text-gray-900 dark:text-gray-100">并行类型</th>
              <th className="px-4 py-2 text-left text-gray-900 dark:text-gray-100">显存占用</th>
              <th className="px-4 py-2 text-left text-gray-900 dark:text-gray-100">通信开销</th>
              <th className="px-4 py-2 text-left text-gray-900 dark:text-gray-100">扩展性</th>
              <th className="px-4 py-2 text-left text-gray-900 dark:text-gray-100">适用场景</th>
            </tr>
          </thead>
          <tbody className="text-gray-100">
            <tr className="border-b border-gray-200 dark:border-gray-600">
              <td className="px-4 py-3 font-semibold">数据并行</td>
              <td className="px-4 py-3">100%/GPU</td>
              <td className="px-4 py-3 text-yellow-600">中（All-Reduce）</td>
              <td className="px-4 py-3 text-green-600">好（线性）</td>
              <td className="px-4 py-3">中小模型（&lt;13B）</td>
            </tr>
            <tr className="border-b border-gray-200 dark:border-gray-600">
              <td className="px-4 py-3 font-semibold">张量并行</td>
              <td className="px-4 py-3">12.5%-25%/GPU</td>
              <td className="px-4 py-3 text-red-600">高（频繁通信）</td>
              <td className="px-4 py-3 text-yellow-600">中（受带宽限制）</td>
              <td className="px-4 py-3">单节点内大模型</td>
            </tr>
            <tr className="border-b border-gray-200 dark:border-gray-600">
              <td className="px-4 py-3 font-semibold">流水线并行</td>
              <td className="px-4 py-3">25%-50%/GPU</td>
              <td className="px-4 py-3 text-green-600">低（点对点）</td>
              <td className="px-4 py-3 text-red-600">差（气泡时间）</td>
              <td className="px-4 py-3">超深模型</td>
            </tr>
            <tr>
              <td className="px-4 py-3 font-semibold">3D 并行</td>
              <td className="px-4 py-3 text-green-600">5%-15%/GPU</td>
              <td className="px-4 py-3 text-yellow-600">中（混合）</td>
              <td className="px-4 py-3 text-green-600">优（多维度）</td>
              <td className="px-4 py-3 text-orange-600">超大模型（&gt;100B）</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  )
}

// 数据并行可视化
function DataParallelismViz() {
  return (
    <div className="grid grid-cols-4 gap-4">
      {[0, 1, 2, 3].map(i => (
        <motion.div
          key={i}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: i * 0.1 }}
          className="bg-blue-500 rounded-lg p-4 text-white text-center"
        >
          <div className="text-lg font-bold mb-2">GPU {i}</div>
          <div className="text-xs mb-2">完整模型副本</div>
          <div className="bg-blue-700 rounded p-2 text-xs mb-1">参数 θ (100%)</div>
          <div className="bg-blue-600 rounded p-2 text-xs mb-1">梯度 ∇L (100%)</div>
          <div className="bg-blue-800 rounded p-2 text-xs">数据分片 {i+1}/4</div>
        </motion.div>
      ))}
      <div className="col-span-4 text-center text-sm text-gray-300 mt-4">
        ↓ All-Reduce 梯度 ↓ 同步模型参数
      </div>
    </div>
  )
}

// 张量并行可视化
function TensorParallelismViz() {
  return (
    <div className="space-y-4">
      <div className="text-center text-sm text-gray-300 mb-2">
        单层 Attention 矩阵分片（Q/K/V）
      </div>
      <div className="grid grid-cols-8 gap-2">
        {[0, 1, 2, 3, 4, 5, 6, 7].map(i => (
          <motion.div
            key={i}
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: i * 0.05 }}
            className="bg-purple-500 rounded-lg p-3 text-white text-center"
          >
            <div className="text-sm font-bold mb-1">GPU {i}</div>
            <div className="text-xs mb-2">Q/K/V 分片 {i+1}/8</div>
            <div className="bg-purple-700 rounded p-1 text-xs">1/8 矩阵</div>
          </motion.div>
        ))}
      </div>
      <div className="text-center text-sm text-gray-300 mt-4">
        ↓ All-Reduce 激活值 ↓ 每层 2 次通信
      </div>
    </div>
  )
}

// 流水线并行可视化
function PipelineParallelismViz() {
  return (
    <div className="flex items-center justify-between">
      {[
        { stage: 0, layers: 'L0-L7', color: 'green-500' },
        { stage: 1, layers: 'L8-L15', color: 'green-600' },
        { stage: 2, layers: 'L16-L23', color: 'green-700' },
        { stage: 3, layers: 'L24-L31', color: 'green-800' }
      ].map((stage, i) => (
        <div key={i} className="flex items-center">
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: i * 0.15 }}
            className={`bg-${stage.color} rounded-lg p-4 text-white text-center w-32`}
          >
            <div className="text-lg font-bold mb-1">Stage {stage.stage}</div>
            <div className="text-xs mb-2">GPU {stage.stage}</div>
            <div className="bg-green-900 rounded p-2 text-xs">
              Layers {stage.layers}
            </div>
          </motion.div>
          {i < 3 && (
            <div className="mx-2 text-2xl text-gray-400">→</div>
          )}
        </div>
      ))}
    </div>
  )
}

// 3D 并行可视化
function ThreeDParallelismViz() {
  return (
    <div className="space-y-6">
      <div className="text-center text-sm font-semibold text-gray-100 mb-4">
        64 GPU 配置：DP=4 × TP=8 × PP=2
      </div>

      {/* 2 个流水线阶段 */}
      {[0, 1].map(ppStage => (
        <div key={ppStage} className="border-2 border-orange-300 dark:border-orange-700 rounded-lg p-4">
          <div className="text-center font-bold text-orange-700 dark:text-orange-300 mb-3">
            Pipeline Stage {ppStage} (Layers {ppStage === 0 ? '0-15' : '16-31'})
          </div>

          {/* 4 个数据并行组 */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {[0, 1, 2, 3].map(dpRank => (
              <div key={dpRank} className="border border-blue-300 dark:border-blue-700 rounded-lg p-2">
                <div className="text-xs text-center text-blue-700 dark:text-blue-300 font-semibold mb-2">
                  DP Rank {dpRank}
                </div>

                {/* 8 个张量并行 GPU */}
                <div className="grid grid-cols-4 gap-1">
                  {[0, 1, 2, 3, 4, 5, 6, 7].map(tpRank => {
                    const globalGpuId = ppStage * 32 + dpRank * 8 + tpRank
                    return (
                      <motion.div
                        key={tpRank}
                        initial={{ opacity: 0, scale: 0.5 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ delay: globalGpuId * 0.01 }}
                        className="bg-orange-500 rounded text-white text-center p-1"
                      >
                        <div className="text-[8px] font-bold">GPU{globalGpuId}</div>
                        <div className="text-[6px]">TP{tpRank}</div>
                      </motion.div>
                    )
                  })}
                </div>
              </div>
            ))}
          </div>
        </div>
      ))}

      <div className="text-xs text-center text-gray-300 mt-4 space-y-1">
        <p>• <strong>DP (数据并行)</strong>: 4 个副本，跨 DP Rank 同步梯度</p>
        <p>• <strong>TP (张量并行)</strong>: 8-way 分片，每层内通信</p>
        <p>• <strong>PP (流水线并行)</strong>: 2 个阶段，跨 stage 传递激活值</p>
        <p className="font-bold text-orange-600 dark:text-orange-400">显存占用：1/(4×8×2) ≈ 1.5625% 每 GPU</p>
      </div>
    </div>
  )
}
