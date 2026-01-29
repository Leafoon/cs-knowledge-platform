"use client"

import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

type HardwareType = 'cpu-intel' | 'cpu-amd' | 'gpu-nvidia' | 'gpu-amd' | 'habana' | 'aws-inferentia'
type TaskType = 'inference' | 'training'
type LatencyRequirement = 'low' | 'medium' | 'high'

interface BackendRecommendation {
  name: string
  backend: string
  reason: string
  installation: string
  sampleCode: string
  performance: string
  costEfficiency: string
}

const OptimumBackendEcosystem: React.FC = () => {
  const [selectedHardware, setSelectedHardware] = useState<HardwareType>('cpu-intel')
  const [selectedTask, setSelectedTask] = useState<TaskType>('inference')
  const [selectedLatency, setSelectedLatency] = useState<LatencyRequirement>('medium')

  const hardwareOptions = [
    { id: 'cpu-intel' as HardwareType, label: 'Intel CPU', icon: '🔷' },
    { id: 'cpu-amd' as HardwareType, label: 'AMD CPU', icon: '🔶' },
    { id: 'gpu-nvidia' as HardwareType, label: 'NVIDIA GPU', icon: '🟩' },
    { id: 'gpu-amd' as HardwareType, label: 'AMD GPU', icon: '🟥' },
    { id: 'habana' as HardwareType, label: 'Habana Gaudi', icon: '🟦' },
    { id: 'aws-inferentia' as HardwareType, label: 'AWS Inferentia', icon: '🟧' },
  ]

  const getRecommendation = (): BackendRecommendation => {
    // 决策逻辑
    if (selectedHardware === 'cpu-intel') {
      return {
        name: 'OpenVINO',
        backend: 'optimum[openvino]',
        reason: 'Intel CPU 原生支持，针对 x86 架构优化，INT8 量化效果好',
        installation: 'pip install optimum[openvino]',
        sampleCode: `from optimum.intel import OVModelForSequenceClassification

model = OVModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    export=True
)`,
        performance: '3.7-4.0x 加速（vs PyTorch）',
        costEfficiency: '高（CPU 成本低）',
      }
    } else if (selectedHardware === 'cpu-amd' || (selectedHardware === 'gpu-nvidia' && selectedTask === 'inference')) {
      return {
        name: 'ONNX Runtime',
        backend: 'optimum[onnxruntime-gpu]',
        reason: '跨平台支持最好，CPU/GPU 自动切换，生态成熟',
        installation: 'pip install optimum[onnxruntime-gpu]',
        sampleCode: `from optimum.onnxruntime import ORTModelForSequenceClassification

model = ORTModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    export=True,
    provider="CUDAExecutionProvider"  # GPU
)`,
        performance: selectedHardware === 'gpu-nvidia' ? '1.3-2.7x 加速（GPU）' : '2.8-4.2x 加速（CPU）',
        costEfficiency: selectedHardware === 'gpu-nvidia' ? '中等' : '高',
      }
    } else if (selectedHardware === 'gpu-nvidia' && selectedTask === 'training') {
      return {
        name: 'PyTorch (原生)',
        backend: 'transformers',
        reason: '训练灵活性最高，支持 FSDP/DeepSpeed，调试方便',
        installation: 'pip install transformers accelerate',
        sampleCode: `from transformers import Trainer, TrainingArguments

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./output",
        fp16=True,  # 混合精度
    )
)
trainer.train()`,
        performance: '基准性能（1.0x）',
        costEfficiency: '中等（需 GPU）',
      }
    } else if (selectedHardware === 'habana') {
      return {
        name: 'Habana Gaudi',
        backend: 'optimum[habana]',
        reason: '训练加速器，性价比优于 A100，支持 BF16',
        installation: 'pip install optimum[habana]',
        sampleCode: `from optimum.habana import GaudiConfig, GaudiTrainer

gaudi_config = GaudiConfig()
trainer = GaudiTrainer(
    model=model,
    args=training_args,
    gaudi_config=gaudi_config
)`,
        performance: '1.6x 加速（vs A100）',
        costEfficiency: '极高（训练成本低 40%）',
      }
    } else if (selectedHardware === 'aws-inferentia') {
      return {
        name: 'AWS Neuron',
        backend: 'optimum[neuron]',
        reason: 'AWS 云端推理优化，成本最低，适合大规模部署',
        installation: 'pip install optimum[neuron]',
        sampleCode: `from optimum.neuron import NeuronModelForSequenceClassification

model = NeuronModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    export=True,
    batch_size=1
)`,
        performance: '1.3x 吞吐量（vs GPU）',
        costEfficiency: '极高（成本效率 10x+）',
      }
    } else {
      return {
        name: 'ONNX Runtime',
        backend: 'optimum[onnxruntime]',
        reason: '通用后端，兼容性最好',
        installation: 'pip install optimum[onnxruntime]',
        sampleCode: `from optimum.onnxruntime import ORTModelForSequenceClassification

model = ORTModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    export=True
)`,
        performance: '2.8x 加速',
        costEfficiency: '高',
      }
    }
  }

  const recommendation = getRecommendation()

  const backendEcosystem = [
    {
      name: 'ONNX Runtime',
      hardware: ['CPU', 'NVIDIA GPU', 'AMD GPU'],
      tasks: ['推理'],
      color: 'bg-blue-500',
      icon: '🔷',
    },
    {
      name: 'OpenVINO',
      hardware: ['Intel CPU', 'Intel GPU'],
      tasks: ['推理'],
      color: 'bg-indigo-500',
      icon: '🔷',
    },
    {
      name: 'Habana',
      hardware: ['Gaudi', 'Gaudi2'],
      tasks: ['训练', '推理'],
      color: 'bg-purple-500',
      icon: '🟦',
    },
    {
      name: 'AWS Neuron',
      hardware: ['Inferentia', 'Trainium'],
      tasks: ['推理', '训练'],
      color: 'bg-orange-500',
      icon: '🟧',
    },
    {
      name: 'BetterTransformer',
      hardware: ['PyTorch 原生'],
      tasks: ['推理'],
      color: 'bg-green-500',
      icon: '⚡',
    },
  ]

  return (
    <div className="w-full space-y-6 my-8">
      {/* 标题 */}
      <div className="text-center">
        <h3 className="text-2xl font-bold mb-2">Optimum 后端生态系统</h3>
        <p className="text-gray-300">
          根据硬件和任务自动推荐最优后端
        </p>
      </div>

      {/* 配置选择 */}
      <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg space-y-4">
        {/* 硬件选择 */}
        <div>
          <h4 className="font-semibold mb-3">1. 选择硬件类型</h4>
          <div className="grid grid-cols-3 md:grid-cols-6 gap-2">
            {hardwareOptions.map((hw) => (
              <button
                key={hw.id}
                onClick={() => setSelectedHardware(hw.id)}
                className={`p-3 rounded-lg text-sm font-medium transition-all ${
                  selectedHardware === hw.id
                    ? 'bg-blue-500 text-white shadow-lg scale-105'
                    : 'bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600'
                }`}
              >
                <div className="text-2xl mb-1">{hw.icon}</div>
                <div className="text-xs">{hw.label}</div>
              </button>
            ))}
          </div>
        </div>

        {/* 任务选择 */}
        <div>
          <h4 className="font-semibold mb-3">2. 选择任务类型</h4>
          <div className="flex gap-2">
            <button
              onClick={() => setSelectedTask('inference')}
              className={`flex-1 py-3 rounded-lg font-medium transition-all ${
                selectedTask === 'inference'
                  ? 'bg-green-500 text-white'
                  : 'bg-gray-100 dark:bg-gray-700'
              }`}
            >
              🚀 推理
            </button>
            <button
              onClick={() => setSelectedTask('training')}
              className={`flex-1 py-3 rounded-lg font-medium transition-all ${
                selectedTask === 'training'
                  ? 'bg-purple-500 text-white'
                  : 'bg-gray-100 dark:bg-gray-700'
              }`}
            >
              🏋️ 训练
            </button>
          </div>
        </div>

        {/* 延迟要求 */}
        <div>
          <h4 className="font-semibold mb-3">3. 延迟要求</h4>
          <div className="flex gap-2">
            {(['low', 'medium', 'high'] as LatencyRequirement[]).map((latency) => (
              <button
                key={latency}
                onClick={() => setSelectedLatency(latency)}
                className={`flex-1 py-2 rounded-lg text-sm font-medium transition-all ${
                  selectedLatency === latency
                    ? 'bg-orange-500 text-white'
                    : 'bg-gray-100 dark:bg-gray-700'
                }`}
              >
                {latency === 'low' && '⚡ 低延迟'}
                {latency === 'medium' && '⚖️ 中等'}
                {latency === 'high' && '💰 成本优先'}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* 推荐结果 */}
      <AnimatePresence mode="wait">
        <motion.div
          key={`${selectedHardware}-${selectedTask}-${selectedLatency}`}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -20 }}
          className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-xl p-6 border-2 border-blue-500 shadow-lg"
        >
          <div className="flex items-start gap-4">
            <div className="text-5xl">🎯</div>
            <div className="flex-1">
              <h4 className="text-2xl font-bold mb-2">
                推荐后端：{recommendation.name}
              </h4>
              <p className="text-gray-100 mb-4">
                {recommendation.reason}
              </p>

              <div className="grid md:grid-cols-2 gap-4 mb-4">
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <div className="text-sm text-gray-300 mb-1">
                    性能提升
                  </div>
                  <div className="text-xl font-bold text-green-600 dark:text-green-400">
                    {recommendation.performance}
                  </div>
                </div>
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <div className="text-sm text-gray-300 mb-1">
                    成本效率
                  </div>
                  <div className="text-xl font-bold text-blue-600 dark:text-blue-400">
                    {recommendation.costEfficiency}
                  </div>
                </div>
              </div>

              {/* 安装命令 */}
              <div className="mb-4">
                <div className="text-sm font-semibold mb-2">安装：</div>
                <pre className="bg-gray-900 text-gray-100 p-3 rounded text-xs overflow-x-auto">
                  {recommendation.installation}
                </pre>
              </div>

              {/* 示例代码 */}
              <div>
                <div className="text-sm font-semibold mb-2">示例代码：</div>
                <pre className="bg-gray-900 text-gray-100 p-3 rounded text-xs overflow-x-auto">
                  {recommendation.sampleCode}
                </pre>
              </div>
            </div>
          </div>
        </motion.div>
      </AnimatePresence>

      {/* 后端生态一览 */}
      <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h4 className="text-lg font-bold mb-4">Optimum 支持的后端</h4>
        <div className="space-y-3">
          {backendEcosystem.map((backend, idx) => (
            <div
              key={idx}
              className="flex items-center gap-4 p-4 bg-gray-50 dark:bg-gray-900 rounded-lg"
            >
              <div className="text-3xl">{backend.icon}</div>
              <div className="flex-1">
                <div className="font-semibold">{backend.name}</div>
                <div className="text-sm text-gray-300">
                  硬件: {backend.hardware.join(', ')}
                </div>
              </div>
              <div className="flex gap-2">
                {backend.tasks.map((task, tidx) => (
                  <span
                    key={tidx}
                    className={`px-3 py-1 rounded-full text-xs font-medium ${
                      task === '推理'
                        ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300'
                        : 'bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300'
                    }`}
                  >
                    {task}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* 决策树可视化 */}
      <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h4 className="text-lg font-bold mb-4">后端选择决策树</h4>
        <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4 font-mono text-xs overflow-x-auto">
          <pre>{`硬件类型？
├─ Intel CPU
│  └─ 推理 → OpenVINO (4.0x 加速, INT8)
│
├─ AMD CPU
│  └─ 推理 → ONNX Runtime (2.8x 加速)
│
├─ NVIDIA GPU
│  ├─ 推理 → ONNX Runtime (2.7x) / TensorRT (3.2x)
│  └─ 训练 → PyTorch (FSDP/DeepSpeed)
│
├─ Habana Gaudi
│  └─ 训练 + 推理 → Optimum Habana (1.6x vs A100)
│
└─ AWS Inferentia
   └─ 云端推理 → AWS Neuron (成本效率 10x+)`}</pre>
        </div>
      </div>
    </div>
  )
}

export default OptimumBackendEcosystem
