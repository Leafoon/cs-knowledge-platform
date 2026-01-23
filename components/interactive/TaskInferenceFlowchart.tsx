'use client'

import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { ArrowRight, Database, Cpu, Zap, CheckCircle } from 'lucide-react'

interface FlowStep {
  id: string
  title: string
  description: string
  icon: React.ElementType
  details: string[]
  color: string
}

const INFERENCE_STEPS: FlowStep[] = [
  {
    id: 'input',
    title: '1. 输入处理',
    description: '接收原始输入（文本/图像/音频）',
    icon: Database,
    details: [
      '接收用户输入数据',
      '自动检测输入类型',
      '进行初步格式验证',
      '传递给 Tokenizer'
    ],
    color: 'from-blue-400 to-blue-600'
  },
  {
    id: 'tokenize',
    title: '2. Token化',
    description: 'Tokenizer 将输入转换为模型可理解的格式',
    icon: Zap,
    details: [
      '文本 → Token IDs (词表映射)',
      '添加 Special Tokens ([CLS], [SEP])',
      '生成 Attention Mask (区分padding)',
      '构建 Token Type IDs (区分句子)',
      '返回 input_ids, attention_mask 等张量'
    ],
    color: 'from-green-400 to-green-600'
  },
  {
    id: 'model',
    title: '3. 模型推理',
    description: 'Model 前向传播计算',
    icon: Cpu,
    details: [
      'Embedding 层：Token → 向量',
      'Transformer 层：Self-Attention + FFN',
      '任务头：分类/生成/序列标注',
      '输出 logits / hidden_states',
      '可选：返回 attentions 用于可视化'
    ],
    color: 'from-purple-400 to-purple-600'
  },
  {
    id: 'postprocess',
    title: '4. 后处理',
    description: '将模型输出转换为最终结果',
    icon: CheckCircle,
    details: [
      'Logits → Probabilities (Softmax)',
      '解码 Token IDs → 文本 (Tokenizer.decode)',
      '应用任务特定规则（阈值过滤、NMS等）',
      '格式化输出（JSON结构）',
      '返回用户友好的结果'
    ],
    color: 'from-pink-400 to-pink-600'
  }
]

const TASK_EXAMPLES = {
  'text-classification': {
    input: '"This movie is amazing!"',
    output: '{"label": "POSITIVE", "score": 0.9998}'
  },
  'ner': {
    input: '"Apple CEO Tim Cook announced..."',
    output: '[{"entity": "ORG", "word": "Apple"}, {"entity": "PER", "word": "Tim Cook"}]'
  },
  'question-answering': {
    input: 'Q: "What is AI?", Context: "..."',
    output: '{"answer": "Artificial Intelligence", "start": 10, "end": 34}'
  },
  'text-generation': {
    input: '"Once upon a time"',
    output: '"Once upon a time, in a faraway land, there lived a brave knight..."'
  }
}

export default function TaskInferenceFlowchart() {
  const [activeStep, setActiveStep] = useState<string>('input')
  const [selectedTask, setSelectedTask] = useState<keyof typeof TASK_EXAMPLES>('text-classification')

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-blue-50 rounded-xl border border-slate-200">
      <h3 className="text-2xl font-bold text-center mb-6 text-slate-800">
        🔄 Pipeline 推理流程图
      </h3>

      {/* 任务选择 */}
      <div className="mb-6 flex gap-2 flex-wrap justify-center">
        {Object.keys(TASK_EXAMPLES).map((task) => (
          <button
            key={task}
            onClick={() => setSelectedTask(task as keyof typeof TASK_EXAMPLES)}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              selectedTask === task
                ? 'bg-gradient-to-r from-blue-500 to-purple-500 text-white shadow-lg'
                : 'bg-white text-slate-600 border border-slate-200 hover:border-blue-300'
            }`}
          >
            {task}
          </button>
        ))}
      </div>

      {/* 流程步骤 */}
      <div className="relative mb-8">
        {/* 连接线 */}
        <div className="absolute top-16 left-0 right-0 h-1 bg-gradient-to-r from-blue-300 via-purple-300 to-pink-300 hidden md:block" />

        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 relative">
          {INFERENCE_STEPS.map((step, idx) => {
            const Icon = step.icon
            const isActive = activeStep === step.id
            
            return (
              <div key={step.id} className="relative">
                {/* 步骤卡片 */}
                <motion.div
                  onHoverStart={() => setActiveStep(step.id)}
                  className={`p-4 rounded-xl cursor-pointer transition-all ${
                    isActive
                      ? 'bg-white shadow-2xl border-2 border-blue-400'
                      : 'bg-white/80 border border-slate-200 hover:shadow-lg'
                  }`}
                  whileHover={{ y: -4 }}
                >
                  {/* 图标 */}
                  <div className={`w-12 h-12 mx-auto mb-3 rounded-full bg-gradient-to-br ${step.color} flex items-center justify-center relative z-10`}>
                    <Icon className="w-6 h-6 text-white" />
                  </div>

                  {/* 标题 */}
                  <h4 className="font-bold text-center text-slate-800 mb-2">
                    {step.title}
                  </h4>
                  <p className="text-xs text-center text-slate-600 mb-3">
                    {step.description}
                  </p>

                  {/* 详细步骤（仅激活时显示） */}
                  {isActive && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      className="mt-3 pt-3 border-t border-slate-200"
                    >
                      <ul className="space-y-1">
                        {step.details.map((detail, detailIdx) => (
                          <motion.li
                            key={detailIdx}
                            initial={{ opacity: 0, x: -10 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: detailIdx * 0.05 }}
                            className="text-xs text-slate-700 flex items-start gap-2"
                          >
                            <span className="text-blue-500 mt-0.5">▸</span>
                            <span>{detail}</span>
                          </motion.li>
                        ))}
                      </ul>
                    </motion.div>
                  )}
                </motion.div>

                {/* 箭头（除最后一个） */}
                {idx < INFERENCE_STEPS.length - 1 && (
                  <div className="hidden md:flex items-center justify-center absolute top-12 -right-3 z-20">
                    <ArrowRight className="w-6 h-6 text-purple-400" />
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </div>

      {/* 示例输入输出 */}
      <div className="bg-white rounded-xl p-6 border border-slate-200">
        <h4 className="font-bold text-slate-800 mb-4 flex items-center gap-2">
          <span className="text-lg">💻</span>
          {selectedTask} 示例
        </h4>

        <div className="grid md:grid-cols-2 gap-4">
          {/* 输入 */}
          <div>
            <div className="text-xs font-bold text-slate-600 mb-2 uppercase">Input</div>
            <div className="bg-slate-50 border border-slate-200 rounded-lg p-3 font-mono text-sm text-slate-700">
              {TASK_EXAMPLES[selectedTask].input}
            </div>
          </div>

          {/* 输出 */}
          <div>
            <div className="text-xs font-bold text-slate-600 mb-2 uppercase">Output</div>
            <div className="bg-green-50 border border-green-200 rounded-lg p-3 font-mono text-sm text-green-700 whitespace-pre-wrap">
              {TASK_EXAMPLES[selectedTask].output}
            </div>
          </div>
        </div>

        {/* Pipeline 代码 */}
        <div className="mt-4 bg-slate-900 rounded-lg p-4 overflow-x-auto">
          <pre className="text-xs text-slate-200">
            <code>{`from transformers import pipeline

# 创建 Pipeline（自动完成上述 4 个步骤）
pipe = pipeline("${selectedTask}")

# 一行代码推理
result = pipe(${TASK_EXAMPLES[selectedTask].input})

print(result)  # ${TASK_EXAMPLES[selectedTask].output}`}</code>
          </pre>
        </div>
      </div>

      {/* 关键概念 */}
      <div className="mt-6 bg-blue-50 border border-blue-200 rounded-lg p-4">
        <h5 className="font-bold text-blue-800 mb-2 text-sm">💡 关键概念</h5>
        <ul className="text-xs text-blue-700 space-y-1">
          <li><strong>Pipeline 自动化</strong>：上述 4 个步骤由 Pipeline 自动完成，无需手动调用</li>
          <li><strong>Tokenizer 对应性</strong>：必须使用与模型配对的 Tokenizer（词表一致）</li>
          <li><strong>批处理优化</strong>：Pipeline 支持批量输入，自动处理 padding 和 batching</li>
          <li><strong>设备自动迁移</strong>：Pipeline 会自动将数据迁移到模型所在设备（CPU/GPU）</li>
        </ul>
      </div>
    </div>
  )
}
