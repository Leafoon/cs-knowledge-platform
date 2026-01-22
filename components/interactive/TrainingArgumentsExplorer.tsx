'use client'

import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { Settings, Zap, Save, BarChart3, Cpu, ChevronDown, ChevronUp } from 'lucide-react'

interface ParamCategory {
  id: string
  name: string
  icon: React.ReactNode
  color: string
  params: Array<{
    name: string
    type: string
    default: string
    description: string
  }>
}

export default function TrainingArgumentsExplorer() {
  const [activeCategory, setActiveCategory] = useState('basic')
  const [expandedParam, setExpandedParam] = useState<string | null>(null)

  const categories: ParamCategory[] = [
    {
      id: 'basic',
      name: '基础参数',
      icon: <Settings className="w-5 h-5" />,
      color: 'blue',
      params: [
        { name: 'output_dir', type: 'str', default: '"./results"', description: '模型和检查点保存目录（必需）' },
        { name: 'num_train_epochs', type: 'int', default: '3', description: '训练轮数' },
        { name: 'per_device_train_batch_size', type: 'int', default: '8', description: '每个 GPU 的训练批次大小' },
        { name: 'learning_rate', type: 'float', default: '5e-5', description: '初始学习率' },
        { name: 'weight_decay', type: 'float', default: '0.01', description: 'L2 正则化系数' }
      ]
    },
    {
      id: 'optimization',
      name: '优化相关',
      icon: <Zap className="w-5 h-5" />,
      color: 'purple',
      params: [
        { name: 'fp16', type: 'bool', default: 'False', description: '启用 FP16 混合精度（Nvidia GPU）' },
        { name: 'bf16', type: 'bool', default: 'False', description: '启用 BF16 混合精度（Ampere+ GPU）' },
        { name: 'gradient_accumulation_steps', type: 'int', default: '1', description: '梯度累积步数（模拟大批次）' },
        { name: 'max_grad_norm', type: 'float', default: '1.0', description: '梯度裁剪阈值' },
        { name: 'gradient_checkpointing', type: 'bool', default: 'False', description: '启用梯度检查点（节省显存）' }
      ]
    },
    {
      id: 'eval_save',
      name: '评估与保存',
      icon: <Save className="w-5 h-5" />,
      color: 'green',
      params: [
        { name: 'evaluation_strategy', type: 'str', default: '"no"', description: '评估策略: no/steps/epoch' },
        { name: 'eval_steps', type: 'int', default: '500', description: '每 N 步评估一次' },
        { name: 'save_strategy', type: 'str', default: '"steps"', description: '保存策略: no/steps/epoch' },
        { name: 'save_steps', type: 'int', default: '500', description: '每 N 步保存检查点' },
        { name: 'save_total_limit', type: 'int', default: 'None', description: '最多保留 N 个检查点' },
        { name: 'load_best_model_at_end', type: 'bool', default: 'False', description: '训练结束加载最佳模型' }
      ]
    },
    {
      id: 'logging',
      name: '日志记录',
      icon: <BarChart3 className="w-5 h-5" />,
      color: 'orange',
      params: [
        { name: 'logging_dir', type: 'str', default: '"./logs"', description: 'TensorBoard 日志目录' },
        { name: 'logging_steps', type: 'int', default: '500', description: '每 N 步记录日志' },
        { name: 'report_to', type: 'list', default: '["tensorboard"]', description: '上报平台: tensorboard/wandb/mlflow' }
      ]
    },
    {
      id: 'distributed',
      name: '分布式训练',
      icon: <Cpu className="w-5 h-5" />,
      color: 'pink',
      params: [
        { name: 'local_rank', type: 'int', default: '-1', description: '本地进程排名（自动设置）' },
        { name: 'deepspeed', type: 'str', default: 'None', description: 'DeepSpeed 配置文件路径' },
        { name: 'fsdp', type: 'str', default: '""', description: 'FSDP 策略配置' },
        { name: 'ddp_find_unused_parameters', type: 'bool', default: 'False', description: '查找未使用的参数' }
      ]
    }
  ]

  const activeData = categories.find(c => c.id === activeCategory)!

  return (
    <div className="my-8 p-6 bg-gradient-to-br from-amber-50 to-orange-50 dark:from-slate-900 dark:to-orange-950 rounded-xl border border-slate-200 dark:border-slate-700">
      {/* Header */}
      <div className="mb-6">
        <h3 className="text-xl font-bold text-slate-900 dark:text-white flex items-center gap-2">
          <Settings className="w-5 h-5 text-orange-500" />
          TrainingArguments 参数浏览器
        </h3>
        <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">
          100+ 参数分类详解
        </p>
      </div>

      {/* Category Tabs */}
      <div className="flex flex-wrap gap-2 mb-6">
        {categories.map(category => (
          <button
            key={category.id}
            onClick={() => setActiveCategory(category.id)}
            className={`px-4 py-2 rounded-lg font-semibold transition-all flex items-center gap-2 ${
              activeCategory === category.id
                ? `bg-${category.color}-500 text-white shadow-lg`
                : 'bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-400 border border-slate-200 dark:border-slate-700'
            }`}
          >
            {category.icon}
            {category.name}
            <span className="text-xs opacity-75">({category.params.length})</span>
          </button>
        ))}
      </div>

      {/* Parameters List */}
      <motion.div
        key={activeCategory}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
        className="space-y-2"
      >
        {activeData.params.map(param => {
          const isExpanded = expandedParam === param.name
          return (
            <div
              key={param.name}
              className="bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 overflow-hidden"
            >
              <button
                onClick={() => setExpandedParam(isExpanded ? null : param.name)}
                className="w-full p-4 text-left hover:bg-slate-50 dark:hover:bg-slate-700/50 transition-colors"
              >
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <code className="font-mono font-bold text-sm text-slate-900 dark:text-white">
                        {param.name}
                      </code>
                      <span className={`px-2 py-0.5 rounded text-xs bg-${activeData.color}-100 dark:bg-${activeData.color}-900/30 text-${activeData.color}-600 dark:text-${activeData.color}-400`}>
                        {param.type}
                      </span>
                    </div>
                    <p className="text-sm text-slate-600 dark:text-slate-400">
                      {param.description}
                    </p>
                  </div>
                  {isExpanded ? <ChevronUp className="w-5 h-5 text-slate-400" /> : <ChevronDown className="w-5 h-5 text-slate-400" />}
                </div>
              </button>

              {isExpanded && (
                <motion.div
                  initial={{ height: 0 }}
                  animate={{ height: 'auto' }}
                  exit={{ height: 0 }}
                  className="px-4 pb-4 border-t border-slate-200 dark:border-slate-700"
                >
                  <div className="pt-3 space-y-2">
                    <div>
                      <span className="text-xs font-semibold text-slate-600 dark:text-slate-400">默认值:</span>
                      <code className="ml-2 px-2 py-1 bg-slate-100 dark:bg-slate-700 rounded text-sm font-mono text-slate-900 dark:text-white">
                        {param.default}
                      </code>
                    </div>
                    <div>
                      <span className="text-xs font-semibold text-slate-600 dark:text-slate-400">示例:</span>
                      <pre className="mt-1 p-2 bg-slate-900 rounded text-xs text-slate-100 overflow-auto">
                        <code>{`training_args = TrainingArguments(
    ${param.name}=${param.default}
)`}</code>
                      </pre>
                    </div>
                  </div>
                </motion.div>
              )}
            </div>
          )
        })}
      </motion.div>
    </div>
  )
}
