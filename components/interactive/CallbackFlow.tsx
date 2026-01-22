'use client'

import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { Play, Square, Save, BarChart, CheckCircle2 } from 'lucide-react'

interface CallbackEvent {
  id: string
  name: string
  trigger: string
  icon: React.ReactNode
  color: string
  examples: string[]
}

export default function CallbackFlow() {
  const [activePhase, setActivePhase] = useState<'train' | 'epoch' | 'step'>('train')

  const events: Record<string, CallbackEvent[]> = {
    train: [
      {
        id: 'init_end',
        name: 'on_init_end',
        trigger: 'Trainer 初始化完成',
        icon: <CheckCircle2 className="w-4 h-4" />,
        color: 'slate',
        examples: ['初始化 WandB', '加载检查点']
      },
      {
        id: 'train_begin',
        name: 'on_train_begin',
        trigger: '训练开始前',
        icon: <Play className="w-4 h-4" />,
        color: 'green',
        examples: ['记录开始时间', '初始化指标']
      },
      {
        id: 'train_end',
        name: 'on_train_end',
        trigger: '训练结束后',
        icon: <Square className="w-4 h-4" />,
        color: 'red',
        examples: ['保存最终模型', '关闭 WandB']
      }
    ],
    epoch: [
      {
        id: 'epoch_begin',
        name: 'on_epoch_begin',
        trigger: '每个 epoch 开始',
        icon: <Play className="w-4 h-4" />,
        color: 'blue',
        examples: ['打印 epoch 信息', '重置统计量']
      },
      {
        id: 'epoch_end',
        name: 'on_epoch_end',
        trigger: '每个 epoch 结束',
        icon: <Square className="w-4 h-4" />,
        color: 'purple',
        examples: ['计算 epoch 指标', '检查早停条件']
      }
    ],
    step: [
      {
        id: 'step_begin',
        name: 'on_step_begin',
        trigger: '每步开始前',
        icon: <Play className="w-4 h-4" />,
        color: 'cyan',
        examples: ['准备批次数据']
      },
      {
        id: 'step_end',
        name: 'on_step_end',
        trigger: '每步结束后',
        icon: <CheckCircle2 className="w-4 h-4" />,
        color: 'green',
        examples: ['记录损失值', '更新进度条']
      },
      {
        id: 'log',
        name: 'on_log',
        trigger: '日志记录时',
        icon: <BarChart className="w-4 h-4" />,
        color: 'orange',
        examples: ['上传 TensorBoard', '打印指标']
      },
      {
        id: 'save',
        name: 'on_save',
        trigger: '保存检查点时',
        icon: <Save className="w-4 h-4" />,
        color: 'indigo',
        examples: ['备份模型', '保存中间结果']
      },
      {
        id: 'evaluate',
        name: 'on_evaluate',
        trigger: '评估后',
        icon: <BarChart className="w-4 h-4" />,
        color: 'pink',
        examples: ['记录验证指标', '更新最佳模型']
      }
    ]
  }

  return (
    <div className="my-8 p-6 bg-gradient-to-br from-rose-50 to-pink-50 dark:from-slate-900 dark:to-rose-950 rounded-xl border border-slate-200 dark:border-slate-700">
      {/* Header */}
      <div className="mb-6">
        <h3 className="text-xl font-bold text-slate-900 dark:text-white">
          Callback 生命周期事件
        </h3>
        <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">
          在特定时刻插入自定义逻辑
        </p>
      </div>

      {/* Phase Selector */}
      <div className="flex gap-2 mb-6">
        <button
          onClick={() => setActivePhase('train')}
          className={`flex-1 py-3 px-4 rounded-lg font-semibold transition-all ${
            activePhase === 'train'
              ? 'bg-green-500 text-white shadow-lg'
              : 'bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-400 border border-slate-200 dark:border-slate-700'
          }`}
        >
          训练级别
        </button>
        <button
          onClick={() => setActivePhase('epoch')}
          className={`flex-1 py-3 px-4 rounded-lg font-semibold transition-all ${
            activePhase === 'epoch'
              ? 'bg-blue-500 text-white shadow-lg'
              : 'bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-400 border border-slate-200 dark:border-slate-700'
          }`}
        >
          Epoch 级别
        </button>
        <button
          onClick={() => setActivePhase('step')}
          className={`flex-1 py-3 px-4 rounded-lg font-semibold transition-all ${
            activePhase === 'step'
              ? 'bg-purple-500 text-white shadow-lg'
              : 'bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-400 border border-slate-200 dark:border-slate-700'
          }`}
        >
          Step 级别
        </button>
      </div>

      {/* Events List */}
      <motion.div
        key={activePhase}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
        className="space-y-3"
      >
        {events[activePhase].map((event, idx) => (
          <motion.div
            key={event.id}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: idx * 0.1 }}
            className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 hover:shadow-md transition-shadow"
          >
            <div className="flex items-start gap-4">
              <div className={`p-3 rounded-lg bg-${event.color}-100 dark:bg-${event.color}-900/30 text-${event.color}-600 dark:text-${event.color}-400`}>
                {event.icon}
              </div>
              <div className="flex-1">
                <div className="flex items-start justify-between mb-2">
                  <div>
                    <code className="font-mono font-bold text-sm text-slate-900 dark:text-white">
                      {event.name}
                    </code>
                    <p className="text-xs text-slate-600 dark:text-slate-400 mt-1">
                      {event.trigger}
                    </p>
                  </div>
                </div>
                <div className="flex flex-wrap gap-2 mt-2">
                  {event.examples.map((example, i) => (
                    <span
                      key={i}
                      className="px-2 py-1 bg-slate-100 dark:bg-slate-700 rounded text-xs text-slate-700 dark:text-slate-300"
                    >
                      {example}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          </motion.div>
        ))}
      </motion.div>

      {/* Code Example */}
      <div className="mt-6 p-4 bg-slate-900 rounded-lg">
        <div className="text-xs text-slate-400 mb-2">自定义 Callback 示例</div>
        <pre className="text-sm text-slate-100 font-mono overflow-auto">
          <code>{`class CustomCallback(TrainerCallback):
    def ${events[activePhase][0].name}(self, args, state, control, **kwargs):
        print(f"${events[activePhase][0].trigger}")
        # 添加自定义逻辑...

trainer = Trainer(
    model=model,
    callbacks=[CustomCallback()]
)`}</code>
        </pre>
      </div>
    </div>
  )
}
