'use client'

import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { Zap, Clock, Target, Code, TrendingUp, Settings } from 'lucide-react'

export default function PTQvsQATComparison() {
  const [selectedMethod, setSelectedMethod] = useState<'PTQ' | 'QAT' | null>(null)

  const methods = {
    PTQ: {
      name: '训练后量化 (PTQ)',
      color: 'from-blue-500 to-cyan-500',
      icon: Zap,
      process: [
        { step: '加载预训练模型', time: '10s' },
        { step: '校准数据统计', time: '2-5min' },
        { step: '量化权重', time: '1min' },
        { step: '保存量化模型', time: '5s' },
      ],
      metrics: {
        time: '5-10 分钟',
        accuracy: '2-5% ↓',
        cost: '低',
        complexity: '简单',
        useCase: '快速部署、资源受限',
      },
      formula: '\\mathbf{W}_{\\text{quant}} = \\arg\\min_{\\mathbf{W}_q \\in \\mathcal{Q}} \\| \\mathbf{W} - \\mathbf{W}_q \\|_F',
    },
    QAT: {
      name: '量化感知训练 (QAT)',
      color: 'from-purple-500 to-pink-500',
      icon: TrendingUp,
      process: [
        { step: '初始化模型', time: '10s' },
        { step: '插入伪量化节点', time: '5s' },
        { step: '完整训练流程', time: '数小时' },
        { step: '移除伪量化', time: '5s' },
      ],
      metrics: {
        time: '数小时',
        accuracy: '<1% ↓',
        cost: '高（需GPU训练）',
        complexity: '复杂',
        useCase: '精度敏感任务',
      },
      formula: '\\min_{\\mathbf{W}} \\mathcal{L}(\\mathbf{W}) + \\lambda \\cdot \\text{Quant}(\\mathbf{W})',
    },
  }

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-blue-50 rounded-xl border border-slate-200">
      <h3 className="text-2xl font-bold text-center mb-6 text-slate-800">
        训练后量化 vs 量化感知训练
      </h3>

      {/* 方法选择 */}
      <div className="grid grid-cols-2 gap-4 mb-8">
        {(Object.keys(methods) as Array<'PTQ' | 'QAT'>).map((key) => {
          const method = methods[key]
          const Icon = method.icon
          return (
            <motion.button
              key={key}
              onClick={() => setSelectedMethod(selectedMethod === key ? null : key)}
              className={`p-6 rounded-xl border-2 transition-all ${
                selectedMethod === key
                  ? 'border-blue-500 bg-white shadow-lg'
                  : 'border-slate-300 bg-white/50 hover:border-blue-300'
              }`}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <div className="flex items-center gap-3 mb-3">
                <div className={`w-12 h-12 rounded-lg bg-gradient-to-br ${method.color} flex items-center justify-center`}>
                  <Icon className="w-6 h-6 text-white" />
                </div>
                <h4 className="text-lg font-bold text-slate-800">{method.name}</h4>
              </div>
              <div className="text-sm text-slate-600 text-left">
                {key === 'PTQ' ? '在已训练模型上直接量化' : '训练时模拟量化行为'}
              </div>
            </motion.button>
          )
        })}
      </div>

      {/* 详细对比 */}
      {selectedMethod && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="space-y-6"
        >
          {/* 流程图 */}
          <div className="bg-white p-6 rounded-xl border border-slate-200">
            <h4 className="font-bold text-slate-800 mb-4 flex items-center gap-2">
              <Settings className="w-5 h-5 text-blue-500" />
              {methods[selectedMethod].name} 流程
            </h4>
            <div className="space-y-3">
              {methods[selectedMethod].process.map((item, idx) => (
                <motion.div
                  key={idx}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: idx * 0.1 }}
                  className="flex items-center gap-4"
                >
                  <div className={`w-8 h-8 rounded-full bg-gradient-to-br ${methods[selectedMethod].color} flex items-center justify-center text-white font-bold text-sm`}>
                    {idx + 1}
                  </div>
                  <div className="flex-1 p-3 bg-slate-50 rounded-lg">
                    <div className="font-medium text-slate-800">{item.step}</div>
                    <div className="text-xs text-slate-500 mt-1">耗时: {item.time}</div>
                  </div>
                  {idx < methods[selectedMethod].process.length - 1 && (
                    <div className="w-px h-8 bg-slate-300" />
                  )}
                </motion.div>
              ))}
            </div>
          </div>

          {/* 性能指标 */}
          <div className="bg-white p-6 rounded-xl border border-slate-200">
            <h4 className="font-bold text-slate-800 mb-4 flex items-center gap-2">
              <Target className="w-5 h-5 text-green-500" />
              性能指标
            </h4>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              {Object.entries(methods[selectedMethod].metrics).map(([key, value], idx) => {
                const icons = {
                  time: Clock,
                  accuracy: Target,
                  cost: TrendingUp,
                  complexity: Code,
                  useCase: Zap,
                }
                const labels = {
                  time: '时间成本',
                  accuracy: '精度损失',
                  cost: '计算成本',
                  complexity: '复杂度',
                  useCase: '适用场景',
                }
                const IconComponent = icons[key as keyof typeof icons]
                return (
                  <motion.div
                    key={key}
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: idx * 0.05 }}
                    className="p-4 bg-gradient-to-br from-slate-50 to-blue-50 rounded-lg"
                  >
                    <div className="flex items-center gap-2 mb-2">
                      {IconComponent && <IconComponent className="w-4 h-4 text-blue-500" />}
                      <div className="text-xs font-medium text-slate-600">
                        {labels[key as keyof typeof labels]}
                      </div>
                    </div>
                    <div className="text-sm font-bold text-slate-800">{value}</div>
                  </motion.div>
                )
              })}
            </div>
          </div>

          {/* 数学公式 */}
          <div className="bg-gradient-to-br from-blue-50 to-indigo-50 p-6 rounded-xl border border-blue-200">
            <h4 className="font-bold text-slate-800 mb-3">优化目标</h4>
            <div className="bg-white p-4 rounded-lg font-mono text-sm text-center text-slate-700 overflow-x-auto">
              ${methods[selectedMethod].formula}$
            </div>
            <div className="mt-3 text-sm text-slate-600">
              {selectedMethod === 'PTQ' ? (
                <div>
                  <strong>后处理优化：</strong>在量化空间 $\mathcal{'Q'}$ 中找到最接近原权重的量化值
                </div>
              ) : (
                <div>
                  <strong>训练时约束：</strong>在损失函数中加入量化惩罚项，训练过程中学习量化友好的权重
                </div>
              )}
            </div>
          </div>
        </motion.div>
      )}

      {/* 对比表格 */}
      {!selectedMethod && (
        <div className="bg-white p-6 rounded-xl border border-slate-200">
          <h4 className="font-bold text-slate-800 mb-4">快速对比</h4>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b-2 border-slate-200">
                  <th className="text-left py-3 px-4 font-bold text-slate-700">维度</th>
                  <th className="text-center py-3 px-4 font-bold text-blue-600">PTQ</th>
                  <th className="text-center py-3 px-4 font-bold text-purple-600">QAT</th>
                </tr>
              </thead>
              <tbody>
                {[
                  { label: '时间成本', ptq: '分钟级', qat: '小时级' },
                  { label: '精度损失', ptq: '2-5%', qat: '<1%' },
                  { label: '需要训练', ptq: '❌', qat: '✅' },
                  { label: '需要GPU', ptq: '可选', qat: '必需' },
                  { label: '复杂度', ptq: '低', qat: '高' },
                ].map((row, idx) => (
                  <tr key={idx} className="border-b border-slate-100 hover:bg-slate-50">
                    <td className="py-3 px-4 font-medium text-slate-700">{row.label}</td>
                    <td className="py-3 px-4 text-center text-blue-600">{row.ptq}</td>
                    <td className="py-3 px-4 text-center text-purple-600">{row.qat}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      <div className="mt-4 text-xs text-slate-500 text-center">
        💡 点击上方卡片查看详细流程和性能指标
      </div>
    </div>
  )
}
