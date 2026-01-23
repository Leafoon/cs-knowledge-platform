'use client'

import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Play, ChevronRight, Zap, AlertTriangle } from 'lucide-react'

export default function AMPWorkflow() {
  const [currentStep, setCurrentStep] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)

  const steps = [
    {
      id: 0,
      title: '前向传播（autocast）',
      code: 'with autocast():\n    outputs = model(input)\n    loss = outputs.loss',
      operations: [
        { name: 'Linear/Conv', precision: 'FP16', reason: '计算密集，Tensor Core 加速' },
        { name: 'Softmax', precision: 'FP32', reason: '数值敏感，防止溢出' },
        { name: 'Loss', precision: 'FP32', reason: '高精度累积' },
      ],
      note: '自动选择精度：GEMM 用 FP16，数值敏感操作用 FP32',
    },
    {
      id: 1,
      title: 'Loss 缩放',
      code: 'scaled_loss = scaler.scale(loss)\n# loss * 65536',
      operations: [
        { name: 'Loss (FP32)', value: '2.345', scaled: '153681.92' },
      ],
      note: '放大 loss 防止梯度下溢（FP16 最小值 6e-5）',
    },
    {
      id: 2,
      title: '反向传播',
      code: 'scaled_loss.backward()\n# 梯度也被放大 65536 倍',
      operations: [
        { name: '原始梯度', value: '1.2e-7 (会下溢)', scaled: '0.0079 (安全)' },
        { name: '中等梯度', value: '0.003', scaled: '196.6' },
        { name: '大梯度', value: '5.2', scaled: '340787' },
      ],
      note: '所有梯度 × scale_factor，避免 FP16 下溢为 0',
    },
    {
      id: 3,
      title: 'Unscale 梯度',
      code: 'scaler.unscale_(optimizer)\n# grad /= 65536',
      operations: [
        { name: '恢复原始值', before: '0.0079', after: '1.2e-7' },
        { name: '检查 inf/NaN', status: 'pass', action: '继续更新' },
      ],
      note: '除以 scale_factor，恢复真实梯度值',
    },
    {
      id: 4,
      title: '梯度裁剪（可选）',
      code: 'torch.nn.utils.clip_grad_norm_(\n    model.parameters(), 1.0)',
      operations: [
        { name: '梯度范数', value: '2.5', clipped: '1.0' },
      ],
      note: '在 unscale 后执行，防止梯度爆炸',
    },
    {
      id: 5,
      title: '参数更新',
      code: 'scaler.step(optimizer)\nscaler.update()',
      operations: [
        { name: 'optimizer.step()', status: 'success' },
        { name: 'scale 动态调整', old: '65536', new: '65536' },
      ],
      note: '更新参数，并根据是否出现 inf/NaN 调整 scale',
    },
  ]

  const playAnimation = async () => {
    setIsPlaying(true)
    for (let i = 0; i <= steps.length - 1; i++) {
      setCurrentStep(i)
      await new Promise(resolve => setTimeout(resolve, 2000))
    }
    setIsPlaying(false)
  }

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-purple-50 rounded-xl shadow-lg">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <Zap className="w-8 h-8 text-purple-600" />
          <h3 className="text-2xl font-bold text-slate-800">AMP 训练流程详解</h3>
        </div>
        <button
          onClick={playAnimation}
          disabled={isPlaying}
          className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:bg-slate-400 flex items-center gap-2"
        >
          <Play className="w-4 h-4" />
          {isPlaying ? '播放中...' : '播放动画'}
        </button>
      </div>

      {/* 步骤指示器 */}
      <div className="flex items-center justify-between mb-8">
        {steps.map((step, idx) => (
          <React.Fragment key={step.id}>
            <button
              onClick={() => setCurrentStep(idx)}
              className={`flex flex-col items-center ${
                currentStep === idx ? 'opacity-100' : 'opacity-40'
              }`}
            >
              <div
                className={`w-12 h-12 rounded-full flex items-center justify-center font-bold text-lg transition-all ${
                  currentStep === idx
                    ? 'bg-purple-600 text-white scale-110 shadow-lg'
                    : currentStep > idx
                    ? 'bg-green-500 text-white'
                    : 'bg-slate-300 text-slate-600'
                }`}
              >
                {idx + 1}
              </div>
              <div className="text-xs text-slate-600 mt-2 text-center max-w-[100px]">
                {step.title.split('（')[0]}
              </div>
            </button>
            {idx < steps.length - 1 && (
              <ChevronRight className="w-6 h-6 text-slate-400" />
            )}
          </React.Fragment>
        ))}
      </div>

      {/* 当前步骤详情 */}
      <AnimatePresence mode="wait">
        <motion.div
          key={currentStep}
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -20 }}
          className="space-y-4"
        >
          {/* 标题 */}
          <div className="bg-white p-5 rounded-lg shadow-lg border-l-4 border-purple-600">
            <h4 className="text-xl font-bold text-purple-900 mb-2">
              步骤 {currentStep + 1}: {steps[currentStep].title}
            </h4>
            <div className="text-sm text-slate-600">{steps[currentStep].note}</div>
          </div>

          {/* 代码 */}
          <div className="bg-slate-900 p-4 rounded-lg">
            <div className="text-xs text-slate-400 mb-2">Python Code:</div>
            <pre className="text-green-400 font-mono text-sm">
              {steps[currentStep].code}
            </pre>
          </div>

          {/* 操作详情 */}
          <div className="bg-white p-5 rounded-lg shadow">
            <h5 className="font-bold text-slate-800 mb-3">详细操作</h5>
            <div className="space-y-3">
              {steps[currentStep].operations.map((op, idx) => (
                <motion.div
                  key={idx}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: idx * 0.1 }}
                  className="flex items-center justify-between p-3 bg-slate-50 rounded border border-slate-200"
                >
                  <div className="font-semibold text-slate-700">{op.name}</div>
                  <div className="flex items-center gap-4">
                    {'precision' in op && (
                      <>
                        <span className={`px-3 py-1 rounded font-mono text-sm ${
                          op.precision === 'FP16'
                            ? 'bg-blue-100 text-blue-700'
                            : 'bg-orange-100 text-orange-700'
                        }`}>
                          {op.precision}
                        </span>
                        <span className="text-xs text-slate-500">{op.reason}</span>
                      </>
                    )}
                    {'value' in op && (
                      <>
                        <span className="font-mono text-slate-600">{op.value}</span>
                        {'scaled' in op && (
                          <>
                            <span className="text-slate-400">→</span>
                            <span className="font-mono font-bold text-purple-600">
                              {op.scaled}
                            </span>
                          </>
                        )}
                      </>
                    )}
                    {'before' in op && (
                      <>
                        <span className="font-mono text-slate-600">{op.before}</span>
                        <span className="text-slate-400">→</span>
                        <span className="font-mono text-green-600">{op.after}</span>
                      </>
                    )}
                    {'old' in op && (
                      <>
                        <span className="font-mono text-slate-600">{op.old}</span>
                        <span className="text-slate-400">→</span>
                        <span className="font-mono text-blue-600">{op.new}</span>
                      </>
                    )}
                    {'status' in op && (
                      <span className={`px-2 py-1 rounded text-xs font-bold ${
                        op.status === 'success' || op.status === 'pass'
                          ? 'bg-green-100 text-green-700'
                          : 'bg-red-100 text-red-700'
                      }`}>
                        {op.status}
                      </span>
                    )}
                    {'clipped' in op && (
                      <>
                        <span className="text-red-600">→</span>
                        <span className="font-mono text-green-600">{op.clipped}</span>
                        <span className="text-xs text-green-600">(裁剪)</span>
                      </>
                    )}
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        </motion.div>
      </AnimatePresence>

      {/* 关键要点 */}
      <div className="mt-6 grid grid-cols-2 gap-4">
        <div className="p-4 bg-blue-50 border border-blue-300 rounded-lg">
          <div className="flex items-center gap-2 mb-2">
            <Zap className="w-5 h-5 text-blue-600" />
            <h5 className="font-bold text-blue-800">为什么需要 Loss Scaling？</h5>
          </div>
          <div className="text-sm text-slate-700">
            FP16 最小正数是 <code className="bg-white px-1 rounded">6.1e-5</code>，
            小于这个值的梯度会<strong>下溢变成 0</strong>。
            通过乘以 65536 (2<sup>16</sup>)，梯度被放大到安全范围。
          </div>
        </div>

        <div className="p-4 bg-green-50 border border-green-300 rounded-lg">
          <div className="flex items-center gap-2 mb-2">
            <AlertTriangle className="w-5 h-5 text-green-600" />
            <h5 className="font-bold text-green-800">动态 Scale 调整</h5>
          </div>
          <div className="text-sm text-slate-700">
            如果检测到 <code className="bg-white px-1 rounded">inf/NaN</code>（梯度爆炸），
            scale 减半；如果连续 2000 步正常，scale 翻倍。自动适应训练状态。
          </div>
        </div>
      </div>
    </div>
  )
}
