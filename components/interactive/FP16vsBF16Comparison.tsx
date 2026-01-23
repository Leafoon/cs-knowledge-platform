'use client'

import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { Gauge, Zap, Shield, AlertTriangle } from 'lucide-react'

export default function FP16vsBF16Comparison() {
  const [selectedMetric, setSelectedMetric] = useState<'range' | 'precision' | 'speed' | 'stability'>('range')

  const metrics = {
    range: {
      title: '动态范围',
      icon: Gauge,
      fp16: {
        value: '6.1e-5 ~ 6.55e4',
        score: 40,
        details: '5 位指数，范围小',
        issue: '梯度 < 6e-5 会下溢为 0',
      },
      bf16: {
        value: '1.2e-38 ~ 3.4e38',
        score: 100,
        details: '8 位指数，与 FP32 相同',
        issue: '无下溢风险',
      },
    },
    precision: {
      title: '数值精度',
      icon: Zap,
      fp16: {
        value: '~3 位有效数字',
        score: 70,
        details: '10 位尾数',
        issue: '累积误差较小',
      },
      bf16: {
        value: '~2 位有效数字',
        score: 50,
        details: '7 位尾数',
        issue: '精度略低，但 DL 可容忍',
      },
    },
    speed: {
      title: '训练速度',
      icon: Zap,
      fp16: {
        value: '2.1x vs FP32',
        score: 85,
        details: 'Tensor Core 加速',
        issue: '需 loss scaling 开销',
      },
      bf16: {
        value: '2.3x vs FP32',
        score: 95,
        details: 'Tensor Core 加速',
        issue: '无额外开销',
      },
    },
    stability: {
      title: '训练稳定性',
      icon: Shield,
      fp16: {
        value: '95% 成功率',
        score: 75,
        details: '偶现 NaN',
        issue: '需监控 loss scaling',
      },
      bf16: {
        value: '100% 成功率',
        score: 100,
        details: '无 NaN 问题',
        issue: '几乎与 FP32 一致',
      },
    },
  }

  const current = metrics[selectedMetric]
  const Icon = current.icon

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-orange-50 rounded-xl shadow-lg">
      <div className="flex items-center gap-3 mb-6">
        <Gauge className="w-8 h-8 text-orange-600" />
        <h3 className="text-2xl font-bold text-slate-800">FP16 vs BF16 深度对比</h3>
      </div>

      {/* 指标选择 */}
      <div className="grid grid-cols-4 gap-3 mb-6">
        {Object.entries(metrics).map(([key, metric]) => {
          const MetricIcon = metric.icon
          return (
            <button
              key={key}
              onClick={() => setSelectedMetric(key as any)}
              className={`p-4 rounded-lg border-2 transition-all ${
                selectedMetric === key
                  ? 'border-orange-600 bg-orange-50 shadow-lg'
                  : 'border-slate-200 bg-white hover:border-orange-300'
              }`}
            >
              <MetricIcon className={`w-6 h-6 mb-2 mx-auto ${
                selectedMetric === key ? 'text-orange-600' : 'text-slate-400'
              }`} />
              <div className={`font-semibold text-sm ${
                selectedMetric === key ? 'text-orange-900' : 'text-slate-600'
              }`}>
                {metric.title}
              </div>
            </button>
          )
        })}
      </div>

      {/* 详细对比 */}
      <motion.div
        key={selectedMetric}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="grid grid-cols-2 gap-6 mb-6"
      >
        {/* FP16 */}
        <div className="bg-white p-6 rounded-lg shadow-lg border-t-4 border-blue-500">
          <div className="flex items-center justify-between mb-4">
            <h4 className="text-xl font-bold text-blue-700">FP16 (Half)</h4>
            <Icon className="w-6 h-6 text-blue-600" />
          </div>

          <div className="space-y-4">
            <div>
              <div className="text-sm text-slate-600 mb-1">{current.title}</div>
              <div className="text-2xl font-bold text-blue-600 mb-2">
                {current.fp16.value}
              </div>
              <div className="text-sm text-slate-700">{current.fp16.details}</div>
            </div>

            {/* 评分条 */}
            <div>
              <div className="text-xs text-slate-600 mb-1">综合评分</div>
              <div className="h-3 bg-slate-100 rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${current.fp16.score}%` }}
                  className="h-full bg-blue-500"
                />
              </div>
              <div className="text-xs text-slate-600 mt-1">{current.fp16.score}/100</div>
            </div>

            {/* 问题 */}
            <div className={`p-3 rounded border-2 ${
              current.fp16.score < 80
                ? 'bg-red-50 border-red-300'
                : 'bg-green-50 border-green-300'
            }`}>
              <div className="flex items-start gap-2">
                {current.fp16.score < 80 ? (
                  <AlertTriangle className="w-4 h-4 text-red-600 mt-0.5" />
                ) : (
                  <Shield className="w-4 h-4 text-green-600 mt-0.5" />
                )}
                <div className="text-sm text-slate-700">{current.fp16.issue}</div>
              </div>
            </div>
          </div>
        </div>

        {/* BF16 */}
        <div className="bg-white p-6 rounded-lg shadow-lg border-t-4 border-green-500">
          <div className="flex items-center justify-between mb-4">
            <h4 className="text-xl font-bold text-green-700">BF16 (BFloat16)</h4>
            <Icon className="w-6 h-6 text-green-600" />
          </div>

          <div className="space-y-4">
            <div>
              <div className="text-sm text-slate-600 mb-1">{current.title}</div>
              <div className="text-2xl font-bold text-green-600 mb-2">
                {current.bf16.value}
              </div>
              <div className="text-sm text-slate-700">{current.bf16.details}</div>
            </div>

            {/* 评分条 */}
            <div>
              <div className="text-xs text-slate-600 mb-1">综合评分</div>
              <div className="h-3 bg-slate-100 rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${current.bf16.score}%` }}
                  className="h-full bg-green-500"
                />
              </div>
              <div className="text-xs text-slate-600 mt-1">{current.bf16.score}/100</div>
            </div>

            {/* 优势 */}
            <div className="p-3 bg-green-50 border-2 border-green-300 rounded">
              <div className="flex items-start gap-2">
                <Shield className="w-4 h-4 text-green-600 mt-0.5" />
                <div className="text-sm text-slate-700">{current.bf16.issue}</div>
              </div>
            </div>
          </div>
        </div>
      </motion.div>

      {/* 全面对比表 */}
      <div className="bg-white p-6 rounded-lg shadow-lg mb-6">
        <h4 className="font-bold text-slate-800 mb-4">技术规格对比</h4>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b-2 border-slate-300">
              <th className="text-left py-2 px-3">特性</th>
              <th className="text-center py-2 px-3">FP16</th>
              <th className="text-center py-2 px-3">BF16</th>
              <th className="text-center py-2 px-3">FP32 (参考)</th>
            </tr>
          </thead>
          <tbody>
            <tr className="border-b border-slate-200">
              <td className="py-3 px-3 font-semibold">总位数</td>
              <td className="py-3 px-3 text-center font-mono">16</td>
              <td className="py-3 px-3 text-center font-mono">16</td>
              <td className="py-3 px-3 text-center font-mono text-slate-400">32</td>
            </tr>
            <tr className="border-b border-slate-200">
              <td className="py-3 px-3 font-semibold">指数位</td>
              <td className="py-3 px-3 text-center font-mono text-blue-600">5</td>
              <td className="py-3 px-3 text-center font-mono text-green-600">8</td>
              <td className="py-3 px-3 text-center font-mono text-slate-400">8</td>
            </tr>
            <tr className="border-b border-slate-200">
              <td className="py-3 px-3 font-semibold">尾数位</td>
              <td className="py-3 px-3 text-center font-mono text-blue-600">10</td>
              <td className="py-3 px-3 text-center font-mono text-green-600">7</td>
              <td className="py-3 px-3 text-center font-mono text-slate-400">23</td>
            </tr>
            <tr className="border-b border-slate-200">
              <td className="py-3 px-3 font-semibold">最小正数</td>
              <td className="py-3 px-3 text-center font-mono text-red-600">6.1e-5</td>
              <td className="py-3 px-3 text-center font-mono text-green-600">1.2e-38</td>
              <td className="py-3 px-3 text-center font-mono text-slate-400">1.2e-38</td>
            </tr>
            <tr className="border-b border-slate-200">
              <td className="py-3 px-3 font-semibold">需要 Loss Scaling</td>
              <td className="py-3 px-3 text-center">
                <span className="px-2 py-1 bg-yellow-100 text-yellow-700 rounded text-xs font-bold">是</span>
              </td>
              <td className="py-3 px-3 text-center">
                <span className="px-2 py-1 bg-green-100 text-green-700 rounded text-xs font-bold">否</span>
              </td>
              <td className="py-3 px-3 text-center text-slate-400">-</td>
            </tr>
            <tr className="border-b border-slate-200">
              <td className="py-3 px-3 font-semibold">适用 GPU</td>
              <td className="py-3 px-3 text-center text-xs">V100, 2080Ti</td>
              <td className="py-3 px-3 text-center text-xs font-semibold text-green-600">A100, H100, 4090</td>
              <td className="py-3 px-3 text-center text-slate-400 text-xs">All</td>
            </tr>
          </tbody>
        </table>
      </div>

      {/* 使用建议 */}
      <div className="grid grid-cols-2 gap-4">
        <div className="p-5 bg-blue-50 border-2 border-blue-300 rounded-lg">
          <h5 className="font-bold text-blue-800 mb-2">FP16 适用场景</h5>
          <ul className="text-sm text-slate-700 space-y-1">
            <li>✓ Volta/Turing GPU（V100, RTX 2080Ti）</li>
            <li>✓ 显存极度受限（比 BF16 稍省 ~5%）</li>
            <li>✓ 模型已验证稳定（无 NaN 风险）</li>
            <li className="text-yellow-600">⚠️ 需要仔细调整 loss scaling</li>
          </ul>
        </div>

        <div className="p-5 bg-green-50 border-2 border-green-400 rounded-lg">
          <h5 className="font-bold text-green-800 mb-2">BF16 适用场景（推荐）</h5>
          <ul className="text-sm text-slate-700 space-y-1">
            <li>✓ Ampere/Hopper GPU（A100, H100, RTX 4090）</li>
            <li>✓ 大规模预训练/微调（稳定性优先）</li>
            <li>✓ 首次实验新模型（无需调参）</li>
            <li className="text-green-600 font-semibold">✓ 深度学习默认首选</li>
          </ul>
        </div>
      </div>
    </div>
  )
}
