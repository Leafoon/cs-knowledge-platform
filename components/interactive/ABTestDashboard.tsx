'use client'

import React, { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { TrendingUp, Users, Zap, CheckCircle2, XCircle, BarChart3 } from 'lucide-react'

interface Variant {
  id: string
  name: string
  description: string
  users: number
  conversions: number
  avgLatency: number
  errorRate: number
}

export default function ABTestDashboard() {
  const [experiment, setExperiment] = useState<'running' | 'completed'>('running')
  const [traffic, setTraffic] = useState({ control: 50, variant: 50 })
  const [selectedMetric, setSelectedMetric] = useState<'conversion' | 'latency' | 'error'>('conversion')

  const [variants, setVariants] = useState<Variant[]>([
    {
      id: 'control',
      name: 'Control (GPT-3.5)',
      description: '原始版本：单轮检索',
      users: 5000,
      conversions: 2500,
      avgLatency: 800,
      errorRate: 2.3
    },
    {
      id: 'variant',
      name: 'Variant (Multi-Query RAG)',
      description: '实验版本：多查询+重排序',
      users: 5000,
      conversions: 3200,
      avgLatency: 1200,
      errorRate: 1.8
    }
  ])

  useEffect(() => {
    if (experiment === 'running') {
      const interval = setInterval(() => {
        setVariants(prev => prev.map(v => ({
          ...v,
          users: v.users + Math.floor(Math.random() * 10),
          conversions: v.conversions + Math.floor(Math.random() * 5),
          avgLatency: v.avgLatency + (Math.random() - 0.5) * 50,
          errorRate: Math.max(0, v.errorRate + (Math.random() - 0.5) * 0.2)
        })))
      }, 2000)
      return () => clearInterval(interval)
    }
  }, [experiment])

  const calculateMetrics = (variant: Variant) => {
    const conversionRate = ((variant.conversions / variant.users) * 100).toFixed(2)
    return { conversionRate }
  }

  const calculateStatisticalSignificance = () => {
    const control = variants[0]
    const variant = variants[1]
    
    const p1 = control.conversions / control.users
    const p2 = variant.conversions / variant.users
    const pooled = (control.conversions + variant.conversions) / (control.users + variant.users)
    
    const se = Math.sqrt(pooled * (1 - pooled) * (1/control.users + 1/variant.users))
    const zScore = Math.abs((p2 - p1) / se)
    
    return {
      zScore: zScore.toFixed(2),
      significant: zScore > 1.96, // 95% confidence
      lift: (((p2 - p1) / p1) * 100).toFixed(1)
    }
  }

  const stats = calculateStatisticalSignificance()
  const controlMetrics = calculateMetrics(variants[0])
  const variantMetrics = calculateMetrics(variants[1])

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-xl border border-blue-200 dark:border-blue-700">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="p-3 bg-blue-500 rounded-lg">
            <BarChart3 className="w-6 h-6 text-white" />
          </div>
          <div>
            <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100">
              A/B 测试仪表盘
            </h3>
            <p className="text-sm text-slate-600 dark:text-slate-400">
              实时监控实验指标与统计显著性
            </p>
          </div>
        </div>
        
        <div className="flex items-center gap-2">
          <div className={`px-3 py-1 rounded-full text-sm font-medium ${
            experiment === 'running'
              ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-300'
              : 'bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-300'
          }`}>
            {experiment === 'running' ? '🟢 运行中' : '⏸️ 已完成'}
          </div>
          <button
            onClick={() => setExperiment(prev => prev === 'running' ? 'completed' : 'running')}
            className="px-4 py-2 bg-blue-500 text-white rounded-lg text-sm hover:bg-blue-600 transition-all"
          >
            {experiment === 'running' ? '停止实验' : '继续实验'}
          </button>
        </div>
      </div>

      {/* 流量分配 */}
      <div className="mb-6 p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
        <div className="font-medium text-slate-800 dark:text-slate-200 mb-3">
          流量分配：Control {traffic.control}% / Variant {traffic.variant}%
        </div>
        <div className="flex gap-1 h-8 rounded-lg overflow-hidden">
          <div
            style={{ width: `${traffic.control}%` }}
            className="bg-blue-400 flex items-center justify-center text-white text-sm font-medium transition-all"
          >
            {traffic.control}%
          </div>
          <div
            style={{ width: `${traffic.variant}%` }}
            className="bg-purple-400 flex items-center justify-center text-white text-sm font-medium transition-all"
          >
            {traffic.variant}%
          </div>
        </div>
        <div className="flex gap-2 mt-3">
          {[
            { label: '5% / 95%', values: { control: 5, variant: 95 } },
            { label: '50% / 50%', values: { control: 50, variant: 50 } },
            { label: '80% / 20%', values: { control: 80, variant: 20 } }
          ].map((preset, idx) => (
            <button
              key={idx}
              onClick={() => setTraffic(preset.values)}
              className="px-3 py-1 text-xs bg-slate-100 dark:bg-slate-700 rounded-full hover:bg-slate-200 dark:hover:bg-slate-600 transition-all"
            >
              {preset.label}
            </button>
          ))}
        </div>
      </div>

      {/* 统计显著性 */}
      <div className={`mb-6 p-4 rounded-lg border ${
        stats.significant
          ? 'bg-green-50 border-green-200 dark:bg-green-900/20 dark:border-green-700'
          : 'bg-yellow-50 border-yellow-200 dark:bg-yellow-900/20 dark:border-yellow-700'
      }`}>
        <div className="flex items-center gap-3 mb-2">
          {stats.significant ? (
            <CheckCircle2 className="w-6 h-6 text-green-600 dark:text-green-400" />
          ) : (
            <XCircle className="w-6 h-6 text-yellow-600 dark:text-yellow-400" />
          )}
          <div className="font-bold text-slate-800 dark:text-slate-200">
            {stats.significant ? '统计显著！' : '统计不显著'}
          </div>
        </div>
        <div className="grid grid-cols-3 gap-4 text-sm">
          <div>
            <div className="text-slate-600 dark:text-slate-400">Z-Score</div>
            <div className="font-bold text-slate-800 dark:text-slate-200">{stats.zScore}</div>
          </div>
          <div>
            <div className="text-slate-600 dark:text-slate-400">提升幅度</div>
            <div className="font-bold text-green-600 dark:text-green-400">+{stats.lift}%</div>
          </div>
          <div>
            <div className="text-slate-600 dark:text-slate-400">置信水平</div>
            <div className="font-bold text-slate-800 dark:text-slate-200">
              {stats.significant ? '95%+' : '<95%'}
            </div>
          </div>
        </div>
      </div>

      {/* 指标选择 */}
      <div className="flex gap-2 mb-4">
        {[
          { id: 'conversion', label: '转化率', icon: <TrendingUp className="w-4 h-4" /> },
          { id: 'latency', label: '延迟', icon: <Zap className="w-4 h-4" /> },
          { id: 'error', label: '错误率', icon: <XCircle className="w-4 h-4" /> }
        ].map((metric) => (
          <button
            key={metric.id}
            onClick={() => setSelectedMetric(metric.id as any)}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all ${
              selectedMetric === metric.id
                ? 'bg-blue-500 text-white'
                : 'bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300 border border-slate-200 dark:border-slate-700'
            }`}
          >
            {metric.icon}
            {metric.label}
          </button>
        ))}
      </div>

      {/* 变体对比 */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {variants.map((variant, idx) => (
          <motion.div
            key={variant.id}
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: idx * 0.1 }}
            className={`p-6 rounded-lg border-2 ${
              idx === 0
                ? 'bg-blue-50 border-blue-200 dark:bg-blue-900/20 dark:border-blue-700'
                : 'bg-purple-50 border-purple-200 dark:bg-purple-900/20 dark:border-purple-700'
            }`}
          >
            <div className="font-bold text-lg mb-2 text-slate-800 dark:text-slate-200">
              {variant.name}
            </div>
            <div className="text-sm text-slate-600 dark:text-slate-400 mb-4">
              {variant.description}
            </div>

            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-sm text-slate-600 dark:text-slate-400">
                  <Users className="w-4 h-4" />
                  用户数
                </div>
                <div className="font-bold text-slate-800 dark:text-slate-200">
                  {variant.users.toLocaleString()}
                </div>
              </div>

              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-sm text-slate-600 dark:text-slate-400">
                  <TrendingUp className="w-4 h-4" />
                  转化率
                </div>
                <div className="font-bold text-green-600 dark:text-green-400">
                  {calculateMetrics(variant).conversionRate}%
                </div>
              </div>

              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-sm text-slate-600 dark:text-slate-400">
                  <Zap className="w-4 h-4" />
                  平均延迟
                </div>
                <div className="font-bold text-slate-800 dark:text-slate-200">
                  {Math.round(variant.avgLatency)}ms
                </div>
              </div>

              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-sm text-slate-600 dark:text-slate-400">
                  <XCircle className="w-4 h-4" />
                  错误率
                </div>
                <div className="font-bold text-red-600 dark:text-red-400">
                  {variant.errorRate.toFixed(2)}%
                </div>
              </div>
            </div>

            {idx === 1 && stats.significant && (
              <div className="mt-4 p-3 bg-green-100 dark:bg-green-900/30 rounded-lg">
                <div className="text-sm font-medium text-green-700 dark:text-green-300">
                  ✅ 实验成功！转化率提升 {stats.lift}%
                </div>
              </div>
            )}
          </motion.div>
        ))}
      </div>

      {/* 建议 */}
      {stats.significant && (
        <div className="mt-6 p-4 bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-lg border border-green-200 dark:border-green-700">
          <div className="font-bold text-slate-800 dark:text-slate-200 mb-2">
            📊 实验建议
          </div>
          <div className="text-sm text-slate-600 dark:text-slate-400 space-y-1">
            <div>• Variant 表现显著优于 Control（Z-Score = {stats.zScore}）</div>
            <div>• 建议逐步扩大 Variant 流量至 100%</div>
            <div>• 继续监控延迟指标，确保用户体验</div>
          </div>
        </div>
      )}
    </div>
  )
}
