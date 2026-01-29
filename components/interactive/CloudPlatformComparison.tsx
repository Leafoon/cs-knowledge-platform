'use client'

import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { Cloud, DollarSign, Zap, Shield, Users, TrendingUp, CheckCircle, XCircle, AlertCircle } from 'lucide-react'

type CloudProvider = 'aws' | 'gcp' | 'azure'

interface ProviderFeature {
  name: string
  rating: number // 1-5
  description: string
}

interface PricingTier {
  name: string
  aws: string
  gcp: string
  azure: string
}

export default function CloudPlatformComparison() {
  const [selectedProvider, setSelectedProvider] = useState<CloudProvider>('aws')
  const [comparisonMode, setComparisonMode] = useState<'features' | 'pricing' | 'performance'>('features')
  
  const providers = {
    aws: {
      name: 'Amazon Web Services',
      logo: '🔶',
      color: 'orange',
      tagline: '成熟稳定的云服务领导者',
      marketShare: '32%',
    },
    gcp: {
      name: 'Google Cloud Platform',
      logo: '🔵',
      color: 'blue',
      tagline: 'AI/ML 和数据分析专家',
      marketShare: '11%',
    },
    azure: {
      name: 'Microsoft Azure',
      logo: '🔷',
      color: 'blue',
      tagline: '企业级混合云首选',
      marketShare: '23%',
    },
  }
  
  const features: Record<string, Record<CloudProvider, ProviderFeature>> = {
    kubernetesService: {
      aws: { name: 'EKS', rating: 4, description: '成熟但配置复杂' },
      gcp: { name: 'GKE', rating: 5, description: 'K8s 原生，自动化程度最高' },
      azure: { name: 'AKS', rating: 4, description: '与 Azure 生态集成好' },
    },
    containerRegistry: {
      aws: { name: 'ECR', rating: 4, description: '与 ECS/EKS 无缝集成' },
      gcp: { name: 'GCR/Artifact Registry', rating: 5, description: '速度快，自动漏洞扫描' },
      azure: { name: 'ACR', rating: 4, description: '支持多区域复制' },
    },
    serverless: {
      aws: { name: 'Lambda', rating: 5, description: '最丰富的触发器和集成' },
      gcp: { name: 'Cloud Functions', rating: 4, description: 'HTTP 函数简单易用' },
      azure: { name: 'Azure Functions', rating: 4, description: 'Durable Functions 支持状态' },
    },
    aiMlPlatform: {
      aws: { name: 'SageMaker', rating: 4, description: '功能全面但学习曲线陡' },
      gcp: { name: 'Vertex AI', rating: 5, description: 'AutoML 和 TensorFlow 原生' },
      azure: { name: 'Azure ML', rating: 4, description: '企业级 MLOps 完善' },
    },
    monitoring: {
      aws: { name: 'CloudWatch', rating: 3, description: '基础但需额外配置' },
      gcp: { name: 'Cloud Monitoring', rating: 5, description: 'Stackdriver 强大易用' },
      azure: { name: 'Azure Monitor', rating: 4, description: 'Application Insights 深度集成' },
    },
    pricing: {
      aws: { name: '按需定价', rating: 3, description: '复杂但灵活，Spot 实例便宜' },
      gcp: { name: '按秒计费', rating: 5, description: '最精细，持续使用折扣' },
      azure: { name: '混合权益', rating: 4, description: 'Windows Server 许可优惠' },
    },
  }
  
  const pricingTiers: PricingTier[] = [
    {
      name: '小型部署 (2 vCPU, 4GB RAM)',
      aws: '$0.0464/h (t3.medium)',
      gcp: '$0.0475/h (e2-medium)',
      azure: '$0.0496/h (B2s)',
    },
    {
      name: '中型部署 (4 vCPU, 16GB RAM)',
      aws: '$0.1856/h (t3.xlarge)',
      gcp: '$0.1900/h (e2-standard-4)',
      azure: '$0.2080/h (D4s_v3)',
    },
    {
      name: 'Kubernetes 集群费用',
      aws: '$0.10/h (控制平面)',
      gcp: '$0.10/h (GKE Autopilot 免费)',
      azure: '免费 (仅付节点费用)',
    },
    {
      name: '负载均衡器',
      aws: '$0.0225/h + 数据传输',
      gcp: '$0.025/h + 规则费',
      azure: '$0.025/h (Basic)',
    },
    {
      name: '容器镜像存储 (100GB)',
      aws: '$10/月',
      gcp: '$5/月',
      azure: '$10/月',
    },
  ]
  
  const performanceMetrics = {
    coldStart: {
      aws: { value: 250, unit: 'ms', rank: 2 },
      gcp: { value: 180, unit: 'ms', rank: 1 },
      azure: { value: 300, unit: 'ms', rank: 3 },
    },
    networkLatency: {
      aws: { value: 12, unit: 'ms', rank: 1 },
      gcp: { value: 15, unit: 'ms', rank: 2 },
      azure: { value: 18, unit: 'ms', rank: 3 },
    },
    scalingSpeed: {
      aws: { value: 45, unit: 's', rank: 2 },
      gcp: { value: 30, unit: 's', rank: 1 },
      azure: { value: 60, unit: 's', rank: 3 },
    },
  }
  
  const colorClasses = {
    orange: { bg: 'bg-orange-500', text: 'text-orange-700', light: 'bg-orange-50', border: 'border-orange-500' },
    blue: { bg: 'bg-blue-500', text: 'text-blue-700', light: 'bg-blue-50', border: 'border-blue-500' },
  }
  
  return (
    <div className="w-full max-w-7xl mx-auto p-6 bg-gradient-to-br from-gray-50 to-blue-50 rounded-xl shadow-lg">
      {/* 标题 */}
      <div className="text-center mb-8">
        <div className="flex items-center justify-center gap-3 mb-3">
          <Cloud className="w-8 h-8 text-indigo-600" />
          <h3 className="text-2xl font-bold text-gray-800">云平台部署对比</h3>
        </div>
        <p className="text-gray-600">全面比较 AWS、GCP、Azure 的 K8s 服务和成本</p>
      </div>

      {/* 对比模式切换 */}
      <div className="flex justify-center gap-3 mb-8">
        {(['features', 'pricing', 'performance'] as const).map((mode) => (
          <button
            key={mode}
            onClick={() => setComparisonMode(mode)}
            className={`px-6 py-2 rounded-lg font-medium transition-colors ${
              comparisonMode === mode
                ? 'bg-indigo-600 text-white shadow-md'
                : 'bg-white text-gray-700 hover:bg-gray-100'
            }`}
          >
            {mode === 'features' && '功能对比'}
            {mode === 'pricing' && '价格对比'}
            {mode === 'performance' && '性能对比'}
          </button>
        ))}
      </div>

      {/* 功能对比视图 */}
      {comparisonMode === 'features' && (
        <div className="space-y-6">
          {/* 提供商卡片 */}
          <div className="grid grid-cols-3 gap-4 mb-6">
            {(Object.keys(providers) as CloudProvider[]).map((provider) => {
              const info = providers[provider]
              const isSelected = selectedProvider === provider
              
              return (
                <motion.div
                  key={provider}
                  whileHover={{ scale: 1.02 }}
                  onClick={() => setSelectedProvider(provider)}
                  className={`p-6 rounded-lg border-2 cursor-pointer transition-all ${
                    isSelected
                      ? 'bg-white border-indigo-500 shadow-lg'
                      : 'bg-white border-gray-200 hover:border-gray-300'
                  }`}
                >
                  <div className="text-center">
                    <div className="text-4xl mb-2">{info.logo}</div>
                    <h4 className="font-bold text-gray-800 mb-1">{info.name}</h4>
                    <p className="text-xs text-gray-600 mb-3">{info.tagline}</p>
                    <div className="flex items-center justify-center gap-2">
                      <Users className="w-4 h-4 text-gray-500" />
                      <span className="text-sm font-semibold text-gray-700">市场份额: {info.marketShare}</span>
                    </div>
                  </div>
                </motion.div>
              )
            })}
          </div>

          {/* 功能详细对比表 */}
          <div className="bg-white rounded-lg shadow overflow-hidden">
            <table className="w-full">
              <thead className="bg-gray-100">
                <tr>
                  <th className="px-6 py-4 text-left text-sm font-semibold text-gray-700">功能</th>
                  <th className="px-6 py-4 text-center text-sm font-semibold text-gray-700">AWS</th>
                  <th className="px-6 py-4 text-center text-sm font-semibold text-gray-700">GCP</th>
                  <th className="px-6 py-4 text-center text-sm font-semibold text-gray-700">Azure</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {Object.entries(features).map(([key, providerFeatures]) => (
                  <tr key={key} className="hover:bg-gray-50">
                    <td className="px-6 py-4">
                      <div className="font-medium text-gray-800">
                        {key === 'kubernetesService' && 'Kubernetes 服务'}
                        {key === 'containerRegistry' && '容器镜像仓库'}
                        {key === 'serverless' && '无服务器函数'}
                        {key === 'aiMlPlatform' && 'AI/ML 平台'}
                        {key === 'monitoring' && '监控与日志'}
                        {key === 'pricing' && '计费模式'}
                      </div>
                    </td>
                    {(Object.keys(providers) as CloudProvider[]).map((provider) => {
                      const feature = providerFeatures[provider]
                      return (
                        <td key={provider} className="px-6 py-4 text-center">
                          <div className="font-semibold text-gray-800 mb-1">{feature.name}</div>
                          <div className="flex items-center justify-center gap-1 mb-1">
                            {Array.from({ length: 5 }).map((_, idx) => (
                              <div
                                key={idx}
                                className={`w-3 h-3 rounded-full ${
                                  idx < feature.rating ? 'bg-yellow-400' : 'bg-gray-200'
                                }`}
                              />
                            ))}
                          </div>
                          <div className="text-xs text-gray-600">{feature.description}</div>
                        </td>
                      )
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* 价格对比视图 */}
      {comparisonMode === 'pricing' && (
        <div className="space-y-6">
          <div className="bg-white rounded-lg shadow overflow-hidden">
            <table className="w-full">
              <thead className="bg-gradient-to-r from-green-500 to-emerald-600 text-white">
                <tr>
                  <th className="px-6 py-4 text-left font-semibold">配置</th>
                  <th className="px-6 py-4 text-center font-semibold">AWS</th>
                  <th className="px-6 py-4 text-center font-semibold">GCP</th>
                  <th className="px-6 py-4 text-center font-semibold">Azure</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {pricingTiers.map((tier, idx) => (
                  <tr key={idx} className="hover:bg-gray-50">
                    <td className="px-6 py-4 font-medium text-gray-800">{tier.name}</td>
                    <td className="px-6 py-4 text-center">
                      <span className="inline-block px-3 py-1 bg-orange-100 text-orange-700 rounded font-semibold">
                        {tier.aws}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-center">
                      <span className="inline-block px-3 py-1 bg-blue-100 text-blue-700 rounded font-semibold">
                        {tier.gcp}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-center">
                      <span className="inline-block px-3 py-1 bg-blue-100 text-blue-700 rounded font-semibold">
                        {tier.azure}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="grid grid-cols-3 gap-4">
            <div className="bg-orange-50 border-l-4 border-orange-500 p-4 rounded-r-lg">
              <DollarSign className="w-6 h-6 text-orange-600 mb-2" />
              <h5 className="font-semibold text-orange-900 mb-1">AWS 成本优势</h5>
              <p className="text-sm text-orange-800">Spot 实例可节省 70-90%，但需处理中断</p>
            </div>
            
            <div className="bg-blue-50 border-l-4 border-blue-500 p-4 rounded-r-lg">
              <DollarSign className="w-6 h-6 text-blue-600 mb-2" />
              <h5 className="font-semibold text-blue-900 mb-1">GCP 成本优势</h5>
              <p className="text-sm text-blue-800">按秒计费 + 持续使用自动折扣 30%</p>
            </div>
            
            <div className="bg-indigo-50 border-l-4 border-indigo-500 p-4 rounded-r-lg">
              <DollarSign className="w-6 h-6 text-indigo-600 mb-2" />
              <h5 className="font-semibold text-indigo-900 mb-1">Azure 成本优势</h5>
              <p className="text-sm text-indigo-800">混合权益：现有 Windows 许可可抵扣 40%</p>
            </div>
          </div>
        </div>
      )}

      {/* 性能对比视图 */}
      {comparisonMode === 'performance' && (
        <div className="space-y-6">
          {Object.entries(performanceMetrics).map(([metric, values]) => (
            <div key={metric} className="bg-white rounded-lg p-6 shadow">
              <h4 className="text-lg font-semibold text-gray-800 mb-4">
                {metric === 'coldStart' && '⚡ 冷启动时间 (越低越好)'}
                {metric === 'networkLatency' && '🌐 网络延迟 (越低越好)'}
                {metric === 'scalingSpeed' && '📈 扩容速度 (越低越好)'}
              </h4>
              
              <div className="grid grid-cols-3 gap-4">
                {(Object.keys(providers) as CloudProvider[]).map((provider) => {
                  const perf = values[provider]
                  const isWinner = perf.rank === 1
                  
                  return (
                    <div
                      key={provider}
                      className={`p-4 rounded-lg border-2 ${
                        isWinner
                          ? 'bg-green-50 border-green-500'
                          : 'bg-gray-50 border-gray-200'
                      }`}
                    >
                      <div className="flex items-center justify-between mb-2">
                        <span className="font-semibold text-gray-700">{providers[provider].name}</span>
                        {isWinner && <CheckCircle className="w-5 h-5 text-green-600" />}
                      </div>
                      
                      <div className="text-3xl font-bold text-gray-800 mb-1">
                        {perf.value}
                        <span className="text-lg text-gray-600 ml-1">{perf.unit}</span>
                      </div>
                      
                      <div className="flex items-center gap-1">
                        {Array.from({ length: 3 }).map((_, idx) => (
                          <div
                            key={idx}
                            className={`flex-1 h-2 rounded ${
                              idx < (4 - perf.rank) ? 'bg-green-400' : 'bg-gray-200'
                            }`}
                          />
                        ))}
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* 推荐建议 */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.3 }}
        className="mt-8 grid grid-cols-3 gap-4"
      >
        <div className="p-4 bg-orange-50 border-l-4 border-orange-500 rounded-r-lg">
          <h5 className="font-semibold text-orange-900 mb-2 flex items-center gap-2">
            <Shield className="w-5 h-5" />
            选择 AWS 如果你需要
          </h5>
          <ul className="text-sm text-orange-800 space-y-1">
            <li>• 最丰富的服务和第三方集成</li>
            <li>• 成熟的企业级支持</li>
            <li>• 全球最多的可用区</li>
          </ul>
        </div>
        
        <div className="p-4 bg-blue-50 border-l-4 border-blue-500 rounded-r-lg">
          <h5 className="font-semibold text-blue-900 mb-2 flex items-center gap-2">
            <Zap className="w-5 h-5" />
            选择 GCP 如果你需要
          </h5>
          <ul className="text-sm text-blue-800 space-y-1">
            <li>• 最佳的 Kubernetes 体验</li>
            <li>• AI/ML 和大数据分析</li>
            <li>• 最优惠的持续使用定价</li>
          </ul>
        </div>
        
        <div className="p-4 bg-indigo-50 border-l-4 border-indigo-500 rounded-r-lg">
          <h5 className="font-semibold text-indigo-900 mb-2 flex items-center gap-2">
            <TrendingUp className="w-5 h-5" />
            选择 Azure 如果你需要
          </h5>
          <ul className="text-sm text-indigo-800 space-y-1">
            <li>• 与微软技术栈集成</li>
            <li>• 混合云和本地部署</li>
            <li>• 企业协议折扣优惠</li>
          </ul>
        </div>
      </motion.div>
    </div>
  )
}
