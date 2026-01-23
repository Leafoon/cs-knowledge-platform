'use client'

import React, { useState } from 'react'
import { motion } from 'framer-motion'

interface ResourceConfig {
  replicas: number
  cpuRequest: number
  memoryRequest: number
  cpuLimit: number
  memoryLimit: number
  gpu: number
  hpaEnabled: boolean
  minReplicas: number
  maxReplicas: number
  targetCPU: number
}

export default function K8sDeploymentVisualizer() {
  const [config, setConfig] = useState<ResourceConfig>({
    replicas: 3,
    cpuRequest: 1000,
    memoryRequest: 2048,
    cpuLimit: 2000,
    memoryLimit: 4096,
    gpu: 0,
    hpaEnabled: false,
    minReplicas: 2,
    maxReplicas: 10,
    targetCPU: 70,
  })

  const [currentUtilization, setCurrentUtilization] = useState({
    cpu: 45,
    memory: 60,
  })

  // 计算当前副本数（基于 HPA）
  const currentReplicas = config.hpaEnabled
    ? Math.min(
        config.maxReplicas,
        Math.max(
          config.minReplicas,
          Math.ceil((currentUtilization.cpu / config.targetCPU) * config.replicas)
        )
      )
    : config.replicas

  // 计算资源状态
  const getResourceStatus = (utilized: number, limit: number) => {
    const percentage = (utilized / limit) * 100
    if (percentage < 50) return { color: 'green', label: '健康' }
    if (percentage < 80) return { color: 'yellow', label: '正常' }
    return { color: 'red', label: '高负载' }
  }

  const cpuStatus = getResourceStatus(currentUtilization.cpu, config.cpuLimit)
  const memoryStatus = getResourceStatus(currentUtilization.memory, config.memoryLimit)

  // 生成 YAML
  const generateYAML = () => {
    return `apiVersion: apps/v1
kind: Deployment
metadata:
  name: transformers-api
  labels:
    app: transformers-api
spec:
  replicas: ${config.replicas}
  selector:
    matchLabels:
      app: transformers-api
  template:
    metadata:
      labels:
        app: transformers-api
    spec:
      containers:
      - name: api
        image: transformers-api:v1.0
        ports:
        - containerPort: 8000
        resources:
          requests:
            cpu: "${config.cpuRequest}m"
            memory: "${config.memoryRequest}Mi"${config.gpu > 0 ? `
            nvidia.com/gpu: ${config.gpu}` : ''}
          limits:
            cpu: "${config.cpuLimit}m"
            memory: "${config.memoryLimit}Mi"${config.gpu > 0 ? `
            nvidia.com/gpu: ${config.gpu}` : ''}
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5${config.hpaEnabled ? `
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: transformers-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: transformers-api
  minReplicas: ${config.minReplicas}
  maxReplicas: ${config.maxReplicas}
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: ${config.targetCPU}` : ''}`
  }

  // 计算成本（示例）
  const calculateMonthlyCost = () => {
    const cpuCost = (config.cpuRequest / 1000) * 0.04 * 730 // $0.04/vCPU/hour
    const memoryCost = (config.memoryRequest / 1024) * 0.005 * 730 // $0.005/GB/hour
    const gpuCost = config.gpu * 0.7 * 730 // $0.70/GPU/hour
    return ((cpuCost + memoryCost + gpuCost) * currentReplicas).toFixed(2)
  }

  return (
    <div className="w-full max-w-7xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-indigo-50 rounded-xl border border-slate-200">
      {/* 标题 */}
      <div className="text-center mb-6">
        <h3 className="text-2xl font-bold text-slate-800 mb-2">
          ☸️ Kubernetes 资源配置可视化
        </h3>
        <p className="text-slate-600">
          交互式配置 Deployment 和 HPA，实时查看资源分配
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* 左侧：配置面板 */}
        <div className="space-y-4">
          <div className="bg-white rounded-lg border border-slate-200 p-5">
            <h4 className="text-lg font-semibold text-slate-800 mb-4">
              📊 资源配置
            </h4>

            {/* 副本数 */}
            <div className="mb-4">
              <label className="block text-sm font-medium text-slate-700 mb-2">
                副本数：{config.replicas}
              </label>
              <input
                type="range"
                min="1"
                max="20"
                value={config.replicas}
                onChange={(e) => setConfig({ ...config, replicas: Number(e.target.value) })}
                className="w-full"
                disabled={config.hpaEnabled}
              />
              {config.hpaEnabled && (
                <div className="text-xs text-amber-600 mt-1">
                  ⚠️ HPA 已启用，副本数由自动扩缩器控制
                </div>
              )}
            </div>

            {/* CPU 请求 */}
            <div className="mb-4">
              <label className="block text-sm font-medium text-slate-700 mb-2">
                CPU 请求：{config.cpuRequest}m ({(config.cpuRequest / 1000).toFixed(2)} vCPU)
              </label>
              <input
                type="range"
                min="100"
                max="4000"
                step="100"
                value={config.cpuRequest}
                onChange={(e) => setConfig({ ...config, cpuRequest: Number(e.target.value) })}
                className="w-full"
              />
            </div>

            {/* CPU 限制 */}
            <div className="mb-4">
              <label className="block text-sm font-medium text-slate-700 mb-2">
                CPU 限制：{config.cpuLimit}m ({(config.cpuLimit / 1000).toFixed(2)} vCPU)
              </label>
              <input
                type="range"
                min="100"
                max="8000"
                step="100"
                value={config.cpuLimit}
                onChange={(e) => setConfig({ ...config, cpuLimit: Number(e.target.value) })}
                className="w-full"
              />
            </div>

            {/* 内存请求 */}
            <div className="mb-4">
              <label className="block text-sm font-medium text-slate-700 mb-2">
                内存请求：{config.memoryRequest}Mi ({(config.memoryRequest / 1024).toFixed(1)} Gi)
              </label>
              <input
                type="range"
                min="512"
                max="16384"
                step="512"
                value={config.memoryRequest}
                onChange={(e) => setConfig({ ...config, memoryRequest: Number(e.target.value) })}
                className="w-full"
              />
            </div>

            {/* 内存限制 */}
            <div className="mb-4">
              <label className="block text-sm font-medium text-slate-700 mb-2">
                内存限制：{config.memoryLimit}Mi ({(config.memoryLimit / 1024).toFixed(1)} Gi)
              </label>
              <input
                type="range"
                min="512"
                max="32768"
                step="512"
                value={config.memoryLimit}
                onChange={(e) => setConfig({ ...config, memoryLimit: Number(e.target.value) })}
                className="w-full"
              />
            </div>

            {/* GPU */}
            <div className="mb-4">
              <label className="block text-sm font-medium text-slate-700 mb-2">
                GPU 数量：{config.gpu}
              </label>
              <input
                type="range"
                min="0"
                max="8"
                value={config.gpu}
                onChange={(e) => setConfig({ ...config, gpu: Number(e.target.value) })}
                className="w-full"
              />
            </div>
          </div>

          {/* HPA 配置 */}
          <div className="bg-white rounded-lg border border-slate-200 p-5">
            <div className="flex items-center justify-between mb-4">
              <h4 className="text-lg font-semibold text-slate-800">
                🔄 自动扩缩容 (HPA)
              </h4>
              <label className="relative inline-flex items-center cursor-pointer">
                <input
                  type="checkbox"
                  checked={config.hpaEnabled}
                  onChange={(e) => setConfig({ ...config, hpaEnabled: e.target.checked })}
                  className="sr-only peer"
                />
                <div className="w-11 h-6 bg-slate-300 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-slate-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
              </label>
            </div>

            {config.hpaEnabled && (
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-2">
                    最小副本数：{config.minReplicas}
                  </label>
                  <input
                    type="range"
                    min="1"
                    max="10"
                    value={config.minReplicas}
                    onChange={(e) => setConfig({ ...config, minReplicas: Number(e.target.value) })}
                    className="w-full"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-2">
                    最大副本数：{config.maxReplicas}
                  </label>
                  <input
                    type="range"
                    min="2"
                    max="50"
                    value={config.maxReplicas}
                    onChange={(e) => setConfig({ ...config, maxReplicas: Number(e.target.value) })}
                    className="w-full"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-2">
                    目标 CPU 利用率：{config.targetCPU}%
                  </label>
                  <input
                    type="range"
                    min="30"
                    max="90"
                    step="5"
                    value={config.targetCPU}
                    onChange={(e) => setConfig({ ...config, targetCPU: Number(e.target.value) })}
                    className="w-full"
                  />
                </div>
              </div>
            )}
          </div>

          {/* 成本估算 */}
          <div className="bg-gradient-to-r from-green-50 to-blue-50 rounded-lg border border-green-200 p-5">
            <h4 className="text-lg font-semibold text-slate-800 mb-3">
              💰 预估月度成本
            </h4>
            <div className="text-4xl font-bold text-green-600 mb-2">
              ${calculateMonthlyCost()}
            </div>
            <div className="text-sm text-slate-600">
              基于 {currentReplicas} 个副本 × 730 小时/月
            </div>
            <div className="mt-3 space-y-1 text-xs text-slate-500">
              <div>• CPU: {(config.cpuRequest / 1000).toFixed(2)} vCPU × ${0.04}/h</div>
              <div>• 内存: {(config.memoryRequest / 1024).toFixed(1)} GB × ${0.005}/h</div>
              {config.gpu > 0 && <div>• GPU: {config.gpu} × ${0.70}/h</div>}
            </div>
          </div>
        </div>

        {/* 右侧：可视化 + YAML */}
        <div className="space-y-4">
          {/* Pod 可视化 */}
          <div className="bg-white rounded-lg border border-slate-200 p-5">
            <h4 className="text-lg font-semibold text-slate-800 mb-4">
              🖥️ Pod 部署可视化
            </h4>
            
            {config.hpaEnabled && (
              <div className="mb-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
                <div className="text-sm font-medium text-blue-900 mb-1">
                  当前副本数：{currentReplicas} / {config.maxReplicas}
                </div>
                <div className="text-xs text-blue-700">
                  CPU 利用率 {currentUtilization.cpu}% → 目标 {config.targetCPU}%
                </div>
              </div>
            )}

            <div className="grid grid-cols-5 gap-2">
              {Array.from({ length: currentReplicas }).map((_, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, scale: 0 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: i * 0.05 }}
                  className="aspect-square bg-gradient-to-br from-green-400 to-blue-500 rounded-lg flex items-center justify-center text-white text-xs font-bold shadow-md"
                >
                  Pod
                  <br />
                  {i + 1}
                </motion.div>
              ))}
            </div>

            {/* 模拟负载控制 */}
            {config.hpaEnabled && (
              <div className="mt-4 space-y-3">
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-2">
                    模拟 CPU 负载：{currentUtilization.cpu}%
                  </label>
                  <input
                    type="range"
                    min="10"
                    max="100"
                    value={currentUtilization.cpu}
                    onChange={(e) => setCurrentUtilization({ ...currentUtilization, cpu: Number(e.target.value) })}
                    className="w-full"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-2">
                    模拟内存负载：{currentUtilization.memory}%
                  </label>
                  <input
                    type="range"
                    min="10"
                    max="100"
                    value={currentUtilization.memory}
                    onChange={(e) => setCurrentUtilization({ ...currentUtilization, memory: Number(e.target.value) })}
                    className="w-full"
                  />
                </div>
              </div>
            )}
          </div>

          {/* 资源状态 */}
          <div className="bg-white rounded-lg border border-slate-200 p-5">
            <h4 className="text-lg font-semibold text-slate-800 mb-4">
              📈 资源利用率
            </h4>
            
            <div className="space-y-4">
              {/* CPU */}
              <div>
                <div className="flex justify-between text-sm mb-2">
                  <span className="font-medium text-slate-700">CPU</span>
                  <span className={`font-semibold text-${cpuStatus.color}-600`}>
                    {cpuStatus.label}
                  </span>
                </div>
                <div className="w-full bg-slate-200 rounded-full h-4 overflow-hidden">
                  <motion.div
                    className={`h-full bg-gradient-to-r from-${cpuStatus.color}-400 to-${cpuStatus.color}-600`}
                    initial={{ width: 0 }}
                    animate={{ width: `${(currentUtilization.cpu / config.cpuLimit) * 100}%` }}
                    transition={{ duration: 0.5 }}
                  />
                </div>
                <div className="text-xs text-slate-500 mt-1">
                  {currentUtilization.cpu}% / {config.cpuLimit}m limit
                </div>
              </div>

              {/* 内存 */}
              <div>
                <div className="flex justify-between text-sm mb-2">
                  <span className="font-medium text-slate-700">内存</span>
                  <span className={`font-semibold text-${memoryStatus.color}-600`}>
                    {memoryStatus.label}
                  </span>
                </div>
                <div className="w-full bg-slate-200 rounded-full h-4 overflow-hidden">
                  <motion.div
                    className={`h-full bg-gradient-to-r from-${memoryStatus.color}-400 to-${memoryStatus.color}-600`}
                    initial={{ width: 0 }}
                    animate={{ width: `${(currentUtilization.memory / config.memoryLimit) * 100}%` }}
                    transition={{ duration: 0.5 }}
                  />
                </div>
                <div className="text-xs text-slate-500 mt-1">
                  {currentUtilization.memory}% / {config.memoryLimit}Mi limit
                </div>
              </div>
            </div>
          </div>

          {/* YAML 输出 */}
          <div className="bg-slate-900 rounded-lg p-5 text-white">
            <h4 className="text-lg font-semibold mb-3">📄 生成的 YAML</h4>
            <pre className="text-xs overflow-x-auto">
              <code>{generateYAML()}</code>
            </pre>
          </div>
        </div>
      </div>

      {/* 最佳实践提示 */}
      <div className="mt-6 p-4 bg-amber-50 border border-amber-200 rounded-lg">
        <div className="text-sm font-medium text-amber-900 mb-2">
          💡 配置最佳实践
        </div>
        <ul className="text-sm text-amber-800 space-y-1">
          <li>• <strong>Requests vs Limits</strong>：Requests 用于调度，Limits 防止资源耗尽</li>
          <li>• <strong>CPU</strong>：通常设置 Limit = 2x Request，避免 throttling</li>
          <li>• <strong>内存</strong>：OOM 会导致 Pod 被杀死，Limit 应预留缓冲</li>
          <li>• <strong>HPA</strong>：目标 CPU 70-80%，过低浪费资源，过高响应慢</li>
          <li>• <strong>GPU</strong>：GPU 不能超分，Request = Limit</li>
        </ul>
      </div>
    </div>
  )
}
