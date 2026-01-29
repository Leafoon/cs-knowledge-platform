'use client'

import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Server, Database, Network, Shield, Activity, Loader, Globe, Box, HardDrive } from 'lucide-react'

type Component = 'ingress' | 'service' | 'deployment' | 'pod' | 'configmap' | 'secret' | 'pvc'

export default function KubernetesArchitectureDiagram() {
  const [selectedComponent, setSelectedComponent] = useState<Component | null>(null)
  const [showDataFlow, setShowDataFlow] = useState(false)
  
  const components = {
    ingress: {
      name: 'Ingress',
      icon: Globe,
      description: 'HTTP/HTTPS 路由和负载均衡',
      color: 'purple',
      yaml: `apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: langserve-ingress
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt
spec:
  tls:
    - hosts: [api.example.com]
      secretName: tls-cert
  rules:
    - host: api.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: langserve-svc
                port:
                  number: 80`,
      details: [
        '管理外部访问到集群内服务',
        '支持 TLS/SSL 证书终止',
        '基于路径和域名的路由规则',
        '通常与 NGINX Ingress Controller 配合',
      ]
    },
    service: {
      name: 'Service',
      icon: Network,
      description: '为 Pod 提供稳定的网络入口',
      color: 'blue',
      yaml: `apiVersion: v1
kind: Service
metadata:
  name: langserve-svc
spec:
  selector:
    app: langserve
  ports:
    - port: 80
      targetPort: 8000
  type: LoadBalancer`,
      details: [
        'ClusterIP: 集群内部访问（默认）',
        'NodePort: 通过节点 IP + 端口访问',
        'LoadBalancer: 云平台负载均衡器',
        '自动负载均衡到多个 Pod',
      ]
    },
    deployment: {
      name: 'Deployment',
      icon: Server,
      description: '管理 Pod 的副本数和更新策略',
      color: 'green',
      yaml: `apiVersion: apps/v1
kind: Deployment
metadata:
  name: langserve-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: langserve
  template:
    metadata:
      labels:
        app: langserve
    spec:
      containers:
        - name: app
          image: langserve:1.0
          ports:
            - containerPort: 8000`,
      details: [
        '声明期望的 Pod 副本数量',
        '滚动更新（零停机部署）',
        '自动重启失败的 Pod',
        '支持回滚到历史版本',
      ]
    },
    pod: {
      name: 'Pod',
      icon: Box,
      description: 'Kubernetes 最小部署单元',
      color: 'orange',
      yaml: `apiVersion: v1
kind: Pod
metadata:
  name: langserve-pod-abc123
  labels:
    app: langserve
spec:
  containers:
    - name: langserve
      image: langserve:1.0
      resources:
        requests:
          cpu: 250m
          memory: 512Mi
        limits:
          cpu: 500m
          memory: 1Gi`,
      details: [
        '包含一个或多个容器',
        '共享网络命名空间（localhost 互通）',
        '共享存储卷',
        '由 Deployment/ReplicaSet 管理',
      ]
    },
    configmap: {
      name: 'ConfigMap',
      icon: Database,
      description: '存储非敏感配置数据',
      color: 'yellow',
      yaml: `apiVersion: v1
kind: ConfigMap
metadata:
  name: langserve-config
data:
  LOG_LEVEL: "info"
  MAX_WORKERS: "4"
  REDIS_HOST: "redis-svc"
  APP_CONFIG: |
    {
      "timeout": 120,
      "retries": 3
    }`,
      details: [
        '以键值对形式存储配置',
        '可挂载为环境变量或文件',
        '修改后需重启 Pod 生效',
        '适合非敏感数据（日志级别、端点 URL）',
      ]
    },
    secret: {
      name: 'Secret',
      icon: Shield,
      description: '存储敏感数据（密码、密钥）',
      color: 'red',
      yaml: `apiVersion: v1
kind: Secret
metadata:
  name: langserve-secrets
type: Opaque
data:
  # Base64 编码
  OPENAI_API_KEY: c2stLi4u
  DB_PASSWORD: cGFzc3dvcmQ=
stringData:
  # 明文（自动编码）
  REDIS_URL: redis://user:pass@host`,
      details: [
        'Base64 编码存储（非加密！）',
        '建议使用外部密钥管理（Vault、AWS Secrets Manager）',
        '可挂载为环境变量或文件',
        '限制 RBAC 权限访问',
      ]
    },
    pvc: {
      name: 'PersistentVolumeClaim',
      icon: HardDrive,
      description: '持久化存储卷声明',
      color: 'indigo',
      yaml: `apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: langserve-data
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
  storageClassName: fast-ssd`,
      details: [
        'ReadWriteOnce: 单节点读写',
        'ReadOnlyMany: 多节点只读',
        'ReadWriteMany: 多节点读写（NFS）',
        '云平台自动创建持久卷（EBS、GCE PD）',
      ]
    },
  }
  
  const colorClasses: Record<string, string> = {
    purple: 'bg-purple-500',
    blue: 'bg-blue-500',
    green: 'bg-green-500',
    orange: 'bg-orange-500',
    yellow: 'bg-yellow-500',
    red: 'bg-red-500',
    indigo: 'bg-indigo-500',
  }
  
  return (
    <div className="w-full max-w-7xl mx-auto p-6 bg-gradient-to-br from-gray-50 to-blue-50 rounded-xl shadow-lg">
      {/* 标题 */}
      <div className="text-center mb-8">
        <div className="flex items-center justify-center gap-3 mb-3">
          <Activity className="w-8 h-8 text-indigo-600" />
          <h3 className="text-2xl font-bold text-gray-800">Kubernetes 架构概览</h3>
        </div>
        <p className="text-gray-600">点击组件查看详细说明和 YAML 配置</p>
      </div>

      {/* 控制按钮 */}
      <div className="flex justify-center gap-3 mb-8">
        <button
          onClick={() => setShowDataFlow(!showDataFlow)}
          className={`px-4 py-2 rounded-lg font-medium transition-colors ${
            showDataFlow
              ? 'bg-indigo-600 text-white'
              : 'bg-white text-gray-700 hover:bg-gray-100'
          }`}
        >
          {showDataFlow ? '隐藏数据流' : '显示数据流'}
        </button>
        <button
          onClick={() => setSelectedComponent(null)}
          className="px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 font-medium"
        >
          重置选择
        </button>
      </div>

      {/* 架构图 */}
      <div className="relative bg-white rounded-lg p-8 shadow-inner mb-6">
        <div className="space-y-6">
          {/* Ingress 层 */}
          <div className="flex justify-center">
            <ComponentCard
              component="ingress"
              info={components.ingress}
              selected={selectedComponent === 'ingress'}
              onClick={() => setSelectedComponent('ingress')}
              colorClass={colorClasses[components.ingress.color]}
            />
          </div>

          {/* 箭头 */}
          <div className="flex justify-center">
            <motion.div
              animate={showDataFlow ? { y: [0, 10, 0] } : {}}
              transition={{ duration: 1.5, repeat: Infinity }}
              className="text-indigo-600 text-2xl"
            >
              ↓
            </motion.div>
          </div>

          {/* Service 层 */}
          <div className="flex justify-center">
            <ComponentCard
              component="service"
              info={components.service}
              selected={selectedComponent === 'service'}
              onClick={() => setSelectedComponent('service')}
              colorClass={colorClasses[components.service.color]}
            />
          </div>

          {/* 箭头 */}
          <div className="flex justify-center">
            <motion.div
              animate={showDataFlow ? { y: [0, 10, 0] } : {}}
              transition={{ duration: 1.5, repeat: Infinity, delay: 0.5 }}
              className="text-indigo-600 text-2xl"
            >
              ↓
            </motion.div>
          </div>

          {/* Deployment 层 */}
          <div className="flex justify-center">
            <ComponentCard
              component="deployment"
              info={components.deployment}
              selected={selectedComponent === 'deployment'}
              onClick={() => setSelectedComponent('deployment')}
              colorClass={colorClasses[components.deployment.color]}
            />
          </div>

          {/* 箭头 */}
          <div className="flex justify-center">
            <motion.div
              animate={showDataFlow ? { y: [0, 10, 0] } : {}}
              transition={{ duration: 1.5, repeat: Infinity, delay: 1 }}
              className="text-indigo-600 text-2xl"
            >
              ↓
            </motion.div>
          </div>

          {/* Pod 层（3个副本） */}
          <div className="flex justify-center gap-4">
            {[1, 2, 3].map((num) => (
              <motion.div
                key={num}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: num * 0.1 }}
              >
                <ComponentCard
                  component="pod"
                  info={{ ...components.pod, name: `Pod ${num}` }}
                  selected={selectedComponent === 'pod'}
                  onClick={() => setSelectedComponent('pod')}
                  colorClass={colorClasses[components.pod.color]}
                  mini
                />
              </motion.div>
            ))}
          </div>

          {/* 配置和存储层 */}
          <div className="grid grid-cols-3 gap-4 mt-8">
            <ComponentCard
              component="configmap"
              info={components.configmap}
              selected={selectedComponent === 'configmap'}
              onClick={() => setSelectedComponent('configmap')}
              colorClass={colorClasses[components.configmap.color]}
              mini
            />
            <ComponentCard
              component="secret"
              info={components.secret}
              selected={selectedComponent === 'secret'}
              onClick={() => setSelectedComponent('secret')}
              colorClass={colorClasses[components.secret.color]}
              mini
            />
            <ComponentCard
              component="pvc"
              info={components.pvc}
              selected={selectedComponent === 'pvc'}
              onClick={() => setSelectedComponent('pvc')}
              colorClass={colorClasses[components.pvc.color]}
              mini
            />
          </div>
        </div>
      </div>

      {/* 详细信息面板 */}
      <AnimatePresence>
        {selectedComponent && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            className="grid grid-cols-2 gap-6"
          >
            {/* 左侧：说明 */}
            <div className="bg-white rounded-lg p-6 shadow">
              <h4 className="text-lg font-semibold text-gray-800 mb-4">
                {components[selectedComponent].name} 详解
              </h4>
              <ul className="space-y-2">
                {components[selectedComponent].details.map((detail, idx) => (
                  <li key={idx} className="flex items-start gap-2 text-sm text-gray-700">
                    <div className="w-1.5 h-1.5 rounded-full bg-indigo-500 mt-1.5 flex-shrink-0" />
                    {detail}
                  </li>
                ))}
              </ul>
            </div>

            {/* 右侧：YAML 配置 */}
            <div className="bg-gray-900 text-gray-100 rounded-lg p-6 shadow overflow-hidden">
              <div className="flex items-center gap-2 mb-4">
                <div className="flex gap-1.5">
                  <div className="w-3 h-3 rounded-full bg-red-500" />
                  <div className="w-3 h-3 rounded-full bg-yellow-500" />
                  <div className="w-3 h-3 rounded-full bg-green-500" />
                </div>
                <span className="text-sm text-gray-400 ml-2">
                  {selectedComponent}.yaml
                </span>
              </div>

              <pre className="text-xs overflow-x-auto leading-relaxed">
                <code>{components[selectedComponent].yaml}</code>
              </pre>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* 提示 */}
      {!selectedComponent && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="text-center p-8 bg-blue-50 rounded-lg border-2 border-dashed border-blue-300"
        >
          <Loader className="w-8 h-8 text-blue-600 mx-auto mb-3" />
          <p className="text-blue-800 font-medium">点击任意组件查看详细配置</p>
        </motion.div>
      )}
    </div>
  )
}

// 组件卡片
function ComponentCard({ 
  component, 
  info, 
  selected, 
  onClick, 
  colorClass,
  mini = false
}: { 
  component: Component
  info: any
  selected: boolean
  onClick: () => void
  colorClass: string
  mini?: boolean
}) {
  const Icon = info.icon
  
  return (
    <motion.div
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
      onClick={onClick}
      className={`cursor-pointer transition-all ${
        mini ? 'p-3' : 'p-4'
      } rounded-lg border-2 ${
        selected
          ? 'bg-indigo-50 border-indigo-500 shadow-lg'
          : 'bg-white border-gray-200 hover:border-gray-300'
      }`}
    >
      <div className={`${mini ? 'w-10 h-10' : 'w-12 h-12'} ${colorClass} rounded-lg flex items-center justify-center mb-2 mx-auto`}>
        <Icon className={`${mini ? 'w-5 h-5' : 'w-6 h-6'} text-white`} />
      </div>
      <h5 className={`${mini ? 'text-sm' : 'text-base'} font-semibold text-center text-gray-800 mb-1`}>
        {info.name}
      </h5>
      {!mini && (
        <p className="text-xs text-gray-600 text-center">
          {info.description}
        </p>
      )}
    </motion.div>
  )
}
