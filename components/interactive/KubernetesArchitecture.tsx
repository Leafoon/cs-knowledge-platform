'use client';

import React, { useState, useMemo } from 'react';

type K8sResource = {
  id: string;
  type: string;
  name: string;
  replicas?: number;
  description: string;
  icon: string;
  color: string;
  connections: string[];
};

export default function KubernetesArchitecture() {
  const [selectedResource, setSelectedResource] = useState<string | null>(null);
  const [showConnections, setShowConnections] = useState(true);

  const resources: K8sResource[] = useMemo(() => [
    {
      id: 'ingress',
      type: 'Ingress',
      name: 'langserve-ingress',
      description: 'HTTP(S) 路由、SSL 终止、域名映射',
      icon: '🌐',
      color: 'blue',
      connections: ['service']
    },
    {
      id: 'service',
      type: 'Service',
      name: 'langserve-service',
      description: 'LoadBalancer，负载均衡到 Pod',
      icon: '⚖️',
      color: 'green',
      connections: ['deployment']
    },
    {
      id: 'deployment',
      type: 'Deployment',
      name: 'langserve-deployment',
      replicas: 3,
      description: '管理 Pod 副本，滚动更新',
      icon: '🚀',
      color: 'purple',
      connections: ['pod1', 'pod2', 'pod3']
    },
    {
      id: 'pod1',
      type: 'Pod',
      name: 'langserve-pod-1',
      description: 'Container: langserve-app:latest',
      icon: '📦',
      color: 'yellow',
      connections: ['secret', 'configmap']
    },
    {
      id: 'pod2',
      type: 'Pod',
      name: 'langserve-pod-2',
      description: 'Container: langserve-app:latest',
      icon: '📦',
      color: 'yellow',
      connections: ['secret', 'configmap']
    },
    {
      id: 'pod3',
      type: 'Pod',
      name: 'langserve-pod-3',
      description: 'Container: langserve-app:latest',
      icon: '📦',
      color: 'yellow',
      connections: ['secret', 'configmap']
    },
    {
      id: 'hpa',
      type: 'HPA',
      name: 'langserve-hpa',
      description: '水平自动扩展 (2-10 副本)',
      icon: '📈',
      color: 'orange',
      connections: ['deployment']
    },
    {
      id: 'secret',
      type: 'Secret',
      name: 'langserve-secrets',
      description: 'OPENAI_API_KEY, 数据库密码',
      icon: '🔐',
      color: 'red',
      connections: []
    },
    {
      id: 'configmap',
      type: 'ConfigMap',
      name: 'langserve-config',
      description: '环境变量、配置文件',
      icon: '⚙️',
      color: 'cyan',
      connections: []
    }
  ], []);

  const selectedResourceData = useMemo(
    () => resources.find(r => r.id === selectedResource),
    [selectedResource, resources]
  );

  const getColorClasses = (color: string) => {
    const colors: Record<string, { bg: string; border: string; text: string; shadow: string }> = {
      blue: { bg: 'bg-blue-100 dark:bg-blue-900/30', border: 'border-blue-500', text: 'text-blue-700 dark:text-blue-300', shadow: 'shadow-blue-500/50' },
      green: { bg: 'bg-green-100 dark:bg-green-900/30', border: 'border-green-500', text: 'text-green-700 dark:text-green-300', shadow: 'shadow-green-500/50' },
      purple: { bg: 'bg-purple-100 dark:bg-purple-900/30', border: 'border-purple-500', text: 'text-purple-700 dark:text-purple-300', shadow: 'shadow-purple-500/50' },
      yellow: { bg: 'bg-yellow-100 dark:bg-yellow-900/30', border: 'border-yellow-500', text: 'text-yellow-700 dark:text-yellow-300', shadow: 'shadow-yellow-500/50' },
      orange: { bg: 'bg-orange-100 dark:bg-orange-900/30', border: 'border-orange-500', text: 'text-orange-700 dark:text-orange-300', shadow: 'shadow-orange-500/50' },
      red: { bg: 'bg-red-100 dark:bg-red-900/30', border: 'border-red-500', text: 'text-red-700 dark:text-red-300', shadow: 'shadow-red-500/50' },
      cyan: { bg: 'bg-cyan-100 dark:bg-cyan-900/30', border: 'border-cyan-500', text: 'text-cyan-700 dark:text-cyan-300', shadow: 'shadow-cyan-500/50' }
    };
    return colors[color] || colors.blue;
  };

  const renderResource = (resource: K8sResource, x: number, y: number, scale: number = 1) => {
    const colors = getColorClasses(resource.color);
    const isSelected = selectedResource === resource.id;
    const isConnected = selectedResource && 
      (selectedResourceData?.connections.includes(resource.id) || 
       resource.connections.includes(selectedResource));

    return (
      <div
        key={resource.id}
        className={`absolute cursor-pointer transition-all ${
          isSelected 
            ? `z-20 scale-110 ${colors.shadow} shadow-2xl` 
            : isConnected && showConnections
            ? 'z-10 scale-105'
            : 'z-0'
        }`}
        style={{
          left: `${x}%`,
          top: `${y}%`,
          transform: `translate(-50%, -50%) scale(${scale})`
        }}
        onClick={() => setSelectedResource(resource.id)}
      >
        <div className={`p-4 rounded-xl border-2 ${
          isSelected 
            ? `${colors.border} ${colors.bg}` 
            : isConnected && showConnections
            ? `${colors.border} bg-white dark:bg-gray-800`
            : 'border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800'
        } hover:shadow-lg transition-all min-w-[160px]`}>
          <div className="text-3xl mb-2 text-center">{resource.icon}</div>
          <div className={`font-bold text-sm text-center mb-1 ${isSelected ? colors.text : 'text-gray-800 dark:text-gray-200'}`}>
            {resource.type}
          </div>
          <div className="text-xs text-gray-500 dark:text-gray-400 text-center font-mono">
            {resource.name}
          </div>
          {resource.replicas && (
            <div className="mt-2 text-center">
              <span className={`px-2 py-1 rounded-full text-xs font-bold ${colors.bg} ${colors.text}`}>
                ×{resource.replicas}
              </span>
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="my-8 p-8 bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-900 rounded-2xl shadow-xl border border-gray-200 dark:border-gray-700">
      <h3 className="text-2xl font-bold mb-2 bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
        Kubernetes 部署架构
      </h3>
      <p className="text-sm text-gray-600 dark:text-gray-400 mb-6">
        探索 LangServe 在 Kubernetes 集群中的资源编排和自动化部署
      </p>

      <div className="flex gap-3 mb-6">
        <button
          onClick={() => setShowConnections(!showConnections)}
          className={`px-4 py-2 rounded-lg font-semibold transition-all ${
            showConnections
              ? 'bg-blue-500 text-white shadow-lg'
              : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 shadow-md hover:shadow-lg'
          }`}
        >
          {showConnections ? '✅ 显示连接' : '⬜ 隐藏连接'}
        </button>
        <button
          onClick={() => setSelectedResource(null)}
          className="px-4 py-2 rounded-lg font-semibold bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 shadow-md hover:shadow-lg transition-all"
        >
          🔄 重置选择
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-lg" style={{ minHeight: '600px' }}>
          <h4 className="font-bold mb-4 text-gray-800 dark:text-gray-200">K8s 资源拓扑</h4>
          <div className="relative bg-gradient-to-br from-blue-50 to-purple-50 dark:from-blue-900/10 dark:to-purple-900/10 rounded-xl p-6" style={{ height: '520px' }}>
            {/* Connections */}
            {showConnections && selectedResource && (
              <svg className="absolute inset-0 w-full h-full pointer-events-none" style={{ zIndex: 5 }}>
                {selectedResourceData?.connections.map(targetId => {
                  const target = resources.find(r => r.id === targetId);
                  if (!target) return null;
                  return (
                    <line
                      key={targetId}
                      x1="50%"
                      y1="20%"
                      x2="50%"
                      y2="80%"
                      stroke="#3b82f6"
                      strokeWidth="3"
                      strokeDasharray="8,4"
                      opacity="0.6"
                    />
                  );
                })}
              </svg>
            )}

            {/* Resources */}
            {renderResource(resources[0], 50, 8, 1)}  {/* Ingress */}
            {renderResource(resources[1], 50, 20, 1)}  {/* Service */}
            {renderResource(resources[2], 50, 35, 1)}  {/* Deployment */}
            {renderResource(resources[3], 25, 55, 0.9)} {/* Pod 1 */}
            {renderResource(resources[4], 50, 55, 0.9)} {/* Pod 2 */}
            {renderResource(resources[5], 75, 55, 0.9)} {/* Pod 3 */}
            {renderResource(resources[6], 85, 35, 0.85)} {/* HPA */}
            {renderResource(resources[7], 25, 75, 0.85)} {/* Secret */}
            {renderResource(resources[8], 75, 75, 0.85)} {/* ConfigMap */}
          </div>
        </div>

        <div className="lg:col-span-1">
          <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-lg sticky top-4">
            <h4 className="font-bold mb-4 text-gray-800 dark:text-gray-200 flex items-center gap-2">
              <span className="text-xl">📋</span>
              资源详情
            </h4>
            {selectedResourceData ? (
              <div className="space-y-4">
                <div className="text-center">
                  <div className="text-5xl mb-3">{selectedResourceData.icon}</div>
                  <div className="font-bold text-lg text-gray-800 dark:text-gray-200 mb-1">
                    {selectedResourceData.type}
                  </div>
                  <div className="text-sm text-gray-500 dark:text-gray-400 font-mono">
                    {selectedResourceData.name}
                  </div>
                </div>
                
                <div className="p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
                  <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">描述</div>
                  <div className="text-sm text-gray-700 dark:text-gray-300">
                    {selectedResourceData.description}
                  </div>
                </div>

                {selectedResourceData.replicas && (
                  <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
                    <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">副本数</div>
                    <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">
                      {selectedResourceData.replicas}
                    </div>
                  </div>
                )}

                {selectedResourceData.connections.length > 0 && (
                  <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                    <div className="text-xs text-gray-500 dark:text-gray-400 mb-2">连接到</div>
                    <div className="space-y-2">
                      {selectedResourceData.connections.map(connId => {
                        const conn = resources.find(r => r.id === connId);
                        return conn ? (
                          <div key={connId} className="flex items-center gap-2 text-sm">
                            <span className="text-lg">{conn.icon}</span>
                            <span className="text-gray-700 dark:text-gray-300">{conn.type}</span>
                          </div>
                        ) : null;
                      })}
                    </div>
                  </div>
                )}

                {selectedResourceData.type === 'Deployment' && (
                  <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
                    <div className="text-xs text-gray-500 dark:text-gray-400 mb-2">部署策略</div>
                    <div className="text-sm text-gray-700 dark:text-gray-300">
                      <div>• 滚动更新 (RollingUpdate)</div>
                      <div>• MaxSurge: 1</div>
                      <div>• MaxUnavailable: 0</div>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center py-12 text-gray-400 dark:text-gray-500">
                <div className="text-5xl mb-3">👈</div>
                <div className="text-sm">点击左侧资源查看详情</div>
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="mt-6 grid grid-cols-4 gap-4">
        <div className="p-4 bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/30 dark:to-blue-800/30 rounded-xl shadow-md border border-blue-200 dark:border-blue-700">
          <div className="flex items-center gap-3 mb-2">
            <div className="w-10 h-10 bg-blue-500 rounded-full flex items-center justify-center shadow-lg">
              <span className="text-white text-xl">🚀</span>
            </div>
            <div className="font-semibold text-gray-800 dark:text-gray-200 text-sm">自动扩展</div>
          </div>
          <div className="text-xs text-gray-600 dark:text-gray-400">
            HPA 根据 CPU/内存自动调整副本数
          </div>
        </div>
        <div className="p-4 bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/30 dark:to-green-800/30 rounded-xl shadow-md border border-green-200 dark:border-green-700">
          <div className="flex items-center gap-3 mb-2">
            <div className="w-10 h-10 bg-green-500 rounded-full flex items-center justify-center shadow-lg">
              <span className="text-white text-xl">💚</span>
            </div>
            <div className="font-semibold text-gray-800 dark:text-gray-200 text-sm">健康检查</div>
          </div>
          <div className="text-xs text-gray-600 dark:text-gray-400">
            Liveness & Readiness Probes
          </div>
        </div>
        <div className="p-4 bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-900/30 dark:to-purple-800/30 rounded-xl shadow-md border border-purple-200 dark:border-purple-700">
          <div className="flex items-center gap-3 mb-2">
            <div className="w-10 h-10 bg-purple-500 rounded-full flex items-center justify-center shadow-lg">
              <span className="text-white text-xl">🔄</span>
            </div>
            <div className="font-semibold text-gray-800 dark:text-gray-200 text-sm">滚动更新</div>
          </div>
          <div className="text-xs text-gray-600 dark:text-gray-400">
            零停机部署，渐进式发布
          </div>
        </div>
        <div className="p-4 bg-gradient-to-br from-red-50 to-red-100 dark:from-red-900/30 dark:to-red-800/30 rounded-xl shadow-md border border-red-200 dark:border-red-700">
          <div className="flex items-center gap-3 mb-2">
            <div className="w-10 h-10 bg-red-500 rounded-full flex items-center justify-center shadow-lg">
              <span className="text-white text-xl">🔐</span>
            </div>
            <div className="font-semibold text-gray-800 dark:text-gray-200 text-sm">密钥管理</div>
          </div>
          <div className="text-xs text-gray-600 dark:text-gray-400">
            Secret 加密存储敏感信息
          </div>
        </div>
      </div>

      <div className="mt-6 p-6 bg-gradient-to-r from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20 rounded-2xl border-l-4 border-yellow-500 shadow-lg">
        <div className="flex items-start gap-4">
          <div className="flex-shrink-0 w-10 h-10 bg-yellow-500 rounded-full flex items-center justify-center shadow-lg">
            <span className="text-white text-xl">💡</span>
          </div>
          <div>
            <h4 className="font-bold text-gray-800 dark:text-gray-100 mb-2">部署最佳实践</h4>
            <div className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
              <div>• 使用 <strong>Deployment</strong> 而非直接创建 Pod，便于版本管理和回滚</div>
              <div>• 配置 <strong>HPA</strong> 实现自动扩缩容，应对流量波动</div>
              <div>• 设置 <strong>Resource Limits</strong> 防止单个 Pod 占用过多资源</div>
              <div>• 使用 <strong>Readiness Probe</strong> 确保流量只路由到就绪的 Pod</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
