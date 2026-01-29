'use client';

import React, { useState, useMemo } from 'react';

type ArchitectureLayer = {
  id: string;
  name: string;
  components: {
    name: string;
    description: string;
    icon: string;
    color: string;
  }[];
};

export default function DeploymentArchitecture() {
  const [selectedLayer, setSelectedLayer] = useState<string>('application');

  const layers: ArchitectureLayer[] = useMemo(() => [
    {
      id: 'client',
      name: '客户端层',
      components: [
        {
          name: 'Web 应用',
          description: 'React/Vue 前端应用，通过 HTTP 调用 API',
          icon: '🌐',
          color: 'blue'
        },
        {
          name: 'Python SDK',
          description: 'RemoteRunnable 客户端，原生调用远程链',
          icon: '🐍',
          color: 'green'
        },
        {
          name: 'cURL/Postman',
          description: 'RESTful API 测试工具',
          icon: '🔧',
          color: 'orange'
        }
      ]
    },
    {
      id: 'gateway',
      name: '网关层',
      components: [
        {
          name: 'Nginx',
          description: '反向代理、负载均衡、SSL 终止',
          icon: '🚪',
          color: 'purple'
        },
        {
          name: 'API Gateway',
          description: '认证、限流、日志、监控',
          icon: '🛡️',
          color: 'red'
        },
        {
          name: 'CDN',
          description: '静态资源缓存、边缘加速',
          icon: '🌍',
          color: 'cyan'
        }
      ]
    },
    {
      id: 'application',
      name: '应用层',
      components: [
        {
          name: 'LangServe',
          description: 'FastAPI 应用，部署 LCEL 链和 LangGraph 图',
          icon: '⚡',
          color: 'yellow'
        },
        {
          name: 'Uvicorn',
          description: 'ASGI 服务器，处理异步请求',
          icon: '🚀',
          color: 'blue'
        },
        {
          name: 'Worker Pool',
          description: '多进程 Worker，并发处理请求',
          icon: '👷',
          color: 'green'
        }
      ]
    },
    {
      id: 'service',
      name: '服务层',
      components: [
        {
          name: 'LLM API',
          description: 'OpenAI/Anthropic/Grok API 调用',
          icon: '🤖',
          color: 'purple'
        },
        {
          name: 'Vector DB',
          description: 'Pinecone/Weaviate 向量检索',
          icon: '🔍',
          color: 'orange'
        },
        {
          name: 'Redis',
          description: '缓存层，提升响应速度',
          icon: '💾',
          color: 'red'
        }
      ]
    },
    {
      id: 'monitoring',
      name: '监控层',
      components: [
        {
          name: 'LangSmith',
          description: 'Trace 追踪、评估、监控',
          icon: '📊',
          color: 'blue'
        },
        {
          name: 'Prometheus',
          description: '指标收集、时序数据库',
          icon: '📈',
          color: 'orange'
        },
        {
          name: 'Grafana',
          description: '可视化仪表板、告警',
          icon: '📉',
          color: 'cyan'
        }
      ]
    }
  ], []);

  const currentLayer = useMemo(
    () => layers.find(layer => layer.id === selectedLayer)!,
    [selectedLayer, layers]
  );

  const getColorClasses = (color: string) => {
    const colors: Record<string, { bg: string; border: string; text: string }> = {
      blue: { bg: 'bg-blue-100 dark:bg-blue-900/30', border: 'border-blue-500', text: 'text-blue-700 dark:text-blue-300' },
      green: { bg: 'bg-green-100 dark:bg-green-900/30', border: 'border-green-500', text: 'text-green-700 dark:text-green-300' },
      purple: { bg: 'bg-purple-100 dark:bg-purple-900/30', border: 'border-purple-500', text: 'text-purple-700 dark:text-purple-300' },
      orange: { bg: 'bg-orange-100 dark:bg-orange-900/30', border: 'border-orange-500', text: 'text-orange-700 dark:text-orange-300' },
      red: { bg: 'bg-red-100 dark:bg-red-900/30', border: 'border-red-500', text: 'text-red-700 dark:text-red-300' },
      yellow: { bg: 'bg-yellow-100 dark:bg-yellow-900/30', border: 'border-yellow-500', text: 'text-yellow-700 dark:text-yellow-300' },
      cyan: { bg: 'bg-cyan-100 dark:bg-cyan-900/30', border: 'border-cyan-500', text: 'text-cyan-700 dark:text-cyan-300' }
    };
    return colors[color] || colors.blue;
  };

  return (
    <div className="my-8 p-8 bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-900 rounded-2xl shadow-xl border border-gray-200 dark:border-gray-700">
      <h3 className="text-2xl font-bold mb-2 bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
        LangServe 部署架构
      </h3>
      <p className="text-sm text-gray-600 dark:text-gray-400 mb-6">
        探索 LangServe 在生产环境中的完整技术栈和系统架构
      </p>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-1">
          <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-lg sticky top-4">
            <h4 className="font-bold mb-4 text-gray-800 dark:text-gray-200 flex items-center gap-2">
              <span className="text-xl">🏗️</span>
              架构层级
            </h4>
            <div className="space-y-2">
              {layers.map((layer, idx) => (
                <button
                  key={layer.id}
                  onClick={() => setSelectedLayer(layer.id)}
                  className={`w-full p-3 rounded-xl text-left transition-all border-2 ${
                    selectedLayer === layer.id
                      ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20 shadow-lg scale-105'
                      : 'border-gray-200 dark:border-gray-600 bg-gray-50 dark:bg-gray-700 hover:shadow-md'
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <div className="flex-shrink-0 w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-500 text-white rounded-full flex items-center justify-center font-bold shadow-lg">
                      {layers.length - idx}
                    </div>
                    <div className="font-semibold text-gray-800 dark:text-gray-200">
                      {layer.name}
                    </div>
                  </div>
                </button>
              ))}
            </div>

            <div className="mt-6 p-4 bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-xl border-l-4 border-blue-500">
              <div className="text-xs text-gray-600 dark:text-gray-400 mb-1">数据流向</div>
              <div className="text-sm text-gray-700 dark:text-gray-300 font-semibold">
                Client → Gateway → App → Services
              </div>
            </div>
          </div>
        </div>

        <div className="lg:col-span-2">
          <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-lg mb-6">
            <h4 className="font-bold mb-4 text-gray-800 dark:text-gray-200">
              {currentLayer.name} - 组件详情
            </h4>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {currentLayer.components.map((component) => {
                const colors = getColorClasses(component.color);
                return (
                  <div
                    key={component.name}
                    className={`p-5 rounded-xl border-2 ${colors.border} ${colors.bg} hover:shadow-lg transition-all`}
                  >
                    <div className="text-4xl mb-3">{component.icon}</div>
                    <div className={`font-bold mb-2 ${colors.text}`}>
                      {component.name}
                    </div>
                    <div className="text-xs text-gray-600 dark:text-gray-400 leading-relaxed">
                      {component.description}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-lg">
            <h4 className="font-bold mb-4 text-gray-800 dark:text-gray-200 flex items-center gap-2">
              <span className="text-xl">🔄</span>
              完整请求流程
            </h4>
            <div className="space-y-3">
              <div className="flex items-center gap-3 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg border-l-4 border-blue-500">
                <div className="flex-shrink-0 w-8 h-8 bg-blue-500 text-white rounded-full flex items-center justify-center font-bold">1</div>
                <div className="text-sm text-gray-700 dark:text-gray-300">
                  <strong>客户端发起请求</strong>：Web/SDK 调用 <code className="bg-blue-100 dark:bg-blue-900/40 px-2 py-0.5 rounded">/translate/invoke</code>
                </div>
              </div>
              <div className="flex items-center gap-3 p-3 bg-purple-50 dark:bg-purple-900/20 rounded-lg border-l-4 border-purple-500">
                <div className="flex-shrink-0 w-8 h-8 bg-purple-500 text-white rounded-full flex items-center justify-center font-bold">2</div>
                <div className="text-sm text-gray-700 dark:text-gray-300">
                  <strong>网关处理</strong>：Nginx 负载均衡，API Gateway 认证 + 限流
                </div>
              </div>
              <div className="flex items-center gap-3 p-3 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg border-l-4 border-yellow-500">
                <div className="flex-shrink-0 w-8 h-8 bg-yellow-500 text-white rounded-full flex items-center justify-center font-bold">3</div>
                <div className="text-sm text-gray-700 dark:text-gray-300">
                  <strong>LangServe 处理</strong>：FastAPI 接收请求，Pydantic 验证，执行链
                </div>
              </div>
              <div className="flex items-center gap-3 p-3 bg-green-50 dark:bg-green-900/20 rounded-lg border-l-4 border-green-500">
                <div className="flex-shrink-0 w-8 h-8 bg-green-500 text-white rounded-full flex items-center justify-center font-bold">4</div>
                <div className="text-sm text-gray-700 dark:text-gray-300">
                  <strong>服务调用</strong>：查 Redis 缓存 → 调用 LLM API → 向量检索（如需）
                </div>
              </div>
              <div className="flex items-center gap-3 p-3 bg-orange-50 dark:bg-orange-900/20 rounded-lg border-l-4 border-orange-500">
                <div className="flex-shrink-0 w-8 h-8 bg-orange-500 text-white rounded-full flex items-center justify-center font-bold">5</div>
                <div className="text-sm text-gray-700 dark:text-gray-300">
                  <strong>监控追踪</strong>：LangSmith 记录 Trace，Prometheus 记录指标
                </div>
              </div>
              <div className="flex items-center gap-3 p-3 bg-cyan-50 dark:bg-cyan-900/20 rounded-lg border-l-4 border-cyan-500">
                <div className="flex-shrink-0 w-8 h-8 bg-cyan-500 text-white rounded-full flex items-center justify-center font-bold">6</div>
                <div className="text-sm text-gray-700 dark:text-gray-300">
                  <strong>返回响应</strong>：结果经网关返回客户端，缓存结果到 Redis
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="mt-6 grid grid-cols-3 gap-4">
        <div className="p-5 bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-xl shadow-md border border-green-200 dark:border-green-700">
          <div className="flex items-center gap-3 mb-2">
            <div className="w-10 h-10 bg-green-500 rounded-full flex items-center justify-center shadow-lg">
              <span className="text-white text-xl">⚡</span>
            </div>
            <div className="font-bold text-gray-800 dark:text-gray-200">高性能</div>
          </div>
          <div className="text-sm text-gray-600 dark:text-gray-400">
            FastAPI 异步处理 + Uvicorn + Worker Pool
          </div>
        </div>
        <div className="p-5 bg-gradient-to-br from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20 rounded-xl shadow-md border border-blue-200 dark:border-blue-700">
          <div className="flex items-center gap-3 mb-2">
            <div className="w-10 h-10 bg-blue-500 rounded-full flex items-center justify-center shadow-lg">
              <span className="text-white text-xl">🔒</span>
            </div>
            <div className="font-bold text-gray-800 dark:text-gray-200">高安全</div>
          </div>
          <div className="text-sm text-gray-600 dark:text-gray-400">
            API 认证 + 限流 + 输入验证 + HTTPS
          </div>
        </div>
        <div className="p-5 bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-xl shadow-md border border-purple-200 dark:border-purple-700">
          <div className="flex items-center gap-3 mb-2">
            <div className="w-10 h-10 bg-purple-500 rounded-full flex items-center justify-center shadow-lg">
              <span className="text-white text-xl">📊</span>
            </div>
            <div className="font-bold text-gray-800 dark:text-gray-200">可观测</div>
          </div>
          <div className="text-sm text-gray-600 dark:text-gray-400">
            LangSmith Trace + Prometheus + Grafana
          </div>
        </div>
      </div>
    </div>
  );
}
