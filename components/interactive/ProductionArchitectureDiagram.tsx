'use client';

import React, { useState } from 'react';

type ComponentType = 'gateway' | 'langserve' | 'router' | 'rag' | 'agent' | 'llm' | 'vectordb' | 'cache' | 'monitoring';

interface ArchitectureComponent {
  id: ComponentType;
  name: string;
  description: string;
  technologies: string[];
  metrics: {
    latency?: string;
    availability?: string;
    throughput?: string;
  };
  responsibilities: string[];
}

const ProductionArchitectureDiagram: React.FC = () => {
  const [selectedComponent, setSelectedComponent] = useState<ComponentType | null>(null);
  const [showDataFlow, setShowDataFlow] = useState(false);

  const components: Record<ComponentType, ArchitectureComponent> = {
    gateway: {
      id: 'gateway',
      name: 'API Gateway',
      description: '统一入口，负责认证、限流、路由',
      technologies: ['Kong', 'Nginx', 'AWS API Gateway'],
      metrics: { latency: '< 10ms', availability: '99.99%', throughput: '10K req/s' },
      responsibilities: [
        'JWT 认证与授权',
        '速率限制（100 req/min/user）',
        'CORS 与安全头配置',
        '请求日志与追踪',
      ],
    },
    langserve: {
      id: 'langserve',
      name: 'LangServe (FastAPI)',
      description: '核心应用服务器，部署 LangChain 链/图',
      technologies: ['FastAPI', 'Uvicorn', 'Pydantic'],
      metrics: { latency: '~ 50ms', availability: '99.95%', throughput: '5K req/s' },
      responsibilities: [
        '链/图编排与执行',
        '请求参数验证',
        '错误处理与重试',
        '流式响应支持',
      ],
    },
    router: {
      id: 'router',
      name: 'Query Router',
      description: '智能路由，根据问题复杂度分配处理流程',
      technologies: ['LangChain', 'Embeddings Classifier'],
      metrics: { latency: '~ 30ms', availability: '99.9%' },
      responsibilities: [
        '简单问题 → 直接 LLM',
        '知识库问题 → RAG Pipeline',
        '工具调用 → Agent',
        '分类准确率 > 95%',
      ],
    },
    rag: {
      id: 'rag',
      name: 'RAG Pipeline',
      description: '检索增强生成流程',
      technologies: ['LangChain', 'Chroma/Pinecone', 'Reranker'],
      metrics: { latency: '~ 800ms', availability: '99.9%' },
      responsibilities: [
        'Query 改写与扩展',
        '向量检索 Top-K',
        'Contextual Compression',
        'Cohere Rerank',
      ],
    },
    agent: {
      id: 'agent',
      name: 'LangGraph Agent',
      description: '复杂任务编排，工具调用',
      technologies: ['LangGraph', 'Tool APIs', 'Checkpointer'],
      metrics: { latency: '~ 3s', availability: '99.8%' },
      responsibilities: [
        'ReAct Agent 决策',
        '工具调用与结果解析',
        '状态持久化',
        'Human-in-the-Loop',
      ],
    },
    llm: {
      id: 'llm',
      name: 'LLM Pool',
      description: '多提供商 LLM 负载均衡',
      technologies: ['GPT-4', 'Claude', 'Gemini', 'Fallback Chain'],
      metrics: { latency: '~ 2s', availability: '99.99%' },
      responsibilities: [
        '主 Provider: GPT-4',
        'Fallback 1: GPT-3.5',
        'Fallback 2: Claude',
        '熔断与降级',
      ],
    },
    vectordb: {
      id: 'vectordb',
      name: 'Vector Database',
      description: '向量存储与检索',
      technologies: ['Chroma', 'Pinecone', 'Weaviate'],
      metrics: { latency: '~ 50ms', availability: '99.95%' },
      responsibilities: [
        '文档 Embeddings 存储',
        'Cosine 相似度检索',
        'Metadata 过滤',
        '索引优化',
      ],
    },
    cache: {
      id: 'cache',
      name: 'Cache Layer',
      description: '多级缓存降低成本与延迟',
      technologies: ['Redis', 'LRU Cache', 'CDN'],
      metrics: { latency: '~ 5ms', availability: '99.99%' },
      responsibilities: [
        'L1: 内存缓存 (1000条)',
        'L2: Redis (1M条, 7天TTL)',
        '缓存命中率 > 60%',
        '智能预热与失效',
      ],
    },
    monitoring: {
      id: 'monitoring',
      name: 'Monitoring & Observability',
      description: '可观测性与监控',
      technologies: ['Prometheus', 'Grafana', 'LangSmith', 'Sentry'],
      metrics: { availability: '99.99%' },
      responsibilities: [
        'Prometheus 指标收集',
        'LangSmith Tracing',
        'Sentry 错误追踪',
        '告警与通知',
      ],
    },
  };

  const dataFlowSteps = [
    { from: 'User', to: 'gateway', label: 'HTTPS Request' },
    { from: 'gateway', to: 'langserve', label: 'Authenticated' },
    { from: 'langserve', to: 'router', label: 'Query Classification' },
    { from: 'router', to: 'rag', label: 'Knowledge Query' },
    { from: 'rag', to: 'vectordb', label: 'Vector Search' },
    { from: 'rag', to: 'llm', label: 'Context + Prompt' },
    { from: 'llm', to: 'cache', label: 'Check Cache' },
    { from: 'cache', to: 'User', label: 'Response' },
  ];

  const selected = selectedComponent ? components[selectedComponent] : null;

  return (
    <div className="w-full max-w-7xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 rounded-xl shadow-lg">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-2xl font-bold text-gray-800">生产架构图</h3>
        
        <button
          onClick={() => setShowDataFlow(!showDataFlow)}
          className={`px-4 py-2 rounded-lg font-semibold transition-all ${
            showDataFlow
              ? 'bg-gradient-to-r from-blue-500 to-indigo-600 text-white'
              : 'bg-white text-gray-700 border border-gray-300'
          }`}
        >
          {showDataFlow ? '隐藏数据流' : '显示数据流'}
        </button>
      </div>

      {/* 架构图 */}
      <div className="bg-white rounded-lg p-8 shadow-lg mb-6">
        <div className="grid grid-cols-1 gap-6">
          {/* 第1层: Gateway */}
          <div className="flex justify-center">
            <ComponentBox
              component={components.gateway}
              isSelected={selectedComponent === 'gateway'}
              onClick={() => setSelectedComponent('gateway')}
            />
          </div>

          <div className="flex justify-center">
            <div className="w-0.5 h-8 bg-gradient-to-b from-blue-400 to-blue-600"></div>
          </div>

          {/* 第2层: LangServe */}
          <div className="flex justify-center">
            <ComponentBox
              component={components.langserve}
              isSelected={selectedComponent === 'langserve'}
              onClick={() => setSelectedComponent('langserve')}
            />
          </div>

          <div className="flex justify-center">
            <div className="w-0.5 h-8 bg-gradient-to-b from-purple-400 to-purple-600"></div>
          </div>

          {/* 第3层: Router */}
          <div className="flex justify-center">
            <ComponentBox
              component={components.router}
              isSelected={selectedComponent === 'router'}
              onClick={() => setSelectedComponent('router')}
            />
          </div>

          <div className="flex justify-center space-x-20">
            <div className="w-0.5 h-8 bg-gradient-to-b from-green-400 to-green-600"></div>
            <div className="w-0.5 h-8 bg-gradient-to-b from-yellow-400 to-yellow-600"></div>
            <div className="w-0.5 h-8 bg-gradient-to-b from-red-400 to-red-600"></div>
          </div>

          {/* 第4层: RAG, Agent, Direct LLM */}
          <div className="grid grid-cols-3 gap-4">
            <ComponentBox
              component={components.rag}
              isSelected={selectedComponent === 'rag'}
              onClick={() => setSelectedComponent('rag')}
            />
            <ComponentBox
              component={components.agent}
              isSelected={selectedComponent === 'agent'}
              onClick={() => setSelectedComponent('agent')}
            />
            <div className="flex flex-col justify-center">
              <ComponentBox
                component={components.llm}
                isSelected={selectedComponent === 'llm'}
                onClick={() => setSelectedComponent('llm')}
                small
              />
            </div>
          </div>

          <div className="flex justify-center space-x-32">
            <div className="w-0.5 h-8 bg-gradient-to-b from-indigo-400 to-indigo-600"></div>
            <div className="w-0.5 h-8 bg-gradient-to-b from-pink-400 to-pink-600"></div>
          </div>

          {/* 第5层: VectorDB, Cache, Monitoring */}
          <div className="grid grid-cols-3 gap-4">
            <ComponentBox
              component={components.vectordb}
              isSelected={selectedComponent === 'vectordb'}
              onClick={() => setSelectedComponent('vectordb')}
              small
            />
            <ComponentBox
              component={components.cache}
              isSelected={selectedComponent === 'cache'}
              onClick={() => setSelectedComponent('cache')}
              small
            />
            <ComponentBox
              component={components.monitoring}
              isSelected={selectedComponent === 'monitoring'}
              onClick={() => setSelectedComponent('monitoring')}
              small
            />
          </div>
        </div>
      </div>

      {/* 组件详情 */}
      {selected && (
        <div className="bg-white rounded-lg p-6 shadow-lg mb-6">
          <div className="flex items-start justify-between mb-4">
            <div>
              <h4 className="text-xl font-bold text-gray-800">{selected.name}</h4>
              <p className="text-gray-600 mt-1">{selected.description}</p>
            </div>
            <button
              onClick={() => setSelectedComponent(null)}
              className="text-gray-400 hover:text-gray-600"
            >
              ✕
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* 技术栈 */}
            <div>
              <h5 className="font-semibold text-gray-700 mb-2">技术栈</h5>
              <div className="flex flex-wrap gap-2">
                {selected.technologies.map((tech, idx) => (
                  <span
                    key={idx}
                    className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800"
                  >
                    {tech}
                  </span>
                ))}
              </div>
            </div>

            {/* 性能指标 */}
            <div>
              <h5 className="font-semibold text-gray-700 mb-2">性能指标</h5>
              <div className="space-y-1 text-sm">
                {selected.metrics.latency && (
                  <p><span className="text-gray-600">延迟:</span> <span className="font-medium">{selected.metrics.latency}</span></p>
                )}
                {selected.metrics.availability && (
                  <p><span className="text-gray-600">可用性:</span> <span className="font-medium">{selected.metrics.availability}</span></p>
                )}
                {selected.metrics.throughput && (
                  <p><span className="text-gray-600">吞吐量:</span> <span className="font-medium">{selected.metrics.throughput}</span></p>
                )}
              </div>
            </div>
          </div>

          {/* 职责 */}
          <div className="mt-4">
            <h5 className="font-semibold text-gray-700 mb-2">核心职责</h5>
            <ul className="space-y-1">
              {selected.responsibilities.map((resp, idx) => (
                <li key={idx} className="text-sm text-gray-600 flex items-start">
                  <span className="mr-2">•</span>
                  <span>{resp}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}

      {/* 数据流 */}
      {showDataFlow && (
        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg p-6 shadow-lg">
          <h4 className="text-lg font-semibold text-gray-800 mb-4">典型请求数据流</h4>
          <div className="space-y-3">
            {dataFlowSteps.map((step, idx) => (
              <div key={idx} className="flex items-center space-x-4">
                <div className="flex-shrink-0 w-8 h-8 bg-blue-500 text-white rounded-full flex items-center justify-center font-semibold text-sm">
                  {idx + 1}
                </div>
                <div className="flex-1 bg-white rounded-lg p-3 shadow">
                  <div className="flex items-center justify-between">
                    <span className="font-medium text-gray-800">{step.from}</span>
                    <span className="text-sm text-gray-500">{step.label}</span>
                    <span className="font-medium text-gray-800">{step.to}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

interface ComponentBoxProps {
  component: ArchitectureComponent;
  isSelected: boolean;
  onClick: () => void;
  small?: boolean;
}

const ComponentBox: React.FC<ComponentBoxProps> = ({ component, isSelected, onClick, small }) => {
  const getGradient = (id: ComponentType): string => {
    const gradients: Record<ComponentType, string> = {
      gateway: 'from-blue-500 to-blue-700',
      langserve: 'from-purple-500 to-purple-700',
      router: 'from-green-500 to-green-700',
      rag: 'from-yellow-500 to-orange-600',
      agent: 'from-red-500 to-pink-600',
      llm: 'from-indigo-500 to-indigo-700',
      vectordb: 'from-cyan-500 to-cyan-700',
      cache: 'from-teal-500 to-teal-700',
      monitoring: 'from-pink-500 to-rose-700',
    };
    return gradients[id];
  };

  return (
    <button
      onClick={onClick}
      className={`${small ? 'p-3' : 'p-4'} rounded-lg font-semibold text-white transition-all shadow-lg hover:shadow-xl ${
        isSelected ? 'scale-110 ring-4 ring-yellow-400' : ''
      } bg-gradient-to-r ${getGradient(component.id)}`}
    >
      <div className={`${small ? 'text-sm' : 'text-base'}`}>{component.name}</div>
      {!small && component.metrics.latency && (
        <div className="text-xs mt-1 opacity-90">{component.metrics.latency}</div>
      )}
    </button>
  );
};

export default ProductionArchitectureDiagram;
