"use client";

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Server, Code, Globe, Zap, FileCode, PlayCircle } from 'lucide-react';

const LangServeArchitecture: React.FC = () => {
  const [activeLayer, setActiveLayer] = useState<'client' | 'langserve' | 'langchain' | null>(null);

  const layers = [
    {
      id: 'client' as const,
      name: '客户端层',
      color: 'blue',
      icon: Globe,
      description: '多种语言/平台调用',
      components: [
        { name: 'Python Client', detail: 'RemoteRunnable' },
        { name: 'JavaScript Client', detail: 'fetch / EventSource' },
        { name: 'curl', detail: 'HTTP 请求' },
        { name: 'Swagger UI', detail: '/docs 自动文档' },
      ],
    },
    {
      id: 'langserve' as const,
      name: 'LangServe 层',
      color: 'green',
      icon: Server,
      description: 'API 网关 + 路由',
      components: [
        { name: '/invoke', detail: '单次调用' },
        { name: '/batch', detail: '批量调用' },
        { name: '/stream', detail: '流式输出（SSE）' },
        { name: '/playground', detail: '在线测试 UI' },
      ],
    },
    {
      id: 'langchain' as const,
      name: 'LangChain 层',
      color: 'purple',
      icon: Code,
      description: '业务逻辑 + 链编排',
      components: [
        { name: 'Prompt Template', detail: '提示模板' },
        { name: 'LLM', detail: 'ChatOpenAI / Claude' },
        { name: 'Output Parser', detail: '结果解析' },
        { name: 'Chain / Graph', detail: 'LCEL / LangGraph' },
      ],
    },
  ];

  const getColor = (color: string, variant: 'border' | 'bg' | 'text' | 'hover') => {
    const colors: Record<string, Record<string, string>> = {
      blue: {
        border: 'border-blue-300',
        bg: 'bg-blue-50',
        text: 'text-blue-700',
        hover: 'hover:border-blue-500',
      },
      green: {
        border: 'border-green-300',
        bg: 'bg-green-50',
        text: 'text-green-700',
        hover: 'hover:border-green-500',
      },
      purple: {
        border: 'border-purple-300',
        bg: 'bg-purple-50',
        text: 'text-purple-700',
        hover: 'hover:border-purple-500',
      },
    };
    return colors[color]?.[variant] || '';
  };

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-blue-50 rounded-xl shadow-lg">
      <div className="mb-6">
        <h3 className="text-2xl font-bold text-gray-800 mb-2 flex items-center gap-2">
          <Server className="w-6 h-6 text-indigo-600" />
          LangServe 架构图
        </h3>
        <p className="text-gray-600">从客户端到 LangChain 的完整调用链路</p>
      </div>

      {/* 架构层级 */}
      <div className="space-y-6 mb-6">
        {layers.map((layer, idx) => {
          const Icon = layer.icon;
          const isActive = activeLayer === layer.id;
          
          return (
            <motion.div
              key={layer.id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: idx * 0.1 }}
              onMouseEnter={() => setActiveLayer(layer.id)}
              onMouseLeave={() => setActiveLayer(null)}
              className={`p-6 rounded-lg border-2 cursor-pointer transition-all ${
                getColor(layer.color, 'border')
              } ${
                getColor(layer.color, 'bg')
              } ${
                isActive ? 'scale-105 shadow-lg' : ''
              } ${getColor(layer.color, 'hover')}`}
            >
              <div className="flex items-start gap-4">
                <div className={`p-3 rounded-lg bg-white shadow ${getColor(layer.color, 'text')}`}>
                  <Icon className="w-6 h-6" />
                </div>
                <div className="flex-grow">
                  <h4 className="text-xl font-semibold text-gray-800 mb-1">{layer.name}</h4>
                  <p className="text-sm text-gray-600 mb-3">{layer.description}</p>
                  
                  <div className="grid grid-cols-2 gap-3">
                    {layer.components.map((comp) => (
                      <div
                        key={comp.name}
                        className="p-3 bg-white rounded border border-gray-200 hover:shadow transition-shadow"
                      >
                        <p className="text-sm font-semibold text-gray-800">{comp.name}</p>
                        <p className="text-xs text-gray-500">{comp.detail}</p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </motion.div>
          );
        })}
      </div>

      {/* 数据流动演示 */}
      <div className="p-6 bg-white rounded-lg shadow border border-gray-200">
        <h4 className="text-lg font-semibold text-gray-800 mb-4 flex items-center gap-2">
          <Zap className="w-5 h-5 text-yellow-500" />
          请求流程演示
        </h4>
        
        <div className="space-y-4">
          {/* 步骤 1 */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
            className="flex items-center gap-4"
          >
            <div className="flex-shrink-0 w-8 h-8 bg-blue-500 text-white rounded-full flex items-center justify-center font-bold">
              1
            </div>
            <div className="flex-grow p-3 bg-blue-50 rounded border border-blue-200">
              <p className="text-sm font-semibold text-gray-800">客户端发送请求</p>
              <code className="text-xs text-gray-600">POST /translate/invoke {{&quot;input&quot;: {{&quot;text&quot;: &quot;Hello&quot;}}}}</code>
            </div>
          </motion.div>

          {/* 箭头 */}
          <div className="flex justify-center">
            <div className="w-0.5 h-8 bg-gray-300" />
          </div>

          {/* 步骤 2 */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.4 }}
            className="flex items-center gap-4"
          >
            <div className="flex-shrink-0 w-8 h-8 bg-green-500 text-white rounded-full flex items-center justify-center font-bold">
              2
            </div>
            <div className="flex-grow p-3 bg-green-50 rounded border border-green-200">
              <p className="text-sm font-semibold text-gray-800">LangServe 路由解析</p>
              <code className="text-xs text-gray-600">add_routes(app, chain, path="/translate")</code>
            </div>
          </motion.div>

          {/* 箭头 */}
          <div className="flex justify-center">
            <div className="w-0.5 h-8 bg-gray-300" />
          </div>

          {/* 步骤 3 */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.6 }}
            className="flex items-center gap-4"
          >
            <div className="flex-shrink-0 w-8 h-8 bg-purple-500 text-white rounded-full flex items-center justify-center font-bold">
              3
            </div>
            <div className="flex-grow p-3 bg-purple-50 rounded border border-purple-200">
              <p className="text-sm font-semibold text-gray-800">LangChain 执行链</p>
              <code className="text-xs text-gray-600">chain.invoke(input) → Prompt | LLM | Parser</code>
            </div>
          </motion.div>

          {/* 箭头 */}
          <div className="flex justify-center">
            <div className="w-0.5 h-8 bg-gray-300" />
          </div>

          {/* 步骤 4 */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.8 }}
            className="flex items-center gap-4"
          >
            <div className="flex-shrink-0 w-8 h-8 bg-blue-500 text-white rounded-full flex items-center justify-center font-bold">
              4
            </div>
            <div className="flex-grow p-3 bg-blue-50 rounded border border-blue-200">
              <p className="text-sm font-semibold text-gray-800">返回结果给客户端</p>
              <code className="text-xs text-gray-600">{{&quot;output&quot;: {{&quot;content&quot;: &quot;Bonjour&quot;}}}}</code>
            </div>
          </motion.div>
        </div>
      </div>

      {/* 优势说明 */}
      <div className="mt-6 p-4 bg-white rounded-lg border border-indigo-200">
        <h4 className="font-semibold text-gray-800 mb-2 flex items-center gap-2">
          <PlayCircle className="w-5 h-5 text-indigo-600" />
          LangServe 核心优势
        </h4>
        <ul className="text-sm text-gray-600 space-y-1">
          <li>• <strong>一行部署</strong>：add_routes() 即可将链变成 API</li>
          <li>• <strong>自动文档</strong>：访问 /docs 查看完整 OpenAPI 规范</li>
          <li>• <strong>多端点支持</strong>：invoke、batch、stream、playground 开箱即用</li>
          <li>• <strong>跨语言调用</strong>：Python、JavaScript、curl 等任意语言</li>
          <li>• <strong>流式响应</strong>：SSE（Server-Sent Events）实时输出</li>
          <li>• <strong>批处理优化</strong>：自动聚合请求提升吞吐量</li>
        </ul>
      </div>
    </div>
  );
};

export default LangServeArchitecture;
