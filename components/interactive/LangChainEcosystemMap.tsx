'use client';

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface Component {
  id: string;
  name: string;
  description: string;
  category: 'core' | 'graph' | 'observability' | 'deployment' | 'community';
  connections: string[];
  features: string[];
  installCommand: string;
}

const components: Component[] = [
  {
    id: 'langchain-core',
    name: 'LangChain Core',
    description: '基础抽象与 LCEL，所有组件的基石',
    category: 'core',
    connections: ['langchain-community', 'langgraph', 'langserve'],
    features: ['Runnable 协议', 'LCEL 管道', '消息格式', '输出解析'],
    installCommand: 'pip install langchain-core'
  },
  {
    id: 'langchain-community',
    name: 'LangChain Community',
    description: '第三方集成（模型、向量库、工具）',
    category: 'community',
    connections: ['langchain-core'],
    features: ['100+ 集成', '向量数据库', '文档加载器', '外部工具'],
    installCommand: 'pip install langchain-community'
  },
  {
    id: 'langgraph',
    name: 'LangGraph',
    description: '状态图与复杂控制流',
    category: 'graph',
    connections: ['langchain-core'],
    features: ['StateGraph', '条件边', 'Checkpointing', 'Human-in-the-loop'],
    installCommand: 'pip install langgraph'
  },
  {
    id: 'langsmith',
    name: 'LangSmith',
    description: '追踪、评估、监控平台',
    category: 'observability',
    connections: ['langchain-core', 'langgraph'],
    features: ['Tracing', '数据集管理', '自动评估', '性能监控'],
    installCommand: 'export LANGCHAIN_TRACING_V2=true'
  },
  {
    id: 'langserve',
    name: 'LangServe',
    description: '一键部署 REST API',
    category: 'deployment',
    connections: ['langchain-core', 'langgraph'],
    features: ['FastAPI 集成', '流式支持', 'Playground', 'OpenAPI 文档'],
    installCommand: 'pip install langserve[all]'
  },
  {
    id: 'hub',
    name: 'LangChain Hub',
    description: '提示模板仓库',
    category: 'community',
    connections: ['langchain-core'],
    features: ['模板共享', '版本管理', 'hub.pull()', 'hub.push()'],
    installCommand: 'langchainhub package included'
  }
];

const categoryColors = {
  core: 'from-blue-500 to-cyan-500',
  graph: 'from-purple-500 to-pink-500',
  observability: 'from-green-500 to-emerald-500',
  deployment: 'from-orange-500 to-red-500',
  community: 'from-yellow-500 to-amber-500'
};

const categoryLabels = {
  core: '核心抽象',
  graph: '状态图',
  observability: '可观测性',
  deployment: '部署',
  community: '生态集成'
};

export default function LangChainEcosystemMap() {
  const [selectedComponent, setSelectedComponent] = useState<Component | null>(null);
  const [hoveredComponent, setHoveredComponent] = useState<string | null>(null);

  const getPosition = (index: number, total: number) => {
    const angle = (index * 2 * Math.PI) / total - Math.PI / 2;
    const radius = 180;
    return {
      x: Math.cos(angle) * radius,
      y: Math.sin(angle) * radius
    };
  };

  const isConnected = (comp1: string, comp2: string) => {
    const component = components.find(c => c.id === comp1);
    return component?.connections.includes(comp2) || false;
  };

  return (
    <div className="w-full bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 rounded-2xl p-8 shadow-2xl">
      <div className="text-center mb-8">
        <h3 className="text-2xl font-bold text-white mb-2">
          LangChain 生态系统架构图
        </h3>
        <p className="text-slate-400">
          点击组件查看详细信息，悬停查看连接关系
        </p>
      </div>

      {/* 主可视化区域 */}
      <div className="relative h-[500px] flex items-center justify-center">
        {/* 中心核心 */}
        <div className="absolute flex items-center justify-center">
          <div className="w-32 h-32 rounded-full bg-gradient-to-br from-blue-600 to-cyan-600 flex items-center justify-center shadow-2xl">
            <div className="text-center">
              <div className="text-white font-bold text-lg">LangChain</div>
              <div className="text-blue-200 text-xs">Core</div>
            </div>
          </div>
        </div>

        {/* 连接线 */}
        <svg className="absolute inset-0 w-full h-full pointer-events-none">
          {components.map((comp1, i) => {
            const pos1 = getPosition(i, components.length);
            return comp1.connections.map(connId => {
              const comp2Index = components.findIndex(c => c.id === connId);
              if (comp2Index === -1) return null;
              const pos2 = getPosition(comp2Index, components.length);
              
              const isHighlighted = 
                hoveredComponent === comp1.id || 
                hoveredComponent === connId ||
                selectedComponent?.id === comp1.id ||
                selectedComponent?.id === connId;

              return (
                <motion.line
                  key={`${comp1.id}-${connId}`}
                  x1={250 + pos1.x}
                  y1={250 + pos1.y}
                  x2={250 + pos2.x}
                  y2={250 + pos2.y}
                  stroke={isHighlighted ? '#60a5fa' : '#334155'}
                  strokeWidth={isHighlighted ? 3 : 1.5}
                  strokeDasharray={isHighlighted ? '0' : '5,5'}
                  initial={{ pathLength: 0 }}
                  animate={{ pathLength: 1 }}
                  transition={{ duration: 1.5, delay: i * 0.1 }}
                />
              );
            });
          })}
        </svg>

        {/* 组件节点 */}
        {components.map((component, index) => {
          const position = getPosition(index, components.length);
          const isSelected = selectedComponent?.id === component.id;
          const isHovered = hoveredComponent === component.id;

          return (
            <motion.div
              key={component.id}
              className="absolute"
              style={{
                left: '50%',
                top: '50%',
                transform: `translate(calc(-50% + ${position.x}px), calc(-50% + ${position.y}px))`
              }}
              initial={{ scale: 0, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ delay: index * 0.15, type: 'spring' }}
            >
              <motion.button
                className={`
                  relative w-28 h-28 rounded-xl cursor-pointer
                  bg-gradient-to-br ${categoryColors[component.category]}
                  shadow-lg hover:shadow-2xl transition-all
                  ${isSelected ? 'ring-4 ring-white' : ''}
                `}
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => setSelectedComponent(
                  isSelected ? null : component
                )}
                onMouseEnter={() => setHoveredComponent(component.id)}
                onMouseLeave={() => setHoveredComponent(null)}
              >
                <div className="p-3 text-center h-full flex flex-col justify-center">
                  <div className="text-white font-bold text-sm mb-1">
                    {component.name.split(' ').map((word, i) => (
                      <div key={i}>{word}</div>
                    ))}
                  </div>
                  <div className="text-xs text-white/80">
                    {categoryLabels[component.category]}
                  </div>
                </div>

                {/* 脉冲效果 */}
                {isHovered && (
                  <motion.div
                    className="absolute inset-0 rounded-xl bg-white"
                    initial={{ opacity: 0.3, scale: 1 }}
                    animate={{ opacity: 0, scale: 1.3 }}
                    transition={{ repeat: Infinity, duration: 1.5 }}
                  />
                )}
              </motion.button>
            </motion.div>
          );
        })}
      </div>

      {/* 详细信息面板 */}
      <AnimatePresence>
        {selectedComponent && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            className="mt-8 bg-slate-800 rounded-xl p-6 border border-slate-700"
          >
            <div className="flex justify-between items-start mb-4">
              <div>
                <h4 className={`
                  text-2xl font-bold bg-gradient-to-r ${categoryColors[selectedComponent.category]}
                  bg-clip-text text-transparent
                `}>
                  {selectedComponent.name}
                </h4>
                <p className="text-slate-400 mt-1">
                  {selectedComponent.description}
                </p>
              </div>
              <button
                onClick={() => setSelectedComponent(null)}
                className="text-slate-400 hover:text-white"
              >
                ✕
              </button>
            </div>

            <div className="grid md:grid-cols-2 gap-6">
              {/* 核心功能 */}
              <div>
                <h5 className="text-sm font-semibold text-slate-300 mb-3">
                  核心功能
                </h5>
                <div className="space-y-2">
                  {selectedComponent.features.map((feature, i) => (
                    <motion.div
                      key={i}
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: i * 0.1 }}
                      className="flex items-center gap-2"
                    >
                      <div className={`
                        w-2 h-2 rounded-full
                        bg-gradient-to-r ${categoryColors[selectedComponent.category]}
                      `} />
                      <span className="text-slate-300 text-sm">{feature}</span>
                    </motion.div>
                  ))}
                </div>
              </div>

              {/* 安装命令 */}
              <div>
                <h5 className="text-sm font-semibold text-slate-300 mb-3">
                  安装/配置
                </h5>
                <div className="bg-slate-900 rounded-lg p-4 font-mono text-sm">
                  <code className="text-green-400">
                    {selectedComponent.installCommand}
                  </code>
                </div>

                {selectedComponent.connections.length > 0 && (
                  <div className="mt-4">
                    <h5 className="text-sm font-semibold text-slate-300 mb-2">
                      依赖关系
                    </h5>
                    <div className="flex flex-wrap gap-2">
                      {selectedComponent.connections.map(connId => {
                        const conn = components.find(c => c.id === connId);
                        return conn ? (
                          <span
                            key={connId}
                            className={`
                              px-3 py-1 rounded-full text-xs font-medium
                              bg-gradient-to-r ${categoryColors[conn.category]}
                              text-white
                            `}
                          >
                            {conn.name}
                          </span>
                        ) : null;
                      })}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* 图例 */}
      <div className="mt-6 flex flex-wrap justify-center gap-4">
        {Object.entries(categoryLabels).map(([key, label]) => (
          <div key={key} className="flex items-center gap-2">
            <div className={`
              w-4 h-4 rounded
              bg-gradient-to-r ${categoryColors[key as keyof typeof categoryColors]}
            `} />
            <span className="text-slate-400 text-sm">{label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
