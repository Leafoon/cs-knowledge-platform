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
    installCommand: 'pip install langchainhub'
  }
];

// 优化后的专业色彩系统 (更柔和、更具科技感)
const themeColors = {
  core: { text: 'text-blue-600 dark:text-blue-400', border: 'border-blue-500/30', bg: 'bg-blue-50/50 dark:bg-blue-500/10', glow: 'shadow-blue-500/20' },
  graph: { text: 'text-purple-600 dark:text-purple-400', border: 'border-purple-500/30', bg: 'bg-purple-50/50 dark:bg-purple-500/10', glow: 'shadow-purple-500/20' },
  observability: { text: 'text-emerald-600 dark:text-emerald-400', border: 'border-emerald-500/30', bg: 'bg-emerald-50/50 dark:bg-emerald-500/10', glow: 'shadow-emerald-500/20' },
  deployment: { text: 'text-orange-600 dark:text-orange-400', border: 'border-orange-500/30', bg: 'bg-orange-50/50 dark:bg-orange-500/10', glow: 'shadow-orange-500/20' },
  community: { text: 'text-amber-600 dark:text-amber-400', border: 'border-amber-500/30', bg: 'bg-amber-50/50 dark:bg-amber-500/10', glow: 'shadow-amber-500/20' }
};

const categoryLabels = {
  core: '核心抽象',
  graph: '状态图',
  observability: '可观测性',
  deployment: '部署',
  community: '生态集成'
};

// 绝对画布配置 (彻底解决相对定位坍缩问题)
const CANVAS_SIZE = 640;
const CENTER = CANVAS_SIZE / 2;
const RADIUS = 220;

export default function LangChainEcosystemMap() {
  const [selectedComponent, setSelectedComponent] = useState<Component | null>(null);
  const [hoveredComponent, setHoveredComponent] = useState<string | null>(null);

  // 精准计算绝对像素坐标 (中心点坐标)
  const nodePositions: Record<string, { x: number, y: number }> = {};
  const orbitNodes = components.filter(c => c.id !== 'langchain-core');
  
  components.forEach(comp => {
    if (comp.id === 'langchain-core') {
      nodePositions[comp.id] = { x: CENTER, y: CENTER };
    } else {
      const index = orbitNodes.findIndex(c => c.id === comp.id);
      const angle = (index * 2 * Math.PI) / orbitNodes.length - Math.PI / 2;
      nodePositions[comp.id] = {
        x: CENTER + Math.cos(angle) * RADIUS,
        y: CENTER + Math.sin(angle) * RADIUS
      };
    }
  });

  const isLineHighlighted = (comp1: string, comp2: string) => {
    if (!hoveredComponent && !selectedComponent) return false;
    const activeId = hoveredComponent || selectedComponent?.id;
    return activeId === comp1 || activeId === comp2;
  };

  return (
    <div className="w-full bg-white dark:bg-[#0A0A0A] rounded-3xl p-6 sm:p-10 shadow-2xl border border-slate-200/60 dark:border-slate-800/60 transition-colors duration-300">
      
      {/* 头部区域 */}
      <div className="text-center mb-8">
        <h3 className="text-3xl font-extrabold text-slate-900 dark:text-white mb-3 tracking-tight">
          LangChain 生态架构
        </h3>
        <p className="text-sm sm:text-base text-slate-500 dark:text-slate-400 max-w-2xl mx-auto">
          点击节点探索组件能力，悬停查看依赖链路。全组件协同构建现代化 AI Agent 应用。
        </p>
      </div>

      {/* 画布区域：限制最小宽度并允许横向滚动，确保坐标绝对正确 */}
      <div className="w-full overflow-x-auto overflow-y-hidden no-scrollbar py-10">
        <div 
          className="relative mx-auto" 
          style={{ width: CANVAS_SIZE, height: CANVAS_SIZE }}
        >
          {/* 背景雷达圈 (增强科技感) */}
          <div className="absolute inset-0 pointer-events-none flex items-center justify-center opacity-30 dark:opacity-20">
            <div className="w-[440px] h-[440px] rounded-full border border-dashed border-slate-400 dark:border-slate-500 animate-[spin_60s_linear_infinite]" />
          </div>

          {/* SVG 连线层 */}
          <svg 
            width={CANVAS_SIZE} 
            height={CANVAS_SIZE} 
            className="absolute inset-0 pointer-events-none z-0"
          >
            {components.map((comp) => (
              comp.connections.map(targetId => {
                const pos1 = nodePositions[comp.id];
                const pos2 = nodePositions[targetId];
                if (!pos1 || !pos2) return null;
                
                const highlighted = isLineHighlighted(comp.id, targetId);

                return (
                  <motion.line
                    key={`${comp.id}-${targetId}`}
                    x1={pos1.x}
                    y1={pos1.y}
                    x2={pos2.x}
                    y2={pos2.y}
                    // 默认状态使用极细的半透明线，高亮时加粗变色
                    stroke={highlighted ? (document.documentElement.classList.contains('dark') ? '#94a3b8' : '#64748b') : (document.documentElement.classList.contains('dark') ? '#33415580' : '#cbd5e180')}
                    strokeWidth={highlighted ? 2 : 1.5}
                    strokeDasharray={highlighted ? '0' : '4,6'}
                    initial={{ pathLength: 0, opacity: 0 }}
                    animate={{ pathLength: 1, opacity: 1 }}
                    transition={{ duration: 1, ease: "easeOut" }}
                  />
                );
              })
            ))}
          </svg>

          {/* DOM 节点层 */}
          {components.map((component, index) => {
            const pos = nodePositions[component.id];
            const isSelected = selectedComponent?.id === component.id;
            const isCore = component.id === 'langchain-core';
            const size = isCore ? 140 : 120; // 定义节点的宽/高绝对像素
            const colors = themeColors[component.category];

            return (
              <motion.div
                key={component.id}
                className="absolute z-10"
                style={{
                  // 通过中心点坐标减去自身一半尺寸，实现绝对居中定位
                  left: pos.x - size / 2,
                  top: pos.y - size / 2,
                  width: size,
                  height: size
                }}
                initial={{ scale: 0, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ delay: index * 0.1, type: 'spring', stiffness: 200, damping: 20 }}
              >
                <button
                  className={`
                    w-full h-full flex flex-col items-center justify-center cursor-pointer
                    transition-all duration-300 backdrop-blur-xl
                    ${isCore ? 'rounded-full' : 'rounded-2xl'}
                    bg-white/80 dark:bg-[#111111]/90 
                    border ${colors.border}
                    hover:-translate-y-1 hover:shadow-xl hover:${colors.glow}
                    ${isSelected ? `ring-2 ring-offset-4 dark:ring-offset-[#0A0A0A] ring-${colors.border.split('-')[1]}-400` : 'shadow-sm dark:shadow-none'}
                  `}
                  onClick={() => setSelectedComponent(isSelected ? null : component)}
                  onMouseEnter={() => setHoveredComponent(component.id)}
                  onMouseLeave={() => setHoveredComponent(null)}
                >
                  <div className="p-2 text-center flex flex-col items-center">
                    {/* 小圆点装饰 */}
                    <div className={`w-2 h-2 rounded-full mb-2 ${colors.bg.split(' ')[0]} border ${colors.border}`} />
                    
                    <div className={`font-bold text-[14px] leading-tight mb-2 text-slate-800 dark:text-slate-100`}>
                      {component.name.split(' ').map((word, i) => (
                        <div key={i}>{word}</div>
                      ))}
                    </div>
                    
                    <div className={`text-[10px] px-2 py-0.5 rounded-full border ${colors.border} ${colors.text} bg-transparent`}>
                      {categoryLabels[component.category]}
                    </div>
                  </div>
                </button>
              </motion.div>
            );
          })}
        </div>
      </div>

      {/* 图例区域 */}
      <div className="flex flex-wrap justify-center gap-6 mt-4 mb-8">
        {Object.entries(categoryLabels).map(([key, label]) => {
          const catColor = themeColors[key as keyof typeof themeColors];
          return (
            <div key={key} className="flex items-center gap-2">
              <div className={`w-3 h-3 rounded-md ${catColor.bg.split(' ')[0]} dark:${catColor.bg.split(' ')[1]} border ${catColor.border}`} />
              <span className="text-slate-600 dark:text-slate-400 text-sm font-medium">{label}</span>
            </div>
          );
        })}
      </div>

      {/* 底部详细信息面板 */}
      <AnimatePresence mode="wait">
        {selectedComponent && (
          <motion.div
            initial={{ opacity: 0, height: 0, y: 20 }}
            animate={{ opacity: 1, height: 'auto', y: 0 }}
            exit={{ opacity: 0, height: 0, y: 10 }}
            transition={{ duration: 0.3 }}
            className="overflow-hidden"
          >
            <div className="bg-slate-50 dark:bg-[#111111] rounded-2xl p-6 lg:p-8 border border-slate-200 dark:border-slate-800 shadow-inner">
              <div className="flex justify-between items-start mb-6">
                <div>
                  <h4 className={`text-2xl font-bold mb-2 flex items-center gap-3 ${themeColors[selectedComponent.category].text}`}>
                    {selectedComponent.name}
                    <span className="text-xs font-normal px-2 py-1 rounded bg-slate-200/50 dark:bg-slate-800 text-slate-600 dark:text-slate-400">
                      {selectedComponent.id}
                    </span>
                  </h4>
                  <p className="text-slate-600 dark:text-slate-400">
                    {selectedComponent.description}
                  </p>
                </div>
                <button
                  onClick={() => setSelectedComponent(null)}
                  className="p-2 text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 transition-colors bg-white dark:bg-slate-800 rounded-full border border-slate-200 dark:border-slate-700"
                >
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <line x1="18" y1="6" x2="6" y2="18"></line>
                    <line x1="6" y1="6" x2="18" y2="18"></line>
                  </svg>
                </button>
              </div>

              <div className="grid md:grid-cols-2 gap-8">
                {/* 核心功能 */}
                <div className="space-y-4">
                  <h5 className="text-sm font-bold text-slate-800 dark:text-slate-200 uppercase tracking-wider flex items-center gap-2">
                    <div className={`w-1.5 h-4 rounded-full ${themeColors[selectedComponent.category].bg.split(' ')[0]}`} />
                    核心功能特性
                  </h5>
                  <div className="grid grid-cols-2 gap-3">
                    {selectedComponent.features.map((feature, i) => (
                      <div key={i} className="flex items-start gap-2 text-slate-600 dark:text-slate-400 text-sm bg-white dark:bg-[#1A1A1A] p-3 rounded-xl border border-slate-100 dark:border-slate-800">
                        <svg className={`w-4 h-4 mt-0.5 shrink-0 ${themeColors[selectedComponent.category].text}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                        </svg>
                        <span>{feature}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* 安装与依赖 */}
                <div className="space-y-4">
                  <h5 className="text-sm font-bold text-slate-800 dark:text-slate-200 uppercase tracking-wider flex items-center gap-2">
                    <div className={`w-1.5 h-4 rounded-full ${themeColors[selectedComponent.category].bg.split(' ')[0]}`} />
                    快速起步
                  </h5>
                  
                  {/* Terminal */}
                  <div className="bg-[#0A0A0A] dark:bg-black rounded-xl p-4 font-mono text-sm flex items-center gap-3 border border-slate-800">
                    <span className="text-slate-600 select-none">❯</span>
                    <code className="text-emerald-400 flex-1">
                      {selectedComponent.installCommand}
                    </code>
                    <button 
                      className="text-slate-500 hover:text-white transition-colors p-1.5 bg-white/5 rounded-md"
                      onClick={() => navigator.clipboard.writeText(selectedComponent.installCommand)}
                      title="复制"
                    >
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                      </svg>
                    </button>
                  </div>

                  {selectedComponent.connections.length > 0 && (
                    <div className="pt-2">
                      <div className="text-xs text-slate-500 mb-3 font-medium">相关组件关联</div>
                      <div className="flex flex-wrap gap-2">
                        {selectedComponent.connections.map(connId => {
                          const conn = components.find(c => c.id === connId);
                          return conn ? (
                            <span
                              key={connId}
                              className={`px-3 py-1.5 rounded-lg text-xs font-semibold bg-white dark:bg-[#1A1A1A] border shadow-sm ${themeColors[conn.category].border} ${themeColors[conn.category].text}`}
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
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}