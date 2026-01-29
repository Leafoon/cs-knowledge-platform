"use client";

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

type Layer = 'application' | 'chains' | 'components' | 'models';

interface ArchitectureNode {
  id: string;
  layer: Layer;
  title: string;
  description: string;
  examples: string[];
  color: string;
}

const architectureNodes: ArchitectureNode[] = [
  {
    id: 'app',
    layer: 'application',
    title: '应用层 (Application)',
    description: '基于 LangChain 构建的完整应用',
    examples: ['聊天机器人', 'RAG 系统', '代码助手', 'Agent 工作流'],
    color: 'from-purple-500 to-pink-500'
  },
  {
    id: 'chains',
    layer: 'chains',
    title: '链层 (Chains & Graphs)',
    description: '组合多个组件完成复杂任务',
    examples: ['LCEL 链', 'LangGraph', 'RouterChain', 'SequentialChain'],
    color: 'from-blue-500 to-cyan-500'
  },
  {
    id: 'components',
    layer: 'components',
    title: '组件层 (Components)',
    description: 'LangChain 核心可复用组件',
    examples: ['Prompts', 'OutputParsers', 'Retrievers', 'Memory', 'Tools'],
    color: 'from-green-500 to-emerald-500'
  },
  {
    id: 'models',
    layer: 'models',
    title: '模型层 (Models)',
    description: '与 LLM 提供商的集成接口',
    examples: ['OpenAI', 'Anthropic', 'Google', 'Ollama', 'HuggingFace'],
    color: 'from-orange-500 to-red-500'
  }
];

const dataFlowSteps = [
  { from: 'application', to: 'chains', label: '用户请求' },
  { from: 'chains', to: 'components', label: '分解任务' },
  { from: 'components', to: 'models', label: 'API 调用' },
  { from: 'models', to: 'components', label: 'LLM 响应' },
  { from: 'components', to: 'chains', label: '解析结果' },
  { from: 'chains', to: 'application', label: '返回用户' }
];

export default function LangChainArchitectureFlow() {
  const [selectedLayer, setSelectedLayer] = useState<Layer | null>(null);
  const [showDataFlow, setShowDataFlow] = useState(false);
  const [currentFlowStep, setCurrentFlowStep] = useState(0);

  const startDataFlow = () => {
    setShowDataFlow(true);
    setCurrentFlowStep(0);
    const interval = setInterval(() => {
      setCurrentFlowStep(prev => {
        if (prev >= dataFlowSteps.length - 1) {
          clearInterval(interval);
          setTimeout(() => setShowDataFlow(false), 1000);
          return prev;
        }
        return prev + 1;
      });
    }, 800);
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-8 bg-gradient-to-br from-slate-50 to-blue-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl border-2 border-slate-200 dark:border-slate-700">
      <div className="text-center mb-8">
        <h3 className="text-3xl font-bold text-slate-800 dark:text-white mb-3">
          LangChain 架构分层
        </h3>
        <p className="text-slate-600 dark:text-slate-300 text-lg mb-4">
          从底层模型到上层应用的完整技术栈
        </p>
        <button
          onClick={startDataFlow}
          disabled={showDataFlow}
          className="px-6 py-3 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-xl font-semibold shadow-lg hover:shadow-xl transform hover:scale-105 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {showDataFlow ? '数据流动中...' : '▶ 演示数据流'}
        </button>
      </div>

      <div className="space-y-6 relative">
        {/* Architecture Layers */}
        {architectureNodes.map((node, index) => (
          <motion.div
            key={node.id}
            initial={{ opacity: 0, x: -50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.1 }}
            className="relative"
          >
            <motion.div
              onClick={() => setSelectedLayer(selectedLayer === node.layer ? null : node.layer)}
              className={`
                cursor-pointer p-6 rounded-2xl border-2 transition-all
                ${selectedLayer === node.layer 
                  ? 'border-blue-500 shadow-2xl scale-[1.02]' 
                  : 'border-slate-300 dark:border-slate-600 hover:border-blue-400 hover:shadow-lg'
                }
                bg-white dark:bg-slate-800
              `}
              whileHover={{ scale: 1.01 }}
              whileTap={{ scale: 0.99 }}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <div className={`w-16 h-16 rounded-xl bg-gradient-to-br ${node.color} flex items-center justify-center text-white font-bold text-2xl shadow-lg`}>
                    {index + 1}
                  </div>
                  <div>
                    <h4 className="text-xl font-bold text-slate-800 dark:text-white mb-1">
                      {node.title}
                    </h4>
                    <p className="text-slate-600 dark:text-slate-300">
                      {node.description}
                    </p>
                  </div>
                </div>
                <svg 
                  className={`w-6 h-6 text-slate-400 transition-transform ${selectedLayer === node.layer ? 'rotate-180' : ''}`}
                  fill="none" 
                  stroke="currentColor" 
                  viewBox="0 0 24 24"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </div>

              <AnimatePresence>
                {selectedLayer === node.layer && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.3 }}
                    className="mt-4 pt-4 border-t border-slate-200 dark:border-slate-700 overflow-hidden"
                  >
                    <div className="flex flex-wrap gap-2">
                      {node.examples.map((example, i) => (
                        <span 
                          key={i}
                          className={`px-4 py-2 rounded-lg bg-gradient-to-r ${node.color} text-white text-sm font-semibold shadow-md`}
                        >
                          {example}
                        </span>
                      ))}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>

            {/* Data Flow Animation */}
            {index < architectureNodes.length - 1 && showDataFlow && (
              <AnimatePresence>
                {(currentFlowStep === index * 2 || currentFlowStep === index * 2 + 1) && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="absolute left-1/2 transform -translate-x-1/2 z-10"
                    style={{ 
                      top: currentFlowStep % 2 === 0 ? '100%' : 'calc(100% + 20px)',
                      marginTop: currentFlowStep % 2 === 0 ? '8px' : '0'
                    }}
                  >
                    <motion.div
                      animate={{ y: [0, 20, 0] }}
                      transition={{ repeat: Infinity, duration: 0.6 }}
                      className="flex flex-col items-center"
                    >
                      <div className="px-4 py-2 bg-blue-500 text-white rounded-lg font-semibold text-sm shadow-lg mb-2">
                        {dataFlowSteps[currentFlowStep]?.label}
                      </div>
                      <svg className="w-6 h-6 text-blue-500" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M10 3a1 1 0 011 1v10.586l2.293-2.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 111.414-1.414L9 14.586V4a1 1 0 011-1z" clipRule="evenodd" />
                      </svg>
                    </motion.div>
                  </motion.div>
                )}
              </AnimatePresence>
            )}

            {/* Connection Line */}
            {index < architectureNodes.length - 1 && !showDataFlow && (
              <div className="flex justify-center py-2">
                <div className="w-0.5 h-8 bg-gradient-to-b from-slate-300 to-slate-400 dark:from-slate-600 dark:to-slate-500" />
              </div>
            )}
          </motion.div>
        ))}
      </div>

      {/* Legend */}
      <div className="mt-8 p-6 bg-blue-50 dark:bg-slate-800 rounded-xl border border-blue-200 dark:border-slate-600">
        <h4 className="text-lg font-bold text-slate-800 dark:text-white mb-3 flex items-center gap-2">
          <svg className="w-5 h-5 text-blue-500" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
          </svg>
          架构设计理念
        </h4>
        <div className="grid md:grid-cols-2 gap-4 text-sm text-slate-700 dark:text-slate-300">
          <div className="flex items-start gap-2">
            <span className="text-green-500 font-bold">✓</span>
            <span><strong>分层解耦</strong>：每层职责清晰，便于替换与扩展</span>
          </div>
          <div className="flex items-start gap-2">
            <span className="text-green-500 font-bold">✓</span>
            <span><strong>组合优先</strong>：通过组合而非继承实现功能</span>
          </div>
          <div className="flex items-start gap-2">
            <span className="text-green-500 font-bold">✓</span>
            <span><strong>统一接口</strong>：Runnable 协议贯穿所有层级</span>
          </div>
          <div className="flex items-start gap-2">
            <span className="text-green-500 font-bold">✓</span>
            <span><strong>可观测性</strong>：LangSmith 追踪每层数据流动</span>
          </div>
        </div>
      </div>
    </div>
  );
}
