'use client';

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

type Layer = 'application' | 'chains' | 'components' | 'models';

interface ArchitectureNode {
  id: string;
  layer: Layer;
  title: string;
  description: string;
  examples: string[];
  // 优化后的专业色彩系统 (明亮模式柔和，暗黑模式发光)
  theme: {
    base: string; // Tailwind base color (e.g., 'purple')
    border: string; // Tailwind border class (e.g., 'border-purple-500')
    bg: string; // Tailwind background class (e.g., 'bg-purple-50/50')
    text: string; // Tailwind text class (e.g., 'text-purple-600')
    gradient: string; // Stronger gradient for tags (e.g., 'from-purple-400')
  };
}

const architectureNodes: ArchitectureNode[] = [
  {
    id: 'app',
    layer: 'application',
    title: 'Application 应用层',
    description: '面向终端用户的完整 AI 业务应用',
    examples: ['智能客服', 'RAG 知识库', '代码 Copilot', '多 Agent 协同'],
    theme: { base: 'purple', border: 'border-purple-500/50', bg: 'bg-purple-100/40 dark:bg-purple-950/20', text: 'text-purple-700 dark:text-purple-300', gradient: 'from-purple-500 to-pink-500' }
  },
  {
    id: 'chains',
    layer: 'chains',
    title: 'Chains & Graphs 链与图层',
    description: '编排复杂逻辑，将组件串联为工作流',
    examples: ['LCEL', 'LangGraph', 'RouterChain', 'SequentialChain'],
    theme: { base: 'blue', border: 'border-blue-500/50', bg: 'bg-blue-100/40 dark:bg-blue-950/20', text: 'text-blue-700 dark:text-blue-300', gradient: 'from-blue-500 to-cyan-500' }
  },
  {
    id: 'components',
    layer: 'components',
    title: 'Components 组件层',
    description: '构建 LLM 应用的核心标准化积木',
    examples: ['Prompts', 'OutputParsers', 'Retrievers', 'Memory', 'Tools'],
    theme: { base: 'emerald', border: 'border-emerald-500/50', bg: 'bg-emerald-100/40 dark:bg-emerald-950/20', text: 'text-emerald-700 dark:text-emerald-300', gradient: 'from-green-500 to-emerald-500' }
  },
  {
    id: 'models',
    layer: 'models',
    title: 'Models 模型层',
    description: '标准化的大语言模型与向量模型接口',
    examples: ['OpenAI', 'Anthropic', 'Google', 'Ollama', 'HuggingFace'],
    theme: { base: 'orange', border: 'border-orange-500/50', bg: 'bg-orange-100/40 dark:bg-orange-950/20', text: 'text-orange-700 dark:text-orange-300', gradient: 'from-orange-500 to-red-500' }
  }
];

export default function LangChainArchitectureFlow() {
  const [expandedLayer, setExpandedLayer] = useState<Layer | null>('chains');
  const [isPlaying, setIsPlaying] = useState(false);
  
  // 动画状态
  const [activeNodeIndex, setActiveNodeIndex] = useState<number | null>(null);
  const [activePipeIndex, setActivePipeIndex] = useState<number | null>(null);
  const [flowStatus, setFlowStatus] = useState<string>("系统就绪");
  const [flowDirection, setFlowDirection] = useState<'down' | 'up' | null>(null);

  // 异步流光动画引擎
  const startDataFlow = async () => {
    if (isPlaying) return;
    setIsPlaying(true);
    setExpandedLayer(null); // 折叠面板以突出流光

    const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));
    const speed = 700;

    // --- 下行：请求链路 ---
    setFlowDirection('down');
    setFlowStatus("↓ 接收用户请求...");
    setActiveNodeIndex(0);
    await delay(speed);

    for (let i = 0; i < architectureNodes.length - 1; i++) {
      setFlowStatus(`↓ 传递至 ${architectureNodes[i + 1].title.split(' ')[0]} 层...`);
      setActivePipeIndex(i);
      await delay(speed / 2);
      
      setActiveNodeIndex(i + 1);
      setActivePipeIndex(null);
      await delay(speed);
    }

    // --- 触底：API 调用 ---
    setFlowStatus("⚡ 模型推理中...");
    await delay(1000);

    // --- 上行：响应链路 ---
    setFlowDirection('up');
    for (let i = architectureNodes.length - 1; i > 0; i--) {
      setFlowStatus(`↑ 响应返回至 ${architectureNodes[i - 1].title.split(' ')[0]} 层...`);
      setActivePipeIndex(i - 1); // 管道复用
      await delay(speed / 2);

      setActiveNodeIndex(i - 1);
      setActivePipeIndex(null);
      await delay(speed);
    }

    setFlowStatus("✓ 处理完成，返回结果");
    await delay(1200);

    // 重置状态
    setIsPlaying(false);
    setActiveNodeIndex(null);
    setFlowDirection(null);
    setFlowStatus("系统就绪");
  };

  return (
    // 增加彩色渐变背景
    <div className="w-full max-w-5xl mx-auto p-6 sm:p-10 bg-gradient-to-br from-purple-50 via-sky-50 to-emerald-50 dark:from-slate-950 dark:via-indigo-950 dark:to-teal-950 rounded-3xl border border-slate-200 dark:border-slate-800 shadow-2xl transition-colors duration-300">
      
      {/* 彩色标题与控制台 */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-end mb-12 gap-6 pb-6 border-b border-slate-100 dark:border-slate-800/60">
        <div>
          <h3 className="text-3xl font-extrabold text-slate-900 dark:text-white mb-2 tracking-tight">
            LangChain 架构拓扑
          </h3>
          <p className="text-slate-500 dark:text-slate-400">
            面向大模型应用的垂直分层技术栈
          </p>
        </div>
        <button
          onClick={startDataFlow}
          disabled={isPlaying}
          className={`
            flex items-center gap-2 px-5 py-2.5 rounded-xl font-medium transition-all duration-300
            ${isPlaying 
              ? 'bg-slate-100 dark:bg-slate-800 text-slate-400' 
              : 'bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white shadow-lg hover:shadow-xl hover:-translate-y-0.5'
            }
          `}
        >
          {isPlaying ? (
            <svg className="w-4 h-4 animate-spin" viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <circle cx="12" cy="12" r="10" strokeWidth="3" className="opacity-25"/>
              <path d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" className="opacity-75" fill="currentColor"/>
            </svg>
          ) : (
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2.5">
              <path strokeLinecap="round" strokeLinejoin="round" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
            </svg>
          )}
          {isPlaying ? '数据流转中' : '▶ 模拟数据流'}
        </button>
      </div>

      {/* 彩色监控条 */}
      <div className={`
        mb-8 px-4 py-3 rounded-lg border font-mono text-sm flex items-center justify-between transition-all duration-500
        ${isPlaying 
          ? 'bg-blue-50/50 dark:bg-blue-950/10 border-blue-200 dark:border-blue-800 text-blue-700 dark:text-blue-300' 
          : 'bg-slate-50 dark:bg-black/20 border-slate-200 dark:border-slate-800 text-slate-500'
        }
      `}>
        <div className="flex items-center gap-3">
          <div className="relative flex h-2.5 w-2.5">
            {isPlaying && <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-75"></span>}
            <span className={`relative inline-flex rounded-full h-2.5 w-2.5 ${isPlaying ? 'bg-blue-500' : 'bg-slate-400'}`}></span>
          </div>
          <span>Status: {flowStatus}</span>
        </div>
        <span className="opacity-50 text-xs">API: v1</span>
      </div>

      {/* 玻璃态层级栈 */}
      <div className="relative">
        {architectureNodes.map((node, index) => {
          const isActiveNode = activeNodeIndex === index;
          const isExpanded = expandedLayer === node.layer;

          return (
            <div key={node.id} className="relative flex flex-col items-center">
              {/* 卡片主体：增加玻璃态 backdrop-blur */}
              <motion.div
                className={`
                  w-full relative overflow-hidden transition-all duration-500 cursor-pointer
                  rounded-2xl border bg-white/70 dark:bg-black/60 backdrop-blur-xl
                  ${isActiveNode 
                    ? `border-${node.theme.base}-400 dark:border-${node.theme.base}-500 shadow-xl shadow-${node.theme.base}-500/10 scale-[1.01] z-10` 
                    : 'border-slate-100 dark:border-slate-800/80 hover:border-slate-200 dark:hover:border-slate-700'
                  }
                `}
                onClick={() => !isPlaying && setExpandedLayer(isExpanded ? null : node.layer)}
              >
                {/* 内部多彩渐变：在激活时透出 */}
                <div className={`absolute inset-0 opacity-10 ${isActiveNode ? `bg-gradient-to-r ${node.theme.gradient}` : ''}`} />

                <div className="p-5 sm:p-6 flex items-start gap-4 sm:gap-6">
                  {/* 多彩数字图标 */}
                  <div className={`shrink-0 w-12 h-12 rounded-full bg-gradient-to-br ${node.theme.gradient} flex items-center justify-center text-white font-bold text-xl shadow-md`}>
                    {index + 1}
                  </div>
                  
                  <div className="flex-1">
                    <div className="flex justify-between items-center mb-1">
                      <h4 className={`text-xl font-extrabold transition-colors ${isActiveNode ? node.theme.text : 'text-slate-800 dark:text-slate-200'}`}>
                        {node.title}
                      </h4>
                      
                      {/* 激活指示/展开箭头 */}
                      <div className="ml-4 shrink-0 flex items-center justify-center w-8 h-8 rounded-lg bg-slate-50 dark:bg-slate-900 border border-slate-100 dark:border-slate-800">
                        {isActiveNode ? (
                          <motion.div animate={{ scale: [1, 1.2, 1] }} transition={{ repeat: Infinity, duration: 1.5 }} className={`w-3 h-3 rounded-full bg-${node.theme.base}-500`} />
                        ) : (
                          <svg className={`w-4 h-4 text-slate-400 transition-transform ${isExpanded ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                          </svg>
                        )}
                      </div>
                    </div>
                    <p className="text-slate-500 dark:text-slate-400 text-sm">
                      {node.description}
                    </p>

                    {/* 展开内容 */}
                    <AnimatePresence>
                      {isExpanded && !isPlaying && (
                        <motion.div
                          initial={{ height: 0, opacity: 0 }}
                          animate={{ height: 'auto', opacity: 1 }}
                          exit={{ height: 0, opacity: 0 }}
                          transition={{ duration: 0.3 }}
                          className="overflow-hidden"
                        >
                          <div className="pt-5 mt-5 border-t border-slate-100 dark:border-slate-800/60">
                            <div className="flex flex-wrap gap-2.5">
                              {node.examples.map((example, i) => (
                                <span key={i} className={`px-4 py-2 text-xs font-semibold rounded-lg bg-white dark:bg-slate-800 border ${node.theme.border} ${node.theme.text} shadow-sm dark:shadow-none`}>
                                  {example}
                                </span>
                              ))}
                            </div>
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>
                </div>
              </motion.div>

              {/* 层级之间的流光管道 */}
              {index < architectureNodes.length - 1 && (
                <div className="h-10 flex items-center justify-center relative">
                  <div className="w-0.5 h-full bg-slate-100 dark:bg-slate-800" />
                  
                  {/* 流光效果：增加光晕 */}
                  <AnimatePresence>
                    {activePipeIndex === index && (
                      <motion.div
                        initial={{ height: "0%", y: flowDirection === 'down' ? "-50%" : "50%" }}
                        animate={{ height: "100%", y: "0%" }}
                        exit={{ opacity: 0 }}
                        transition={{ duration: 0.3 }}
                        className={`absolute w-1 rounded-full bg-blue-500 shadow-[0_0_8px_rgba(59,130,246,0.6)]`}
                      />
                    )}
                  </AnimatePresence>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* 多彩图例与设计哲学 */}
      <div className="mt-12 p-6 bg-slate-50 dark:bg-black/20 rounded-2xl border border-slate-100 dark:border-slate-800">
        <h4 className="text-sm font-bold text-slate-800 dark:text-slate-200 uppercase tracking-wider mb-4">
          架构设计准则
        </h4>
        <div className="grid sm:grid-cols-2 gap-y-3 gap-x-6 text-sm">
          {[
            ['分层解耦', '每层职责清晰，支持无缝替换底层模型'],
            ['Runnable 协议', '所有组件实现统一接口，支持流式输出'],
            ['组合优先', '通过管道符号 (|) 组合而非继承实现功能'],
            ['全链路追踪', '基于 LangSmith 实现从应用端到模型的透明追踪']
          ].map(([title, desc], i) => (
            <div key={i} className="flex flex-col gap-1">
              <span className="font-semibold text-slate-700 dark:text-slate-300">{title}</span>
              <span className="text-slate-500 dark:text-slate-500">{desc}</span>
            </div>
          ))}
        </div>
      </div>

    </div>
  );
}