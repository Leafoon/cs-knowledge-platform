'use client';

import React, { useState, useMemo } from 'react';

type ArchType = 'lcel' | 'langgraph' | 'comparison';

type Edge = {
  from: string;
  to: string;
  label?: string;
  type?: 'default' | 'loop' | 'condition';
};

type Node = {
  id: string;
  label: string;
  x: number;
  y: number;
  color: string;
  type?: 'start' | 'process' | 'decision' | 'end';
};

export default function LangGraphArchitectureDiagram() {
  const [selectedArch, setSelectedArch] = useState<ArchType>('langgraph');

  const architectures = useMemo(() => ({
    lcel: {
      title: 'LCEL（线性/并行流程）',
      color: '#3b82f6',
      gradient: 'from-blue-500 to-blue-600',
      nodes: [
        { id: 'input', label: '输入', x: 80, y: 80, color: '#60a5fa', type: 'process' },
        { id: 'prompt', label: 'Prompt', x: 230, y: 80, color: '#3b82f6', type: 'process' },
        { id: 'llm', label: 'LLM', x: 380, y: 80, color: '#2563eb', type: 'process' },
        { id: 'parser', label: 'Parser', x: 530, y: 80, color: '#1d4ed8', type: 'process' },
        { id: 'output', label: '输出', x: 680, y: 80, color: '#1e40af', type: 'process' }
      ],
      edges: [
        { from: 'input', to: 'prompt', type: 'default' },
        { from: 'prompt', to: 'llm', type: 'default' },
        { from: 'llm', to: 'parser', type: 'default' },
        { from: 'parser', to: 'output', type: 'default' }
      ] as Edge[],
      features: ['单向流动', '无状态', '不支持循环', '简单场景']
    },
    langgraph: {
      title: 'LangGraph（状态图）',
      color: '#10b981',
      gradient: 'from-emerald-500 to-green-600',
      nodes: [
        { id: 'start', label: 'START', x: 120, y: 50, color: '#10b981', type: 'start' },
        { id: 'agent', label: 'Agent', x: 350, y: 50, color: '#3b82f6', type: 'process' },
        { id: 'tools', label: 'Tools', x: 350, y: 170, color: '#8b5cf6', type: 'process' },
        { id: 'check', label: '条件判断', x: 580, y: 50, color: '#f59e0b', type: 'decision' },
        { id: 'end', label: 'END', x: 580, y: 170, color: '#ef4444', type: 'end' }
      ],
      edges: [
        { from: 'start', to: 'agent', type: 'default' },
        { from: 'agent', to: 'check', type: 'default' },
        { from: 'check', to: 'tools', label: '需要工具', type: 'condition' },
        { from: 'tools', to: 'agent', label: '循环', type: 'loop' },
        { from: 'check', to: 'end', label: '完成', type: 'condition' }
      ] as Edge[],
      features: ['支持循环', '有状态', '条件路由', '复杂场景']
    }
  }), []);

  const comparisonData = useMemo(() => [
    { aspect: '流程控制', lcel: '线性/并行', langgraph: '循环/条件/动态' },
    { aspect: '状态管理', lcel: '无状态', langgraph: '持久化状态' },
    { aspect: '人工介入', lcel: '不支持', langgraph: 'Human-in-the-Loop' },
    { aspect: '执行恢复', lcel: '不支持', langgraph: 'Checkpoint 恢复' },
    { aspect: '适用场景', lcel: '简单链式任务', langgraph: '复杂 Agent 系统' },
    { aspect: '学习曲线', lcel: '低', langgraph: '中等' }
  ], []);

  const currentArch = useMemo(() => 
    selectedArch === 'lcel' ? architectures.lcel :
    selectedArch === 'langgraph' ? architectures.langgraph :
    null
  , [selectedArch, architectures]);

  return (
    <div className="my-8 p-8 bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-900 rounded-2xl shadow-xl border border-gray-200 dark:border-gray-700">
      <div className="mb-6">
        <h3 className="text-2xl font-bold mb-2 bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
          LangGraph vs LCEL 架构对比
        </h3>
        <p className="text-sm text-gray-600 dark:text-gray-400 mb-6">
          探索两种不同的编排范式：线性流程 vs 状态图
        </p>
        
        <div className="flex gap-3 mb-6">
          <button
            onClick={() => setSelectedArch('lcel')}
            className={`px-6 py-3 rounded-xl font-semibold transition-all transform hover:scale-105 ${
              selectedArch === 'lcel'
                ? 'bg-gradient-to-r from-blue-500 to-blue-600 text-white shadow-lg shadow-blue-500/50'
                : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 shadow-md hover:shadow-lg'
            }`}
          >
            <span className="flex items-center gap-2">
              📊 LCEL 架构
            </span>
          </button>
          <button
            onClick={() => setSelectedArch('langgraph')}
            className={`px-6 py-3 rounded-xl font-semibold transition-all transform hover:scale-105 ${
              selectedArch === 'langgraph'
                ? 'bg-gradient-to-r from-emerald-500 to-green-600 text-white shadow-lg shadow-green-500/50'
                : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 shadow-md hover:shadow-lg'
            }`}
          >
            <span className="flex items-center gap-2">
              🔄 LangGraph 架构
            </span>
          </button>
          <button
            onClick={() => setSelectedArch('comparison')}
            className={`px-6 py-3 rounded-xl font-semibold transition-all transform hover:scale-105 ${
              selectedArch === 'comparison'
                ? 'bg-gradient-to-r from-purple-500 to-pink-600 text-white shadow-lg shadow-purple-500/50'
                : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 shadow-md hover:shadow-lg'
            }`}
          >
            <span className="flex items-center gap-2">
              📋 对比表格
            </span>
          </button>
        </div>
      </div>

      {currentArch && (
        <div className="space-y-6">
          <div className={`p-6 rounded-2xl bg-gradient-to-r ${currentArch.gradient} shadow-lg`}>
            <h4 className="text-xl font-bold text-white mb-2">
              {currentArch.title}
            </h4>
            <p className="text-white/80 text-sm">点击节点查看详细信息</p>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-lg overflow-x-auto">
            <svg width="1000" height="300" viewBox="0 0 1000 300" className="w-full min-w-[1000px]">
              <defs>
                <filter id="shadow" x="-50%" y="-50%" width="200%" height="200%">
                  <feDropShadow dx="0" dy="4" stdDeviation="4" floodOpacity="0.3"/>
                </filter>
                <linearGradient id="blueGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                  <stop offset="0%" style={{stopColor: '#3b82f6', stopOpacity: 1}} />
                  <stop offset="100%" style={{stopColor: '#2563eb', stopOpacity: 1}} />
                </linearGradient>
                <linearGradient id="greenGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                  <stop offset="0%" style={{stopColor: '#10b981', stopOpacity: 1}} />
                  <stop offset="100%" style={{stopColor: '#059669', stopOpacity: 1}} />
                </linearGradient>
                <linearGradient id="purpleGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                  <stop offset="0%" style={{stopColor: '#8b5cf6', stopOpacity: 1}} />
                  <stop offset="100%" style={{stopColor: '#7c3aed', stopOpacity: 1}} />
                </linearGradient>
                <linearGradient id="orangeGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                  <stop offset="0%" style={{stopColor: '#f59e0b', stopOpacity: 1}} />
                  <stop offset="100%" style={{stopColor: '#d97706', stopOpacity: 1}} />
                </linearGradient>
                <linearGradient id="redGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                  <stop offset="0%" style={{stopColor: '#ef4444', stopOpacity: 1}} />
                  <stop offset="100%" style={{stopColor: '#dc2626', stopOpacity: 1}} />
                </linearGradient>
                <marker id="arrowDefault" markerWidth="12" markerHeight="12" refX="11" refY="3" orient="auto">
                  <polygon points="0 0, 12 3, 0 6" fill="#94a3b8" />
                </marker>
                <marker id="arrowLoop" markerWidth="12" markerHeight="12" refX="11" refY="3" orient="auto">
                  <polygon points="0 0, 12 3, 0 6" fill="#8b5cf6" />
                </marker>
                <marker id="arrowCondition" markerWidth="12" markerHeight="12" refX="11" refY="3" orient="auto">
                  <polygon points="0 0, 12 3, 0 6" fill="#3b82f6" />
                </marker>
              </defs>

              {currentArch.edges.map((edge, idx) => {
                const fromNode = currentArch.nodes.find(n => n.id === edge.from);
                const toNode = currentArch.nodes.find(n => n.id === edge.to);
                
                if (!fromNode || !toNode) return null;
                
                const isLoop = edge.type === 'loop';
                const isCondition = edge.type === 'condition';
                const fromCenterX = fromNode.x + 60;
                const fromCenterY = fromNode.y + 25;
                const toCenterX = toNode.x + 60;
                const toCenterY = toNode.y + 25;
                
                return (
                  <g key={idx}>
                    {isLoop ? (
                      <>
                        <path
                          d={`M ${fromCenterX} ${fromCenterY} 
                              C ${fromCenterX - 80} ${fromCenterY + 100}, 
                                ${toCenterX - 80} ${toCenterY + 100}, 
                                ${toCenterX} ${toCenterY}`}
                          fill="none"
                          stroke="#8b5cf6"
                          strokeWidth="3"
                          strokeDasharray="8,4"
                          markerEnd="url(#arrowLoop)"
                          opacity="0.8"
                        />
                        {edge.label && (
                          <text
                            x={fromCenterX - 90}
                            y={fromCenterY + 100}
                            fontSize="13"
                            fill="#8b5cf6"
                            fontWeight="600"
                            textAnchor="middle"
                          >
                            {edge.label}
                          </text>
                        )}
                      </>
                    ) : (
                      <>
                        <line
                          x1={fromCenterX}
                          y1={fromCenterY}
                          x2={toCenterX}
                          y2={toCenterY}
                          stroke={isCondition ? '#3b82f6' : '#94a3b8'}
                          strokeWidth="3"
                          markerEnd={isCondition ? 'url(#arrowCondition)' : 'url(#arrowDefault)'}
                          opacity="0.7"
                        />
                        {edge.label && (
                          <text
                            x={(fromCenterX + toCenterX) / 2}
                            y={(fromCenterY + toCenterY) / 2 - 10}
                            fontSize="13"
                            fill="#3b82f6"
                            fontWeight="600"
                            textAnchor="middle"
                          >
                            {edge.label}
                          </text>
                        )}
                      </>
                    )}
                  </g>
                );
              })}
              
              {currentArch.nodes.map((node) => {
                const nodeType = node.type || 'process';
                const fillColor = node.color;
                const isStart = nodeType === 'start';
                const isEnd = nodeType === 'end';
                const isDecision = nodeType === 'decision';
                
                return (
                  <g key={node.id} className="cursor-pointer" style={{transition: 'transform 0.2s'}}>
                    {isDecision ? (
                      <>
                        <path
                          d={`M ${node.x + 60} ${node.y} 
                              L ${node.x + 120} ${node.y + 25} 
                              L ${node.x + 60} ${node.y + 50} 
                              L ${node.x} ${node.y + 25} Z`}
                          fill={fillColor}
                          filter="url(#shadow)"
                        />
                        <text
                          x={node.x + 60}
                          y={node.y + 30}
                          textAnchor="middle"
                          fill="white"
                          fontSize="13"
                          fontWeight="700"
                        >
                          {node.label}
                        </text>
                      </>
                    ) : (
                      <>
                        <rect
                          x={node.x}
                          y={node.y}
                          width="120"
                          height="50"
                          rx={isStart || isEnd ? '25' : '12'}
                          fill={fillColor}
                          filter="url(#shadow)"
                        />
                        <rect
                          x={node.x}
                          y={node.y}
                          width="120"
                          height="50"
                          rx={isStart || isEnd ? '25' : '12'}
                          fill="url(#fff)"
                          opacity="0.1"
                        />
                        <text
                          x={node.x + 60}
                          y={node.y + 30}
                          textAnchor="middle"
                          fill="white"
                          fontSize="14"
                          fontWeight="700"
                        >
                          {node.label}
                        </text>
                      </>
                    )}
                  </g>
                );
              })}
            </svg>
          </div>
          
          <div className="grid grid-cols-2 gap-3 mt-6">
            {currentArch.features.map((feature, idx) => (
              <div
                key={idx}
                className="flex items-center gap-3 px-4 py-3 bg-gradient-to-r from-white to-gray-50 dark:from-gray-700 dark:to-gray-800 rounded-xl shadow-md border border-gray-200 dark:border-gray-600"
              >
                <div className="flex-shrink-0 w-8 h-8 rounded-full bg-green-500 flex items-center justify-center shadow-lg">
                  <span className="text-white font-bold text-lg">✓</span>
                </div>
                <span className="text-sm font-medium text-gray-700 dark:text-gray-300">{feature}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {selectedArch === 'comparison' && (
        <div className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-xl overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full border-collapse">
              <thead>
                <tr className="bg-gradient-to-r from-gray-100 to-gray-200 dark:from-gray-700 dark:to-gray-800">
                  <th className="border-2 border-gray-300 dark:border-gray-600 px-6 py-4 text-left font-bold text-gray-800 dark:text-gray-100">
                    对比维度
                  </th>
                  <th className="border-2 border-gray-300 dark:border-gray-600 px-6 py-4 text-left font-bold text-blue-600 dark:text-blue-400">
                    LCEL
                  </th>
                  <th className="border-2 border-gray-300 dark:border-gray-600 px-6 py-4 text-left font-bold text-green-600 dark:text-green-400">
                    LangGraph
                  </th>
                </tr>
              </thead>
              <tbody>
                {comparisonData.map((row, idx) => (
                  <tr 
                    key={idx} 
                    className="hover:bg-blue-50 dark:hover:bg-blue-900/20 transition-colors"
                  >
                    <td className="border-2 border-gray-300 dark:border-gray-600 px-6 py-4 font-semibold text-gray-700 dark:text-gray-300 bg-gray-50 dark:bg-gray-800">
                      {row.aspect}
                    </td>
                    <td className="border-2 border-gray-300 dark:border-gray-600 px-6 py-4 text-gray-600 dark:text-gray-400">
                      {row.lcel}
                    </td>
                    <td className="border-2 border-gray-300 dark:border-gray-600 px-6 py-4 text-gray-600 dark:text-gray-400">
                      {row.langgraph}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      <div className="mt-6 p-6 bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-2xl border-l-4 border-blue-500 shadow-lg">
        <div className="flex items-start gap-4">
          <div className="flex-shrink-0 w-10 h-10 bg-blue-500 rounded-full flex items-center justify-center shadow-lg">
            <span className="text-white text-xl">💡</span>
          </div>
          <div>
            <h4 className="font-bold text-gray-800 dark:text-gray-100 mb-2">关键区别</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300 leading-relaxed">
              LangGraph 支持循环、条件路由、持久化状态，
              适合构建需要多轮交互、工具调用、人工介入的复杂 Agent 系统。
              LCEL 更适合简单的链式处理流程。
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
