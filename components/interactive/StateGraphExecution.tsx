'use client';

import React, { useState, useMemo } from 'react';

type ExecutionStep = {
  step: number;
  node: string;
  state: {
    messages: number;
    iteration: number;
    status: string;
  };
  action: string;
};

export default function StateGraphExecution() {
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  const executionSteps: ExecutionStep[] = useMemo(() => [
    {
      step: 0,
      node: 'START',
      state: { messages: 1, iteration: 0, status: '初始化' },
      action: '接收用户输入'
    },
    {
      step: 1,
      node: 'agent',
      state: { messages: 2, iteration: 1, status: '调用 LLM' },
      action: 'LLM 决定使用 calculator 工具'
    },
    {
      step: 2,
      node: 'should_continue',
      state: { messages: 2, iteration: 1, status: '条件判断' },
      action: '检测到 tool_calls，路由到 tools'
    },
    {
      step: 3,
      node: 'tools',
      state: { messages: 3, iteration: 1, status: '执行工具' },
      action: '执行 calculator(25 * 4) = 100'
    },
    {
      step: 4,
      node: 'agent',
      state: { messages: 4, iteration: 2, status: '调用 LLM' },
      action: 'LLM 决定使用 search 工具'
    },
    {
      step: 5,
      node: 'should_continue',
      state: { messages: 4, iteration: 2, status: '条件判断' },
      action: '检测到 tool_calls，路由到 tools'
    },
    {
      step: 6,
      node: 'tools',
      state: { messages: 5, iteration: 2, status: '执行工具' },
      action: '执行 search(LangGraph)'
    },
    {
      step: 7,
      node: 'agent',
      state: { messages: 6, iteration: 3, status: '调用 LLM' },
      action: 'LLM 生成最终答案'
    },
    {
      step: 8,
      node: 'should_continue',
      state: { messages: 6, iteration: 3, status: '条件判断' },
      action: '无 tool_calls，路由到 END'
    },
    {
      step: 9,
      node: 'END',
      state: { messages: 6, iteration: 3, status: '完成' },
      action: '返回最终结果'
    }
  ], []);

  const handlePlay = () => {
    if (currentStep >= executionSteps.length - 1) {
      setCurrentStep(0);
    }
    setIsPlaying(true);
  };

  const handlePause = () => {
    setIsPlaying(false);
  };

  const handleReset = () => {
    setCurrentStep(0);
    setIsPlaying(false);
  };

  const handleNext = () => {
    if (currentStep < executionSteps.length - 1) {
      setCurrentStep(currentStep + 1);
    }
  };

  const handlePrev = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  React.useEffect(() => {
    if (!isPlaying) return;
    
    const timer = setTimeout(() => {
      if (currentStep < executionSteps.length - 1) {
        setCurrentStep(currentStep + 1);
      } else {
        setIsPlaying(false);
      }
    }, 1500);

    return () => clearTimeout(timer);
  }, [isPlaying, currentStep, executionSteps.length]);

  const currentExecution = useMemo(() => 
    executionSteps[currentStep]
  , [currentStep, executionSteps]);

  const graphNodes = useMemo(() => [
    { id: 'START', label: 'START', x: 80, y: 100, color: '#10b981', type: 'start' },
    { id: 'agent', label: 'agent', x: 380, y: 100, color: '#3b82f6', type: 'process' },
    { id: 'should_continue', label: '条件判断', x: 700, y: 100, color: '#f59e0b', type: 'decision' },
    { id: 'tools', label: 'tools', x: 380, y: 240, color: '#8b5cf6', type: 'process' },
    { id: 'END', label: 'END', x: 700, y: 240, color: '#ef4444', type: 'end' }
  ], []);

  const graphEdges = useMemo(() => [
    { from: 'START', to: 'agent', fromX: 200, fromY: 125, toX: 380, toY: 125 },
    { from: 'agent', to: 'should_continue', fromX: 500, fromY: 125, toX: 700, toY: 125 },
    { from: 'should_continue', to: 'tools', fromX: 760, fromY: 150, toX: 500, toY: 240, label: 'continue', type: 'condition' },
    { from: 'tools', to: 'agent', fromX: 380, fromY: 240, toX: 440, toY: 150, label: 'loop', type: 'loop' },
    { from: 'should_continue', to: 'END', fromX: 760, fromY: 150, toX: 760, toY: 240, label: 'end', type: 'condition' }
  ], []);

  const getNodeColor = (nodeId: string) => {
    const node = graphNodes.find(n => n.id === nodeId);
    return node?.color || '#6b7280';
  };

  return (
    <div className="my-8 p-8 bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-900 rounded-2xl shadow-xl border border-gray-200 dark:border-gray-700">
      <h3 className="text-2xl font-bold mb-2 bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
        StateGraph 执行流程演示
      </h3>
      <p className="text-sm text-gray-600 dark:text-gray-400 mb-6">
        观察状态图如何在节点间流转，执行复杂的 Agent 逻辑
      </p>

      <div className="mb-6 flex flex-wrap gap-3">
        <button
          onClick={isPlaying ? handlePause : handlePlay}
          className="px-6 py-3 bg-gradient-to-r from-green-500 to-emerald-600 text-white rounded-xl font-semibold hover:shadow-lg hover:scale-105 transition-all shadow-md"
        >
          {isPlaying ? '⏸ 暂停' : '▶ 播放'}
        </button>
        <button
          onClick={handlePrev}
          disabled={currentStep === 0}
          className="px-6 py-3 bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-xl font-semibold hover:shadow-lg hover:scale-105 disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:scale-100 transition-all shadow-md"
        >
          ← 上一步
        </button>
        <button
          onClick={handleNext}
          disabled={currentStep === executionSteps.length - 1}
          className="px-6 py-3 bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-xl font-semibold hover:shadow-lg hover:scale-105 disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:scale-100 transition-all shadow-md"
        >
          下一步 →
        </button>
        <button
          onClick={handleReset}
          className="px-6 py-3 bg-gradient-to-r from-gray-500 to-gray-600 text-white rounded-xl font-semibold hover:shadow-lg hover:scale-105 transition-all shadow-md"
        >
          🔄 重置
        </button>
        <div className="ml-auto px-6 py-3 bg-white dark:bg-gray-800 rounded-xl shadow-md border border-gray-200 dark:border-gray-600">
          <span className="font-semibold text-gray-700 dark:text-gray-300">
            步骤 <span className="text-blue-600 dark:text-blue-400">{currentStep + 1}</span> / {executionSteps.length}
          </span>
        </div>
      </div>

      <div className="mb-8 bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-lg overflow-x-auto">
        <svg width="950" height="350" viewBox="0 0 950 350" className="w-full min-w-[950px]">
          <defs>
            <filter id="node-shadow" x="-50%" y="-50%" width="200%" height="200%">
              <feDropShadow dx="0" dy="4" stdDeviation="6" floodOpacity="0.3"/>
            </filter>
            <marker id="arrow-default" markerWidth="12" markerHeight="12" refX="11" refY="3" orient="auto">
              <polygon points="0 0, 12 3, 0 6" fill="#94a3b8" />
            </marker>
            <marker id="arrow-loop" markerWidth="12" markerHeight="12" refX="11" refY="3" orient="auto">
              <polygon points="0 0, 12 3, 0 6" fill="#8b5cf6" />
            </marker>
            <marker id="arrow-condition" markerWidth="12" markerHeight="12" refX="11" refY="3" orient="auto">
              <polygon points="0 0, 12 3, 0 6" fill="#3b82f6" />
            </marker>
          </defs>

          {graphEdges.map((edge, idx) => {
            const isActive = 
              (edge.from === currentExecution.node && currentStep < executionSteps.length - 1) ||
              (executionSteps[currentStep - 1]?.node === edge.from && 
               executionSteps[currentStep]?.node === edge.to);

            const isLoop = edge.type === 'loop';
            const isCondition = edge.type === 'condition';

            return (
              <g key={idx}>
                {isLoop ? (
                  <path
                    d={`M ${edge.fromX} ${edge.fromY} 
                        C ${edge.fromX - 120} ${edge.fromY + 70}, 
                          ${edge.toX - 120} ${edge.toY - 70}, 
                          ${edge.toX} ${edge.toY}`}
                    fill="none"
                    stroke={isActive ? '#8b5cf6' : '#cbd5e1'}
                    strokeWidth={isActive ? '4' : '2'}
                    strokeDasharray="8,4"
                    markerEnd="url(#arrow-loop)"
                    opacity={isActive ? '1' : '0.4'}
                  />
                ) : (
                  <line
                    x1={edge.fromX}
                    y1={edge.fromY}
                    x2={edge.toX}
                    y2={edge.toY}
                    stroke={isActive ? (isCondition ? '#3b82f6' : '#94a3b8') : '#e2e8f0'}
                    strokeWidth={isActive ? '4' : '2'}
                    markerEnd={isCondition ? 'url(#arrow-condition)' : 'url(#arrow-default)'}
                    opacity={isActive ? '1' : '0.4'}
                  />
                )}

                {edge.label && (
                  <text
                    x={isLoop ? edge.fromX - 130 : (edge.fromX + edge.toX) / 2}
                    y={isLoop ? (edge.fromY + edge.toY) / 2 : edge.fromY - 10}
                    fontSize="12"
                    fill={isActive ? '#3b82f6' : '#94a3b8'}
                    fontWeight="600"
                    textAnchor="middle"
                  >
                    {edge.label}
                  </text>
                )}
              </g>
            );
          })}

          {graphNodes.map((node) => {
            const isActive = node.id === currentExecution.node;
            const nodeType = node.type;

            return (
              <g key={node.id}>
                {nodeType === 'decision' ? (
                  <>
                    <path
                      d={`M ${node.x + 60} ${node.y} 
                          L ${node.x + 120} ${node.y + 25} 
                          L ${node.x + 60} ${node.y + 50} 
                          L ${node.x} ${node.y + 25} Z`}
                      fill={node.color}
                      opacity={isActive ? '1' : '0.5'}
                      filter="url(#node-shadow)"
                      stroke={isActive ? '#fbbf24' : 'none'}
                      strokeWidth={isActive ? '4' : '0'}
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
                      rx={nodeType === 'start' || nodeType === 'end' ? '25' : '12'}
                      fill={node.color}
                      opacity={isActive ? '1' : '0.5'}
                      filter="url(#node-shadow)"
                      stroke={isActive ? '#fbbf24' : 'none'}
                      strokeWidth={isActive ? '4' : '0'}
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
                {isActive && (
                  <circle
                    cx={node.x + 60}
                    cy={node.y - 15}
                    r="8"
                    fill="#fbbf24"
                    filter="url(#node-shadow)"
                  >
                    <animate attributeName="opacity" values="1;0.3;1" dur="1s" repeatCount="indefinite" />
                  </circle>
                )}
              </g>
            );
          })}
        </svg>
      </div>

      <div className="grid grid-cols-2 gap-6 mb-6">
        <div className="p-6 bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/30 dark:to-blue-800/30 rounded-2xl shadow-lg border border-blue-200 dark:border-blue-700">
          <div className="flex items-center gap-3 mb-3">
            <div className="w-10 h-10 bg-blue-500 rounded-full flex items-center justify-center shadow-lg">
              <span className="text-white text-lg">📍</span>
            </div>
            <h4 className="font-bold text-gray-800 dark:text-gray-100">当前节点</h4>
          </div>
          <div className="text-3xl font-bold mb-2" style={{ color: getNodeColor(currentExecution.node) }}>
            {currentExecution.node}
          </div>
          <div className="text-sm text-gray-700 dark:text-gray-300 leading-relaxed">
            {currentExecution.action}
          </div>
        </div>

        <div className="p-6 bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/30 dark:to-green-800/30 rounded-2xl shadow-lg border border-green-200 dark:border-green-700">
          <div className="flex items-center gap-3 mb-3">
            <div className="w-10 h-10 bg-green-500 rounded-full flex items-center justify-center shadow-lg">
              <span className="text-white text-lg">📊</span>
            </div>
            <h4 className="font-bold text-gray-800 dark:text-gray-100">状态快照</h4>
          </div>
          <div className="space-y-2">
            <div className="flex justify-between items-center p-2 bg-white/50 dark:bg-gray-800/50 rounded-lg">
              <span className="text-sm text-gray-600 dark:text-gray-400">消息数：</span>
              <span className="font-mono font-bold text-green-600 dark:text-green-400 text-lg">
                {currentExecution.state.messages}
              </span>
            </div>
            <div className="flex justify-between items-center p-2 bg-white/50 dark:bg-gray-800/50 rounded-lg">
              <span className="text-sm text-gray-600 dark:text-gray-400">迭代次数：</span>
              <span className="font-mono font-bold text-green-600 dark:text-green-400 text-lg">
                {currentExecution.state.iteration}
              </span>
            </div>
            <div className="flex justify-between items-center p-2 bg-white/50 dark:bg-gray-800/50 rounded-lg">
              <span className="text-sm text-gray-600 dark:text-gray-400">状态：</span>
              <span className="font-mono font-bold text-green-600 dark:text-green-400 text-lg">
                {currentExecution.state.status}
              </span>
            </div>
          </div>
        </div>
      </div>

      <div className="p-6 bg-white dark:bg-gray-800 rounded-2xl shadow-lg">
        <h4 className="font-bold mb-4 text-gray-800 dark:text-gray-200 flex items-center gap-2">
          <span className="text-xl">📜</span>
          执行历史
        </h4>
        <div className="space-y-2 max-h-48 overflow-y-auto pr-2">
          {executionSteps.slice(0, currentStep + 1).map((step, idx) => (
            <div
              key={idx}
              className={`p-3 rounded-xl transition-all ${
                idx === currentStep
                  ? 'bg-gradient-to-r from-yellow-100 to-yellow-200 dark:from-yellow-900/40 dark:to-yellow-800/40 border-2 border-yellow-400 scale-105 shadow-md'
                  : 'bg-gray-50 dark:bg-gray-700 border border-gray-200 dark:border-gray-600'
              }`}
            >
              <div className="flex items-center gap-3">
                <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center font-bold text-white ${
                  idx === currentStep ? 'bg-yellow-500' : 'bg-gray-400'
                }`}>
                  {step.step}
                </div>
                <div className="flex-1">
                  <div className="font-semibold" style={{ color: getNodeColor(step.node) }}>
                    {step.node}
                  </div>
                  <div className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                    {step.action}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="mt-6 p-6 bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-2xl border-l-4 border-purple-500 shadow-lg">
        <div className="flex items-start gap-4">
          <div className="flex-shrink-0 w-10 h-10 bg-purple-500 rounded-full flex items-center justify-center shadow-lg">
            <span className="text-white text-xl">💡</span>
          </div>
          <div>
            <h4 className="font-bold text-gray-800 dark:text-gray-100 mb-2">关键机制</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300 leading-relaxed">
              每个节点执行后更新状态，条件边根据状态决定下一个节点。
              状态在整个执行过程中持久保存，支持循环与回溯。
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
