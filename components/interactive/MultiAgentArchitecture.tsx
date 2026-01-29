'use client';

import React, { useState, useMemo } from 'react';

type AgentType = 'researcher' | 'writer' | 'editor' | 'supervisor';

type Message = {
  from: AgentType | 'user';
  to: AgentType | 'user';
  content: string;
  timestamp: number;
};

export default function MultiAgentArchitecture() {
  const [currentStep, setCurrentStep] = useState(0);
  const [selectedAgent, setSelectedAgent] = useState<AgentType | null>(null);

  const agents = useMemo(() => [
    {
      type: 'supervisor' as AgentType,
      name: 'Supervisor',
      color: '#8b5cf6',
      role: '任务分配与协调',
      capabilities: ['分析任务', '分配给专业 Agent', '汇总结果']
    },
    {
      type: 'researcher' as AgentType,
      name: 'Researcher',
      color: '#3b82f6',
      role: '信息收集',
      capabilities: ['网络搜索', '数据收集', '信息提取']
    },
    {
      type: 'writer' as AgentType,
      name: 'Writer',
      color: '#10b981',
      role: '内容创作',
      capabilities: ['撰写文章', '生成报告', '内容组织']
    },
    {
      type: 'editor' as AgentType,
      name: 'Editor',
      color: '#f59e0b',
      role: '内容编辑',
      capabilities: ['语法检查', '风格优化', '质量审核']
    }
  ], []);

  const workflow: Message[] = useMemo(() => [
    {
      from: 'user',
      to: 'supervisor',
      content: '请写一篇关于 LangGraph 的博客文章',
      timestamp: 0
    },
    {
      from: 'supervisor',
      to: 'researcher',
      content: '收集 LangGraph 的相关信息和文档',
      timestamp: 1
    },
    {
      from: 'researcher',
      to: 'supervisor',
      content: '已收集：核心概念、使用示例、最佳实践',
      timestamp: 2
    },
    {
      from: 'supervisor',
      to: 'writer',
      content: '基于研究结果撰写博客文章',
      timestamp: 3
    },
    {
      from: 'writer',
      to: 'supervisor',
      content: '已完成 800 字初稿',
      timestamp: 4
    },
    {
      from: 'supervisor',
      to: 'editor',
      content: '请审核并优化文章',
      timestamp: 5
    },
    {
      from: 'editor',
      to: 'supervisor',
      content: '已完成编辑，质量良好',
      timestamp: 6
    },
    {
      from: 'supervisor',
      to: 'user',
      content: '博客文章已完成',
      timestamp: 7
    }
  ], []);

  const currentMessage = useMemo(() => 
    workflow[currentStep]
  , [workflow, currentStep]);

  const handleNext = () => {
    if (currentStep < workflow.length - 1) {
      setCurrentStep(currentStep + 1);
    }
  };

  const handlePrev = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  const handleReset = () => {
    setCurrentStep(0);
    setSelectedAgent(null);
  };

  const getAgentColor = (agentType: AgentType | 'user') => {
    if (agentType === 'user') return '#6b7280';
    const agent = agents.find(a => a.type === agentType);
    return agent?.color || '#6b7280';
  };

  const getAgentPosition = (agentType: AgentType | 'user') => {
    switch (agentType) {
      case 'supervisor': return { x: 300, y: 50 };
      case 'researcher': return { x: 100, y: 150 };
      case 'writer': return { x: 300, y: 150 };
      case 'editor': return { x: 500, y: 150 };
      case 'user': return { x: 300, y: 250 };
      default: return { x: 0, y: 0 };
    }
  };

  const selectedAgentData = useMemo(() => 
    selectedAgent ? agents.find(a => a.type === selectedAgent) : null
  , [selectedAgent, agents]);

  return (
    <div className="my-8 p-6 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
      <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-gray-100">
        Multi-Agent 协作架构（Supervisor 模式）
      </h3>

      <div className="mb-6 flex gap-2">
        <button
          onClick={handlePrev}
          disabled={currentStep === 0}
          className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
        >
          ← 上一步
        </button>
        <button
          onClick={handleNext}
          disabled={currentStep === workflow.length - 1}
          className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
        >
          下一步 →
        </button>
        <button
          onClick={handleReset}
          className="px-4 py-2 bg-gray-500 text-white rounded hover:bg-gray-600 transition-colors"
        >
          🔄 重置
        </button>
        <div className="ml-auto px-4 py-2 bg-gray-100 dark:bg-gray-700 rounded">
          步骤 {currentStep + 1} / {workflow.length}
        </div>
      </div>

      <svg width="650" height="320" className="border border-gray-300 dark:border-gray-600 rounded bg-gray-50 dark:bg-gray-900 mb-6">
        {currentStep > 0 && (
          <line
            x1={getAgentPosition(currentMessage.from).x + 50}
            y1={getAgentPosition(currentMessage.from).y + 30}
            x2={getAgentPosition(currentMessage.to).x + 50}
            y2={getAgentPosition(currentMessage.to).y + 30}
            stroke="#fbbf24"
            strokeWidth="3"
            markerEnd="url(#arrow-active)"
          />
        )}

        {[...agents, { type: 'user' as AgentType | 'user', name: 'User', color: '#6b7280', role: '用户', capabilities: [] }].map((agent) => {
          const pos = getAgentPosition(agent.type);
          const isActive =
            agent.type === currentMessage.from ||
            agent.type === currentMessage.to;
          
          return (
            <g
              key={agent.type}
              onClick={() => agent.type !== 'user' && setSelectedAgent(agent.type as AgentType)}
              className="cursor-pointer"
            >
              <rect
                x={pos.x}
                y={pos.y}
                width="100"
                height="60"
                rx="8"
                fill={agent.color}
                opacity={isActive ? '1' : '0.5'}
                stroke={isActive ? '#fbbf24' : 'none'}
                strokeWidth={isActive ? '3' : '0'}
              />
              <text
                x={pos.x + 50}
                y={pos.y + 30}
                textAnchor="middle"
                fill="white"
                fontSize="14"
                fontWeight="600"
              >
                {agent.name}
              </text>
              <text
                x={pos.x + 50}
                y={pos.y + 48}
                textAnchor="middle"
                fill="white"
                fontSize="10"
              >
                {agent.role}
              </text>
            </g>
          );
        })}

        <defs>
          <marker
            id="arrow-active"
            markerWidth="10"
            markerHeight="10"
            refX="9"
            refY="3"
            orient="auto"
          >
            <polygon points="0 0, 10 3, 0 6" fill="#fbbf24" />
          </marker>
        </defs>
      </svg>

      <div className="p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded mb-4">
        <div className="flex items-start gap-3">
          <div
            className="w-3 h-3 rounded-full mt-1"
            style={{ backgroundColor: getAgentColor(currentMessage.from) }}
          ></div>
          <div className="flex-1">
            <div className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-1">
              {currentMessage.from === 'user' ? 'User' : agents.find(a => a.type === currentMessage.from)?.name} →{' '}
              {currentMessage.to === 'user' ? 'User' : agents.find(a => a.type === currentMessage.to)?.name}
            </div>
            <div className="text-gray-800 dark:text-gray-200">
              {currentMessage.content}
            </div>
          </div>
        </div>
      </div>

      {selectedAgentData && (
        <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded mb-4">
          <h4 className="font-semibold mb-2 text-gray-800 dark:text-gray-200">
            {selectedAgentData.name} 详情
          </h4>
          <div className="text-sm text-gray-600 dark:text-gray-400 mb-2">
            角色：{selectedAgentData.role}
          </div>
          <div className="text-sm text-gray-700 dark:text-gray-300">
            <strong>能力：</strong>
            <ul className="mt-1 ml-4 space-y-1">
              {selectedAgentData.capabilities.map((cap, idx) => (
                <li key={idx}>• {cap}</li>
              ))}
            </ul>
          </div>
        </div>
      )}

      <div className="p-4 bg-gray-50 dark:bg-gray-700 rounded">
        <h4 className="font-semibold mb-2 text-gray-800 dark:text-gray-200">
          消息历史
        </h4>
        <div className="space-y-2 max-h-40 overflow-y-auto">
          {workflow.slice(0, currentStep + 1).map((msg, idx) => (
            <div
              key={idx}
              className={`text-xs p-2 rounded ${
                idx === currentStep
                  ? 'bg-yellow-100 dark:bg-yellow-900/30 font-semibold'
                  : 'bg-white dark:bg-gray-800'
              }`}
            >
              <div className="flex items-center gap-2 mb-1">
                <div
                  className="w-2 h-2 rounded-full"
                  style={{ backgroundColor: getAgentColor(msg.from) }}
                ></div>
                <span className="text-gray-500 dark:text-gray-400">
                  {msg.from === 'user' ? 'User' : agents.find(a => a.type === msg.from)?.name}
                </span>
                <span className="text-gray-400">→</span>
                <div
                  className="w-2 h-2 rounded-full"
                  style={{ backgroundColor: getAgentColor(msg.to) }}
                ></div>
                <span className="text-gray-500 dark:text-gray-400">
                  {msg.to === 'user' ? 'User' : agents.find(a => a.type === msg.to)?.name}
                </span>
              </div>
              <div className="text-gray-700 dark:text-gray-300 ml-4">
                {msg.content}
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="mt-4 grid grid-cols-4 gap-2">
        {agents.map((agent) => (
          <div
            key={agent.type}
            className="p-2 rounded text-center text-xs"
            style={{ backgroundColor: `${agent.color}20` }}
          >
            <div
              className="w-8 h-8 rounded-full mx-auto mb-1"
              style={{ backgroundColor: agent.color }}
            ></div>
            <div className="font-semibold text-gray-800 dark:text-gray-200">
              {agent.name}
            </div>
          </div>
        ))}
      </div>

      <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-900/20 rounded text-sm text-gray-700 dark:text-gray-300">
        <strong>Supervisor 模式优势：</strong>
        <ul className="mt-2 space-y-1 ml-4 text-xs">
          <li>• 清晰的任务分配与协调机制</li>
          <li>• 每个 Agent 专注于特定领域</li>
          <li>• 易于扩展新的专业 Agent</li>
          <li>• 支持复杂多步骤工作流</li>
        </ul>
      </div>
    </div>
  );
}
