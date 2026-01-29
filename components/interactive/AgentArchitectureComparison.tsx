'use client';

import React, { useState, useMemo } from 'react';

type ArchType = 'chain' | 'react' | 'planning';

export default function AgentArchitectureComparison() {
  const [selectedType, setSelectedType] = useState<ArchType>('react');

  const architectures = useMemo(() => ({
    chain: {
      title: 'Chain（固定流程）',
      description: '预定义的执行路径，每个步骤固定',
      steps: [
        { id: 1, name: '接收输入', color: '#3b82f6' },
        { id: 2, name: '提示模板', color: '#3b82f6' },
        { id: 3, name: '调用 LLM', color: '#3b82f6' },
        { id: 4, name: '输出解析', color: '#3b82f6' },
        { id: 5, name: '返回结果', color: '#3b82f6' }
      ],
      features: [
        { text: '流程固定', positive: false },
        { text: '不能动态决策', positive: false },
        { text: '简单可预测', positive: true },
        { text: '性能稳定', positive: true }
      ]
    },
    react: {
      title: 'ReAct Agent（动态决策）',
      description: '通过 Thought-Action-Observation 循环自主决策',
      steps: [
        { id: 1, name: 'Thought: 分析问题', color: '#10b981' },
        { id: 2, name: 'Action: 选择工具', color: '#f59e0b' },
        { id: 3, name: 'Observation: 观察结果', color: '#8b5cf6' },
        { id: 4, name: 'Thought: 继续思考', color: '#10b981' },
        { id: 5, name: 'Final Answer', color: '#ef4444' }
      ],
      features: [
        { text: '自主决策', positive: true },
        { text: '动态工具调用', positive: true },
        { text: '适应复杂任务', positive: true },
        { text: '可能不稳定', positive: false }
      ]
    },
    planning: {
      title: 'Planning Agent（先计划后执行）',
      description: '先制定完整计划，再逐步执行',
      steps: [
        { id: 1, name: 'Plan: 制定计划', color: '#3b82f6' },
        { id: 2, name: 'Execute Step 1', color: '#10b981' },
        { id: 3, name: 'Execute Step 2', color: '#10b981' },
        { id: 4, name: 'Execute Step 3', color: '#10b981' },
        { id: 5, name: 'Summarize', color: '#8b5cf6' }
      ],
      features: [
        { text: '结构化执行', positive: true },
        { text: '可追溯性强', positive: true },
        { text: '适合长任务', positive: true },
        { text: '缺乏灵活性', positive: false }
      ]
    }
  }), []);

  const currentArch = useMemo(() => 
    architectures[selectedType]
  , [selectedType, architectures]);

  const comparisonTable = useMemo(() => [
    { aspect: '决策方式', chain: '预定义', react: '动态', planning: '计划驱动' },
    { aspect: '工具调用', chain: '固定位置', react: '自主选择', planning: '按计划执行' },
    { aspect: '灵活性', chain: '低', react: '高', planning: '中' },
    { aspect: '可控性', chain: '高', react: '低', planning: '中' },
    { aspect: '适用场景', chain: '简单任务', react: '开放式问题', planning: '复杂长任务' }
  ], []);

  return (
    <div className="my-8 p-6 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
      <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-gray-100">
        Agent 架构对比
      </h3>

      <div className="flex gap-2 mb-6">
        <button
          onClick={() => setSelectedType('chain')}
          className={`px-4 py-2 rounded transition-colors ${
            selectedType === 'chain'
              ? 'bg-blue-500 text-white'
              : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
          }`}
        >
          Chain
        </button>
        <button
          onClick={() => setSelectedType('react')}
          className={`px-4 py-2 rounded transition-colors ${
            selectedType === 'react'
              ? 'bg-green-500 text-white'
              : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
          }`}
        >
          ReAct Agent
        </button>
        <button
          onClick={() => setSelectedType('planning')}
          className={`px-4 py-2 rounded transition-colors ${
            selectedType === 'planning'
              ? 'bg-purple-500 text-white'
              : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
          }`}
        >
          Planning Agent
        </button>
      </div>

      <div className="mb-6 p-4 bg-gray-50 dark:bg-gray-700 rounded">
        <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-2">
          {currentArch.title}
        </h4>
        <p className="text-sm text-gray-600 dark:text-gray-400">
          {currentArch.description}
        </p>
      </div>

      <div className="mb-6 flex justify-center items-center gap-4">
        {currentArch.steps.map((step, idx) => (
          <div key={step.id} className="flex items-center gap-2">
            <div
              className="px-4 py-3 rounded-lg text-white text-center min-w-[120px]"
              style={{ backgroundColor: step.color }}
            >
              <div className="text-xs font-semibold">Step {step.id}</div>
              <div className="text-sm mt-1">{step.name}</div>
            </div>
            {idx < currentArch.steps.length - 1 && (
              <div className="text-gray-400 text-2xl">→</div>
            )}
          </div>
        ))}
      </div>

      <div className="grid grid-cols-2 gap-3 mb-6">
        {currentArch.features.map((feature, idx) => (
          <div
            key={idx}
            className={`flex items-center gap-2 px-3 py-2 rounded ${
              feature.positive
                ? 'bg-green-50 dark:bg-green-900/20'
                : 'bg-red-50 dark:bg-red-900/20'
            }`}
          >
            <span className={feature.positive ? 'text-green-500' : 'text-red-500'}>
              {feature.positive ? '✓' : '✗'}
            </span>
            <span className="text-sm text-gray-700 dark:text-gray-300">
              {feature.text}
            </span>
          </div>
        ))}
      </div>

      <div className="overflow-x-auto">
        <table className="w-full border-collapse text-sm">
          <thead>
            <tr className="bg-gray-100 dark:bg-gray-700">
              <th className="border border-gray-300 dark:border-gray-600 px-3 py-2 text-left">
                对比维度
              </th>
              <th className="border border-gray-300 dark:border-gray-600 px-3 py-2 text-left">
                Chain
              </th>
              <th className="border border-gray-300 dark:border-gray-600 px-3 py-2 text-left">
                ReAct
              </th>
              <th className="border border-gray-300 dark:border-gray-600 px-3 py-2 text-left">
                Planning
              </th>
            </tr>
          </thead>
          <tbody>
            {comparisonTable.map((row, idx) => (
              <tr key={idx} className="hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors">
                <td className="border border-gray-300 dark:border-gray-600 px-3 py-2 font-medium">
                  {row.aspect}
                </td>
                <td className="border border-gray-300 dark:border-gray-600 px-3 py-2">
                  {row.chain}
                </td>
                <td className="border border-gray-300 dark:border-gray-600 px-3 py-2">
                  {row.react}
                </td>
                <td className="border border-gray-300 dark:border-gray-600 px-3 py-2">
                  {row.planning}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-900/20 rounded text-sm text-gray-700 dark:text-gray-300">
        <strong>选择建议：</strong>
        <ul className="mt-2 space-y-1 ml-4">
          <li>• <strong>Chain</strong>: 固定流程、可预测的任务（如数据转换、格式化）</li>
          <li>• <strong>ReAct</strong>: 需要动态决策、工具调用的开放式问题（如客服、研究助手）</li>
          <li>• <strong>Planning</strong>: 复杂多步骤任务，需要清晰计划和追溯（如报告生成、项目分析）</li>
        </ul>
      </div>
    </div>
  );
}
