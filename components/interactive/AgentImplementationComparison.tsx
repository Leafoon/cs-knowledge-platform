'use client';

import React, { useState } from 'react';

type ImplType = 'prompting' | 'framework' | 'langgraph' | 'custom';

export default function AgentImplementationComparison() {
  const [selected, setSelected] = useState<ImplType>('langgraph');

  const implementations = {
    prompting: {
      name: '纯 Prompt 工程',
      complexity: '⭐',
      speed: '⚡⚡⚡⚡⚡',
      flexibility: '⭐⭐⭐',
      maintainability: '⭐',
      description: '直接在prompt中描述Agent行为，依赖LLM原生能力',
      useCases: ['简单问答', '快速原型', '单轮工具调用'],
      examples: 'GPTs、AutoGPT提示工程',
      color: '#3b82f6',
    },
    framework: {
      name: '第三方框架',
      complexity: '⭐⭐⭐',
      speed: '⚡⚡⚡',
      flexibility: '⭐⭐⭐⭐',
      maintainability: '⭐⭐⭐',
      description: '使用现成Agent框架（LangChain、LlamaIndex等）快速搭建',
      useCases: ['产品原型', '中等复杂度', '快速开发'],
      examples: 'LangChain Agent、LlamaIndex Agent',
      color: '#10b981',
    },
    langgraph: {
      name: 'LangGraph 显式控制流',
      complexity: '⭐⭐⭐⭐',
      speed: '⚡⚡',
      flexibility: '⭐⭐⭐⭐⭐',
      maintainability: '⭐⭐⭐⭐',
      description: '用图结构显式定义Agent状态和流转，支持复杂控制流',
      useCases: ['生产级Agent', '多步推理', '人机协作'],
      examples: 'LangGraph、AutoGPT with planning',
      color: '#8b5cf6',
    },
    custom: {
      name: '完全自定义实现',
      complexity: '⭐⭐⭐⭐⭐',
      speed: '⚡',
      flexibility: '⭐⭐⭐⭐⭐',
      maintainability: '⭐⭐',
      description: '从零开始手写所有推理和控制逻辑',
      useCases: ['特殊需求', '研究项目', '极致优化'],
      examples: '研究论文复现、定制化Agent',
      color: '#ef4444',
    },
  };

  const current = implementations[selected];

  return (
    <div className="my-8 p-6 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
      <h3 className="text-lg font-semibold mb-6 text-gray-900 dark:text-gray-100">
        Agent 实现方式对比
      </h3>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
        {(Object.keys(implementations) as ImplType[]).map((key) => (
          <button
            key={key}
            onClick={() => setSelected(key)}
            className={`p-4 rounded-lg border-2 text-left transition-all ${
              selected === key
                ? 'border-opacity-100'
                : 'border-gray-200 dark:border-gray-700 bg-transparent'
            }`}
            style={{
              borderColor: selected === key ? implementations[key].color : undefined,
              backgroundColor: selected === key ? `${implementations[key].color}15` : undefined,
            }}
          >
            <div className="font-semibold text-gray-900 dark:text-gray-100 mb-1">
              {implementations[key].name}
            </div>
            <div className="text-xs text-gray-500 dark:text-gray-400">
              复杂度: {implementations[key].complexity}
            </div>
          </button>
        ))}
      </div>

      <div className="p-5 rounded-lg bg-gray-50 dark:bg-gray-700">
        <h4
          className="text-xl font-semibold mb-3"
          style={{ color: current.color }}
        >
          {current.name}
        </h4>
        <p className="text-gray-700 dark:text-gray-300 mb-4">{current.description}</p>

        <div className="grid grid-cols-2 gap-6 mb-4">
          <div>
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div className="text-gray-600 dark:text-gray-400">实现复杂度</div>
              <div>{current.complexity}</div>
              <div className="text-gray-600 dark:text-gray-400">运行速度</div>
              <div>{current.speed}</div>
              <div className="text-gray-600 dark:text-gray-400">灵活性</div>
              <div>{current.flexibility}</div>
              <div className="text-gray-600 dark:text-gray-400">可维护性</div>
              <div>{current.maintainability}</div>
            </div>
          </div>
          <div>
            <div className="mb-2">
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">适用场景：</span>
              <ul className="mt-1 space-y-1">
                {current.useCases.map((useCase, idx) => (
                  <li
                    key={idx}
                    className="text-sm text-gray-600 dark:text-gray-400 pl-4 relative"
                  >
                    <span className="absolute left-0 top-1 text-blue-600">•</span>
                    {useCase}
                  </li>
                ))}
              </ul>
            </div>
            <div className="mt-2">
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                典型例子：
              </span>
              <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                {current.examples}
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-900/20 rounded text-sm text-gray-700 dark:text-gray-300">
        <strong>推荐选择：</strong>
        <ul className="mt-2 space-y-1 ml-4">
          <li>• 原型验证：<strong>纯 Prompt + 第三方框架</strong></li>
          <li>• 生产环境：<strong>LangGraph + 框架</strong> 兼顾灵活性和可维护性</li>
          <li>• 特殊需求：<strong>完全自定义</strong></li>
        </ul>
      </div>
    </div>
  );
}
