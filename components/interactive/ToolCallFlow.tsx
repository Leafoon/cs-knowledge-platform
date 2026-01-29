'use client';

import React, { useState, useMemo } from 'react';

type ToolCallStep = {
  step: number;
  phase: 'thought' | 'action' | 'observation' | 'error' | 'final';
  content: string;
  toolName?: string;
  toolInput?: string;
  toolOutput?: string;
  success?: boolean;
};

export default function ToolCallFlow() {
  const [currentStep, setCurrentStep] = useState(0);
  const [showRetry, setShowRetry] = useState(false);

  const toolCallSteps: ToolCallStep[] = useMemo(() => [
    {
      step: 0,
      phase: 'thought',
      content: '用户询问："北京天气如何？25 * 4 是多少？"我需要调用两个工具：get_weather 和 calculator'
    },
    {
      step: 1,
      phase: 'action',
      content: '决定先调用 get_weather 工具',
      toolName: 'get_weather',
      toolInput: '{ "city": "Beijing" }'
    },
    {
      step: 2,
      phase: 'observation',
      content: '工具返回结果',
      toolName: 'get_weather',
      toolOutput: 'Sunny, 25°C',
      success: true
    },
    {
      step: 3,
      phase: 'action',
      content: '现在调用 calculator 工具',
      toolName: 'calculator',
      toolInput: '{ "expression": "25 * 4" }'
    },
    {
      step: 4,
      phase: 'observation',
      content: '工具返回结果',
      toolName: 'calculator',
      toolOutput: '100',
      success: true
    },
    {
      step: 5,
      phase: 'final',
      content: '综合两个工具的结果，生成最终答案：北京天气晴朗，25°C。25 * 4 = 100。'
    }
  ], []);

  const errorScenario: ToolCallStep[] = useMemo(() => [
    {
      step: 0,
      phase: 'thought',
      content: '用户询问："执行 SQL: SELECT * FROM users"'
    },
    {
      step: 1,
      phase: 'action',
      content: '调用 database_query 工具',
      toolName: 'database_query',
      toolInput: '{ "sql": "SELECT * FROM users" }'
    },
    {
      step: 2,
      phase: 'error',
      content: '工具执行失败',
      toolName: 'database_query',
      toolOutput: 'ToolException: Connection timeout',
      success: false
    },
    {
      step: 3,
      phase: 'thought',
      content: '工具调用失败，尝试重试机制'
    },
    {
      step: 4,
      phase: 'action',
      content: '重试 database_query（第 2 次尝试）',
      toolName: 'database_query',
      toolInput: '{ "sql": "SELECT * FROM users" }'
    },
    {
      step: 5,
      phase: 'observation',
      content: '重试成功',
      toolName: 'database_query',
      toolOutput: '[{"id": 1, "name": "Alice"}, ...]',
      success: true
    },
    {
      step: 6,
      phase: 'final',
      content: '成功获取数据，返回用户'
    }
  ], []);

  const currentScenario = useMemo(() => 
    showRetry ? errorScenario : toolCallSteps
  , [showRetry, errorScenario, toolCallSteps]);

  const currentData = useMemo(() => 
    currentScenario[currentStep]
  , [currentStep, currentScenario]);

  const handleNext = () => {
    if (currentStep < currentScenario.length - 1) {
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
  };

  const getPhaseColor = (phase: string) => {
    switch (phase) {
      case 'thought': return '#3b82f6';
      case 'action': return '#f59e0b';
      case 'observation': return '#10b981';
      case 'error': return '#ef4444';
      case 'final': return '#8b5cf6';
      default: return '#6b7280';
    }
  };

  const getPhaseLabel = (phase: string) => {
    switch (phase) {
      case 'thought': return '💭 Thought';
      case 'action': return '⚡ Action';
      case 'observation': return '👁 Observation';
      case 'error': return '❌ Error';
      case 'final': return '✅ Final Answer';
      default: return 'Unknown';
    }
  };

  return (
    <div className="my-8 p-6 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
      <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-gray-100">
        工具调用流程演示
      </h3>

      <div className="mb-6 flex gap-2">
        <button
          onClick={() => {
            setShowRetry(false);
            setCurrentStep(0);
          }}
          className={`px-4 py-2 rounded transition-colors ${
            !showRetry
              ? 'bg-green-500 text-white'
              : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
          }`}
        >
          正常流程
        </button>
        <button
          onClick={() => {
            setShowRetry(true);
            setCurrentStep(0);
          }}
          className={`px-4 py-2 rounded transition-colors ${
            showRetry
              ? 'bg-red-500 text-white'
              : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
          }`}
        >
          错误重试场景
        </button>
      </div>

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
          disabled={currentStep === currentScenario.length - 1}
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
          步骤 {currentStep + 1} / {currentScenario.length}
        </div>
      </div>

      <div
        className="p-6 rounded-lg mb-6 transition-all"
        style={{ backgroundColor: `${getPhaseColor(currentData.phase)}20` }}
      >
        <div className="flex items-center gap-3 mb-3">
          <div
            className="px-4 py-2 rounded-full text-white font-semibold text-sm"
            style={{ backgroundColor: getPhaseColor(currentData.phase) }}
          >
            {getPhaseLabel(currentData.phase)}
          </div>
          <div className="text-sm text-gray-500 dark:text-gray-400">
            Step {currentData.step}
          </div>
        </div>

        <div className="text-gray-800 dark:text-gray-200 mb-4">
          {currentData.content}
        </div>

        {currentData.toolName && (
          <div className="space-y-2">
            <div className="p-3 bg-white dark:bg-gray-800 rounded border border-gray-300 dark:border-gray-600">
              <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">
                工具名称
              </div>
              <div className="font-mono text-sm text-blue-600 dark:text-blue-400">
                {currentData.toolName}
              </div>
            </div>

            {currentData.toolInput && (
              <div className="p-3 bg-white dark:bg-gray-800 rounded border border-gray-300 dark:border-gray-600">
                <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">
                  输入参数
                </div>
                <div className="font-mono text-sm text-gray-700 dark:text-gray-300">
                  {currentData.toolInput}
                </div>
              </div>
            )}

            {currentData.toolOutput && (
              <div
                className={`p-3 rounded border ${
                  currentData.success
                    ? 'bg-green-50 dark:bg-green-900/20 border-green-500'
                    : 'bg-red-50 dark:bg-red-900/20 border-red-500'
                }`}
              >
                <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">
                  {currentData.success ? '✓ 输出结果' : '✗ 错误信息'}
                </div>
                <div className="font-mono text-sm text-gray-700 dark:text-gray-300">
                  {currentData.toolOutput}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      <div className="mb-4 p-4 bg-gray-50 dark:bg-gray-700 rounded">
        <h4 className="font-semibold mb-2 text-gray-800 dark:text-gray-200">
          执行历史
        </h4>
        <div className="space-y-1 max-h-32 overflow-y-auto">
          {currentScenario.slice(0, currentStep + 1).map((step, idx) => (
            <div
              key={idx}
              className={`text-xs p-2 rounded ${
                idx === currentStep
                  ? 'bg-yellow-100 dark:bg-yellow-900/30 font-semibold'
                  : 'bg-white dark:bg-gray-800'
              }`}
            >
              <span
                className="inline-block w-2 h-2 rounded-full mr-2"
                style={{ backgroundColor: getPhaseColor(step.phase) }}
              ></span>
              <span className="text-gray-500 dark:text-gray-400">Step {step.step}:</span>{' '}
              {getPhaseLabel(step.phase)}
              {step.toolName && ` → ${step.toolName}`}
            </div>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-3 gap-3">
        <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded text-center">
          <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
            {currentScenario.filter(s => s.phase === 'thought').length}
          </div>
          <div className="text-xs text-gray-600 dark:text-gray-400 mt-1">
            思考次数
          </div>
        </div>
        <div className="p-3 bg-orange-50 dark:bg-orange-900/20 rounded text-center">
          <div className="text-2xl font-bold text-orange-600 dark:text-orange-400">
            {currentScenario.filter(s => s.phase === 'action').length}
          </div>
          <div className="text-xs text-gray-600 dark:text-gray-400 mt-1">
            工具调用
          </div>
        </div>
        <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded text-center">
          <div className="text-2xl font-bold text-red-600 dark:text-red-400">
            {currentScenario.filter(s => s.phase === 'error').length}
          </div>
          <div className="text-xs text-gray-600 dark:text-gray-400 mt-1">
            错误次数
          </div>
        </div>
      </div>

      <div className="mt-4 p-3 bg-purple-50 dark:bg-purple-900/20 rounded text-sm text-gray-700 dark:text-gray-300">
        <strong>工具调用最佳实践：</strong>
        <ul className="mt-2 space-y-1 ml-4 text-xs">
          <li>• 使用 Pydantic schema 定义工具参数类型</li>
          <li>• 提供清晰的 docstring 和参数描述</li>
          <li>• 实现错误处理和重试机制</li>
          <li>• 验证输入参数防止注入攻击</li>
          <li>• 记录工具调用日志便于调试</li>
        </ul>
      </div>
    </div>
  );
}
