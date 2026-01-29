'use client';

import React, { useState } from 'react';

type ErrorType = 'parameter' | 'timeout' | 'service' | 'permission';
type RecoveryStrategy = 'retry' | 'fallback' | 'skip' | 'escalate';

interface ErrorScenario {
  id: number;
  error: string;
  type: ErrorType;
  severity: 'low' | 'medium' | 'high';
  primaryAction: RecoveryStrategy;
  fallbackAction?: RecoveryStrategy;
  finalAction?: RecoveryStrategy;
}

export default function ErrorRecoveryFlowDiagram() {
  const [selectedScenario, setSelectedScenario] = useState(0);
  const [recoveryStep, setRecoveryStep] = useState(0);

  const scenarios: ErrorScenario[] = [
    {
      id: 1,
      error: '搜索工具参数错误 (TypeError)',
      type: 'parameter',
      severity: 'medium',
      primaryAction: 'retry',
      fallbackAction: 'fallback',
      finalAction: 'skip'
    },
    {
      id: 2,
      error: 'API 请求超时 (TimeoutError)',
      type: 'timeout',
      severity: 'medium',
      primaryAction: 'retry',
      fallbackAction: 'fallback',
      finalAction: 'escalate'
    },
    {
      id: 3,
      error: '服务暂时不可用 (503)',
      type: 'service',
      severity: 'high',
      primaryAction: 'fallback',
      fallbackAction: 'retry',
      finalAction: 'escalate'
    },
    {
      id: 4,
      error: '权限不足 (403 Forbidden)',
      type: 'permission',
      severity: 'high',
      primaryAction: 'skip',
      fallbackAction: 'escalate'
    }
  ];

  const strategyInfo: Record<RecoveryStrategy, { label: string; icon: string; color: string; description: string }> = {
    retry: {
      label: '重试',
      icon: '🔄',
      color: 'from-blue-500 to-blue-600',
      description: '使用指数退避策略重试（最多3次）'
    },
    fallback: {
      label: '降级',
      icon: '⚠️',
      color: 'from-yellow-500 to-yellow-600',
      description: '切换到备用工具或简化的实现'
    },
    skip: {
      label: '跳过',
      icon: '⏭️',
      color: 'from-gray-500 to-gray-600',
      description: '跳过非关键步骤，继续执行'
    },
    escalate: {
      label: '升级',
      icon: '🆙',
      color: 'from-red-500 to-red-600',
      description: '上报人工或触发告警'
    }
  };

  const current = scenarios[selectedScenario];
  const recoveryChain = [
    current.primaryAction,
    current.fallbackAction,
    current.finalAction
  ].filter(Boolean) as RecoveryStrategy[];

  const nextStep = () => {
    if (recoveryStep < recoveryChain.length - 1) {
      setRecoveryStep(recoveryStep + 1);
    }
  };

  const reset = () => {
    setRecoveryStep(0);
  };

  const selectScenario = (index: number) => {
    setSelectedScenario(index);
    setRecoveryStep(0);
  };

  return (
    <div className="my-8 p-8 bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-900 rounded-2xl shadow-xl border border-gray-200 dark:border-gray-700">
      <h3 className="text-2xl font-bold mb-4 bg-gradient-to-r from-red-600 to-orange-600 bg-clip-text text-transparent">
        Tool Error Recovery：容错机制
      </h3>
      <p className="text-sm text-gray-600 dark:text-gray-400 mb-6">
        演示 Agent 如何从工具调用失败中自动恢复
      </p>

      {/* 场景选择 */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
        {scenarios.map((scenario, index) => (
          <button
            key={scenario.id}
            onClick={() => selectScenario(index)}
            className={`p-4 rounded-xl border-2 transition-all ${
              selectedScenario === index
                ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/30 shadow-lg'
                : 'border-gray-200 dark:border-gray-700 hover:border-blue-300'
            }`}
          >
            <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">场景 {scenario.id}</div>
            <div className="text-sm font-semibold text-gray-800 dark:text-gray-100 mb-2">
              {scenario.error.split('(')[0]}
            </div>
            <div className={`inline-block px-2 py-1 rounded text-xs font-semibold ${
              scenario.severity === 'high'
                ? 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400'
                : scenario.severity === 'medium'
                ? 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400'
                : 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400'
            }`}>
              {scenario.severity}
            </div>
          </button>
        ))}
      </div>

      {/* 错误信息 */}
      <div className="bg-red-50 dark:bg-red-900/20 border-l-4 border-red-500 rounded-lg p-4 mb-6">
        <div className="flex items-start gap-3">
          <span className="text-2xl">❌</span>
          <div>
            <h4 className="font-bold text-red-700 dark:text-red-400 mb-1">错误发生</h4>
            <p className="text-sm text-red-600 dark:text-red-300">{current.error}</p>
            <div className="mt-2 text-xs text-red-500">
              类型: {current.type} | 严重程度: {current.severity}
            </div>
          </div>
        </div>
      </div>

      {/* 恢复流程 */}
      <div className="bg-white dark:bg-gray-800 rounded-xl p-6 mb-6 shadow-lg">
        <h4 className="font-bold text-lg mb-4 text-gray-800 dark:text-gray-100">恢复策略链</h4>
        
        <div className="flex items-center justify-between mb-6">
          {recoveryChain.map((strategy, index) => {
            const info = strategyInfo[strategy];
            const isActive = index === recoveryStep;
            const isPassed = index < recoveryStep;
            
            return (
              <React.Fragment key={index}>
                <div className="flex flex-col items-center">
                  <div
                    className={`w-24 h-24 rounded-xl flex flex-col items-center justify-center transition-all ${
                      isActive
                        ? `bg-gradient-to-br ${info.color} scale-110 shadow-lg text-white`
                        : isPassed
                        ? 'bg-green-500 text-white'
                        : 'bg-gray-200 dark:bg-gray-700 text-gray-500'
                    }`}
                  >
                    <span className="text-3xl mb-1">{info.icon}</span>
                    <span className="text-xs font-bold">{info.label}</span>
                  </div>
                  <div className="mt-2 text-center max-w-[120px]">
                    <p className="text-xs text-gray-600 dark:text-gray-400">
                      {info.description}
                    </p>
                  </div>
                </div>
                {index < recoveryChain.length - 1 && (
                  <div className="flex-1 h-1 mx-4 bg-gray-300 dark:bg-gray-600 rounded-full overflow-hidden">
                    <div
                      className={`h-full transition-all duration-500 ${
                        isPassed ? 'bg-green-500 w-full' : 'w-0'
                      }`}
                    />
                  </div>
                )}
              </React.Fragment>
            );
          })}
        </div>

        {/* 当前策略详情 */}
        {recoveryChain.length > 0 && (
          <div className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-lg p-4">
            <div className="flex items-center gap-3 mb-3">
              <span className="text-3xl">{strategyInfo[recoveryChain[recoveryStep]].icon}</span>
              <div>
                <h5 className="font-bold text-gray-800 dark:text-gray-100">
                  当前策略: {strategyInfo[recoveryChain[recoveryStep]].label}
                </h5>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  {strategyInfo[recoveryChain[recoveryStep]].description}
                </p>
              </div>
            </div>
            
            {recoveryChain[recoveryStep] === 'retry' && (
              <div className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                <div>• 第1次重试: 等待 2秒</div>
                <div>• 第2次重试: 等待 4秒</div>
                <div>• 第3次重试: 等待 8秒</div>
                <div className="text-yellow-600 dark:text-yellow-400 mt-2">
                  ⚠️ 如果3次重试都失败，将执行下一个策略
                </div>
              </div>
            )}
            
            {recoveryChain[recoveryStep] === 'fallback' && (
              <div className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                <div>• 主工具: DuckDuckGo Search</div>
                <div>• 备用工具: Wikipedia Search</div>
                <div className="text-blue-600 dark:text-blue-400 mt-2">
                  💡 尝试使用功能相似的备用工具
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* 控制按钮 */}
      <div className="flex gap-3">
        <button
          onClick={nextStep}
          disabled={recoveryStep >= recoveryChain.length - 1}
          className="px-6 py-3 bg-gradient-to-r from-blue-500 to-purple-500 text-white rounded-xl font-semibold shadow-lg hover:shadow-xl disabled:opacity-50 disabled:cursor-not-allowed transition-all"
        >
          {recoveryStep === 0 ? '▶️ 开始恢复' : '➡️ 下一个策略'}
        </button>
        
        <button
          onClick={reset}
          className="px-6 py-3 bg-gradient-to-r from-gray-500 to-gray-600 text-white rounded-xl font-semibold shadow-lg hover:shadow-xl transition-all"
        >
          🔁 重置
        </button>
      </div>

      {/* 最佳实践 */}
      <div className="mt-6 grid md:grid-cols-2 gap-4">
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-md">
          <h5 className="font-bold text-gray-800 dark:text-gray-100 mb-2">✅ 推荐做法</h5>
          <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
            <li>• 使用指数退避避免过载</li>
            <li>• 为关键工具配置备用方案</li>
            <li>• 记录所有错误和恢复尝试</li>
            <li>• 设置最大重试次数防止死循环</li>
          </ul>
        </div>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-md">
          <h5 className="font-bold text-gray-800 dark:text-gray-100 mb-2">❌ 避免做法</h5>
          <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
            <li>• 无限重试导致资源耗尽</li>
            <li>• 忽略错误继续执行</li>
            <li>• 所有错误使用相同策略</li>
            <li>• 不记录失败原因</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
