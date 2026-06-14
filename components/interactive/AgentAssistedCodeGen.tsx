'use client';

import { useState, useEffect } from 'react';

const flowSteps = [
  {
    id: 'input',
    name: '用户输入',
    icon: '💬',
    color: 'blue',
    description: '描述内核需求和约束',
  },
  {
    id: 'llm',
    name: 'LLM 生成',
    icon: '🤖',
    color: 'purple',
    description: 'AI 生成初始内核代码',
  },
  {
    id: 'compile',
    name: '编译检查',
    icon: '⚙️',
    color: 'yellow',
    description: '验证语法和类型正确性',
  },
  {
    id: 'test',
    name: '正确性测试',
    icon: '✅',
    color: 'green',
    description: '与参考实现对比验证',
  },
  {
    id: 'profile',
    name: '性能分析',
    icon: '📊',
    color: 'orange',
    description: '测量性能指标',
  },
  {
    id: 'optimize',
    name: '优化迭代',
    icon: '🔄',
    color: 'red',
    description: '根据反馈优化内核',
  },
];

const conversationHistory = [
  { role: 'user', content: '实现一个高效的 GEMM 内核，支持 FP16 计算' },
  { role: 'assistant', content: '我将为您生成一个 TileLang GEMM 内核...' },
  { role: 'system', content: '编译成功，开始正确性测试...' },
  { role: 'assistant', content: '测试通过！吞吐量达到 cuBLAS 的 92%' },
  { role: 'system', content: '性能分析完成，发现内存访问可以优化' },
  { role: 'assistant', content: '已优化内存访问模式，性能提升 8%' },
];

export default function AgentAssistedCodeGen() {
  const [currentStep, setCurrentStep] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const [iteration, setIteration] = useState(1);

  const startDemo = () => {
    setIsRunning(true);
    setCurrentStep(0);
    setIteration(1);
  };

  const resetDemo = () => {
    setIsRunning(false);
    setCurrentStep(0);
    setIteration(1);
  };

  useEffect(() => {
    if (!isRunning) return;
    if (currentStep < flowSteps.length - 1) {
      const timer = setTimeout(() => {
        setCurrentStep(prev => prev + 1);
      }, 1500);
      return () => clearTimeout(timer);
    } else {
      setIsRunning(false);
    }
  }, [isRunning, currentStep]);

  return (
    <div className="p-6 bg-gray-900 rounded-xl text-white">
      <h2 className="text-xl font-bold mb-2">Agent 辅助代码生成</h2>
      <p className="text-gray-400 text-sm mb-4">LLM 生成内核 → 测试 → 优化的迭代流程</p>

      <div className="flex items-center gap-4 mb-6">
        <button
          onClick={startDemo}
          disabled={isRunning}
          className={`px-4 py-2 rounded text-sm font-medium transition-all ${
            isRunning ? 'bg-gray-600 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700'
          }`}
        >
          {isRunning ? '运行中...' : '开始演示'}
        </button>
        <button
          onClick={resetDemo}
          className="px-4 py-2 rounded text-sm font-medium bg-gray-700 hover:bg-gray-600"
        >
          重置
        </button>
        <div className="ml-auto text-xs text-gray-400">
          迭代次数: {iteration}
        </div>
      </div>

      <div className="flex items-center justify-between mb-8 px-4">
        {flowSteps.map((step, idx) => {
          const isActive = currentStep === idx;
          const isCompleted = currentStep > idx;

          return (
            <div key={idx} className="flex items-center">
              <div className="flex flex-col items-center">
                <div
                  className={`w-14 h-14 rounded-full flex items-center justify-center text-xl transition-all ${
                    isCompleted
                      ? 'bg-green-600'
                      : isActive
                      ? `bg-${step.color}-600 ring-4 ring-${step.color}-400/30 animate-pulse`
                      : 'bg-gray-700'
                  }`}
                >
                  {isCompleted ? '✓' : step.icon}
                </div>
                <div className={`mt-2 text-[10px] text-center max-w-[80px] ${
                  isActive ? 'text-white' : 'text-gray-500'
                }`}>
                  {step.name}
                </div>
              </div>
              {idx < flowSteps.length - 1 && (
                <div className={`w-8 h-0.5 mx-1 ${
                  isCompleted ? 'bg-green-500' : 'bg-gray-700'
                }`} />
              )}
            </div>
          );
        })}
      </div>

      <div className="grid grid-cols-2 gap-6">
        <div className="bg-gray-800 rounded-lg overflow-hidden">
          <div className="bg-gray-700 px-4 py-2 text-xs font-bold text-gray-300 border-b border-gray-600">
            对话记录
          </div>
          <div className="p-4 space-y-3 max-h-64 overflow-y-auto">
            {conversationHistory.map((msg, idx) => (
              <div
                key={idx}
                className={`flex ${msg.role === 'user' ? 'justify-start' : 'justify-end'}`}
              >
                <div
                  className={`max-w-[80%] rounded-lg px-3 py-2 text-xs ${
                    msg.role === 'user'
                      ? 'bg-blue-600 text-white'
                      : msg.role === 'assistant'
                      ? 'bg-gray-700 text-gray-300'
                      : 'bg-gray-600 text-gray-400 text-[10px]'
                  }`}
                >
                  {msg.content}
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="space-y-4">
          <div className="bg-gray-800 rounded-lg p-4">
            <h3 className="text-sm font-bold text-gray-300 mb-3">当前状态</h3>
            <div className="space-y-2">
              <div className="flex justify-between text-xs">
                <span className="text-gray-400">步骤</span>
                <span className="text-white">{flowSteps[currentStep]?.name || '完成'}</span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-gray-400">描述</span>
                <span className="text-gray-300">{flowSteps[currentStep]?.description || '-'}</span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-gray-400">状态</span>
                <span className={isRunning ? 'text-yellow-400' : 'text-green-400'}>
                  {isRunning ? '运行中' : currentStep >= flowSteps.length - 1 ? '完成' : '等待'}
                </span>
              </div>
            </div>
          </div>

          <div className="bg-gray-800 rounded-lg p-4">
            <h3 className="text-sm font-bold text-gray-300 mb-3">优化历史</h3>
            <div className="space-y-2 text-xs">
              <div className="flex items-center gap-2">
                <span className="text-green-400">✓</span>
                <span className="text-gray-400">初始实现 - 85% cuBLAS</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-green-400">✓</span>
                <span className="text-gray-400">优化内存访问 - 92% cuBLAS</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-yellow-400">→</span>
                <span className="text-gray-400">尝试 Tile 大小调整...</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
