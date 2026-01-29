'use client';

import React, { useState } from 'react';

type StepStatus = 'pending' | 'planning' | 'executing' | 'completed' | 'failed';

interface PlanStep {
  id: number;
  description: string;
  tool: string;
  status: StepStatus;
  result?: string;
}

export default function PlanExecuteFlowDiagram() {
  const [currentPhase, setCurrentPhase] = useState<'plan' | 'execute' | 'replan' | 'done'>('plan');
  const [currentStep, setCurrentStep] = useState(0);

  const examplePlan: PlanStep[] = [
    { id: 1, description: '搜索 LangGraph 最新文档', tool: 'search', status: 'pending' },
    { id: 2, description: '分析核心特性', tool: 'analyzer', status: 'pending' },
    { id: 3, description: '生成代码示例', tool: 'code_gen', status: 'pending' },
    { id: 4, description: '撰写技术博客', tool: 'writer', status: 'pending' }
  ];

  const [steps, setSteps] = useState<PlanStep[]>(examplePlan);

  const phases = [
    { key: 'plan' as const, label: '规划阶段', color: 'bg-blue-500', icon: '📋' },
    { key: 'execute' as const, label: '执行阶段', color: 'bg-green-500', icon: '⚙️' },
    { key: 'replan' as const, label: '重新规划', color: 'bg-yellow-500', icon: '🔄' },
    { key: 'done' as const, label: '完成', color: 'bg-purple-500', icon: '✅' }
  ];

  const simulateExecution = () => {
    if (currentPhase === 'plan') {
      setCurrentPhase('execute');
      setCurrentStep(0);
      executeNextStep();
    } else if (currentPhase === 'execute' && currentStep < steps.length - 1) {
      executeNextStep();
    } else if (currentPhase === 'execute' && currentStep >= steps.length - 1) {
      setCurrentPhase('done');
    }
  };

  const executeNextStep = () => {
    const newSteps = [...steps];
    if (currentStep < newSteps.length) {
      newSteps[currentStep].status = 'executing';
      setSteps(newSteps);
      
      setTimeout(() => {
        newSteps[currentStep].status = Math.random() > 0.8 ? 'failed' : 'completed';
        newSteps[currentStep].result = newSteps[currentStep].status === 'completed' 
          ? '✓ 执行成功' 
          : '✗ 执行失败';
        setSteps(newSteps);
        setCurrentStep(currentStep + 1);
      }, 1000);
    }
  };

  const triggerReplan = () => {
    setCurrentPhase('replan');
    setTimeout(() => {
      const newSteps = steps.map(s => ({ ...s, status: 'pending' as StepStatus, result: undefined }));
      setSteps(newSteps);
      setCurrentStep(0);
      setCurrentPhase('plan');
    }, 1500);
  };

  const reset = () => {
    setSteps(examplePlan.map(s => ({ ...s, status: 'pending' as StepStatus, result: undefined })));
    setCurrentStep(0);
    setCurrentPhase('plan');
  };

  return (
    <div className="my-8 p-8 bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-900 rounded-2xl shadow-xl border border-gray-200 dark:border-gray-700">
      <h3 className="text-2xl font-bold mb-4 bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
        Plan-and-Execute 执行流程
      </h3>
      <p className="text-sm text-gray-600 dark:text-gray-400 mb-6">
        演示任务规划、执行、失败重规划的完整流程
      </p>

      {/* 阶段指示器 */}
      <div className="flex items-center justify-between mb-8">
        {phases.map((phase, index) => (
          <React.Fragment key={phase.key}>
            <div className="flex flex-col items-center">
              <div
                className={`w-16 h-16 rounded-full flex items-center justify-center text-2xl transition-all ${
                  currentPhase === phase.key
                    ? `${phase.color} scale-110 shadow-lg`
                    : currentPhase === 'done' || phases.findIndex(p => p.key === currentPhase) > index
                    ? 'bg-green-500'
                    : 'bg-gray-300 dark:bg-gray-600'
                }`}
              >
                {phase.icon}
              </div>
              <span className="mt-2 text-xs font-semibold text-gray-700 dark:text-gray-300">
                {phase.label}
              </span>
            </div>
            {index < phases.length - 1 && (
              <div className="flex-1 h-1 mx-4 bg-gray-300 dark:bg-gray-600 rounded-full overflow-hidden">
                <div
                  className={`h-full transition-all duration-500 ${
                    phases.findIndex(p => p.key === currentPhase) > index ? 'bg-green-500 w-full' : 'w-0'
                  }`}
                />
              </div>
            )}
          </React.Fragment>
        ))}
      </div>

      {/* 执行计划可视化 */}
      <div className="bg-white dark:bg-gray-800 rounded-xl p-6 mb-6 shadow-lg">
        <h4 className="font-bold text-lg mb-4 text-gray-800 dark:text-gray-100">执行计划</h4>
        <div className="space-y-3">
          {steps.map((step, index) => (
            <div
              key={step.id}
              className={`flex items-center gap-4 p-4 rounded-lg border-2 transition-all ${
                step.status === 'executing'
                  ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20 animate-pulse'
                  : step.status === 'completed'
                  ? 'border-green-500 bg-green-50 dark:bg-green-900/20'
                  : step.status === 'failed'
                  ? 'border-red-500 bg-red-50 dark:bg-red-900/20'
                  : 'border-gray-200 dark:border-gray-700'
              }`}
            >
              <div className="flex-shrink-0 w-10 h-10 rounded-full bg-gradient-to-br from-blue-500 to-purple-500 flex items-center justify-center text-white font-bold">
                {step.id}
              </div>
              <div className="flex-1">
                <div className="font-semibold text-gray-800 dark:text-gray-100">{step.description}</div>
                <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                  工具: <span className="font-mono bg-gray-200 dark:bg-gray-700 px-2 py-0.5 rounded">{step.tool}</span>
                </div>
                {step.result && (
                  <div className={`text-sm mt-2 ${step.status === 'completed' ? 'text-green-600' : 'text-red-600'}`}>
                    {step.result}
                  </div>
                )}
              </div>
              <div className="flex-shrink-0">
                {step.status === 'pending' && <span className="text-2xl">⏳</span>}
                {step.status === 'executing' && <span className="text-2xl animate-spin">⚙️</span>}
                {step.status === 'completed' && <span className="text-2xl">✅</span>}
                {step.status === 'failed' && <span className="text-2xl">❌</span>}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* 控制按钮 */}
      <div className="flex gap-3">
        <button
          onClick={simulateExecution}
          disabled={currentPhase === 'done'}
          className="px-6 py-3 bg-gradient-to-r from-blue-500 to-purple-500 text-white rounded-xl font-semibold shadow-lg hover:shadow-xl disabled:opacity-50 disabled:cursor-not-allowed transition-all"
        >
          {currentPhase === 'plan' ? '▶️ 开始执行' : '➡️ 执行下一步'}
        </button>
        
        <button
          onClick={triggerReplan}
          disabled={currentPhase === 'plan' || currentPhase === 'done'}
          className="px-6 py-3 bg-gradient-to-r from-yellow-500 to-orange-500 text-white rounded-xl font-semibold shadow-lg hover:shadow-xl disabled:opacity-50 disabled:cursor-not-allowed transition-all"
        >
          🔄 触发重新规划
        </button>
        
        <button
          onClick={reset}
          className="px-6 py-3 bg-gradient-to-r from-gray-500 to-gray-600 text-white rounded-xl font-semibold shadow-lg hover:shadow-xl transition-all"
        >
          🔁 重置
        </button>
      </div>

      {/* 说明 */}
      <div className="mt-6 p-4 bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-xl border-l-4 border-blue-500">
        <h4 className="font-bold text-gray-800 dark:text-gray-100 mb-2">💡 流程说明</h4>
        <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
          <li><strong>规划阶段:</strong> Planner 分析任务并生成执行计划</li>
          <li><strong>执行阶段:</strong> Executor 按顺序执行每个步骤</li>
          <li><strong>重新规划:</strong> 当步骤失败时，Planner 根据已完成的工作重新制定计划</li>
          <li><strong>完成:</strong> 所有步骤成功执行，输出最终结果</li>
        </ul>
      </div>
    </div>
  );
}
