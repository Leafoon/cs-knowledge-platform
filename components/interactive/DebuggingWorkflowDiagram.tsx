'use client';

import { useState } from 'react';

const workflowSteps = [
  {
    id: 'issue',
    name: '发现问题',
    icon: '🐛',
    color: 'red',
    description: '内核运行异常或性能不达标',
    actions: ['检查错误日志', '运行单元测试', '对比预期输出'],
  },
  {
    id: 'ir_dump',
    name: 'IR Dump',
    icon: '📄',
    color: 'yellow',
    description: '导出中间表示进行分析',
    actions: ['tile_lang.dump_ir()', '查看 TIR 结构', '检查分块策略'],
  },
  {
    id: 'check_types',
    name: '类型检查',
    icon: '🔍',
    color: 'blue',
    description: '验证数据类型和维度',
    actions: ['检查 dtype 匹配', '验证 shape 一致性', '确认内存布局'],
  },
  {
    id: 'profile',
    name: '性能分析',
    icon: '📊',
    color: 'purple',
    description: '定位性能瓶颈',
    actions: ['NVTX 标记', 'SM 占用率', '内存带宽分析'],
  },
  {
    id: 'fix',
    name: '修复优化',
    icon: '🔧',
    color: 'green',
    description: '根据分析结果优化内核',
    actions: ['调整 Tile 大小', '优化内存访问', '重写调度策略'],
  },
];

export default function DebuggingWorkflowDiagram() {
  const [activeStep, setActiveStep] = useState<number>(0);
  const [completedSteps, setCompletedSteps] = useState<number[]>([]);

  const handleStepClick = (idx: number) => {
    setActiveStep(idx);
  };

  const markCompleted = () => {
    if (!completedSteps.includes(activeStep)) {
      setCompletedSteps([...completedSteps, activeStep]);
    }
    if (activeStep < workflowSteps.length - 1) {
      setActiveStep(activeStep + 1);
    }
  };

  const resetWorkflow = () => {
    setActiveStep(0);
    setCompletedSteps([]);
  };

  return (
    <div className="p-6 bg-gray-900 rounded-xl text-white">
      <h2 className="text-xl font-bold mb-2">调试工作流</h2>
      <p className="text-gray-400 text-sm mb-4">Issue → IR Dump → Check Types → Profile → Fix</p>

      <div className="flex items-center justify-between mb-8 px-4">
        {workflowSteps.map((step, idx) => {
          const isActive = activeStep === idx;
          const isCompleted = completedSteps.includes(idx);

          return (
            <div key={idx} className="flex items-center">
              <div
                className={`relative flex flex-col items-center cursor-pointer transition-all ${
                  isActive ? 'scale-110' : ''
                }`}
                onClick={() => handleStepClick(idx)}
              >
                <div
                  className={`w-16 h-16 rounded-full flex items-center justify-center text-2xl transition-all ${
                    isCompleted
                      ? 'bg-green-600'
                      : isActive
                      ? `bg-${step.color}-600 ring-4 ring-${step.color}-400/30`
                      : 'bg-gray-700 hover:bg-gray-600'
                  }`}
                >
                  {isCompleted ? '✓' : step.icon}
                </div>
                <div className={`mt-2 text-xs font-medium ${
                  isActive ? 'text-white' : 'text-gray-400'
                }`}>
                  {step.name}
                </div>
              </div>
              {idx < workflowSteps.length - 1 && (
                <div className={`w-12 h-0.5 mx-2 ${
                  completedSteps.includes(idx) ? 'bg-green-500' : 'bg-gray-700'
                }`} />
              )}
            </div>
          );
        })}
      </div>

      <div className="bg-gray-800 rounded-lg p-6 mb-6">
        <div className="flex items-center gap-3 mb-4">
          <span className="text-3xl">{workflowSteps[activeStep].icon}</span>
          <div>
            <h3 className={`text-lg font-bold text-${workflowSteps[activeStep].color}-400`}>
              {workflowSteps[activeStep].name}
            </h3>
            <p className="text-sm text-gray-400">{workflowSteps[activeStep].description}</p>
          </div>
        </div>

        <div className="grid grid-cols-3 gap-4">
          {workflowSteps[activeStep].actions.map((action, idx) => (
            <div key={idx} className="bg-gray-700 rounded-lg p-3">
              <div className="flex items-center gap-2">
                <span className={`text-${workflowSteps[activeStep].color}-400`}>●</span>
                <span className="text-sm text-gray-300">{action}</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="flex items-center gap-4">
        <button
          onClick={markCompleted}
          className={`px-4 py-2 rounded text-sm font-medium transition-all ${
            completedSteps.includes(activeStep)
              ? 'bg-green-600 text-white'
              : 'bg-blue-600 hover:bg-blue-700 text-white'
          }`}
        >
          {completedSteps.includes(activeStep) ? '已完成' : '标记完成并继续'}
        </button>
        <button
          onClick={resetWorkflow}
          className="px-4 py-2 rounded text-sm font-medium bg-gray-700 hover:bg-gray-600 text-gray-300"
        >
          重置流程
        </button>
        <div className="ml-auto text-sm text-gray-400">
          进度: {completedSteps.length}/{workflowSteps.length} 步骤完成
        </div>
      </div>
    </div>
  );
}
