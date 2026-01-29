"use client";

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

type Phase = 'planning' | 'coding' | 'testing' | 'review' | 'complete';

type Agent = {
  id: string;
  name: string;
  role: string;
  color: string;
};

type StepInfo = {
  phase: Phase;
  agent: string;
  action: string;
  output: string;
};

const agents: Agent[] = [
  { id: 'planner', name: 'Planner', role: '架构设计', color: '#8B5CF6' },
  { id: 'coder', name: 'Coder', role: '代码实现', color: '#3B82F6' },
  { id: 'tester', name: 'Tester', role: '测试验证', color: '#10B981' },
  { id: 'reviewer', name: 'Reviewer', role: '代码审查', color: '#F59E0B' },
];

const workflow: StepInfo[] = [
  {
    phase: 'planning',
    agent: 'planner',
    action: '分析需求：实现一个 LRU 缓存',
    output: '设计方案：哈希表 + 双向链表，O(1) 读写',
  },
  {
    phase: 'coding',
    agent: 'coder',
    action: '根据设计实现代码',
    output: '完成 LRUCache 类，包含 get() 和 put() 方法',
  },
  {
    phase: 'testing',
    agent: 'tester',
    action: '编写并运行测试用例',
    output: '发现边界条件 Bug：capacity=1 时未正确淘汰',
  },
  {
    phase: 'coding',
    agent: 'coder',
    action: '修复测试发现的 Bug',
    output: '修正边界条件处理逻辑',
  },
  {
    phase: 'testing',
    agent: 'tester',
    action: '重新运行测试',
    output: '所有测试通过 ✓',
  },
  {
    phase: 'review',
    agent: 'reviewer',
    action: '代码审查',
    output: '建议：添加类型注解，优化变量命名',
  },
  {
    phase: 'coding',
    agent: 'coder',
    action: '根据审查意见优化代码',
    output: '添加类型注解，重命名 _remove → _remove_node',
  },
  {
    phase: 'complete',
    agent: 'reviewer',
    action: '最终确认',
    output: '代码质量达标，可以合并 ✓',
  },
];

export default function MultiAgentCodeGenFlow() {
  const [currentStep, setCurrentStep] = useState(0);
  const [completedSteps, setCompletedSteps] = useState<number[]>([]);
  const [isPlaying, setIsPlaying] = useState(false);

  const executeNextStep = () => {
    if (currentStep >= workflow.length) return;

    setCompletedSteps(prev => [...prev, currentStep]);
    
    setTimeout(() => {
      setCurrentStep(prev => prev + 1);
      if (currentStep + 1 < workflow.length) {
        // 继续自动播放
      } else {
        setIsPlaying(false);
      }
    }, 2000);
  };

  const autoPlay = () => {
    setIsPlaying(true);
    setCurrentStep(0);
    setCompletedSteps([]);
    
    let step = 0;
    const interval = setInterval(() => {
      if (step >= workflow.length) {
        clearInterval(interval);
        setIsPlaying(false);
        return;
      }
      
      setCompletedSteps(prev => [...prev, step]);
      step++;
      setCurrentStep(step);
    }, 2500);
  };

  const reset = () => {
    setIsPlaying(false);
    setCurrentStep(0);
    setCompletedSteps([]);
  };

  const currentPhase = currentStep < workflow.length ? workflow[currentStep].phase : 'complete';

  const getPhaseLabel = (phase: Phase) => {
    const labels: Record<Phase, string> = {
      planning: '规划阶段',
      coding: '编码阶段',
      testing: '测试阶段',
      review: '审查阶段',
      complete: '完成',
    };
    return labels[phase];
  };

  const getAgentColor = (agentId: string) => {
    return agents.find(a => a.id === agentId)?.color || '#6B7280';
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-cyan-50 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold text-center mb-2 text-slate-800">
        Multi-Agent 代码生成流程
      </h3>
      <p className="text-center text-slate-600 mb-6">
        协作式开发：规划 → 编码 → 测试 → 审查 → 迭代改进
      </p>

      {/* 阶段进度条 */}
      <div className="flex items-center justify-between mb-8 bg-white rounded-lg p-4 shadow-md">
        {(['planning', 'coding', 'testing', 'review', 'complete'] as Phase[]).map((phase, index) => {
          const isActive = phase === currentPhase;
          const isPassed = completedSteps.some(s => workflow[s]?.phase === phase);

          return (
            <React.Fragment key={phase}>
              <div className="flex flex-col items-center">
                <div
                  className={`w-12 h-12 rounded-full flex items-center justify-center font-bold text-sm transition-all ${
                    isActive
                      ? 'bg-blue-600 text-white scale-110 shadow-lg'
                      : isPassed
                      ? 'bg-green-500 text-white'
                      : 'bg-slate-200 text-slate-500'
                  }`}
                >
                  {isPassed ? '✓' : index + 1}
                </div>
                <div className="text-xs mt-2 font-medium text-slate-700">
                  {getPhaseLabel(phase)}
                </div>
              </div>
              {index < 4 && (
                <div className="flex-1 h-1 bg-slate-200 mx-2">
                  <div
                    className="h-full bg-blue-600 transition-all"
                    style={{
                      width: isPassed ? '100%' : '0%',
                    }}
                  />
                </div>
              )}
            </React.Fragment>
          );
        })}
      </div>

      {/* Agent 卡片 */}
      <div className="grid grid-cols-4 gap-4 mb-8">
        {agents.map(agent => {
          const isActive =
            currentStep < workflow.length && workflow[currentStep].agent === agent.id;
          const workCount = completedSteps.filter(s => workflow[s].agent === agent.id).length;

          return (
            <motion.div
              key={agent.id}
              animate={{
                scale: isActive ? 1.05 : 1,
                y: isActive ? -5 : 0,
              }}
              className={`bg-white rounded-lg p-4 shadow-md border-2 transition-all ${
                isActive ? 'shadow-xl' : 'shadow-sm'
              }`}
              style={{
                borderColor: isActive ? agent.color : '#E5E7EB',
              }}
            >
              <div
                className="w-12 h-12 rounded-full mx-auto mb-2 flex items-center justify-center text-white font-bold"
                style={{ backgroundColor: agent.color }}
              >
                {agent.name[0]}
              </div>
              <div className="text-center">
                <div className="font-semibold text-sm text-slate-800">{agent.name}</div>
                <div className="text-xs text-slate-500 mt-1">{agent.role}</div>
                <div className="text-xs text-slate-400 mt-2">
                  完成任务: {workCount}
                </div>
              </div>
              {isActive && (
                <div className="mt-2 flex items-center justify-center">
                  <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                  <span className="text-xs text-green-600 ml-1 font-medium">工作中</span>
                </div>
              )}
            </motion.div>
          );
        })}
      </div>

      {/* 工作流时间线 */}
      <div className="bg-white rounded-lg p-6 shadow-md mb-6" style={{ maxHeight: '400px', overflowY: 'auto' }}>
        <h4 className="font-semibold text-lg mb-4 text-slate-800 sticky top-0 bg-white pb-2">
          执行时间线
        </h4>
        <div className="space-y-4">
          {workflow.map((step, index) => {
            const isCompleted = completedSteps.includes(index);
            const isCurrent = index === currentStep && currentStep < workflow.length;
            const agent = agents.find(a => a.id === step.agent);

            return (
              <motion.div
                key={index}
                initial={{ opacity: 0 }}
                animate={{ opacity: isCompleted || isCurrent ? 1 : 0.3 }}
                className="flex gap-4"
              >
                {/* 时间线 */}
                <div className="flex flex-col items-center">
                  <div
                    className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold text-white ${
                      isCompleted ? 'bg-green-500' : isCurrent ? 'bg-blue-600 animate-pulse' : 'bg-slate-300'
                    }`}
                  >
                    {isCompleted ? '✓' : index + 1}
                  </div>
                  {index < workflow.length - 1 && (
                    <div className={`w-0.5 h-16 ${isCompleted ? 'bg-green-500' : 'bg-slate-200'}`} />
                  )}
                </div>

                {/* 内容 */}
                <div className="flex-1 pb-4">
                  <div className="flex items-center gap-2 mb-2">
                    <div
                      className="px-2 py-1 rounded text-xs font-medium text-white"
                      style={{ backgroundColor: agent?.color }}
                    >
                      {agent?.name}
                    </div>
                    <span className="text-xs text-slate-500">{getPhaseLabel(step.phase)}</span>
                  </div>
                  <div className="bg-slate-50 rounded-lg p-3">
                    <div className="text-sm font-medium text-slate-800 mb-1">{step.action}</div>
                    {(isCompleted || isCurrent) && (
                      <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        className="text-xs text-slate-600 mt-2 pt-2 border-t border-slate-200"
                      >
                        → {step.output}
                      </motion.div>
                    )}
                  </div>
                </div>
              </motion.div>
            );
          })}
        </div>
      </div>

      {/* 控制按钮 */}
      <div className="flex justify-center gap-4">
        <button
          onClick={executeNextStep}
          disabled={currentStep >= workflow.length || isPlaying}
          className="px-6 py-2 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 disabled:bg-slate-300 disabled:cursor-not-allowed transition-colors"
        >
          执行下一步
        </button>
        <button
          onClick={autoPlay}
          disabled={isPlaying}
          className="px-6 py-2 bg-green-600 text-white rounded-lg font-medium hover:bg-green-700 disabled:bg-slate-300 disabled:cursor-not-allowed transition-colors"
        >
          {isPlaying ? '自动播放中...' : '自动播放'}
        </button>
        <button
          onClick={reset}
          className="px-6 py-2 bg-slate-600 text-white rounded-lg font-medium hover:bg-slate-700 transition-colors"
        >
          重置
        </button>
      </div>

      {/* 说明 */}
      <div className="mt-6 p-4 bg-indigo-50 rounded-lg border border-indigo-200">
        <div className="text-sm text-slate-700">
          <span className="font-semibold">协作优势：</span>
          每个 Agent 专注自己的专业领域；通过迭代循环（测试 → 修复 → 审查 → 优化）逐步提升代码质量；
          形成闭环反馈机制，确保输出的可靠性。
        </div>
      </div>
    </div>
  );
}
