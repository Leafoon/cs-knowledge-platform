"use client";

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

type Agent = {
  id: string;
  name: string;
  role: string;
  color: string;
};

type Task = {
  id: number;
  description: string;
  assignedTo: string | null;
  status: 'pending' | 'assigned' | 'executing' | 'done';
  result?: string;
};

const agents: Agent[] = [
  { id: 'researcher', name: 'Researcher', role: '搜索与信息收集', color: '#3B82F6' },
  { id: 'analyst', name: 'Analyst', role: '数据分析与处理', color: '#10B981' },
  { id: 'writer', name: 'Writer', role: '内容撰写与编辑', color: '#8B5CF6' },
];

const initialTasks: Task[] = [
  { id: 1, description: '搜索 LangChain 最新文档', assignedTo: null, status: 'pending' },
  { id: 2, description: '分析多 Agent 架构模式', assignedTo: null, status: 'pending' },
  { id: 3, description: '撰写技术博客文章', assignedTo: null, status: 'pending' },
];

export default function SupervisorRoutingFlow() {
  const [tasks, setTasks] = useState<Task[]>(initialTasks);
  const [currentStep, setCurrentStep] = useState(0);
  const [supervisorThinking, setSupervisorThinking] = useState<string>('');

  const steps = [
    {
      task: 1,
      agent: 'researcher',
      thinking: '任务需要搜索能力 → 分配给 Researcher',
      result: '找到官方文档和示例代码',
    },
    {
      task: 2,
      agent: 'analyst',
      thinking: '任务需要分析能力 → 分配给 Analyst',
      result: '识别出 Supervisor、Hierarchical、Collaborative 三种模式',
    },
    {
      task: 3,
      agent: 'writer',
      thinking: '任务需要写作能力 → 分配给 Writer',
      result: '生成结构化博客内容',
    },
  ];

  const executeStep = () => {
    if (currentStep >= steps.length) return;

    const step = steps[currentStep];
    setSupervisorThinking(step.thinking);

    // 更新任务状态：assigned
    setTimeout(() => {
      setTasks(prev => prev.map(t =>
        t.id === step.task ? { ...t, assignedTo: step.agent, status: 'assigned' } : t
      ));
    }, 500);

    // 更新任务状态：executing
    setTimeout(() => {
      setTasks(prev => prev.map(t =>
        t.id === step.task ? { ...t, status: 'executing' } : t
      ));
    }, 1500);

    // 更新任务状态：done
    setTimeout(() => {
      setTasks(prev => prev.map(t =>
        t.id === step.task ? { ...t, status: 'done', result: step.result } : t
      ));
      setSupervisorThinking('');
      setCurrentStep(prev => prev + 1);
    }, 3000);
  };

  const reset = () => {
    setTasks(initialTasks);
    setCurrentStep(0);
    setSupervisorThinking('');
  };

  const getAgentColor = (agentId: string) => {
    return agents.find(a => a.id === agentId)?.color || '#6B7280';
  };

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-blue-50 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold text-center mb-2 text-slate-800">
        Supervisor 路由流程
      </h3>
      <p className="text-center text-slate-600 mb-6">
        中心化调度：Supervisor 分析任务并路由到合适的 Agent
      </p>

      {/* Supervisor 节点 */}
      <div className="flex justify-center mb-8">
        <div className="relative">
          <div className="bg-gradient-to-br from-amber-500 to-orange-500 text-white px-8 py-4 rounded-lg shadow-lg">
            <div className="text-center">
              <div className="text-sm font-semibold uppercase tracking-wide">Supervisor</div>
              <div className="text-xs mt-1">任务调度中心</div>
            </div>
          </div>
          
          {/* Supervisor 思考气泡 */}
          <AnimatePresence>
            {supervisorThinking && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="absolute -top-16 left-1/2 -translate-x-1/2 bg-white px-4 py-2 rounded-lg shadow-md border-2 border-amber-300 whitespace-nowrap"
              >
                <div className="text-xs text-slate-700">{supervisorThinking}</div>
                <div className="absolute -bottom-2 left-1/2 -translate-x-1/2 w-0 h-0 border-l-8 border-l-transparent border-r-8 border-r-transparent border-t-8 border-t-white"></div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>

      {/* 连接线和 Agents */}
      <div className="grid grid-cols-3 gap-6 mb-8">
        {agents.map((agent, index) => {
          const assignedTask = tasks.find(t => t.assignedTo === agent.id);
          const isActive = assignedTask?.status === 'executing';

          return (
            <div key={agent.id} className="flex flex-col items-center">
              {/* 向下的连接线 */}
              <div className="h-12 w-0.5 bg-slate-300 mb-4 relative">
                <AnimatePresence>
                  {assignedTask && assignedTask.status !== 'pending' && (
                    <motion.div
                      initial={{ height: 0 }}
                      animate={{ height: '100%' }}
                      exit={{ height: 0 }}
                      className="w-full absolute top-0 left-0"
                      style={{ backgroundColor: agent.color }}
                    />
                  )}
                </AnimatePresence>
              </div>

              {/* Agent 卡片 */}
              <motion.div
                animate={{
                  scale: isActive ? 1.05 : 1,
                  boxShadow: isActive ? '0 10px 30px rgba(0,0,0,0.2)' : '0 4px 6px rgba(0,0,0,0.1)',
                }}
                className="w-full bg-white rounded-lg p-4 border-2 transition-all"
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
                </div>

                {/* Agent 状态 */}
                {assignedTask && (
                  <div className="mt-3 pt-3 border-t border-slate-200">
                    <div className="flex items-center justify-center gap-2">
                      <div
                        className={`w-2 h-2 rounded-full ${
                          assignedTask.status === 'executing'
                            ? 'bg-green-500 animate-pulse'
                            : assignedTask.status === 'done'
                            ? 'bg-blue-500'
                            : 'bg-yellow-500'
                        }`}
                      />
                      <span className="text-xs font-medium">
                        {assignedTask.status === 'executing'
                          ? '执行中...'
                          : assignedTask.status === 'done'
                          ? '已完成'
                          : '已分配'}
                      </span>
                    </div>
                  </div>
                )}
              </motion.div>
            </div>
          );
        })}
      </div>

      {/* 任务列表 */}
      <div className="bg-white rounded-lg p-6 shadow-md mb-6">
        <h4 className="font-semibold text-lg mb-4 text-slate-800">任务队列</h4>
        <div className="space-y-3">
          {tasks.map(task => {
            const agent = agents.find(a => a.id === task.assignedTo);
            return (
              <motion.div
                key={task.id}
                layout
                className="flex items-center gap-4 p-3 bg-slate-50 rounded-lg border border-slate-200"
              >
                <div className="flex-shrink-0 w-6 h-6 rounded-full bg-slate-200 flex items-center justify-center text-xs font-bold text-slate-600">
                  {task.id}
                </div>
                <div className="flex-1">
                  <div className="text-sm font-medium text-slate-800">{task.description}</div>
                  {task.result && (
                    <div className="text-xs text-slate-600 mt-1">✓ {task.result}</div>
                  )}
                </div>
                <div className="flex-shrink-0">
                  {task.assignedTo ? (
                    <div
                      className="px-3 py-1 rounded-full text-xs font-medium text-white"
                      style={{ backgroundColor: getAgentColor(task.assignedTo) }}
                    >
                      {agent?.name}
                    </div>
                  ) : (
                    <div className="px-3 py-1 rounded-full text-xs font-medium bg-slate-200 text-slate-600">
                      等待分配
                    </div>
                  )}
                </div>
                <div className="flex-shrink-0 w-20 text-right">
                  <span
                    className={`text-xs font-medium ${
                      task.status === 'done'
                        ? 'text-green-600'
                        : task.status === 'executing'
                        ? 'text-blue-600'
                        : task.status === 'assigned'
                        ? 'text-yellow-600'
                        : 'text-slate-400'
                    }`}
                  >
                    {task.status === 'done'
                      ? '✓ 完成'
                      : task.status === 'executing'
                      ? '执行中'
                      : task.status === 'assigned'
                      ? '已分配'
                      : '待处理'}
                  </span>
                </div>
              </motion.div>
            );
          })}
        </div>
      </div>

      {/* 控制按钮 */}
      <div className="flex justify-center gap-4">
        <button
          onClick={executeStep}
          disabled={currentStep >= steps.length}
          className="px-6 py-2 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 disabled:bg-slate-300 disabled:cursor-not-allowed transition-colors"
        >
          {currentStep >= steps.length ? '流程完成' : `执行任务 ${currentStep + 1}`}
        </button>
        <button
          onClick={reset}
          className="px-6 py-2 bg-slate-600 text-white rounded-lg font-medium hover:bg-slate-700 transition-colors"
        >
          重置
        </button>
      </div>

      {/* 说明 */}
      <div className="mt-6 p-4 bg-blue-50 rounded-lg border border-blue-200">
        <div className="text-sm text-slate-700">
          <span className="font-semibold">核心机制：</span>
          Supervisor 分析每个任务的需求，选择最合适的 Agent 执行。所有 Agent 直接向 Supervisor 汇报，无需相互通信。
        </div>
      </div>
    </div>
  );
}
