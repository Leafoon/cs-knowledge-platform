"use client";

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Play, Pause, RotateCcw, User, Bot, CheckCircle, XCircle, Clock, AlertTriangle } from 'lucide-react';

type FlowStep = {
  id: string;
  type: 'agent' | 'interrupt' | 'human' | 'decision';
  title: string;
  description: string;
  status: 'pending' | 'active' | 'completed' | 'approved' | 'rejected';
  duration?: number;
};

type Scenario = {
  id: string;
  name: string;
  description: string;
  steps: FlowStep[];
};

const scenarios: Scenario[] = [
  {
    id: 'basic-approval',
    name: '基础审批流程',
    description: '用户请求删除数据，Agent 暂停等待人工审批',
    steps: [
      { id: 's1', type: 'agent', title: '接收请求', description: '用户: "删除用户ID 12345的数据"', status: 'pending', duration: 500 },
      { id: 's2', type: 'agent', title: '分析请求', description: '检测到高风险操作（delete）', status: 'pending', duration: 800 },
      { id: 's3', type: 'interrupt', title: '触发中断', description: '在execute_action前暂停', status: 'pending', duration: 300 },
      { id: 's4', type: 'human', title: '人工审批', description: '等待管理员批准...', status: 'pending', duration: 0 },
      { id: 's5', type: 'decision', title: '审批决策', description: '批准 / 拒绝', status: 'pending', duration: 500 },
      { id: 's6', type: 'agent', title: '执行操作', description: '根据审批结果执行或拒绝', status: 'pending', duration: 1000 }
    ]
  },
  {
    id: 'multi-level',
    name: '多级审批',
    description: '大额申请需要经理、总监、CEO逐级审批',
    steps: [
      { id: 'm1', type: 'agent', title: '接收请求', description: '申请金额: $150,000', status: 'pending', duration: 500 },
      { id: 'm2', type: 'agent', title: '确定审批级别', description: 'amount >= $100k → CEO审批', status: 'pending', duration: 600 },
      { id: 'm3', type: 'interrupt', title: '触发中断', description: '等待CEO审批', status: 'pending', duration: 300 },
      { id: 'm4', type: 'human', title: 'CEO审批', description: '等待CEO决策...', status: 'pending', duration: 0 },
      { id: 'm5', type: 'decision', title: 'CEO决策', description: '批准 / 拒绝', status: 'pending', duration: 500 },
      { id: 'm6', type: 'agent', title: '执行申请', description: '处理申请并通知用户', status: 'pending', duration: 1200 }
    ]
  },
  {
    id: 'timeout',
    name: '审批超时',
    description: '审批超过5秒自动拒绝',
    steps: [
      { id: 't1', type: 'agent', title: '接收请求', description: '敏感操作请求', status: 'pending', duration: 500 },
      { id: 't2', type: 'interrupt', title: '触发中断', description: '等待审批，设置5秒超时', status: 'pending', duration: 300 },
      { id: 't3', type: 'human', title: '等待审批', description: '倒计时: 5s', status: 'pending', duration: 5000 },
      { id: 't4', type: 'agent', title: '超时检查', description: '检测到审批超时', status: 'pending', duration: 400 },
      { id: 't5', type: 'agent', title: '自动拒绝', description: '操作已自动拒绝', status: 'pending', duration: 600 }
    ]
  }
];

export default function HumanInLoopFlow() {
  const [selectedScenario, setSelectedScenario] = useState<Scenario>(scenarios[0]);
  const [currentStepIndex, setCurrentStepIndex] = useState(-1);
  const [isPlaying, setIsPlaying] = useState(false);
  const [approvalDecision, setApprovalDecision] = useState<'approved' | 'rejected' | null>(null);
  const [timeoutCounter, setTimeoutCounter] = useState(5);

  const resetFlow = () => {
    setCurrentStepIndex(-1);
    setIsPlaying(false);
    setApprovalDecision(null);
    setTimeoutCounter(5);
    setSelectedScenario({
      ...selectedScenario,
      steps: selectedScenario.steps.map(s => ({ ...s, status: 'pending' }))
    });
  };

  const handleApproval = (decision: 'approved' | 'rejected') => {
    setApprovalDecision(decision);
    const updatedSteps = [...selectedScenario.steps];
    if (currentStepIndex >= 0 && currentStepIndex < updatedSteps.length) {
      updatedSteps[currentStepIndex].status = decision;
    }
    setSelectedScenario({ ...selectedScenario, steps: updatedSteps });
    
    // 继续播放
    setTimeout(() => {
      playNextStep();
    }, 500);
  };

  const playNextStep = React.useCallback(() => {
    if (currentStepIndex < selectedScenario.steps.length - 1) {
      const nextIndex = currentStepIndex + 1;
      const nextStep = selectedScenario.steps[nextIndex];
      
      // 更新当前步骤为完成
      if (currentStepIndex >= 0) {
        const updatedSteps = [...selectedScenario.steps];
        if (updatedSteps[currentStepIndex].type !== 'human' && updatedSteps[currentStepIndex].status !== 'approved' && updatedSteps[currentStepIndex].status !== 'rejected') {
          updatedSteps[currentStepIndex].status = 'completed';
        }
        setSelectedScenario(prev => ({ ...prev, steps: updatedSteps }));
      }
      
      // 更新下一步为激活
      const updatedSteps = [...selectedScenario.steps];
      updatedSteps[nextIndex].status = 'active';
      setSelectedScenario(prev => ({ ...prev, steps: updatedSteps }));
      setCurrentStepIndex(nextIndex);
      
      // 处理人工审批步骤
      if (nextStep.type === 'human') {
        if (selectedScenario.id === 'timeout') {
          // 超时场景：启动倒计时
          let counter = 5;
          const interval = setInterval(() => {
            counter--;
            setTimeoutCounter(counter);
            if (counter <= 0) {
              clearInterval(interval);
              // 自动继续到超时检查
              setTimeout(() => {
                playNextStep();
              }, 500);
            }
          }, 1000);
        } else {
          // 普通审批：暂停等待人工操作
          setIsPlaying(false);
        }
      } else {
        // 自动步骤：延迟后继续
        const duration = nextStep.duration || 1000;
        setTimeout(() => {
          if (isPlaying || nextIndex === currentStepIndex + 1) {
            playNextStep();
          }
        }, duration);
      }
    } else {
      // 流程结束
      setIsPlaying(false);
      const updatedSteps = [...selectedScenario.steps];
      if (currentStepIndex >= 0) {
        updatedSteps[currentStepIndex].status = 'completed';
      }
      setSelectedScenario(prev => ({ ...prev, steps: updatedSteps }));
    }
  }, [currentStepIndex, selectedScenario, isPlaying]);

  const togglePlay = () => {
    if (currentStepIndex === -1 || currentStepIndex >= selectedScenario.steps.length - 1) {
      resetFlow();
      setIsPlaying(true);
      setCurrentStepIndex(-1);
      setTimeout(playNextStep, 100);
    } else {
      setIsPlaying(!isPlaying);
      if (!isPlaying) {
        playNextStep();
      }
    }
  };

  const getStepIcon = (step: FlowStep) => {
    switch (step.type) {
      case 'agent': return <Bot className="w-5 h-5" />;
      case 'human': return <User className="w-5 h-5" />;
      case 'interrupt': return <AlertTriangle className="w-5 h-5" />;
      case 'decision': return step.status === 'approved' ? <CheckCircle className="w-5 h-5" /> : <Clock className="w-5 h-5" />;
      default: return <Bot className="w-5 h-5" />;
    }
  };

  const getStepColor = (step: FlowStep) => {
    if (step.status === 'approved') return 'bg-green-500';
    if (step.status === 'rejected') return 'bg-red-500';
    if (step.status === 'active') return 'bg-blue-500';
    if (step.status === 'completed') return 'bg-gray-400';
    return 'bg-gray-300';
  };

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-white rounded-lg shadow-lg">
      <div className="mb-6">
        <h3 className="text-2xl font-bold mb-2">人机协作流程（HITL）可视化</h3>
        <p className="text-gray-600">模拟 LangGraph 中断机制、审批流程与超时处理</p>
      </div>

      {/* 场景选择 */}
      <div className="mb-6">
        <label className="block text-sm font-medium mb-2">选择场景</label>
        <div className="grid grid-cols-3 gap-3">
          {scenarios.map(scenario => (
            <button
              key={scenario.id}
              onClick={() => {
                setSelectedScenario(scenario);
                resetFlow();
              }}
              className={`p-3 rounded border-2 text-left transition-all ${
                selectedScenario.id === scenario.id
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-gray-300 hover:border-blue-300'
              }`}
            >
              <div className="font-semibold text-sm">{scenario.name}</div>
              <div className="text-xs text-gray-600 mt-1">{scenario.description}</div>
            </button>
          ))}
        </div>
      </div>

      {/* 控制按钮 */}
      <div className="flex gap-3 mb-6">
        <button
          onClick={togglePlay}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
        >
          {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
          {currentStepIndex === -1 ? '开始' : isPlaying ? '暂停' : '继续'}
        </button>
        <button
          onClick={resetFlow}
          className="flex items-center gap-2 px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700"
        >
          <RotateCcw className="w-4 h-4" />
          重置
        </button>
      </div>

      {/* 流程可视化 */}
      <div className="space-y-4 mb-6">
        {selectedScenario.steps.map((step, index) => (
          <motion.div
            key={step.id}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.1 }}
            className="relative"
          >
            <div className={`flex items-start gap-4 p-4 rounded-lg border-2 ${
              step.status === 'active' ? 'border-blue-500 bg-blue-50' :
              step.status === 'approved' ? 'border-green-500 bg-green-50' :
              step.status === 'rejected' ? 'border-red-500 bg-red-50' :
              step.status === 'completed' ? 'border-gray-400 bg-gray-50' :
              'border-gray-300'
            }`}>
              {/* 图标 */}
              <div className={`flex-shrink-0 w-10 h-10 rounded-full ${getStepColor(step)} text-white flex items-center justify-center`}>
                {getStepIcon(step)}
              </div>

              {/* 内容 */}
              <div className="flex-1">
                <div className="flex items-center justify-between mb-1">
                  <h4 className="font-semibold">{step.title}</h4>
                  <span className="text-xs px-2 py-1 rounded bg-gray-200">{step.type}</span>
                </div>
                <p className="text-sm text-gray-600">{step.description}</p>

                {/* 审批按钮 */}
                {step.type === 'human' && step.status === 'active' && selectedScenario.id !== 'timeout' && (
                  <div className="mt-3 flex gap-2">
                    <button
                      onClick={() => handleApproval('approved')}
                      className="flex items-center gap-1 px-3 py-1 bg-green-600 text-white rounded hover:bg-green-700 text-sm"
                    >
                      <CheckCircle className="w-4 h-4" />
                      批准
                    </button>
                    <button
                      onClick={() => handleApproval('rejected')}
                      className="flex items-center gap-1 px-3 py-1 bg-red-600 text-white rounded hover:bg-red-700 text-sm"
                    >
                      <XCircle className="w-4 h-4" />
                      拒绝
                    </button>
                  </div>
                )}

                {/* 超时倒计时 */}
                {step.type === 'human' && step.status === 'active' && selectedScenario.id === 'timeout' && (
                  <div className="mt-3 flex items-center gap-2 text-orange-600">
                    <Clock className="w-5 h-5" />
                    <span className="font-semibold text-lg">{timeoutCounter}s</span>
                    <span className="text-sm">剩余时间</span>
                  </div>
                )}

                {/* 状态指示 */}
                {step.status === 'approved' && (
                  <div className="mt-2 text-green-700 text-sm font-medium">✅ 已批准</div>
                )}
                {step.status === 'rejected' && (
                  <div className="mt-2 text-red-700 text-sm font-medium">❌ 已拒绝</div>
                )}
              </div>
            </div>

            {/* 连接线 */}
            {index < selectedScenario.steps.length - 1 && (
              <div className="ml-5 h-6 w-0.5 bg-gray-300"></div>
            )}
          </motion.div>
        ))}
      </div>

      {/* 代码示例 */}
      <div className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
        <div className="text-xs font-mono">
          <div className="text-green-400"># 编译时添加中断点</div>
          <div>app = workflow.compile(</div>
          <div className="ml-4">checkpointer=MemorySaver(),</div>
          <div className="ml-4 text-yellow-400">
            interrupt_before=[<span className="text-orange-400">&quot;execute_action&quot;</span>]
          </div>
          <div>)</div>
          <div className="mt-3 text-green-400"># 运行到中断点</div>
          <div>result = app.invoke(initial_input, config)</div>
          <div className="mt-3 text-green-400"># 人工审批后继续</div>
          <div>
            app.update_state(config, &#123;<span className="text-orange-400">&quot;approved&quot;</span>: <span className="text-blue-400">True</span>&#125;)
          </div>
          <div>final_result = app.invoke(<span className="text-blue-400">None</span>, config)</div>
        </div>
      </div>
    </div>
  );
}
