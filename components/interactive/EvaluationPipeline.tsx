"use client";

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Play, CheckCircle, XCircle, BarChart3, Database, Zap } from 'lucide-react';

interface EvaluationStep {
  id: number;
  name: string;
  description: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  duration?: number;
  result?: string;
}

const EvaluationPipeline: React.FC = () => {
  const [currentStep, setCurrentStep] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const [steps, setSteps] = useState<EvaluationStep[]>([
    {
      id: 1,
      name: '加载数据集',
      description: '从 LangSmith 加载测试数据集',
      status: 'pending',
    },
    {
      id: 2,
      name: '执行链调用',
      description: '对每个示例运行链',
      status: 'pending',
    },
    {
      id: 3,
      name: '应用评估器',
      description: '使用多个评估器计算分数',
      status: 'pending',
    },
    {
      id: 4,
      name: '聚合结果',
      description: '计算平均分数和统计信息',
      status: 'pending',
    },
    {
      id: 5,
      name: '保存到 LangSmith',
      description: '上传评估结果到云端',
      status: 'pending',
    },
  ]);

  const [results, setResults] = useState({
    totalExamples: 0,
    successCount: 0,
    failCount: 0,
    avgScore: 0,
  });

  const runPipeline = async () => {
    setIsRunning(true);
    setCurrentStep(0);

    const stepDurations = [500, 2000, 1500, 800, 600];
    const stepResults = [
      '已加载 50 条测试样本',
      '成功执行 48/50，失败 2/50',
      '3 个评估器已应用：Correctness, Coherence, Relevance',
      '平均分数: 0.87',
      '实验已保存: exp-2024-01-15-12-34',
    ];

    for (let i = 0; i < steps.length; i++) {
      setCurrentStep(i);
      
      // 设置为运行中
      setSteps(prev =>
        prev.map((step, idx) =>
          idx === i ? { ...step, status: 'running' } : step
        )
      );

      // 模拟执行
      await new Promise(resolve => setTimeout(resolve, stepDurations[i]));

      // 设置为完成
      setSteps(prev =>
        prev.map((step, idx) =>
          idx === i
            ? {
                ...step,
                status: 'completed',
                duration: stepDurations[i],
                result: stepResults[i],
              }
            : step
        )
      );

      // 更新结果统计
      if (i === 1) {
        setResults(prev => ({
          ...prev,
          totalExamples: 50,
          successCount: 48,
          failCount: 2,
        }));
      } else if (i === 3) {
        setResults(prev => ({ ...prev, avgScore: 0.87 }));
      }
    }

    setIsRunning(false);
  };

  const reset = () => {
    setSteps(steps.map(step => ({ ...step, status: 'pending', duration: undefined, result: undefined })));
    setCurrentStep(0);
    setIsRunning(false);
    setResults({ totalExamples: 0, successCount: 0, failCount: 0, avgScore: 0 });
  };

  const getStepIcon = (status: string) => {
    switch (status) {
      case 'running':
        return <Zap className="w-5 h-5 text-yellow-500 animate-pulse" />;
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'failed':
        return <XCircle className="w-5 h-5 text-red-500" />;
      default:
        return <div className="w-5 h-5 rounded-full border-2 border-gray-300" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running':
        return 'border-yellow-500 bg-yellow-50';
      case 'completed':
        return 'border-green-500 bg-green-50';
      case 'failed':
        return 'border-red-500 bg-red-50';
      default:
        return 'border-gray-300 bg-white';
    }
  };

  return (
    <div className="w-full max-w-4xl mx-auto p-6 bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl shadow-lg">
      <div className="mb-6">
        <h3 className="text-2xl font-bold text-gray-800 mb-2 flex items-center gap-2">
          <BarChart3 className="w-6 h-6 text-indigo-600" />
          LangSmith 评估流程可视化
        </h3>
        <p className="text-gray-600">观察 evaluate() 函数的完整执行流程</p>
      </div>

      {/* 控制按钮 */}
      <div className="flex gap-3 mb-6">
        <button
          onClick={runPipeline}
          disabled={isRunning}
          className="flex items-center gap-2 px-6 py-3 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors font-medium"
        >
          <Play className="w-4 h-4" />
          {isRunning ? '运行中...' : '开始评估'}
        </button>
        <button
          onClick={reset}
          disabled={isRunning}
          className="px-6 py-3 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 disabled:bg-gray-100 disabled:cursor-not-allowed transition-colors font-medium"
        >
          重置
        </button>
      </div>

      {/* 评估步骤 */}
      <div className="space-y-4 mb-6">
        {steps.map((step, index) => (
          <motion.div
            key={step.id}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.1 }}
            className={`p-4 rounded-lg border-2 transition-all ${getStatusColor(step.status)}`}
          >
            <div className="flex items-start gap-4">
              <div className="flex-shrink-0 mt-1">{getStepIcon(step.status)}</div>
              <div className="flex-grow">
                <div className="flex items-center justify-between mb-1">
                  <h4 className="font-semibold text-gray-800">
                    步骤 {step.id}: {step.name}
                  </h4>
                  {step.duration && (
                    <span className="text-sm text-gray-500">{step.duration}ms</span>
                  )}
                </div>
                <p className="text-sm text-gray-600 mb-2">{step.description}</p>
                <AnimatePresence>
                  {step.result && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      exit={{ opacity: 0, height: 0 }}
                      className="mt-2 p-2 bg-white rounded border border-gray-200"
                    >
                      <p className="text-sm text-gray-700 font-mono">{step.result}</p>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </div>

            {/* 进度条 */}
            {step.status === 'running' && (
              <motion.div
                className="mt-3 h-1 bg-yellow-500 rounded-full"
                initial={{ width: 0 }}
                animate={{ width: '100%' }}
                transition={{ duration: (step.id === 2 ? 2 : step.id === 3 ? 1.5 : 0.5), ease: 'linear' }}
              />
            )}
          </motion.div>
        ))}
      </div>

      {/* 结果统计 */}
      <AnimatePresence>
        {results.totalExamples > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="grid grid-cols-4 gap-4"
          >
            <div className="p-4 bg-white rounded-lg shadow border border-gray-200">
              <div className="flex items-center gap-2 mb-1">
                <Database className="w-4 h-4 text-blue-500" />
                <p className="text-sm text-gray-600">总样本数</p>
              </div>
              <p className="text-2xl font-bold text-gray-800">{results.totalExamples}</p>
            </div>
            <div className="p-4 bg-white rounded-lg shadow border border-gray-200">
              <div className="flex items-center gap-2 mb-1">
                <CheckCircle className="w-4 h-4 text-green-500" />
                <p className="text-sm text-gray-600">成功</p>
              </div>
              <p className="text-2xl font-bold text-green-600">{results.successCount}</p>
            </div>
            <div className="p-4 bg-white rounded-lg shadow border border-gray-200">
              <div className="flex items-center gap-2 mb-1">
                <XCircle className="w-4 h-4 text-red-500" />
                <p className="text-sm text-gray-600">失败</p>
              </div>
              <p className="text-2xl font-bold text-red-600">{results.failCount}</p>
            </div>
            <div className="p-4 bg-white rounded-lg shadow border border-gray-200">
              <div className="flex items-center gap-2 mb-1">
                <BarChart3 className="w-4 h-4 text-indigo-500" />
                <p className="text-sm text-gray-600">平均分数</p>
              </div>
              <p className="text-2xl font-bold text-indigo-600">{results.avgScore.toFixed(2)}</p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* 说明 */}
      <div className="mt-6 p-4 bg-white rounded-lg border border-blue-200">
        <h4 className="font-semibold text-gray-800 mb-2">💡 工作原理</h4>
        <ul className="text-sm text-gray-600 space-y-1">
          <li>• <strong>步骤 1-2</strong>：加载数据集并对每个样本执行链</li>
          <li>• <strong>步骤 3</strong>：应用多个评估器（LLM-as-Judge、距离度量等）</li>
          <li>• <strong>步骤 4-5</strong>：聚合结果并上传到 LangSmith UI 可视化</li>
        </ul>
      </div>
    </div>
  );
};

export default EvaluationPipeline;
