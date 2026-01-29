"use client";

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface Task {
  id: string;
  name: string;
  duration: number;
  color: string;
  result: string;
}

const parallelTasks: Task[] = [
  { id: 'summary', name: '生成摘要', duration: 2000, color: 'bg-blue-500', result: '文章摘要：LangChain 是一个用于构建 LLM 应用的框架...' },
  { id: 'keywords', name: '提取关键词', duration: 1500, color: 'bg-green-500', result: '关键词：LangChain, LCEL, Agent, RAG' },
  { id: 'sentiment', name: '情感分析', duration: 1000, color: 'bg-purple-500', result: '情感：积极 (85%)' },
  { id: 'translate', name: '翻译为英文', duration: 2500, color: 'bg-orange-500', result: 'Translation: LangChain is a framework...' }
];

type ExecutionMode = 'sequential' | 'parallel';

export default function ParallelExecutionDemo() {
  const [mode, setMode] = useState<ExecutionMode>('sequential');
  const [isRunning, setIsRunning] = useState(false);
  const [completedTasks, setCompletedTasks] = useState<Set<string>>(new Set());
  const [startTime, setStartTime] = useState<number>(0);
  const [endTime, setEndTime] = useState<number>(0);
  const [currentTask, setCurrentTask] = useState<number>(0);

  const runSequential = async () => {
    setIsRunning(true);
    setCompletedTasks(new Set());
    setStartTime(Date.now());
    
    for (let i = 0; i < parallelTasks.length; i++) {
      setCurrentTask(i);
      await new Promise(resolve => setTimeout(resolve, parallelTasks[i].duration));
      setCompletedTasks(prev => new Set([...prev, parallelTasks[i].id]));
    }
    
    setEndTime(Date.now());
    setIsRunning(false);
  };

  const runParallel = async () => {
    setIsRunning(true);
    setCompletedTasks(new Set());
    setStartTime(Date.now());
    
    const promises = parallelTasks.map(task => 
      new Promise(resolve => {
        setTimeout(() => {
          setCompletedTasks(prev => new Set([...prev, task.id]));
          resolve(task.id);
        }, task.duration);
      })
    );
    
    await Promise.all(promises);
    setEndTime(Date.now());
    setIsRunning(false);
  };

  const executeWorkflow = () => {
    if (mode === 'sequential') {
      runSequential();
    } else {
      runParallel();
    }
  };

  const getTotalTime = () => {
    if (mode === 'sequential') {
      return parallelTasks.reduce((sum, task) => sum + task.duration, 0);
    } else {
      return Math.max(...parallelTasks.map(t => t.duration));
    }
  };

  const elapsedTime = endTime > 0 ? endTime - startTime : 0;

  return (
    <div className="w-full max-w-6xl mx-auto p-8 bg-gradient-to-br from-cyan-50 to-blue-50 dark:from-slate-900 dark:to-cyan-900 rounded-2xl border-2 border-cyan-200 dark:border-cyan-700 shadow-xl">
      <div className="text-center mb-8">
        <h3 className="text-3xl font-bold text-slate-800 dark:text-white mb-3">
          并行执行 vs 串行执行
        </h3>
        <p className="text-slate-600 dark:text-slate-300">
          RunnableParallel 性能对比演示
        </p>
      </div>

      {/* Mode Selector */}
      <div className="flex justify-center gap-4 mb-8">
        <button
          onClick={() => setMode('sequential')}
          disabled={isRunning}
          className={`
            px-8 py-4 rounded-xl font-bold text-lg transition-all border-2
            ${mode === 'sequential'
              ? 'bg-gradient-to-r from-orange-500 to-red-500 text-white border-orange-600 shadow-lg'
              : 'bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300 border-slate-300 dark:border-slate-600 hover:border-orange-400'
            }
            disabled:opacity-50 disabled:cursor-not-allowed
          `}
        >
          🔗 串行执行 ({getTotalTime() / 1000}s)
        </button>
        <button
          onClick={() => setMode('parallel')}
          disabled={isRunning}
          className={`
            px-8 py-4 rounded-xl font-bold text-lg transition-all border-2
            ${mode === 'parallel'
              ? 'bg-gradient-to-r from-green-500 to-emerald-500 text-white border-green-600 shadow-lg'
              : 'bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300 border-slate-300 dark:border-slate-600 hover:border-green-400'
            }
            disabled:opacity-50 disabled:cursor-not-allowed
          `}
        >
          ⚡ 并行执行 ({getTotalTime() / 1000}s)
        </button>
      </div>

      {/* Code Preview */}
      <div className="mb-8 p-6 bg-slate-900 rounded-xl overflow-hidden">
        <pre className="text-sm text-green-400">
          <code>{mode === 'sequential' ? `# 串行执行（LCEL 默认）
chain = (
    prompt 
    | model  # 等待完成
    | parser  # 等待完成
)

# 总耗时 = sum(各步骤耗时)` : `# 并行执行（RunnableParallel）
from langchain_core.runnables import RunnableParallel

parallel_chain = RunnableParallel({
    "summary": summary_chain,
    "keywords": keyword_chain,
    "sentiment": sentiment_chain,
    "translate": translate_chain
})

# 总耗时 = max(各步骤耗时)`}</code>
        </pre>
      </div>

      {/* Task Visualization */}
      <div className="mb-8 space-y-4">
        {parallelTasks.map((task, index) => (
          <motion.div
            key={task.id}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.1 }}
            className="relative"
          >
            <div className="flex items-center gap-4">
              {/* Task Number */}
              <div className={`
                w-12 h-12 rounded-xl ${task.color} text-white font-bold text-xl 
                flex items-center justify-center shadow-lg
                ${mode === 'sequential' && isRunning && currentTask === index ? 'animate-pulse' : ''}
              `}>
                {index + 1}
              </div>

              {/* Task Info */}
              <div className="flex-1">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-bold text-slate-800 dark:text-white">
                    {task.name}
                  </span>
                  <span className="text-sm text-slate-600 dark:text-slate-400">
                    {task.duration / 1000}s
                  </span>
                </div>

                {/* Progress Bar */}
                <div className="h-3 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
                  <AnimatePresence>
                    {((mode === 'sequential' && isRunning && currentTask === index) ||
                      (mode === 'parallel' && isRunning && !completedTasks.has(task.id))) && (
                      <motion.div
                        initial={{ width: '0%' }}
                        animate={{ width: '100%' }}
                        transition={{ duration: task.duration / 1000, ease: 'linear' }}
                        className={`h-full ${task.color}`}
                      />
                    )}
                    {completedTasks.has(task.id) && (
                      <motion.div
                        initial={{ width: '0%' }}
                        animate={{ width: '100%' }}
                        className={`h-full ${task.color}`}
                      />
                    )}
                  </AnimatePresence>
                </div>
              </div>

              {/* Status Icon */}
              <div className="w-10 h-10 flex items-center justify-center">
                {completedTasks.has(task.id) ? (
                  <motion.svg
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    className="w-8 h-8 text-green-500"
                    fill="currentColor"
                    viewBox="0 0 20 20"
                  >
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                  </motion.svg>
                ) : isRunning && ((mode === 'sequential' && currentTask === index) || mode === 'parallel') ? (
                  <div className="w-6 h-6 border-3 border-blue-500 border-t-transparent rounded-full animate-spin" />
                ) : null}
              </div>
            </div>

            {/* Result Display */}
            <AnimatePresence>
              {completedTasks.has(task.id) && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="mt-3 ml-16 p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-600"
                >
                  <p className="text-sm text-slate-700 dark:text-slate-300">
                    {task.result}
                  </p>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        ))}
      </div>

      {/* Execute Button */}
      <div className="text-center mb-6">
        <button
          onClick={executeWorkflow}
          disabled={isRunning}
          className={`
            px-12 py-5 rounded-xl font-bold text-xl shadow-lg transition-all
            ${isRunning
              ? 'bg-gray-400 cursor-not-allowed'
              : mode === 'sequential'
                ? 'bg-gradient-to-r from-orange-500 to-red-500 hover:from-orange-600 hover:to-red-600 text-white hover:shadow-xl hover:scale-105'
                : 'bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-600 hover:to-emerald-700 text-white hover:shadow-xl hover:scale-105'
            }
          `}
        >
          {isRunning ? '执行中...' : '▶ 开始执行'}
        </button>
      </div>

      {/* Performance Summary */}
      {elapsedTime > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="p-6 bg-gradient-to-r from-emerald-50 to-cyan-50 dark:from-emerald-900/20 dark:to-cyan-900/20 rounded-xl border-2 border-emerald-300 dark:border-emerald-700"
        >
          <div className="text-center">
            <h4 className="text-2xl font-bold text-emerald-700 dark:text-emerald-300 mb-2">
              🎉 执行完成！
            </h4>
            <div className="text-lg text-emerald-600 dark:text-emerald-400">
              总耗时：<span className="font-bold text-2xl">{(elapsedTime / 1000).toFixed(2)}s</span>
            </div>
            {mode === 'parallel' && (
              <div className="mt-3 text-emerald-600 dark:text-emerald-400">
                相比串行加速：<span className="font-bold text-xl">
                  {((getTotalTime() - getTotalTime()) / getTotalTime() * 100).toFixed(0)}% 
                  ({(parallelTasks.reduce((sum, t) => sum + t.duration, 0) / 1000).toFixed(1)}s → {(getTotalTime() / 1000).toFixed(1)}s)
                </span>
              </div>
            )}
          </div>
        </motion.div>
      )}

      {/* Usage Tips */}
      <div className="mt-6 p-5 bg-blue-50 dark:bg-blue-900/20 border-l-4 border-blue-500 rounded-lg">
        <h4 className="text-sm font-bold text-blue-800 dark:text-blue-300 mb-2 flex items-center gap-2">
          <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
          </svg>
          何时使用并行执行？
        </h4>
        <ul className="text-sm text-blue-700 dark:text-blue-200 space-y-1">
          <li>✓ 多个独立任务（生成摘要 + 提取关键词 + 翻译）</li>
          <li>✓ 调用多个模型（GPT-4 + Claude + Gemini）</li>
          <li>✓ 多路检索（向量搜索 + 全文搜索 + 知识图谱）</li>
          <li>✗ 有依赖关系的任务（必须先 A 后 B）</li>
        </ul>
      </div>
    </div>
  );
}
