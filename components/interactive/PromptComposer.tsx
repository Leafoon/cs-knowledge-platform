"use client";

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

type PromptModule = {
  id: string;
  name: string;
  content: string;
  variables: string[];
};

const predefinedModules: PromptModule[] = [
  {
    id: 'role',
    name: '角色定义',
    content: 'You are a {role} with expertise in {domain}.',
    variables: ['role', 'domain']
  },
  {
    id: 'task',
    name: '任务说明',
    content: 'Task: {task}\nRequirements:\n- {requirement1}\n- {requirement2}',
    variables: ['task', 'requirement1', 'requirement2']
  },
  {
    id: 'format',
    name: '输出格式',
    content: 'Output format: {format}\nExample:\n{example}',
    variables: ['format', 'example']
  },
  {
    id: 'tone',
    name: '语气风格',
    content: 'Communication style: {tone}\nTarget audience: {audience}',
    variables: ['tone', 'audience']
  }
];

export default function PromptComposer() {
  const [selectedModules, setSelectedModules] = useState<string[]>(['role', 'task']);
  const [variables, setVariables] = useState<Record<string, string>>({
    role: 'Data Scientist',
    domain: 'machine learning',
    task: 'Analyze customer data',
    requirement1: 'Clean missing values',
    requirement2: 'Identify trends'
  });

  const toggleModule = (moduleId: string) => {
    setSelectedModules(prev =>
      prev.includes(moduleId)
        ? prev.filter(id => id !== moduleId)
        : [...prev, moduleId]
    );
  };

  const updateVariable = (key: string, value: string) => {
    setVariables(prev => ({ ...prev, [key]: value }));
  };

  // 组合最终提示
  const composedPrompt = selectedModules
    .map(moduleId => {
      const module = predefinedModules.find(m => m.id === moduleId);
      if (!module) return '';
      
      let content = module.content;
      module.variables.forEach(variable => {
        const value = variables[variable] || `{${variable}}`;
        content = content.replace(new RegExp(`\\{${variable}\\}`, 'g'), value);
      });
      
      return content;
    })
    .join('\n\n');

  // 获取所有需要的变量
  const allVariables = new Set<string>();
  selectedModules.forEach(moduleId => {
    const module = predefinedModules.find(m => m.id === moduleId);
    module?.variables.forEach(v => allVariables.add(v));
  });

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-700">
      <h3 className="text-2xl font-bold text-slate-900 dark:text-white mb-4">
        可视化提示组合工具
      </h3>
      
      <p className="text-slate-600 dark:text-slate-400 mb-6">
        通过模块化设计构建复杂提示，实现复用和版本管理
      </p>

      <div className="grid lg:grid-cols-2 gap-6">
        {/* 左侧：模块选择和配置 */}
        <div className="space-y-6">
          {/* 模块库 */}
          <div>
            <h4 className="text-lg font-bold text-slate-900 dark:text-white mb-3">
              提示模块库
            </h4>
            <div className="space-y-2">
              {predefinedModules.map((module) => {
                const isSelected = selectedModules.includes(module.id);
                return (
                  <motion.button
                    key={module.id}
                    onClick={() => toggleModule(module.id)}
                    className={`w-full text-left p-4 rounded-lg border-2 transition-all ${
                      isSelected
                        ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                        : 'border-slate-200 dark:border-slate-700 hover:border-blue-300'
                    }`}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <div className="font-semibold text-slate-900 dark:text-white">
                        {module.name}
                      </div>
                      <div className={`w-5 h-5 rounded border-2 flex items-center justify-center ${
                        isSelected
                          ? 'bg-blue-500 border-blue-500'
                          : 'border-slate-400'
                      }`}>
                        {isSelected && (
                          <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                          </svg>
                        )}
                      </div>
                    </div>
                    <div className="text-xs text-slate-600 dark:text-slate-400 font-mono">
                      {module.content.substring(0, 60)}...
                    </div>
                    <div className="mt-2 flex flex-wrap gap-1">
                      {module.variables.map(v => (
                        <span key={v} className="text-xs px-2 py-0.5 bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300 rounded">
                          {v}
                        </span>
                      ))}
                    </div>
                  </motion.button>
                );
              })}
            </div>
          </div>

          {/* 变量配置 */}
          <div>
            <h4 className="text-lg font-bold text-slate-900 dark:text-white mb-3">
              变量配置
            </h4>
            <div className="space-y-3">
              {Array.from(allVariables).map(variable => (
                <div key={variable}>
                  <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">
                    {variable}
                  </label>
                  <input
                    type="text"
                    value={variables[variable] || ''}
                    onChange={(e) => updateVariable(variable, e.target.value)}
                    className="w-full px-3 py-2 rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 text-slate-900 dark:text-white text-sm"
                    placeholder={`输入 ${variable}...`}
                  />
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* 右侧：组合流程和预览 */}
        <div className="space-y-6">
          {/* 组合流程可视化 */}
          <div>
            <h4 className="text-lg font-bold text-slate-900 dark:text-white mb-3">
              组合流程
            </h4>
            <div className="bg-slate-50 dark:bg-slate-800/50 rounded-lg p-4 border border-slate-200 dark:border-slate-700">
              <AnimatePresence mode="popLayout">
                {selectedModules.length === 0 ? (
                  <div className="text-center text-slate-500 dark:text-slate-400 py-8">
                    请从左侧选择模块
                  </div>
                ) : (
                  <div className="space-y-3">
                    {selectedModules.map((moduleId, idx) => {
                      const module = predefinedModules.find(m => m.id === moduleId);
                      if (!module) return null;
                      
                      return (
                        <React.Fragment key={moduleId}>
                          <motion.div
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            exit={{ opacity: 0, x: 20 }}
                            transition={{ delay: idx * 0.1 }}
                            className="flex items-center gap-3"
                          >
                            <div className="w-8 h-8 rounded-full bg-blue-500 text-white flex items-center justify-center font-bold text-sm">
                              {idx + 1}
                            </div>
                            <div className="flex-1 bg-white dark:bg-slate-900 rounded-lg p-3 border border-slate-200 dark:border-slate-700">
                              <div className="text-sm font-semibold text-slate-900 dark:text-white">
                                {module.name}
                              </div>
                              <div className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                                {module.variables.length} 个变量
                              </div>
                            </div>
                          </motion.div>
                          {idx < selectedModules.length - 1 && (
                            <div className="flex justify-center">
                              <div className="text-blue-400 text-2xl">⬇</div>
                            </div>
                          )}
                        </React.Fragment>
                      );
                    })}
                  </div>
                )}
              </AnimatePresence>
            </div>
          </div>

          {/* 最终提示预览 */}
          <div>
            <h4 className="text-lg font-bold text-slate-900 dark:text-white mb-3">
              组合结果
            </h4>
            <div className="bg-slate-900 dark:bg-slate-950 rounded-lg p-4 border border-slate-700">
              <pre className="text-slate-100 text-sm font-mono whitespace-pre-wrap">
                {composedPrompt || '（空提示）'}
              </pre>
            </div>
          </div>

          {/* 代码示例 */}
          <div>
            <h4 className="text-sm font-semibold text-slate-700 dark:text-slate-300 mb-2">
              Python 代码：
            </h4>
            <pre className="bg-slate-900 dark:bg-slate-950 text-slate-100 rounded-lg p-4 text-xs font-mono overflow-x-auto border border-slate-700">
              <code>{`from langchain.prompts import PromptTemplate
from langchain.prompts.pipeline import PipelinePromptTemplate

# 定义模块
${selectedModules.map(id => {
  const m = predefinedModules.find(m => m.id === id);
  return `${id}_template = PromptTemplate.from_template(
    """${m?.content}"""
)`;
}).join('\n\n')}

# 组合
final_template = PromptTemplate.from_template(
    """${composedPrompt}"""
)

pipeline = PipelinePromptTemplate(
    final_prompt=final_template,
    pipeline_prompts=[
${selectedModules.map(id => `        ("${id}", ${id}_template)`).join(',\n')}
    ]
)`}</code>
            </pre>
          </div>
        </div>
      </div>

      {/* 统计信息 */}
      <div className="mt-6 grid grid-cols-3 gap-4">
        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4 text-center">
          <div className="text-3xl font-bold text-blue-600 dark:text-blue-400">
            {selectedModules.length}
          </div>
          <div className="text-sm text-blue-700 dark:text-blue-300 mt-1">
            已选模块
          </div>
        </div>
        <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4 text-center">
          <div className="text-3xl font-bold text-purple-600 dark:text-purple-400">
            {allVariables.size}
          </div>
          <div className="text-sm text-purple-700 dark:text-purple-300 mt-1">
            变量数量
          </div>
        </div>
        <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4 text-center">
          <div className="text-3xl font-bold text-green-600 dark:text-green-400">
            {composedPrompt.length}
          </div>
          <div className="text-sm text-green-700 dark:text-green-300 mt-1">
            字符数
          </div>
        </div>
      </div>
    </div>
  );
}
