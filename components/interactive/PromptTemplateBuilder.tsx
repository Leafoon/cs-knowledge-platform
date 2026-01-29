"use client";

import React, { useState } from 'react';
import { motion } from 'framer-motion';

interface Variable {
  name: string;
  value: string;
}

const templateExamples = [
  {
    name: '简单字符串模板',
    template: 'Translate {text} to {language}',
    variables: [
      { name: 'text', value: 'Hello, world!' },
      { name: 'language', value: 'French' }
    ]
  },
  {
    name: 'ChatPromptTemplate',
    template: 'System: You are a helpful assistant.\nHuman: {question}\nAI:',
    variables: [
      { name: 'question', value: 'What is LangChain?' }
    ]
  },
  {
    name: 'Few-Shot 模板',
    template: 'Example 1: Input: {ex1_input} → Output: {ex1_output}\nExample 2: Input: {ex2_input} → Output: {ex2_output}\n\nNow translate: {input}',
    variables: [
      { name: 'ex1_input', value: 'hello' },
      { name: 'ex1_output', value: 'bonjour' },
      { name: 'ex2_input', value: 'goodbye' },
      { name: 'ex2_output', value: 'au revoir' },
      { name: 'input', value: 'thank you' }
    ]
  }
];

export default function PromptTemplateBuilder() {
  const [selectedExample, setSelectedExample] = useState(0);
  const [customTemplate, setCustomTemplate] = useState(templateExamples[0].template);
  const [variables, setVariables] = useState<Variable[]>(templateExamples[0].variables);
  const [showCode, setShowCode] = useState(false);

  const handleExampleChange = (index: number) => {
    setSelectedExample(index);
    setCustomTemplate(templateExamples[index].template);
    setVariables(templateExamples[index].variables);
  };

  const updateVariable = (index: number, value: string) => {
    const newVars = [...variables];
    newVars[index].value = value;
    setVariables(newVars);
  };

  const renderPrompt = () => {
    let result = customTemplate;
    variables.forEach(v => {
      result = result.replace(new RegExp(`\\{${v.name}\\}`, 'g'), `<span class="bg-blue-200 dark:bg-blue-700 px-1 rounded font-semibold">${v.value}</span>`);
    });
    return result;
  };

  const generateCode = () => {
    const varNames = variables.map(v => v.name).join(', ');
    return `from langchain_core.prompts import PromptTemplate

template = """${customTemplate}"""

prompt = PromptTemplate.from_template(template)

# 使用变量填充
result = prompt.format(
${variables.map(v => `    ${v.name}="${v.value}"`).join(',\n')}
)

print(result)`;
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-8 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-900 rounded-2xl border-2 border-indigo-200 dark:border-indigo-700 shadow-xl">
      <div className="text-center mb-8">
        <h3 className="text-3xl font-bold text-slate-800 dark:text-white mb-3">
          Prompt 模板构建器
        </h3>
        <p className="text-slate-600 dark:text-slate-300">
          实时预览变量替换与代码生成
        </p>
      </div>

      {/* Example Selector */}
      <div className="mb-6">
        <label className="block text-sm font-semibold text-slate-700 dark:text-slate-300 mb-3">
          选择模板示例：
        </label>
        <div className="grid grid-cols-3 gap-3">
          {templateExamples.map((example, index) => (
            <button
              key={index}
              onClick={() => handleExampleChange(index)}
              className={`
                p-4 rounded-xl font-semibold transition-all border-2
                ${selectedExample === index 
                  ? 'bg-indigo-500 text-white border-indigo-600 shadow-lg' 
                  : 'bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300 border-slate-300 dark:border-slate-600 hover:border-indigo-400'
                }
              `}
            >
              {example.name}
            </button>
          ))}
        </div>
      </div>

      {/* Template Editor */}
      <div className="mb-6">
        <label className="block text-sm font-semibold text-slate-700 dark:text-slate-300 mb-3">
          编辑模板（使用 {'{variable}'} 语法）：
        </label>
        <textarea
          value={customTemplate}
          onChange={(e) => setCustomTemplate(e.target.value)}
          className="w-full h-32 p-4 rounded-xl border-2 border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 text-slate-800 dark:text-white font-mono text-sm resize-none focus:outline-none focus:ring-2 focus:ring-indigo-500"
        />
      </div>

      {/* Variables Input */}
      <div className="mb-6">
        <label className="block text-sm font-semibold text-slate-700 dark:text-slate-300 mb-3">
          变量值：
        </label>
        <div className="grid md:grid-cols-2 gap-4">
          {variables.map((variable, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.05 }}
              className="flex items-center gap-3"
            >
              <span className="text-sm font-semibold text-indigo-600 dark:text-indigo-400 min-w-[100px]">
                {variable.name}:
              </span>
              <input
                type="text"
                value={variable.value}
                onChange={(e) => updateVariable(index, e.target.value)}
                className="flex-1 px-4 py-2 rounded-lg border-2 border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 text-slate-800 dark:text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
              />
            </motion.div>
          ))}
        </div>
      </div>

      {/* Rendered Preview */}
      <div className="mb-6 p-6 bg-white dark:bg-slate-800 rounded-xl border-2 border-green-300 dark:border-green-700">
        <div className="flex items-center justify-between mb-3">
          <h4 className="text-lg font-bold text-slate-800 dark:text-white flex items-center gap-2">
            <svg className="w-5 h-5 text-green-500" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
            </svg>
            渲染结果
          </h4>
          <button
            onClick={() => setShowCode(!showCode)}
            className="px-4 py-2 bg-indigo-100 dark:bg-indigo-900 text-indigo-700 dark:text-indigo-300 rounded-lg font-semibold text-sm hover:bg-indigo-200 dark:hover:bg-indigo-800 transition-colors"
          >
            {showCode ? '隐藏代码' : '查看代码'}
          </button>
        </div>
        <div 
          className="text-slate-700 dark:text-slate-200 whitespace-pre-wrap font-mono text-sm leading-relaxed"
          dangerouslySetInnerHTML={{ __html: renderPrompt() }}
        />
      </div>

      {/* Code Generation */}
      {showCode && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          exit={{ opacity: 0, height: 0 }}
          className="p-6 bg-slate-900 rounded-xl overflow-hidden"
        >
          <pre className="text-sm text-green-400 overflow-x-auto">
            <code>{generateCode()}</code>
          </pre>
        </motion.div>
      )}

      {/* Tips */}
      <div className="mt-6 p-5 bg-amber-50 dark:bg-amber-900/20 border-l-4 border-amber-400 rounded-lg">
        <h4 className="text-sm font-bold text-amber-800 dark:text-amber-300 mb-2 flex items-center gap-2">
          <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
          </svg>
          最佳实践
        </h4>
        <ul className="text-sm text-amber-700 dark:text-amber-200 space-y-1">
          <li>• 使用 <code className="bg-amber-200 dark:bg-amber-800 px-1 rounded">f-string</code> 风格变量：<code>{'{variable_name}'}</code></li>
          <li>• ChatPromptTemplate 支持消息级别模板（System、Human、AI）</li>
          <li>• Few-Shot 模板可自动从示例中学习格式</li>
          <li>• 使用 PromptTemplate.partial() 预填充部分变量</li>
        </ul>
      </div>
    </div>
  );
}
