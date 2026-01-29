"use client";

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

type Step = 'input' | 'template' | 'model' | 'parser' | 'output';

const steps: { id: Step; label: string; number: number }[] = [
  { id: 'input', label: '输入数据', number: 1 },
  { id: 'template', label: 'Prompt Template', number: 2 },
  { id: 'model', label: 'Language Model', number: 3 },
  { id: 'parser', label: 'Output Parser', number: 4 },
  { id: 'output', label: '最终输出', number: 5 }
];

const stepDetails: Record<Step, { title: string; description: string; input: string; output: string; code: string }> = {
  input: {
    title: '输入数据',
    description: '用户提供原始请求数据，包含待翻译文本和目标语言。',
    input: '原始请求',
    output: `{
  "text": "Hello",
  "language": "French"
}`,
    code: `# 用户输入
input_data = {
    "text": "Hello",
    "language": "French"
}`
  },
  template: {
    title: 'Prompt Template',
    description: '将输入数据格式化为结构化的提示消息，注入到预定义模板中。',
    input: `{
  "text": "Hello",
  "language": "French"
}`,
    output: `[
  HumanMessage(
    content="Translate to French: Hello"
  )
]`,
    code: `from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_template(
    "Translate to {language}: {text}"
)

messages = template.invoke(input_data)
# 输出: [HumanMessage(content="Translate to French: Hello")]`
  },
  model: {
    title: 'Language Model',
    description: '大语言模型接收格式化后的提示，执行推理并生成响应。',
    input: `[
  HumanMessage(
    content="Translate to French: Hello"
  )
]`,
    output: `AIMessage(
  content="Bonjour"
)`,
    code: `from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4")

ai_message = model.invoke(messages)
# 输出: AIMessage(content="Bonjour")`
  },
  parser: {
    title: 'Output Parser',
    description: '解析 LLM 输出，提取结构化数据或进行格式转换。',
    input: `AIMessage(
  content="Bonjour"
)`,
    output: `{
  "text": "Bonjour",
  "language": "French"
}`,
    code: `from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()

result = parser.invoke(ai_message)
# 输出: "Bonjour"`
  },
  output: {
    title: '最终输出',
    description: '返回经过完整处理流程的最终结果，可供应用程序使用。',
    input: `{
  "text": "Bonjour",
  "language": "French"
}`,
    output: `"Bonjour"`,
    code: `# 完整链式调用
chain = template | model | parser

result = chain.invoke({
    "text": "Hello",
    "language": "French"
})

print(result)  # 输出: "Bonjour"`
  }
};

export default function RunnableCompositionFlow() {
  const [activeStep, setActiveStep] = useState<Step>('input');
  const activeIndex = steps.findIndex(s => s.id === activeStep);
  const detail = stepDetails[activeStep];

  return (
    <div className="w-full max-w-6xl mx-auto p-6 sm:p-8 bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-700 shadow-sm">
      {/* 标题 */}
      <div className="mb-8">
        <h3 className="text-2xl font-bold text-slate-900 dark:text-white mb-2">
          函数组合可视化
        </h3>
        <p className="text-slate-600 dark:text-slate-400">
          理解 LCEL 管道操作符背后的数学原理：
          <code className="mx-2 px-2 py-0.5 bg-slate-100 dark:bg-slate-800 text-blue-600 dark:text-blue-400 rounded text-sm font-mono">
            chain(x) = f₄(f₃(f₂(f₁(x))))
          </code>
        </p>
      </div>

      {/* 流程图 */}
      <div className="mb-8 overflow-x-auto pb-4">
        <div className="flex items-center gap-2 min-w-max">
          {steps.map((step, index) => (
            <React.Fragment key={step.id}>
              {/* 步骤卡片 */}
              <button
                onClick={() => setActiveStep(step.id)}
                className={`
                  group relative flex-shrink-0 w-32 h-32 rounded-lg border-2 transition-all
                  ${activeStep === step.id 
                    ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20 shadow-lg shadow-blue-500/20' 
                    : 'border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 hover:border-blue-400 dark:hover:border-blue-500'
                  }
                `}
              >
                {/* 激活指示器 */}
                {activeStep === step.id && (
                  <motion.div
                    layoutId="activeIndicator"
                    className="absolute -top-1 -right-1 w-3 h-3 bg-blue-500 rounded-full"
                    transition={{ type: "spring", stiffness: 300, damping: 30 }}
                  />
                )}
                
                {/* 内容 */}
                <div className="flex flex-col items-center justify-center h-full p-3">
                  <div className={`
                    w-12 h-12 rounded-full flex items-center justify-center text-white font-bold text-lg mb-2
                    ${activeStep === step.id ? 'bg-blue-500' : 'bg-slate-400 dark:bg-slate-600'}
                  `}>
                    {step.number}
                  </div>
                  <div className={`
                    text-xs font-semibold text-center leading-tight
                    ${activeStep === step.id ? 'text-blue-700 dark:text-blue-300' : 'text-slate-600 dark:text-slate-400'}
                  `}>
                    {step.label}
                  </div>
                </div>
              </button>

              {/* 箭头 */}
              {index < steps.length - 1 && (
                <div className="flex-shrink-0 text-slate-400 dark:text-slate-600">
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                  </svg>
                </div>
              )}
            </React.Fragment>
          ))}
        </div>
      </div>

      {/* 详情面板 */}
      <AnimatePresence mode="wait">
        <motion.div
          key={activeStep}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
          transition={{ duration: 0.2 }}
          className="space-y-6"
        >
          {/* 标题和描述 */}
          <div className="bg-slate-50 dark:bg-slate-800/50 rounded-lg p-5 border border-slate-200 dark:border-slate-700">
            <div className="flex items-center gap-3 mb-3">
              <div className="w-8 h-8 rounded-full bg-blue-500 text-white flex items-center justify-center font-bold">
                {steps.find(s => s.id === activeStep)?.number}
              </div>
              <h4 className="text-lg font-bold text-slate-900 dark:text-white">
                {detail.title}
              </h4>
            </div>
            <p className="text-sm text-slate-600 dark:text-slate-400 leading-relaxed">
              {detail.description}
            </p>
          </div>

          {/* 输入输出 */}
          <div className="grid md:grid-cols-2 gap-4">
            {/* 输入 */}
            <div>
              <div className="flex items-center gap-2 mb-3">
                <div className="w-6 h-6 rounded bg-blue-100 dark:bg-blue-900/30 flex items-center justify-center">
                  <svg className="w-4 h-4 text-blue-600 dark:text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 5l7 7-7 7M5 5l7 7-7 7" />
                  </svg>
                </div>
                <span className="text-sm font-semibold text-slate-700 dark:text-slate-300">输入</span>
              </div>
              <pre className="bg-slate-900 dark:bg-slate-950 text-slate-100 rounded-lg p-4 text-xs font-mono overflow-x-auto border border-slate-700">
{detail.input}
              </pre>
            </div>

            {/* 输出 */}
            <div>
              <div className="flex items-center gap-2 mb-3">
                <div className="w-6 h-6 rounded bg-emerald-100 dark:bg-emerald-900/30 flex items-center justify-center">
                  <svg className="w-4 h-4 text-emerald-600 dark:text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                  </svg>
                </div>
                <span className="text-sm font-semibold text-slate-700 dark:text-slate-300">输出</span>
              </div>
              <pre className="bg-slate-900 dark:bg-slate-950 text-slate-100 rounded-lg p-4 text-xs font-mono overflow-x-auto border border-slate-700">
{detail.output}
              </pre>
            </div>
          </div>

          {/* 代码示例 */}
          <div>
            <div className="flex items-center gap-2 mb-3">
              <div className="w-6 h-6 rounded bg-purple-100 dark:bg-purple-900/30 flex items-center justify-center">
                <svg className="w-4 h-4 text-purple-600 dark:text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
                </svg>
              </div>
              <span className="text-sm font-semibold text-slate-700 dark:text-slate-300">代码实现</span>
            </div>
            <pre className="bg-slate-900 dark:bg-slate-950 text-slate-100 rounded-lg p-4 text-sm font-mono overflow-x-auto border border-slate-700">
              <code className="language-python">
{detail.code}
              </code>
            </pre>
          </div>

          {/* 导航按钮 */}
          <div className="flex items-center justify-between pt-4 border-t border-slate-200 dark:border-slate-700">
            <button
              onClick={() => {
                if (activeIndex > 0) {
                  setActiveStep(steps[activeIndex - 1].id);
                }
              }}
              disabled={activeIndex === 0}
              className="px-4 py-2 rounded-lg font-medium text-sm transition-colors disabled:opacity-30 disabled:cursor-not-allowed
                bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 
                hover:bg-slate-200 dark:hover:bg-slate-700 disabled:hover:bg-slate-100 dark:disabled:hover:bg-slate-800"
            >
              ← 上一步
            </button>
            
            <div className="text-sm text-slate-500 dark:text-slate-400 font-medium">
              {activeIndex + 1} / {steps.length}
            </div>

            <button
              onClick={() => {
                if (activeIndex < steps.length - 1) {
                  setActiveStep(steps[activeIndex + 1].id);
                }
              }}
              disabled={activeIndex === steps.length - 1}
              className="px-4 py-2 rounded-lg font-medium text-sm transition-colors disabled:opacity-30 disabled:cursor-not-allowed
                bg-blue-500 text-white hover:bg-blue-600 disabled:hover:bg-blue-500"
            >
              下一步 →
            </button>
          </div>
        </motion.div>
      </AnimatePresence>
    </div>
  );
}
