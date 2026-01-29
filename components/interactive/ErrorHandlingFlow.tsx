"use client";

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

type ErrorType = 'timeout' | 'rate_limit' | 'api_error' | 'parse_error' | 'validation_error';

interface ErrorScenario {
  type: ErrorType;
  title: string;
  description: string;
  color: string;
  icon: string;
  handlers: {
    strategy: string;
    code: string;
    result: string;
  }[];
}

const errorScenarios: ErrorScenario[] = [
  {
    type: 'timeout',
    title: '超时错误',
    description: 'LLM 响应时间过长',
    color: 'from-orange-500 to-red-500',
    icon: '⏱️',
    handlers: [
      {
        strategy: 'Retry with Timeout',
        code: `chain = (
    prompt 
    | model.with_retry(
        stop_after_attempt=3,
        wait_exponential_multiplier=1
    )
    | parser
)`,
        result: '✓ 指数退避重试 3 次后成功'
      },
      {
        strategy: 'Fallback to Cache',
        code: `from langchain.cache import InMemoryCache
langchain.llm_cache = InMemoryCache()

chain = prompt | model | parser`,
        result: '✓ 从缓存返回历史结果'
      }
    ]
  },
  {
    type: 'rate_limit',
    title: '速率限制',
    description: 'API 调用频率过高',
    color: 'from-yellow-500 to-orange-500',
    icon: '🚦',
    handlers: [
      {
        strategy: 'Rate Limiter',
        code: `from langchain.llms import OpenAI

model = OpenAI(
    max_retries=5,
    request_timeout=60
)

chain = prompt | model | parser`,
        result: '✓ 自动限流并重试'
      },
      {
        strategy: 'Queue System',
        code: `import asyncio
from asyncio import Queue

queue = Queue()
# 每秒最多处理 5 个请求
async def process_with_limit():
    await asyncio.sleep(0.2)
    return await chain.ainvoke(input)`,
        result: '✓ 队列控制并发数'
      }
    ]
  },
  {
    type: 'api_error',
    title: 'API 错误',
    description: 'OpenAI/Anthropic API 返回错误',
    color: 'from-red-500 to-pink-500',
    icon: '⚠️',
    handlers: [
      {
        strategy: 'Fallback Chain',
        code: `fallback_chain = (
    prompt 
    | model.with_fallbacks([
        ChatAnthropic(),
        ChatOpenAI(model="gpt-3.5-turbo")
    ])
    | parser
)`,
        result: '✓ 自动切换到备用模型'
      },
      {
        strategy: 'Error Logging',
        code: `import logging
from langchain.callbacks import StdOutCallbackHandler

chain = (
    prompt 
    | model.with_config({
        "callbacks": [StdOutCallbackHandler()]
    })
    | parser
)`,
        result: '✓ 记录完整错误堆栈'
      }
    ]
  },
  {
    type: 'parse_error',
    title: '解析错误',
    description: 'OutputParser 无法解析 LLM 输出',
    color: 'from-purple-500 to-indigo-500',
    icon: '🔍',
    handlers: [
      {
        strategy: 'OutputFixingParser',
        code: `from langchain.output_parsers import OutputFixingParser

fixing_parser = OutputFixingParser.from_llm(
    parser=original_parser,
    llm=ChatOpenAI()
)

chain = prompt | model | fixing_parser`,
        result: '✓ LLM 自动修复格式错误'
      },
      {
        strategy: 'Retry with Instructions',
        code: `from langchain.output_parsers import RetryWithErrorOutputParser

retry_parser = RetryWithErrorOutputParser.from_llm(
    parser=original_parser,
    llm=model
)

chain = prompt | model | retry_parser`,
        result: '✓ 重新生成符合格式的输出'
      }
    ]
  },
  {
    type: 'validation_error',
    title: '验证错误',
    description: 'Pydantic 模型验证失败',
    color: 'from-blue-500 to-cyan-500',
    icon: '✅',
    handlers: [
      {
        strategy: 'PydanticOutputParser',
        code: `from langchain.output_parsers import PydanticOutputParser

parser = PydanticOutputParser(pydantic_object=MyModel)

prompt = PromptTemplate(
    template="...\\n{format_instructions}",
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)`,
        result: '✓ 提示中包含格式说明'
      },
      {
        strategy: 'with_structured_output',
        code: `from pydantic import BaseModel

class Output(BaseModel):
    name: str
    age: int

chain = (
    prompt 
    | model.with_structured_output(Output)
)`,
        result: '✓ 强制返回结构化对象'
      }
    ]
  }
];

export default function ErrorHandlingFlow() {
  const [selectedError, setSelectedError] = useState<ErrorType>('timeout');
  const [selectedHandler, setSelectedHandler] = useState(0);
  const [isSimulating, setIsSimulating] = useState(false);

  const currentScenario = errorScenarios.find(s => s.type === selectedError)!;

  const simulateError = () => {
    setIsSimulating(true);
    setTimeout(() => setIsSimulating(false), 2000);
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-8 bg-gradient-to-br from-rose-50 to-orange-50 dark:from-slate-900 dark:to-rose-900 rounded-2xl border-2 border-rose-200 dark:border-rose-700 shadow-xl">
      <div className="text-center mb-8">
        <h3 className="text-3xl font-bold text-slate-800 dark:text-white mb-3">
          错误处理与容错策略
        </h3>
        <p className="text-slate-600 dark:text-slate-300">
          生产环境必备的错误处理模式
        </p>
      </div>

      {/* Error Type Selector */}
      <div className="grid grid-cols-5 gap-3 mb-8">
        {errorScenarios.map((scenario) => (
          <motion.button
            key={scenario.type}
            onClick={() => {
              setSelectedError(scenario.type);
              setSelectedHandler(0);
            }}
            className={`
              p-4 rounded-xl border-2 transition-all text-center
              ${selectedError === scenario.type
                ? 'border-rose-500 bg-white dark:bg-slate-800 shadow-lg scale-105'
                : 'border-slate-300 dark:border-slate-600 bg-white/50 dark:bg-slate-800/50 hover:border-rose-400'
              }
            `}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            <div className="text-3xl mb-2">{scenario.icon}</div>
            <div className="text-xs font-semibold text-slate-700 dark:text-slate-300">
              {scenario.title}
            </div>
          </motion.button>
        ))}
      </div>

      {/* Error Description */}
      <motion.div
        key={selectedError}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className={`p-6 rounded-xl bg-gradient-to-r ${currentScenario.color} text-white mb-6 shadow-lg`}
      >
        <div className="flex items-center gap-4 mb-3">
          <span className="text-5xl">{currentScenario.icon}</span>
          <div>
            <h4 className="text-2xl font-bold mb-1">{currentScenario.title}</h4>
            <p className="text-white/90">{currentScenario.description}</p>
          </div>
        </div>
      </motion.div>

      {/* Handler Strategies */}
      <div className="grid md:grid-cols-2 gap-4 mb-6">
        {currentScenario.handlers.map((handler, index) => (
          <motion.div
            key={index}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.1 }}
            onClick={() => setSelectedHandler(index)}
            className={`
              cursor-pointer p-5 rounded-xl border-2 transition-all
              ${selectedHandler === index
                ? 'border-rose-500 bg-white dark:bg-slate-800 shadow-lg'
                : 'border-slate-300 dark:border-slate-600 bg-white/70 dark:bg-slate-800/70 hover:border-rose-400'
              }
            `}
          >
            <h5 className="text-lg font-bold text-slate-800 dark:text-white mb-3 flex items-center gap-2">
              <span className={`w-8 h-8 rounded-lg bg-gradient-to-r ${currentScenario.color} flex items-center justify-center text-white font-bold text-sm`}>
                {index + 1}
              </span>
              {handler.strategy}
            </h5>
            <pre className="text-xs bg-slate-900 text-green-400 p-4 rounded-lg overflow-x-auto mb-3">
              <code>{handler.code}</code>
            </pre>
            <div className="flex items-center gap-2 text-sm text-emerald-600 dark:text-emerald-400 font-semibold">
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
              </svg>
              {handler.result}
            </div>
          </motion.div>
        ))}
      </div>

      {/* Simulation Button */}
      <div className="text-center mb-6">
        <button
          onClick={simulateError}
          disabled={isSimulating}
          className={`
            px-8 py-4 rounded-xl font-bold text-lg shadow-lg transition-all
            ${isSimulating
              ? 'bg-gray-400 cursor-not-allowed'
              : 'bg-gradient-to-r from-rose-500 to-pink-600 hover:from-rose-600 hover:to-pink-700 text-white hover:shadow-xl hover:scale-105'
            }
          `}
        >
          {isSimulating ? '处理中...' : '▶ 模拟错误场景'}
        </button>
      </div>

      <AnimatePresence>
        {isSimulating && (
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            className="p-6 bg-emerald-50 dark:bg-emerald-900/20 border-2 border-emerald-500 rounded-xl"
          >
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 border-4 border-emerald-500 border-t-transparent rounded-full animate-spin" />
              <div>
                <h4 className="text-lg font-bold text-emerald-700 dark:text-emerald-300 mb-1">
                  应用 {currentScenario.handlers[selectedHandler].strategy}
                </h4>
                <p className="text-emerald-600 dark:text-emerald-400">
                  {currentScenario.handlers[selectedHandler].result}
                </p>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Best Practices */}
      <div className="mt-6 p-5 bg-blue-50 dark:bg-blue-900/20 border-l-4 border-blue-500 rounded-lg">
        <h4 className="text-sm font-bold text-blue-800 dark:text-blue-300 mb-3 flex items-center gap-2">
          <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
          </svg>
          生产环境最佳实践
        </h4>
        <ul className="text-sm text-blue-700 dark:text-blue-200 space-y-1.5">
          <li>• 始终使用 <code className="bg-blue-200 dark:bg-blue-800 px-1 rounded">.with_retry()</code> 处理临时性错误</li>
          <li>• 配置 Fallback 链提供备用方案（多模型/缓存）</li>
          <li>• 集成 LangSmith 追踪错误发生位置与频率</li>
          <li>• 使用 OutputFixingParser 自动修复格式错误</li>
          <li>• 设置合理的超时时间（建议 30-60 秒）</li>
        </ul>
      </div>
    </div>
  );
}
