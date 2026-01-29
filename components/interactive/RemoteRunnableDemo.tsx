"use client"

import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Globe, Laptop, ArrowRight, CheckCircle2, Copy, Zap, Shield, Code2 } from 'lucide-react'

interface CodeExample {
  title: string
  local: string
  remote: string
  explanation: string
}

const examples: CodeExample[] = [
  {
    title: "基础调用",
    local: `# 本地链
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
    "翻译成中文: {text}"
)
model = ChatOpenAI()
chain = prompt | model

# 直接调用
result = chain.invoke({"text": "Hello"})
print(result.content)`,
    remote: `# 远程链
from langserve import RemoteRunnable

# 连接到远程服务
chain = RemoteRunnable(
    "http://api.example.com/translator/"
)

# 调用方式完全相同！
result = chain.invoke({"text": "Hello"})
print(result.content)`,
    explanation: "RemoteRunnable 提供与本地链完全相同的接口，无需修改业务代码即可切换。"
  },
  {
    title: "流式输出",
    local: `# 本地流式
chain = prompt | model

for chunk in chain.stream({"text": "Hello"}):
    print(chunk.content, end="", flush=True)`,
    remote: `# 远程流式
chain = RemoteRunnable(
    "http://api.example.com/translator/"
)

for chunk in chain.stream({"text": "Hello"}):
    print(chunk.content, end="", flush=True)`,
    explanation: "流式调用同样保持一致，RemoteRunnable 自动处理 SSE 协议。"
  },
  {
    title: "批量处理",
    local: `# 本地批量
inputs = [
    {"text": "Hello"},
    {"text": "World"},
    {"text": "AI"}
]

results = chain.batch(inputs)
for result in results:
    print(result.content)`,
    remote: `# 远程批量
inputs = [
    {"text": "Hello"},
    {"text": "World"},
    {"text": "AI"}
]

results = chain.batch(inputs)
for result in results:
    print(result.content)`,
    explanation: "批量调用会自动优化网络请求，减少往返次数，提高效率。"
  },
  {
    title: "异步调用",
    local: `# 本地异步
import asyncio

async def process():
    result = await chain.ainvoke(
        {"text": "Hello"}
    )
    print(result.content)

asyncio.run(process())`,
    remote: `# 远程异步
import asyncio

async def process():
    result = await chain.ainvoke(
        {"text": "Hello"}
    )
    print(result.content)

asyncio.run(process())`,
    explanation: "支持 async/await，适用于高并发场景，RemoteRunnable 内部使用 httpx 异步客户端。"
  },
  {
    title: "配置传递",
    local: `# 本地配置
result = chain.invoke(
    {"text": "Hello"},
    config={
        "tags": ["production"],
        "metadata": {"user_id": "123"}
    }
)`,
    remote: `# 远程配置
result = chain.invoke(
    {"text": "Hello"},
    config={
        "tags": ["production"],
        "metadata": {"user_id": "123"}
    }
)`,
    explanation: "配置参数（tags、metadata、callbacks 等）会自动序列化并传递到服务端。"
  }
]

const advantages = [
  {
    icon: Globe,
    title: "统一接口",
    description: "本地和远程使用完全相同的 API，降低学习成本",
    color: "blue"
  },
  {
    icon: Zap,
    title: "自动优化",
    description: "批量请求、连接池、重试机制自动处理",
    color: "green"
  },
  {
    icon: Shield,
    title: "类型安全",
    description: "完整的类型提示和运行时校验",
    color: "purple"
  },
  {
    icon: Code2,
    title: "零重构",
    description: "从本地开发到生产部署无需修改代码",
    color: "orange"
  }
]

export default function RemoteRunnableDemo() {
  const [activeExample, setActiveExample] = useState(0)
  const [copiedSide, setCopiedSide] = useState<'local' | 'remote' | null>(null)
  const [showComparison, setShowComparison] = useState(true)

  const currentExample = examples[activeExample]

  const handleCopy = (code: string, side: 'local' | 'remote') => {
    navigator.clipboard.writeText(code)
    setCopiedSide(side)
    setTimeout(() => setCopiedSide(null), 2000)
  }

  const colorMap = {
    blue: 'from-blue-500 to-cyan-500',
    green: 'from-green-500 to-emerald-500',
    purple: 'from-purple-500 to-pink-500',
    orange: 'from-orange-500 to-red-500'
  }

  return (
    <div className="w-full max-w-7xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-indigo-50 rounded-xl shadow-lg">
      {/* 标题 */}
      <div className="mb-6">
        <h3 className="text-2xl font-bold text-slate-800 mb-2">
          RemoteRunnable 使用演示
        </h3>
        <p className="text-slate-600">
          对比本地链和远程链的调用方式，理解 RemoteRunnable 的核心价值
        </p>
      </div>

      {/* 示例选择器 */}
      <div className="flex gap-2 mb-6 overflow-x-auto pb-2">
        {examples.map((example, index) => (
          <button
            key={example.title}
            onClick={() => setActiveExample(index)}
            className={`px-4 py-2 rounded-lg font-medium transition-all whitespace-nowrap ${
              activeExample === index
                ? 'bg-indigo-600 text-white shadow-lg scale-105'
                : 'bg-white text-slate-700 hover:bg-indigo-50'
            }`}
          >
            {example.title}
          </button>
        ))}
      </div>

      {/* 对比开关 */}
      <div className="flex justify-end mb-4">
        <button
          onClick={() => setShowComparison(!showComparison)}
          className="px-4 py-2 bg-white rounded-lg shadow hover:shadow-md transition-all text-sm font-medium text-slate-700"
        >
          {showComparison ? '隐藏对比' : '显示对比'}
        </button>
      </div>

      <AnimatePresence mode="wait">
        <motion.div
          key={activeExample}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -20 }}
          transition={{ duration: 0.3 }}
        >
          {/* 代码对比 */}
          <div className={`grid ${showComparison ? 'md:grid-cols-2' : 'md:grid-cols-1'} gap-6 mb-6`}>
            {/* 本地链 */}
            {showComparison && (
              <div className="bg-white rounded-lg shadow-md overflow-hidden">
                <div className="bg-gradient-to-r from-slate-700 to-slate-600 px-6 py-4 flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <Laptop className="w-5 h-5 text-white" />
                    <h4 className="font-semibold text-white">本地链</h4>
                  </div>
                  <button
                    onClick={() => handleCopy(currentExample.local, 'local')}
                    className="p-2 hover:bg-slate-600 rounded transition-colors"
                    title="复制代码"
                  >
                    {copiedSide === 'local' ? (
                      <CheckCircle2 className="w-4 h-4 text-green-400" />
                    ) : (
                      <Copy className="w-4 h-4 text-white" />
                    )}
                  </button>
                </div>
                <pre className="bg-slate-900 text-slate-100 p-6 overflow-x-auto text-sm leading-relaxed">
                  {currentExample.local}
                </pre>
              </div>
            )}

            {/* 箭头指示 */}
            {showComparison && (
              <div className="hidden md:flex absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 z-10">
                <motion.div
                  animate={{ x: [0, 10, 0] }}
                  transition={{ repeat: Infinity, duration: 1.5 }}
                  className="bg-white rounded-full p-3 shadow-lg"
                >
                  <ArrowRight className="w-6 h-6 text-indigo-600" />
                </motion.div>
              </div>
            )}

            {/* 远程链 */}
            <div className="bg-white rounded-lg shadow-md overflow-hidden">
              <div className="bg-gradient-to-r from-indigo-600 to-purple-600 px-6 py-4 flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <Globe className="w-5 h-5 text-white" />
                  <h4 className="font-semibold text-white">远程链（RemoteRunnable）</h4>
                </div>
                <button
                  onClick={() => handleCopy(currentExample.remote, 'remote')}
                  className="p-2 hover:bg-indigo-700 rounded transition-colors"
                  title="复制代码"
                >
                  {copiedSide === 'remote' ? (
                    <CheckCircle2 className="w-4 h-4 text-green-400" />
                  ) : (
                    <Copy className="w-4 h-4 text-white" />
                  )}
                </button>
              </div>
              <pre className="bg-slate-900 text-slate-100 p-6 overflow-x-auto text-sm leading-relaxed">
                {currentExample.remote}
              </pre>
            </div>
          </div>

          {/* 解释说明 */}
          <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg p-6 border border-blue-200 mb-6">
            <h5 className="font-semibold text-blue-900 mb-2 flex items-center gap-2">
              <span className="text-xl">📌</span>
              关键要点
            </h5>
            <p className="text-blue-900 leading-relaxed">{currentExample.explanation}</p>
          </div>
        </motion.div>
      </AnimatePresence>

      {/* 核心优势 */}
      <div className="bg-white rounded-lg p-6 shadow-md mb-6">
        <h4 className="text-lg font-bold text-slate-800 mb-4">核心优势</h4>
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
          {advantages.map((advantage, index) => (
            <motion.div
              key={advantage.title}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className="relative overflow-hidden rounded-lg p-4 bg-gradient-to-br from-white to-slate-50 border border-slate-200 hover:shadow-lg transition-shadow"
            >
              <div className={`absolute top-0 right-0 w-20 h-20 bg-gradient-to-br ${colorMap[advantage.color as keyof typeof colorMap]} opacity-10 rounded-full -mr-10 -mt-10`}></div>
              <advantage.icon className={`w-8 h-8 mb-3 bg-gradient-to-r ${colorMap[advantage.color as keyof typeof colorMap]} bg-clip-text text-transparent`} />
              <h5 className="font-semibold text-slate-800 mb-2">{advantage.title}</h5>
              <p className="text-sm text-slate-600 leading-relaxed">{advantage.description}</p>
            </motion.div>
          ))}
        </div>
      </div>

      {/* 网络流程可视化 */}
      <div className="bg-white rounded-lg p-6 shadow-md">
        <h4 className="text-lg font-bold text-slate-800 mb-4">网络请求流程</h4>
        <div className="flex items-center justify-between">
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            className="flex flex-col items-center"
          >
            <div className="w-20 h-20 bg-gradient-to-br from-blue-500 to-blue-600 rounded-full flex items-center justify-center mb-2 shadow-lg">
              <Code2 className="w-10 h-10 text-white" />
            </div>
            <p className="text-sm font-medium text-slate-700">客户端代码</p>
            <p className="text-xs text-slate-500">RemoteRunnable</p>
          </motion.div>

          <motion.div
            animate={{ x: [0, 5, 0] }}
            transition={{ repeat: Infinity, duration: 2 }}
            className="flex-1 mx-4"
          >
            <div className="h-1 bg-gradient-to-r from-blue-500 via-indigo-500 to-purple-500 rounded-full"></div>
            <div className="flex justify-center mt-2">
              <span className="text-xs text-slate-500 bg-white px-2 py-1 rounded shadow">
                HTTP/SSE
              </span>
            </div>
          </motion.div>

          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ delay: 0.2 }}
            className="flex flex-col items-center"
          >
            <div className="w-20 h-20 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-full flex items-center justify-center mb-2 shadow-lg">
              <Globe className="w-10 h-10 text-white" />
            </div>
            <p className="text-sm font-medium text-slate-700">LangServe</p>
            <p className="text-xs text-slate-500">FastAPI 服务</p>
          </motion.div>

          <motion.div
            animate={{ x: [0, 5, 0] }}
            transition={{ repeat: Infinity, duration: 2, delay: 0.5 }}
            className="flex-1 mx-4"
          >
            <div className="h-1 bg-gradient-to-r from-purple-500 via-pink-500 to-red-500 rounded-full"></div>
            <div className="flex justify-center mt-2">
              <span className="text-xs text-slate-500 bg-white px-2 py-1 rounded shadow">
                Local Call
              </span>
            </div>
          </motion.div>

          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ delay: 0.4 }}
            className="flex flex-col items-center"
          >
            <div className="w-20 h-20 bg-gradient-to-br from-green-500 to-emerald-600 rounded-full flex items-center justify-center mb-2 shadow-lg">
              <Laptop className="w-10 h-10 text-white" />
            </div>
            <p className="text-sm font-medium text-slate-700">LangChain</p>
            <p className="text-xs text-slate-500">实际执行链</p>
          </motion.div>
        </div>

        <div className="mt-6 p-4 bg-gradient-to-r from-amber-50 to-yellow-50 rounded-lg border border-amber-200">
          <p className="text-sm text-amber-900 leading-relaxed">
            <strong>透明代理：</strong>RemoteRunnable 作为透明代理，将所有 Runnable 方法（invoke、stream、batch 等）
            映射到对应的 HTTP 端点，实现本地和远程的无缝切换。服务端通过 LangServe 将这些 HTTP 请求还原为本地调用。
          </p>
        </div>
      </div>

      {/* 底部提示 */}
      <div className="mt-6 text-center text-sm text-slate-500">
        <p>
          <code className="px-2 py-1 bg-slate-200 rounded">RemoteRunnable</code> 完全兼容
          <code className="px-2 py-1 bg-slate-200 rounded mx-1">Runnable</code> 协议，
          支持所有标准方法和 LCEL 组合
        </p>
      </div>
    </div>
  )
}
