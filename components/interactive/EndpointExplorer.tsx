"use client"

import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { CheckCircle2, Copy, Play, Loader2 } from 'lucide-react'

interface EndpointConfig {
  name: string
  path: string
  method: string
  description: string
  requestExample: string
  responseExample: string
  curlExample: string
  pythonExample: string
  features: string[]
}

const endpoints: EndpointConfig[] = [
  {
    name: "Invoke",
    path: "/invoke",
    method: "POST",
    description: "同步调用链，等待完整结果返回。适用于需要完整响应的场景。",
    requestExample: `{
  "input": {
    "topic": "人工智能的未来"
  },
  "config": {
    "tags": ["production"]
  }
}`,
    responseExample: `{
  "output": "人工智能的未来将朝着更加通用化、...",
  "metadata": {
    "run_id": "a1b2c3d4-...",
    "feedback_tokens": []
  }
}`,
    curlExample: `curl -X POST "http://localhost:8000/chain/invoke" \\
  -H "Content-Type: application/json" \\
  -d '{
    "input": {"topic": "人工智能的未来"},
    "config": {"tags": ["production"]}
  }'`,
    pythonExample: `from langserve import RemoteRunnable

chain = RemoteRunnable("http://localhost:8000/chain/")
result = chain.invoke({
    "topic": "人工智能的未来"
})
print(result)`,
    features: [
      "同步调用，阻塞等待",
      "返回完整结果",
      "支持配置传递",
      "适合短时任务"
    ]
  },
  {
    name: "Batch",
    path: "/batch",
    method: "POST",
    description: "批量调用链，一次处理多个输入。自动优化批处理性能。",
    requestExample: `{
  "inputs": [
    {"topic": "量子计算"},
    {"topic": "区块链技术"},
    {"topic": "元宇宙"}
  ],
  "config": {
    "max_concurrency": 3
  }
}`,
    responseExample: `{
  "outputs": [
    "量子计算是利用量子力学原理...",
    "区块链技术是一种分布式账本...",
    "元宇宙是一个虚拟的共享空间..."
  ]
}`,
    curlExample: `curl -X POST "http://localhost:8000/chain/batch" \\
  -H "Content-Type: application/json" \\
  -d '{
    "inputs": [
      {"topic": "量子计算"},
      {"topic": "区块链技术"}
    ]
  }'`,
    pythonExample: `from langserve import RemoteRunnable

chain = RemoteRunnable("http://localhost:8000/chain/")
results = chain.batch([
    {"topic": "量子计算"},
    {"topic": "区块链技术"},
    {"topic": "元宇宙"}
])
for result in results:
    print(result)`,
    features: [
      "批量处理多个输入",
      "自动并发优化",
      "减少网络开销",
      "支持并发控制"
    ]
  },
  {
    name: "Stream",
    path: "/stream",
    method: "POST",
    description: "流式返回结果，逐步输出生成内容。适用于长文本生成。",
    requestExample: `{
  "input": {
    "topic": "深度学习的应用"
  }
}`,
    responseExample: `# SSE 流式响应
event: data
data: {"chunk": "深度学习"}

event: data
data: {"chunk": "在计算机视觉"}

event: data
data: {"chunk": "、自然语言处理"}

event: end`,
    curlExample: `curl -X POST "http://localhost:8000/chain/stream" \\
  -H "Content-Type: application/json" \\
  -d '{"input": {"topic": "深度学习的应用"}}' \\
  --no-buffer`,
    pythonExample: `from langserve import RemoteRunnable

chain = RemoteRunnable("http://localhost:8000/chain/")
for chunk in chain.stream({"topic": "深度学习的应用"}):
    print(chunk, end="", flush=True)`,
    features: [
      "实时流式输出",
      "降低首字延迟",
      "支持 SSE 协议",
      "改善用户体验"
    ]
  },
  {
    name: "Stream Events",
    path: "/stream_events",
    method: "POST",
    description: "流式返回详细事件，包含中间步骤和元数据。适用于调试和监控。",
    requestExample: `{
  "input": {
    "query": "什么是 Transformer?"
  },
  "version": "v2"
}`,
    responseExample: `# 事件流
event: metadata
data: {"run_id": "abc123"}

event: on_chat_model_start
data: {"name": "ChatOpenAI"}

event: on_chat_model_stream
data: {"chunk": "Transformer"}

event: on_chat_model_end
data: {"output": {...}}`,
    curlExample: `curl -X POST "http://localhost:8000/chain/stream_events" \\
  -H "Content-Type: application/json" \\
  -d '{
    "input": {"query": "什么是 Transformer?"},
    "version": "v2"
  }'`,
    pythonExample: `from langserve import RemoteRunnable

chain = RemoteRunnable("http://localhost:8000/chain/")
async for event in chain.astream_events(
    {"query": "什么是 Transformer?"},
    version="v2"
):
    print(f"{event['event']}: {event['data']}")`,
    features: [
      "详细事件流",
      "中间步骤可见",
      "支持调试追踪",
      "丰富的元数据"
    ]
  },
  {
    name: "Playground",
    path: "/playground",
    method: "GET",
    description: "Web UI 界面，可视化测试链。提供交互式调试环境。",
    requestExample: `# 直接在浏览器访问
http://localhost:8000/chain/playground`,
    responseExample: `<!DOCTYPE html>
<html>
  <head>
    <title>LangServe Playground</title>
  </head>
  <body>
    <!-- 交互式 UI -->
    <div id="playground">
      <form>
        <input name="topic" />
        <button>Submit</button>
      </form>
      <div id="output"></div>
    </div>
  </body>
</html>`,
    curlExample: `# 在浏览器中打开
open http://localhost:8000/chain/playground

# 或使用 curl 获取 HTML
curl http://localhost:8000/chain/playground`,
    pythonExample: `# 无需代码，直接在浏览器中使用
# 1. 启动服务器: uvicorn server:app
# 2. 访问: http://localhost:8000/chain/playground
# 3. 在 UI 中输入参数并测试`,
    features: [
      "Web UI 界面",
      "可视化测试",
      "无需编码",
      "快速原型验证"
    ]
  }
]

export default function EndpointExplorer() {
  const [activeTab, setActiveTab] = useState(0)
  const [isRunning, setIsRunning] = useState(false)
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null)
  const [codeView, setCodeView] = useState<'curl' | 'python'>('python')

  const currentEndpoint = endpoints[activeTab]

  const handleCopy = (text: string, index: number) => {
    navigator.clipboard.writeText(text)
    setCopiedIndex(index)
    setTimeout(() => setCopiedIndex(null), 2000)
  }

  const handleRun = () => {
    setIsRunning(true)
    setTimeout(() => setIsRunning(false), 2000)
  }

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-blue-50 rounded-xl shadow-lg">
      {/* 标题 */}
      <div className="mb-6">
        <h3 className="text-2xl font-bold text-slate-800 mb-2">
          LangServe 端点探索器
        </h3>
        <p className="text-slate-600">
          交互式学习 LangServe 提供的所有 HTTP 端点及其使用方式
        </p>
      </div>

      {/* 端点选项卡 */}
      <div className="flex gap-2 mb-6 overflow-x-auto pb-2">
        {endpoints.map((endpoint, index) => (
          <button
            key={endpoint.name}
            onClick={() => setActiveTab(index)}
            className={`px-4 py-2 rounded-lg font-medium transition-all whitespace-nowrap ${
              activeTab === index
                ? 'bg-blue-600 text-white shadow-lg scale-105'
                : 'bg-white text-slate-700 hover:bg-blue-50'
            }`}
          >
            {endpoint.name}
            <span className="ml-2 text-xs opacity-75">{endpoint.method}</span>
          </button>
        ))}
      </div>

      <AnimatePresence mode="wait">
        <motion.div
          key={activeTab}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -20 }}
          transition={{ duration: 0.3 }}
          className="space-y-6"
        >
          {/* 端点信息 */}
          <div className="bg-white rounded-lg p-6 shadow-md">
            <div className="flex items-start justify-between mb-4">
              <div>
                <h4 className="text-xl font-bold text-slate-800 mb-2">
                  {currentEndpoint.path}
                </h4>
                <p className="text-slate-600">{currentEndpoint.description}</p>
              </div>
              <span className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-sm font-medium">
                {currentEndpoint.method}
              </span>
            </div>

            {/* 特性列表 */}
            <div className="grid grid-cols-2 gap-3">
              {currentEndpoint.features.map((feature, idx) => (
                <div key={idx} className="flex items-center gap-2">
                  <CheckCircle2 className="w-4 h-4 text-green-600" />
                  <span className="text-sm text-slate-700">{feature}</span>
                </div>
              ))}
            </div>
          </div>

          {/* 请求/响应示例 */}
          <div className="grid md:grid-cols-2 gap-6">
            {/* 请求示例 */}
            <div className="bg-white rounded-lg p-6 shadow-md">
              <div className="flex items-center justify-between mb-4">
                <h5 className="font-semibold text-slate-800">请求示例</h5>
                <button
                  onClick={() => handleCopy(currentEndpoint.requestExample, 0)}
                  className="p-2 hover:bg-slate-100 rounded transition-colors"
                  title="复制代码"
                >
                  {copiedIndex === 0 ? (
                    <CheckCircle2 className="w-4 h-4 text-green-600" />
                  ) : (
                    <Copy className="w-4 h-4 text-slate-600" />
                  )}
                </button>
              </div>
              <pre className="bg-slate-900 text-slate-100 p-4 rounded-lg overflow-x-auto text-sm">
                {currentEndpoint.requestExample}
              </pre>
            </div>

            {/* 响应示例 */}
            <div className="bg-white rounded-lg p-6 shadow-md">
              <div className="flex items-center justify-between mb-4">
                <h5 className="font-semibold text-slate-800">响应示例</h5>
                <button
                  onClick={() => handleCopy(currentEndpoint.responseExample, 1)}
                  className="p-2 hover:bg-slate-100 rounded transition-colors"
                  title="复制代码"
                >
                  {copiedIndex === 1 ? (
                    <CheckCircle2 className="w-4 h-4 text-green-600" />
                  ) : (
                    <Copy className="w-4 h-4 text-slate-600" />
                  )}
                </button>
              </div>
              <pre className="bg-slate-900 text-slate-100 p-4 rounded-lg overflow-x-auto text-sm">
                {currentEndpoint.responseExample}
              </pre>
            </div>
          </div>

          {/* 代码示例 */}
          <div className="bg-white rounded-lg p-6 shadow-md">
            <div className="flex items-center justify-between mb-4">
              <h5 className="font-semibold text-slate-800">客户端调用示例</h5>
              <div className="flex gap-2">
                <button
                  onClick={() => setCodeView('python')}
                  className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
                    codeView === 'python'
                      ? 'bg-blue-600 text-white'
                      : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
                  }`}
                >
                  Python
                </button>
                <button
                  onClick={() => setCodeView('curl')}
                  className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
                    codeView === 'curl'
                      ? 'bg-blue-600 text-white'
                      : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
                  }`}
                >
                  cURL
                </button>
              </div>
            </div>

            <div className="relative">
              <pre className="bg-slate-900 text-slate-100 p-4 rounded-lg overflow-x-auto text-sm">
                {codeView === 'python'
                  ? currentEndpoint.pythonExample
                  : currentEndpoint.curlExample}
              </pre>
              <div className="absolute top-4 right-4 flex gap-2">
                <button
                  onClick={() =>
                    handleCopy(
                      codeView === 'python'
                        ? currentEndpoint.pythonExample
                        : currentEndpoint.curlExample,
                      2
                    )
                  }
                  className="p-2 bg-slate-800 hover:bg-slate-700 rounded transition-colors"
                  title="复制代码"
                >
                  {copiedIndex === 2 ? (
                    <CheckCircle2 className="w-4 h-4 text-green-400" />
                  ) : (
                    <Copy className="w-4 h-4 text-slate-300" />
                  )}
                </button>
                {codeView === 'python' && (
                  <button
                    onClick={handleRun}
                    disabled={isRunning}
                    className="flex items-center gap-2 px-3 py-2 bg-green-600 hover:bg-green-700 disabled:bg-green-400 text-white rounded transition-colors text-sm font-medium"
                    title="模拟运行"
                  >
                    {isRunning ? (
                      <>
                        <Loader2 className="w-4 h-4 animate-spin" />
                        运行中...
                      </>
                    ) : (
                      <>
                        <Play className="w-4 h-4" />
                        运行
                      </>
                    )}
                  </button>
                )}
              </div>
            </div>

            {isRunning && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className="mt-4 p-4 bg-green-50 border border-green-200 rounded-lg"
              >
                <p className="text-sm text-green-800">
                  ✓ 请求成功！查看上方"响应示例"了解返回数据格式。
                </p>
              </motion.div>
            )}
          </div>

          {/* 使用建议 */}
          <div className="bg-gradient-to-r from-amber-50 to-orange-50 rounded-lg p-6 border border-amber-200">
            <h5 className="font-semibold text-amber-900 mb-3 flex items-center gap-2">
              <span className="text-xl">💡</span>
              使用场景建议
            </h5>
            <ul className="space-y-2 text-sm text-amber-900">
              {activeTab === 0 && (
                <>
                  <li>• 短时同步任务（如分类、情感分析）</li>
                  <li>• 需要完整响应后再处理的场景</li>
                  <li>• 简单的 REST API 调用</li>
                </>
              )}
              {activeTab === 1 && (
                <>
                  <li>• 批量数据处理（如批量翻译、摘要）</li>
                  <li>• 需要优化吞吐量的场景</li>
                  <li>• 离线批处理任务</li>
                </>
              )}
              {activeTab === 2 && (
                <>
                  <li>• 长文本生成（如文章、代码生成）</li>
                  <li>• 需要实时反馈的交互式应用</li>
                  <li>• 聊天机器人对话场景</li>
                </>
              )}
              {activeTab === 3 && (
                <>
                  <li>• 复杂链的调试和监控</li>
                  <li>• 需要观察中间步骤的场景</li>
                  <li>• Agent 执行过程追踪</li>
                </>
              )}
              {activeTab === 4 && (
                <>
                  <li>• 快速原型开发和测试</li>
                  <li>• 非技术人员测试链</li>
                  <li>• 演示和教学场景</li>
                </>
              )}
            </ul>
          </div>
        </motion.div>
      </AnimatePresence>

      {/* 底部提示 */}
      <div className="mt-6 text-center text-sm text-slate-500">
        <p>
          所有端点均自动由 <code className="px-2 py-1 bg-slate-200 rounded">add_routes()</code> 生成，
          支持 OpenAPI 规范（访问 <code className="px-2 py-1 bg-slate-200 rounded">/docs</code> 查看完整文档）
        </p>
      </div>
    </div>
  )
}
