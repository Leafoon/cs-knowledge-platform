"use client";

import React, { useState } from 'react';
import { motion } from 'framer-motion';

interface BenchmarkResult {
  method: string;
  throughput: number; // requests per second
  latency: number; // ms
  concurrent: number;
  color: string;
}

const benchmarkData: BenchmarkResult[] = [
  { method: 'invoke()', throughput: 5, latency: 200, concurrent: 1, color: 'from-red-500 to-orange-500' },
  { method: 'batch()', throughput: 15, latency: 150, concurrent: 10, color: 'from-orange-500 to-yellow-500' },
  { method: 'ainvoke()', throughput: 20, latency: 100, concurrent: 1, color: 'from-blue-500 to-cyan-500' },
  { method: 'abatch()', throughput: 50, latency: 80, concurrent: 50, color: 'from-green-500 to-emerald-500' },
  { method: 'astream()', throughput: 60, latency: 50, concurrent: 100, color: 'from-purple-500 to-pink-500' }
];

const codeExamples = {
  'invoke()': `# 同步单次调用
result = chain.invoke({"query": "什么是 LangChain?"})
print(result)

# ❌ 阻塞主线程
# ❌ 无法并发
# ✓ 简单直接`,
  'batch()': `# 同步批处理
results = chain.batch([
    {"query": "什么是 LangChain?"},
    {"query": "什么是 LCEL?"},
    {"query": "什么是 Agent?"}
])

# ✓ 内部并发优化
# ❌ 仍会阻塞主线程`,
  'ainvoke()': `# 异步单次调用
import asyncio

async def process():
    result = await chain.ainvoke({"query": "什么是 LangChain?"})
    return result

result = asyncio.run(process())

# ✓ 非阻塞
# ✓ 可与其他异步任务并发
# ✓ 适合 FastAPI/异步框架`,
  'abatch()': `# 异步批处理
import asyncio

async def process_batch():
    results = await chain.abatch([
        {"query": "什么是 LangChain?"},
        {"query": "什么是 LCEL?"},
        {"query": "什么是 Agent?"}
    ])
    return results

results = asyncio.run(process_batch())

# ✓ 最高吞吐量
# ✓ 非阻塞
# ✓ 生产环境首选`,
  'astream()': `# 异步流式输出
import asyncio

async def process_stream():
    async for chunk in chain.astream({"query": "什么是 LangChain?"}):
        print(chunk, end="", flush=True)

asyncio.run(process_stream())

# ✓ 实时响应
# ✓ 最佳用户体验
# ✓ SSE/WebSocket 必备`
};

export default function AsyncPerformanceComparison() {
  const [selectedMethod, setSelectedMethod] = useState('ainvoke()');
  const [showChart, setShowChart] = useState(true);

  const maxThroughput = Math.max(...benchmarkData.map(d => d.throughput));

  return (
    <div className="w-full max-w-6xl mx-auto p-8 bg-gradient-to-br from-violet-50 to-purple-50 dark:from-slate-900 dark:to-violet-900 rounded-2xl border-2 border-violet-200 dark:border-violet-700 shadow-xl">
      <div className="text-center mb-8">
        <h3 className="text-3xl font-bold text-slate-800 dark:text-white mb-3">
          同步 vs 异步性能对比
        </h3>
        <p className="text-slate-600 dark:text-slate-300 text-lg">
          选择正确的调用方式提升 10x 性能
        </p>
      </div>

      {/* Toggle View */}
      <div className="flex justify-center gap-4 mb-8">
        <button
          onClick={() => setShowChart(true)}
          className={`
            px-6 py-3 rounded-xl font-semibold transition-all
            ${showChart
              ? 'bg-violet-500 text-white shadow-lg'
              : 'bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300 border-2 border-slate-300 dark:border-slate-600'
            }
          `}
        >
          📊 性能图表
        </button>
        <button
          onClick={() => setShowChart(false)}
          className={`
            px-6 py-3 rounded-xl font-semibold transition-all
            ${!showChart
              ? 'bg-violet-500 text-white shadow-lg'
              : 'bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300 border-2 border-slate-300 dark:border-slate-600'
            }
          `}
        >
          💻 代码示例
        </button>
      </div>

      {showChart ? (
        <div className="space-y-6">
          {/* Throughput Chart */}
          <div className="p-6 bg-white dark:bg-slate-800 rounded-xl border-2 border-slate-200 dark:border-slate-700">
            <h4 className="text-xl font-bold text-slate-800 dark:text-white mb-6 flex items-center gap-2">
              <svg className="w-6 h-6 text-violet-500" fill="currentColor" viewBox="0 0 20 20">
                <path d="M2 11a1 1 0 011-1h2a1 1 0 011 1v5a1 1 0 01-1 1H3a1 1 0 01-1-1v-5zM8 7a1 1 0 011-1h2a1 1 0 011 1v9a1 1 0 01-1 1H9a1 1 0 01-1-1V7zM14 4a1 1 0 011-1h2a1 1 0 011 1v12a1 1 0 01-1 1h-2a1 1 0 01-1-1V4z" />
              </svg>
              吞吐量对比 (Requests/秒)
            </h4>
            <div className="space-y-4">
              {benchmarkData.map((data, index) => (
                <motion.div
                  key={data.method}
                  initial={{ opacity: 0, x: -50 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="relative"
                >
                  <div className="flex items-center gap-4 mb-2">
                    <span className="w-32 font-semibold text-slate-700 dark:text-slate-300">
                      {data.method}
                    </span>
                    <div className="flex-1 h-10 bg-slate-100 dark:bg-slate-700 rounded-lg overflow-hidden relative">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${(data.throughput / maxThroughput) * 100}%` }}
                        transition={{ duration: 1, delay: index * 0.1 }}
                        className={`h-full bg-gradient-to-r ${data.color} flex items-center justify-end pr-4`}
                      >
                        <span className="text-white font-bold text-sm">
                          {data.throughput} req/s
                        </span>
                      </motion.div>
                    </div>
                  </div>
                  <div className="ml-36 text-xs text-slate-500 dark:text-slate-400">
                    延迟: {data.latency}ms | 并发: {data.concurrent}
                  </div>
                </motion.div>
              ))}
            </div>
          </div>

          {/* Comparison Table */}
          <div className="overflow-x-auto">
            <table className="w-full bg-white dark:bg-slate-800 rounded-xl border-2 border-slate-200 dark:border-slate-700 overflow-hidden">
              <thead className="bg-violet-100 dark:bg-violet-900">
                <tr>
                  <th className="px-6 py-4 text-left font-bold text-slate-800 dark:text-white">方法</th>
                  <th className="px-6 py-4 text-left font-bold text-slate-800 dark:text-white">吞吐量</th>
                  <th className="px-6 py-4 text-left font-bold text-slate-800 dark:text-white">延迟</th>
                  <th className="px-6 py-4 text-left font-bold text-slate-800 dark:text-white">并发数</th>
                  <th className="px-6 py-4 text-left font-bold text-slate-800 dark:text-white">适用场景</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200 dark:divide-slate-700">
                {benchmarkData.map((data, index) => (
                  <motion.tr
                    key={data.method}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: index * 0.05 }}
                    className="hover:bg-violet-50 dark:hover:bg-violet-900/20 transition-colors"
                  >
                    <td className="px-6 py-4">
                      <code className={`px-3 py-1 bg-gradient-to-r ${data.color} text-white rounded font-semibold`}>
                        {data.method}
                      </code>
                    </td>
                    <td className="px-6 py-4 font-semibold text-slate-700 dark:text-slate-300">
                      {data.throughput} req/s
                    </td>
                    <td className="px-6 py-4 text-slate-600 dark:text-slate-400">
                      {data.latency}ms
                    </td>
                    <td className="px-6 py-4 text-slate-600 dark:text-slate-400">
                      {data.concurrent}
                    </td>
                    <td className="px-6 py-4 text-sm text-slate-600 dark:text-slate-400">
                      {index === 0 && '简单脚本、调试'}
                      {index === 1 && '批量离线处理'}
                      {index === 2 && 'FastAPI、异步应用'}
                      {index === 3 && '生产环境、高并发'}
                      {index === 4 && '聊天机器人、实时交互'}
                    </td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ) : (
        <div>
          {/* Method Selector */}
          <div className="grid grid-cols-5 gap-3 mb-6">
            {benchmarkData.map((data) => (
              <button
                key={data.method}
                onClick={() => setSelectedMethod(data.method)}
                className={`
                  px-4 py-3 rounded-xl font-semibold transition-all border-2
                  ${selectedMethod === data.method
                    ? 'bg-gradient-to-r ' + data.color + ' text-white border-transparent shadow-lg scale-105'
                    : 'bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300 border-slate-300 dark:border-slate-600 hover:border-violet-400'
                  }
                `}
              >
                {data.method}
              </button>
            ))}
          </div>

          {/* Code Display */}
          <div className="p-6 bg-slate-900 rounded-xl">
            <pre className="text-sm text-green-400 overflow-x-auto">
              <code>{codeExamples[selectedMethod as keyof typeof codeExamples]}</code>
            </pre>
          </div>
        </div>
      )}

      {/* Recommendations */}
      <div className="mt-8 grid md:grid-cols-2 gap-6">
        <div className="p-6 bg-green-50 dark:bg-green-900/20 border-l-4 border-green-500 rounded-lg">
          <h4 className="text-lg font-bold text-green-800 dark:text-green-300 mb-3 flex items-center gap-2">
            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
            </svg>
            推荐使用
          </h4>
          <ul className="text-sm text-green-700 dark:text-green-200 space-y-2">
            <li>✓ 生产环境：<code className="bg-green-200 dark:bg-green-800 px-1 rounded">abatch()</code> 或 <code className="bg-green-200 dark:bg-green-800 px-1 rounded">astream()</code></li>
            <li>✓ FastAPI 集成：必须使用异步方法</li>
            <li>✓ 流式聊天：<code className="bg-green-200 dark:bg-green-800 px-1 rounded">astream_events()</code></li>
            <li>✓ 大批量处理：<code className="bg-green-200 dark:bg-green-800 px-1 rounded">abatch()</code> + 分批</li>
          </ul>
        </div>

        <div className="p-6 bg-red-50 dark:bg-red-900/20 border-l-4 border-red-500 rounded-lg">
          <h4 className="text-lg font-bold text-red-800 dark:text-red-300 mb-3 flex items-center gap-2">
            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
            </svg>
            避免使用
          </h4>
          <ul className="text-sm text-red-700 dark:text-red-200 space-y-2">
            <li>✗ 生产环境避免：<code className="bg-red-200 dark:bg-red-800 px-1 rounded">invoke()</code> (阻塞)</li>
            <li>✗ 循环调用 <code className="bg-red-200 dark:bg-red-800 px-1 rounded">invoke()</code>：改用 <code className="bg-red-200 dark:bg-red-800 px-1 rounded">batch()</code></li>
            <li>✗ 在 async 函数中使用同步方法</li>
            <li>✗ 未设置并发限制导致 API 限流</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
