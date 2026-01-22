"use client";

import { motion } from "framer-motion";
import { useState } from "react";

interface PerformanceData {
  method: string;
  time: number;
  speedup: number;
  color: string;
}

const performanceData: PerformanceData[] = [
  { method: "Pipeline (逐条)", time: 12.34, speedup: 1, color: "from-red-500 to-orange-500" },
  { method: "Pipeline (batch=32)", time: 3.21, speedup: 3.8, color: "from-yellow-500 to-amber-500" },
  { method: "手动批处理", time: 0.89, speedup: 13.9, color: "from-green-500 to-emerald-500" }
];

interface Bottleneck {
  name: string;
  impact: "high" | "medium" | "low";
  description: string;
  emoji: string;
}

const bottlenecks: Bottleneck[] = [
  {
    name: "重复模型加载",
    impact: "high",
    description: "每次调用 pipeline() 都会重新初始化模型",
    emoji: "🔴"
  },
  {
    name: "动态 Padding",
    impact: "medium",
    description: "每个样本的序列长度不同，导致计算效率低",
    emoji: "🟡"
  },
  {
    name: "单样本推理",
    impact: "high",
    description: "无法充分利用 GPU 并行计算能力",
    emoji: "🔴"
  },
  {
    name: "Python 循环",
    impact: "low",
    description: "for 循环开销，而非向量化操作",
    emoji: "🟢"
  }
];

export default function PipelinePerformanceAnalyzer() {
  const [selectedMethod, setSelectedMethod] = useState<string | null>(null);

  const maxTime = Math.max(...performanceData.map(d => d.time));

  const getImpactColor = (impact: string) => {
    switch (impact) {
      case "high": return "bg-red-500/20 border-red-500";
      case "medium": return "bg-yellow-500/20 border-yellow-500";
      case "low": return "bg-green-500/20 border-green-500";
      default: return "bg-gray-500/20 border-gray-500";
    }
  };

  return (
    <div className="my-8 p-6 bg-gradient-to-br from-slate-900 to-slate-800 rounded-xl border border-slate-700 shadow-2xl">
      <h3 className="text-2xl font-bold mb-6 text-center bg-gradient-to-r from-red-400 to-green-500 bg-clip-text text-transparent">
        ⚡ Pipeline 性能瓶颈分析
      </h3>

      {/* Performance Comparison */}
      <div className="mb-8">
        <h4 className="text-sm font-semibold text-gray-400 mb-4">📊 性能对比（处理 100 条文本）</h4>
        <div className="space-y-4">
          {performanceData.map((data, index) => (
            <motion.div
              key={data.method}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.2 }}
              whileHover={{ scale: 1.02 }}
              onClick={() => setSelectedMethod(selectedMethod === data.method ? null : data.method)}
              className={`p-4 rounded-lg cursor-pointer transition-all ${
                selectedMethod === data.method
                  ? "bg-slate-700 ring-2 ring-white/50 shadow-xl"
                  : "bg-slate-800 hover:bg-slate-750"
              }`}
            >
              <div className="flex items-center justify-between mb-3">
                <h5 className="text-lg font-semibold text-white">{data.method}</h5>
                <div className="flex items-center gap-3">
                  <span className="text-2xl font-bold text-white">{data.time}s</span>
                  {data.speedup > 1 && (
                    <span className="px-3 py-1 bg-green-500 text-white text-sm font-bold rounded-full">
                      {data.speedup.toFixed(1)}x 加速
                    </span>
                  )}
                </div>
              </div>

              {/* Time Bar */}
              <div className="relative h-8 bg-slate-700 rounded-lg overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${(data.time / maxTime) * 100}%` }}
                  transition={{ duration: 1, delay: index * 0.2 + 0.3 }}
                  className={`h-full bg-gradient-to-r ${data.color} flex items-center justify-end pr-3`}
                >
                  <span className="text-sm font-semibold text-white drop-shadow">
                    {data.time}s
                  </span>
                </motion.div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Bottlenecks */}
      <div className="mb-8">
        <h4 className="text-sm font-semibold text-gray-400 mb-4">🔍 性能瓶颈识别</h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {bottlenecks.map((bottleneck, index) => (
            <motion.div
              key={bottleneck.name}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className={`p-4 rounded-lg border ${getImpactColor(bottleneck.impact)}`}
            >
              <div className="flex items-start gap-3">
                <span className="text-2xl">{bottleneck.emoji}</span>
                <div className="flex-1">
                  <h5 className="font-semibold text-white mb-1">{bottleneck.name}</h5>
                  <p className="text-xs text-gray-300">{bottleneck.description}</p>
                  <div className="mt-2">
                    <span className={`px-2 py-0.5 text-xs font-semibold rounded ${
                      bottleneck.impact === "high"
                        ? "bg-red-500 text-white"
                        : bottleneck.impact === "medium"
                        ? "bg-yellow-500 text-black"
                        : "bg-green-500 text-white"
                    }`}>
                      {bottleneck.impact === "high" ? "高影响" : bottleneck.impact === "medium" ? "中等影响" : "低影响"}
                    </span>
                  </div>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Optimization Recommendations */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="p-4 bg-green-900/20 border border-green-500/50 rounded-lg">
          <h5 className="text-sm font-semibold text-green-400 mb-3">✅ 应该使用 Pipeline</h5>
          <ul className="text-xs text-gray-300 space-y-2">
            <li className="flex items-start gap-2">
              <span className="text-green-400">•</span>
              <span>快速原型开发</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-green-400">•</span>
              <span>单次推理或小批量（&lt; 10 条）</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-green-400">•</span>
              <span>演示 / Jupyter Notebook</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-green-400">•</span>
              <span>吞吐量要求低（&lt; 10 QPS）</span>
            </li>
          </ul>
        </div>

        <div className="p-4 bg-blue-900/20 border border-blue-500/50 rounded-lg">
          <h5 className="text-sm font-semibold text-blue-400 mb-3">🚀 应该使用底层 API</h5>
          <ul className="text-xs text-gray-300 space-y-2">
            <li className="flex items-start gap-2">
              <span className="text-blue-400">•</span>
              <span>生产环境部署</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-blue-400">•</span>
              <span>需要批处理优化</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-blue-400">•</span>
              <span>自定义 post-processing</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-blue-400">•</span>
              <span>高吞吐量需求（&gt; 100 QPS）</span>
            </li>
          </ul>
        </div>
      </div>

      {/* Code Comparison */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1 }}
        className="mt-6 p-4 bg-slate-950 rounded-lg border border-slate-700"
      >
        <h5 className="text-sm font-semibold text-yellow-400 mb-3">💻 优化示例代码</h5>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <div className="text-xs text-red-400 mb-2">❌ 慢（逐条处理）</div>
            <pre className="text-xs text-gray-300 font-mono overflow-x-auto">
{`for text in texts:
    result = pipeline(text)
# 时间: 12.34s`}
            </pre>
          </div>
          <div>
            <div className="text-xs text-green-400 mb-2">✅ 快（批处理）</div>
            <pre className="text-xs text-gray-300 font-mono overflow-x-auto">
{`inputs = tokenizer(texts, ...)
outputs = model(**inputs)
# 时间: 0.89s (13.9x)`}
            </pre>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
