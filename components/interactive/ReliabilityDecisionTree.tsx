"use client";

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { GitBranch, CheckCircle, XCircle, AlertTriangle, Info } from 'lucide-react';

type DecisionNode = {
  id: string;
  question: string;
  description: string;
  options: {
    label: string;
    value: string;
    nextNode?: string;
    recommendation?: string;
  }[];
  recommendation?: {
    strategy: string;
    rationale: string;
    implementation: string[];
    example: string;
  };
};

const decisionTree: Record<string, DecisionNode> = {
  root: {
    id: 'root',
    question: '服务的主要风险是什么？',
    description: '识别当前系统面临的主要可靠性挑战',
    options: [
      { label: 'API 临时不可用', value: 'transient', nextNode: 'transient_failure' },
      { label: '响应延迟过高', value: 'latency', nextNode: 'latency_issue' },
      { label: '成本超预算', value: 'cost', nextNode: 'cost_issue' },
      { label: '结果质量不稳定', value: 'quality', nextNode: 'quality_issue' }
    ]
  },
  transient_failure: {
    id: 'transient_failure',
    question: '故障频率如何？',
    description: '评估瞬时故障的发生频率',
    options: [
      { label: '偶尔发生 (<5%)', value: 'rare', nextNode: 'rare_failure' },
      { label: '频繁发生 (>20%)', value: 'frequent', nextNode: 'frequent_failure' }
    ]
  },
  rare_failure: {
    id: 'rare_failure',
    question: '能否接受短暂重试延迟？',
    description: '重试会增加 1-5 秒延迟',
    options: [
      { label: '可接受', value: 'yes' },
      { label: '不可接受', value: 'no', nextNode: 'no_retry_latency' }
    ],
    recommendation: {
      strategy: '简单重试（Retry）',
      rationale: '偶发故障用指数退避重试即可解决，成本低、实现简单',
      implementation: [
        '使用 .with_retry() 或 tenacity 库',
        '设置 3-5 次重试，指数退避（1s, 2s, 4s...）',
        '只重试瞬时错误（超时、429 限流），不重试永久错误（401, 400）'
      ],
      example: `llm.with_retry(
  stop_after_attempt=3,
  wait_exponential_multiplier=1
)`
    }
  },
  frequent_failure: {
    id: 'frequent_failure',
    question: '有备用服务可用吗？',
    description: '是否有其他 API/模型可作为降级选项',
    options: [
      { label: '有（其他模型/API）', value: 'yes' },
      { label: '无', value: 'no', nextNode: 'no_fallback' }
    ],
    recommendation: {
      strategy: '降级策略（Fallback）',
      rationale: '主服务频繁失败时，自动切换到备用服务保证可用性',
      implementation: [
        '配置多级 fallback: 主模型 -> 备用模型 -> 缓存 -> 默认响应',
        '监控各级使用率，发现主服务异常',
        '备用模型选择更便宜/更快的版本（如 gpt-4o -> gpt-4o-mini）'
      ],
      example: `primary_llm.with_fallbacks([
  fallback_llm,
  cached_response_fn
])`
    }
  },
  no_fallback: {
    id: 'no_fallback',
    question: '能否容忍部分请求失败？',
    description: '是否可以返回错误而非阻塞',
    options: [
      { label: '可以（非关键路径）', value: 'yes' },
      { label: '不可以（关键服务）', value: 'no' }
    ],
    recommendation: {
      strategy: '熔断器（Circuit Breaker）',
      rationale: '快速失败避免雪崩，保护系统资源',
      implementation: [
        '使用 pybreaker 库实现熔断器',
        '失败率达阈值（如 50%）时打开熔断器',
        '半开状态定期尝试恢复',
        '记录熔断事件并告警'
      ],
      example: `@circuit(
  failure_threshold=5,
  recovery_timeout=60
)
def call_llm():
  return llm.invoke(...)
`
    }
  },
  no_retry_latency: {
    id: 'no_retry_latency',
    question: '是否有缓存可用？',
    description: '能否从缓存/数据库返回历史结果',
    options: [
      { label: '有缓存', value: 'yes' },
      { label: '无缓存', value: 'no' }
    ],
    recommendation: {
      strategy: '缓存降级',
      rationale: '失败时返回缓存结果，牺牲时效性换取可用性',
      implementation: [
        '启用 Redis/SQLite 缓存',
        '设置较长 TTL（1-7 天）',
        '主调用失败时查询缓存',
        '返回时标记"缓存结果"'
      ],
      example: `try:
  result = llm.invoke(q)
except:
  result = cache.get(q) or "暂时无法回答"
`
    }
  },
  latency_issue: {
    id: 'latency_issue',
    question: '延迟主要来源是什么？',
    description: '分析延迟瓶颈',
    options: [
      { label: 'LLM API 调用慢', value: 'llm_slow', nextNode: 'llm_slow' },
      { label: '检索/数据库慢', value: 'retrieval_slow', nextNode: 'retrieval_slow' }
    ]
  },
  llm_slow: {
    id: 'llm_slow',
    question: '能否使用更快的模型？',
    description: '如 gpt-4o-mini 比 gpt-4o 快 2-3 倍',
    options: [
      { label: '可以降级模型', value: 'yes' },
      { label: '必须用高质量模型', value: 'no', nextNode: 'need_quality' }
    ],
    recommendation: {
      strategy: '模型路由',
      rationale: '简单任务用快速模型，复杂任务用高质量模型',
      implementation: [
        '按输入长度/复杂度路由',
        '使用 RunnableBranch 实现',
        '监控不同路径的延迟和质量',
        'A/B 测试验证效果'
      ],
      example: `RunnableBranch(
  (is_simple, fast_llm),
  slow_llm  # default
)`
    }
  },
  need_quality: {
    id: 'need_quality',
    question: '能否接受流式输出？',
    description: '用户先看到部分结果',
    options: [
      { label: '可以（聊天场景）', value: 'yes' },
      { label: '不可以（批处理）', value: 'no' }
    ],
    recommendation: {
      strategy: '流式响应（Streaming）',
      rationale: '首 token 延迟低，用户体验更好',
      implementation: [
        '使用 .stream() 或 .astream()',
        '前端实时渲染 token',
        'WebSocket/SSE 传输',
        '显示进度指示器'
      ],
      example: `for chunk in chain.stream(input):
  print(chunk, end="", flush=True)
`
    }
  },
  retrieval_slow: {
    id: 'retrieval_slow',
    question: '检索是否可以并行？',
    description: '多个独立查询可并发执行',
    options: [
      { label: '可以并行', value: 'yes' },
      { label: '必须顺序执行', value: 'no' }
    ],
    recommendation: {
      strategy: '异步并行',
      rationale: '并发执行独立操作，减少总延迟',
      implementation: [
        '使用 asyncio.gather() 并发调用',
        'Runnable.map() 批量处理',
        '注意控制并发数（max_concurrency）',
        '使用 aiohttp 替代 requests'
      ],
      example: `results = await asyncio.gather(
  retriever1.ainvoke(q),
  retriever2.ainvoke(q)
)`
    }
  },
  cost_issue: {
    id: 'cost_issue',
    question: '成本主要来源？',
    description: '识别成本高的环节',
    options: [
      { label: '重复调用相同问题', value: 'repetitive', nextNode: 'repetitive_calls' },
      { label: '使用昂贵模型', value: 'expensive_model', nextNode: 'expensive_model' }
    ]
  },
  repetitive_calls: {
    id: 'repetitive_calls',
    question: '问题完全相同还是语义相似？',
    description: '选择缓存策略',
    options: [
      { label: '完全相同', value: 'exact' },
      { label: '语义相似', value: 'semantic' }
    ],
    recommendation: {
      strategy: '精确缓存',
      rationale: '完全相同的请求直接命中缓存，节省 100% 成本',
      implementation: [
        '启用 RedisCache 或 SQLiteCache',
        '设置合理 TTL（1-7 天）',
        '监控缓存命中率',
        '预热高频问题'
      ],
      example: `set_llm_cache(RedisCache(redis_client))`
    }
  },
  expensive_model: {
    id: 'expensive_model',
    question: '所有任务都需要高质量模型吗？',
    description: '评估任务复杂度差异',
    options: [
      { label: '有简单任务', value: 'mixed' },
      { label: '全是复杂任务', value: 'complex' }
    ],
    recommendation: {
      strategy: '模型路由 + 输出限制',
      rationale: '简单任务用便宜模型，限制输出长度',
      implementation: [
        '按长度/复杂度路由到不同模型',
        '设置 max_tokens 限制输出',
        '优化 prompt 减少输入 tokens',
        '使用 tiktoken 监控 token 使用'
      ],
      example: `ChatOpenAI(
  model="gpt-4o-mini",
  max_tokens=100
)`
    }
  },
  quality_issue: {
    id: 'quality_issue',
    question: '质量问题的表现？',
    description: '诊断质量不稳定原因',
    options: [
      { label: '偶尔输出格式错误', value: 'format_error' },
      { label: '答案准确性波动', value: 'accuracy' }
    ],
    recommendation: {
      strategy: '重试 + 验证',
      rationale: '自动重试格式错误的输出，直到符合要求',
      implementation: [
        '使用 OutputParser 验证格式',
        '格式错误时重试（max 3 次）',
        '使用 Pydantic 强制结构化输出',
        '记录失败案例用于 prompt 优化'
      ],
      example: `parser.parse_with_retry(
  llm_output,
  max_retries=3
)`
    }
  }
};

export default function ReliabilityDecisionTree() {
  const [currentNode, setCurrentNode] = useState<string>('root');
  const [path, setPath] = useState<string[]>(['root']);
  const [selectedOptions, setSelectedOptions] = useState<Record<string, string>>({});

  const node = decisionTree[currentNode];

  const selectOption = (option: typeof node.options[0]) => {
    setSelectedOptions({ ...selectedOptions, [currentNode]: option.value });
    
    if (option.nextNode) {
      setCurrentNode(option.nextNode);
      setPath([...path, option.nextNode]);
    }
  };

  const goBack = () => {
    if (path.length > 1) {
      const newPath = path.slice(0, -1);
      setPath(newPath);
      setCurrentNode(newPath[newPath.length - 1]);
    }
  };

  const reset = () => {
    setCurrentNode('root');
    setPath(['root']);
    setSelectedOptions({});
  };

  return (
    <div className="w-full max-w-4xl mx-auto p-6 bg-white rounded-lg shadow-lg">
      <div className="mb-6">
        <h3 className="text-2xl font-bold mb-2">可靠性工程决策树</h3>
        <p className="text-gray-600">回答问题找到最适合的可靠性策略</p>
      </div>

      {/* 路径面包屑 */}
      <div className="mb-6 flex items-center gap-2 text-sm">
        <GitBranch className="w-4 h-4 text-gray-400" />
        {path.map((nodeId, index) => (
          <React.Fragment key={nodeId}>
            {index > 0 && <span className="text-gray-400">→</span>}
            <span className={index === path.length - 1 ? 'font-semibold text-blue-600' : 'text-gray-600'}>
              {decisionTree[nodeId].question.slice(0, 15)}...
            </span>
          </React.Fragment>
        ))}
      </div>

      {/* 当前问题 */}
      <AnimatePresence mode="wait">
        <motion.div
          key={currentNode}
          initial={{ opacity: 0, x: 50 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -50 }}
          className="mb-6"
        >
          <div className="bg-blue-50 border-2 border-blue-200 rounded-lg p-6 mb-4">
            <h4 className="text-xl font-bold mb-2">{node.question}</h4>
            <p className="text-gray-700">{node.description}</p>
          </div>

          {/* 选项 */}
          <div className="space-y-3">
            {node.options.map((option, index) => (
              <motion.button
                key={option.value}
                onClick={() => selectOption(option)}
                className="w-full p-4 border-2 border-gray-200 rounded-lg hover:border-blue-400 hover:bg-blue-50 transition-all text-left"
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
              >
                <div className="flex items-center justify-between">
                  <span className="font-medium">{option.label}</span>
                  {option.nextNode ? (
                    <span className="text-blue-600">→</span>
                  ) : (
                    <CheckCircle className="w-5 h-5 text-green-600" />
                  )}
                </div>
              </motion.button>
            ))}
          </div>
        </motion.div>
      </AnimatePresence>

      {/* 推荐策略 */}
      {node.recommendation && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-green-50 border-2 border-green-200 rounded-lg p-6 mb-6"
        >
          <div className="flex items-start gap-3 mb-4">
            <CheckCircle className="w-6 h-6 text-green-600 flex-shrink-0 mt-1" />
            <div>
              <h4 className="text-xl font-bold text-green-900 mb-2">
                推荐策略：{node.recommendation.strategy}
              </h4>
              <p className="text-green-800 mb-4">{node.recommendation.rationale}</p>

              <div className="mb-4">
                <h5 className="font-semibold text-green-900 mb-2">实施步骤：</h5>
                <ul className="space-y-1">
                  {node.recommendation.implementation.map((step, i) => (
                    <li key={i} className="flex items-start gap-2 text-sm text-green-800">
                      <span className="text-green-600 font-bold">{i + 1}.</span>
                      <span>{step}</span>
                    </li>
                  ))}
                </ul>
              </div>

              <div>
                <h5 className="font-semibold text-green-900 mb-2">代码示例：</h5>
                <pre className="bg-gray-900 text-green-400 p-3 rounded text-xs overflow-x-auto">
                  {node.recommendation.example}
                </pre>
              </div>
            </div>
          </div>
        </motion.div>
      )}

      {/* 操作按钮 */}
      <div className="flex gap-3">
        {path.length > 1 && (
          <button
            onClick={goBack}
            className="flex-1 px-4 py-2 border-2 border-gray-300 rounded-lg hover:bg-gray-100 transition-colors"
          >
            上一步
          </button>
        )}
        <button
          onClick={reset}
          className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
        >
          重新开始
        </button>
      </div>

      {/* 提示信息 */}
      <div className="mt-6 p-4 bg-yellow-50 border-2 border-yellow-200 rounded-lg">
        <div className="flex items-start gap-3">
          <Info className="w-5 h-5 text-yellow-600 flex-shrink-0 mt-0.5" />
          <div>
            <h5 className="font-semibold text-yellow-900 mb-1">使用提示</h5>
            <ul className="text-sm text-yellow-800 space-y-1">
              <li>• 多个策略可以组合使用（如重试 + 降级 + 缓存）</li>
              <li>• 建议在测试环境验证效果后再上线</li>
              <li>• 监控关键指标：成功率、延迟、成本</li>
              <li>• 定期 review 策略效果并调整</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
