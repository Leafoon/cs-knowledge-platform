'use client';

import React, { useState, useMemo } from 'react';

type EvaluationResult = {
  example_id: string;
  input: string;
  prediction: string;
  reference: string;
  scores: {
    exact_match: number;
    semantic_similarity: number;
    factual_correctness: number;
    relevance: number;
  };
  overall_score: number;
  comments: string;
};

type ExperimentComparison = {
  name: string;
  model: string;
  prompt_version: string;
  avg_score: number;
  avg_latency: number;
  avg_cost: number;
  success_rate: number;
  color: string;
};

export default function EvaluationDashboard() {
  const [selectedTab, setSelectedTab] = useState<'results' | 'comparison'>('results');
  const [selectedExample, setSelectedExample] = useState<string | null>(null);

  const evaluationResults: EvaluationResult[] = useMemo(() => [
    {
      example_id: 'ex_001',
      input: '如何重置密码？',
      prediction: '请访问登录页面，点击"忘记密码"链接，然后按照邮件中的指引操作即可重置密码。',
      reference: '点击登录页面的"忘记密码"链接，按照邮件指引操作。',
      scores: {
        exact_match: 0.0,
        semantic_similarity: 0.92,
        factual_correctness: 1.0,
        relevance: 0.95
      },
      overall_score: 0.88,
      comments: '语义准确，表述略有不同但信息完整'
    },
    {
      example_id: 'ex_002',
      input: '支持哪些支付方式？',
      prediction: '我们支持信用卡、借记卡、PayPal、支付宝和微信支付。',
      reference: '支持信用卡、PayPal、支付宝和微信支付。',
      scores: {
        exact_match: 0.0,
        semantic_similarity: 0.85,
        factual_correctness: 0.9,
        relevance: 1.0
      },
      overall_score: 0.82,
      comments: '提到了借记卡（未在参考中），其他正确'
    },
    {
      example_id: 'ex_003',
      input: '退货政策是什么？',
      prediction: '30 天内可无理由退货。',
      reference: '30 天内无理由退货，需保持商品完好。',
      scores: {
        exact_match: 0.0,
        semantic_similarity: 0.65,
        factual_correctness: 0.7,
        relevance: 0.8
      },
      overall_score: 0.68,
      comments: '缺少关键信息"保持商品完好"'
    },
    {
      example_id: 'ex_004',
      input: '客服工作时间？',
      prediction: '客服工作时间为周一至周五 9:00-18:00，周末及节假日休息。',
      reference: '周一至周五 9:00-18:00。',
      scores: {
        exact_match: 0.0,
        semantic_similarity: 0.88,
        factual_correctness: 1.0,
        relevance: 0.95
      },
      overall_score: 0.90,
      comments: '信息准确且更详细'
    },
    {
      example_id: 'ex_005',
      input: '如何联系人工客服？',
      prediction: '您可以通过在线聊天、拨打客服热线 400-xxx-xxxx 或发送邮件至 support@example.com 联系人工客服。',
      reference: '拨打客服热线 400-xxx-xxxx 或发送邮件至 support@example.com。',
      scores: {
        exact_match: 0.0,
        semantic_similarity: 0.90,
        factual_correctness: 1.0,
        relevance: 1.0
      },
      overall_score: 0.95,
      comments: '增加了在线聊天渠道，信息更全面'
    }
  ], []);

  const experiments: ExperimentComparison[] = useMemo(() => [
    {
      name: 'Baseline (GPT-3.5)',
      model: 'gpt-3.5-turbo',
      prompt_version: 'v1.0',
      avg_score: 0.72,
      avg_latency: 1.2,
      avg_cost: 0.0008,
      success_rate: 0.85,
      color: 'blue'
    },
    {
      name: 'Optimized Prompt (GPT-3.5)',
      model: 'gpt-3.5-turbo',
      prompt_version: 'v2.0',
      avg_score: 0.81,
      avg_latency: 1.3,
      avg_cost: 0.0009,
      success_rate: 0.92,
      color: 'green'
    },
    {
      name: 'GPT-4 Standard',
      model: 'gpt-4',
      prompt_version: 'v1.0',
      avg_score: 0.89,
      avg_latency: 2.8,
      avg_cost: 0.0045,
      success_rate: 0.98,
      color: 'purple'
    },
    {
      name: 'GPT-4 + Few-Shot',
      model: 'gpt-4',
      prompt_version: 'v3.0',
      avg_score: 0.94,
      avg_latency: 3.2,
      avg_cost: 0.0052,
      success_rate: 1.0,
      color: 'orange'
    }
  ], []);

  const avgScoresByMetric = useMemo(() => {
    const totals = evaluationResults.reduce(
      (acc, result) => ({
        exact_match: acc.exact_match + result.scores.exact_match,
        semantic_similarity: acc.semantic_similarity + result.scores.semantic_similarity,
        factual_correctness: acc.factual_correctness + result.scores.factual_correctness,
        relevance: acc.relevance + result.scores.relevance
      }),
      { exact_match: 0, semantic_similarity: 0, factual_correctness: 0, relevance: 0 }
    );
    
    const count = evaluationResults.length;
    return {
      exact_match: totals.exact_match / count,
      semantic_similarity: totals.semantic_similarity / count,
      factual_correctness: totals.factual_correctness / count,
      relevance: totals.relevance / count
    };
  }, [evaluationResults]);

  const selectedExampleData = useMemo(
    () => evaluationResults.find(ex => ex.example_id === selectedExample),
    [selectedExample, evaluationResults]
  );

  const getScoreColor = (score: number) => {
    if (score >= 0.9) return 'text-green-600 dark:text-green-400';
    if (score >= 0.7) return 'text-yellow-600 dark:text-yellow-400';
    return 'text-red-600 dark:text-red-400';
  };

  const getScoreBg = (score: number) => {
    if (score >= 0.9) return 'bg-green-100 dark:bg-green-900/30';
    if (score >= 0.7) return 'bg-yellow-100 dark:bg-yellow-900/30';
    return 'bg-red-100 dark:bg-red-900/30';
  };

  return (
    <div className="my-8 p-8 bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-900 rounded-2xl shadow-xl border border-gray-200 dark:border-gray-700">
      <h3 className="text-2xl font-bold mb-2 bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
        评估仪表板
      </h3>
      <p className="text-sm text-gray-600 dark:text-gray-400 mb-6">
        查看评估结果、对比实验性能，持续优化 LLM 应用质量
      </p>

      <div className="flex gap-3 mb-6">
        <button
          onClick={() => setSelectedTab('results')}
          className={`px-6 py-3 rounded-xl font-semibold transition-all transform hover:scale-105 ${
            selectedTab === 'results'
              ? 'bg-gradient-to-r from-blue-500 to-blue-600 text-white shadow-lg shadow-blue-500/50'
              : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 shadow-md hover:shadow-lg'
          }`}
        >
          <span className="flex items-center gap-2">
            📊 评估结果
          </span>
        </button>
        <button
          onClick={() => setSelectedTab('comparison')}
          className={`px-6 py-3 rounded-xl font-semibold transition-all transform hover:scale-105 ${
            selectedTab === 'comparison'
              ? 'bg-gradient-to-r from-purple-500 to-purple-600 text-white shadow-lg shadow-purple-500/50'
              : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 shadow-md hover:shadow-lg'
          }`}
        >
          <span className="flex items-center gap-2">
            🔬 实验对比
          </span>
        </button>
      </div>

      {selectedTab === 'results' && (
        <div className="space-y-6">
          <div className="grid grid-cols-4 gap-4">
            <div className="p-4 bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/30 dark:to-blue-800/30 rounded-xl shadow-md border border-blue-200 dark:border-blue-700">
              <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">语义相似度</div>
              <div className="text-3xl font-bold text-blue-600 dark:text-blue-400">
                {(avgScoresByMetric.semantic_similarity * 100).toFixed(0)}%
              </div>
            </div>
            <div className="p-4 bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-900/30 dark:to-purple-800/30 rounded-xl shadow-md border border-purple-200 dark:border-purple-700">
              <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">事实准确性</div>
              <div className="text-3xl font-bold text-purple-600 dark:text-purple-400">
                {(avgScoresByMetric.factual_correctness * 100).toFixed(0)}%
              </div>
            </div>
            <div className="p-4 bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/30 dark:to-green-800/30 rounded-xl shadow-md border border-green-200 dark:border-green-700">
              <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">相关性</div>
              <div className="text-3xl font-bold text-green-600 dark:text-green-400">
                {(avgScoresByMetric.relevance * 100).toFixed(0)}%
              </div>
            </div>
            <div className="p-4 bg-gradient-to-br from-orange-50 to-orange-100 dark:from-orange-900/30 dark:to-orange-800/30 rounded-xl shadow-md border border-orange-200 dark:border-orange-700">
              <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">精确匹配</div>
              <div className="text-3xl font-bold text-orange-600 dark:text-orange-400">
                {(avgScoresByMetric.exact_match * 100).toFixed(0)}%
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-lg">
            <h4 className="font-bold mb-4 text-gray-800 dark:text-gray-200">详细评估结果</h4>
            <div className="space-y-3">
              {evaluationResults.map((result) => (
                <div
                  key={result.example_id}
                  className={`p-4 rounded-xl cursor-pointer transition-all border-2 ${
                    selectedExample === result.example_id
                      ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20 shadow-lg'
                      : 'border-gray-200 dark:border-gray-600 bg-gray-50 dark:bg-gray-700 hover:shadow-md'
                  }`}
                  onClick={() => setSelectedExample(result.example_id)}
                >
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex-1">
                      <div className="font-semibold text-gray-800 dark:text-gray-200 mb-1">
                        {result.input}
                      </div>
                      <div className="text-xs text-gray-500 dark:text-gray-400">
                        ID: {result.example_id}
                      </div>
                    </div>
                    <div className={`ml-4 px-4 py-2 rounded-lg ${getScoreBg(result.overall_score)}`}>
                      <div className="text-xs text-gray-600 dark:text-gray-400">总分</div>
                      <div className={`text-2xl font-bold ${getScoreColor(result.overall_score)}`}>
                        {(result.overall_score * 100).toFixed(0)}
                      </div>
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-4 gap-2">
                    {Object.entries(result.scores).map(([key, value]) => (
                      <div key={key} className="bg-white dark:bg-gray-800 rounded-lg p-2">
                        <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">
                          {key === 'exact_match' ? '精确匹配' :
                           key === 'semantic_similarity' ? '语义相似' :
                           key === 'factual_correctness' ? '事实准确' : '相关性'}
                        </div>
                        <div className={`font-bold ${getScoreColor(value)}`}>
                          {(value * 100).toFixed(0)}%
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {selectedExampleData && (
            <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-lg">
              <h4 className="font-bold mb-4 text-gray-800 dark:text-gray-200">样本详情</h4>
              <div className="space-y-4">
                <div>
                  <div className="text-xs text-gray-500 dark:text-gray-400 mb-2">输入问题</div>
                  <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg text-gray-800 dark:text-gray-200">
                    {selectedExampleData.input}
                  </div>
                </div>
                <div>
                  <div className="text-xs text-gray-500 dark:text-gray-400 mb-2">模型预测</div>
                  <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded-lg text-gray-800 dark:text-gray-200">
                    {selectedExampleData.prediction}
                  </div>
                </div>
                <div>
                  <div className="text-xs text-gray-500 dark:text-gray-400 mb-2">参考答案</div>
                  <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded-lg text-gray-800 dark:text-gray-200">
                    {selectedExampleData.reference}
                  </div>
                </div>
                <div>
                  <div className="text-xs text-gray-500 dark:text-gray-400 mb-2">评估意见</div>
                  <div className="p-3 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg text-gray-700 dark:text-gray-300 italic">
                    {selectedExampleData.comments}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {selectedTab === 'comparison' && (
        <div className="space-y-6">
          <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-lg overflow-x-auto">
            <h4 className="font-bold mb-4 text-gray-800 dark:text-gray-200">实验对比表</h4>
            <table className="w-full min-w-[800px]">
              <thead>
                <tr className="bg-gradient-to-r from-blue-500 to-purple-500 text-white">
                  <th className="p-3 text-left rounded-tl-lg">实验名称</th>
                  <th className="p-3 text-left">模型</th>
                  <th className="p-3 text-left">提示版本</th>
                  <th className="p-3 text-center">平均分数</th>
                  <th className="p-3 text-center">平均延迟</th>
                  <th className="p-3 text-center">平均成本</th>
                  <th className="p-3 text-center rounded-tr-lg">成功率</th>
                </tr>
              </thead>
              <tbody>
                {experiments.map((exp, idx) => (
                  <tr
                    key={exp.name}
                    className={`border-b border-gray-200 dark:border-gray-600 ${
                      idx % 2 === 0 ? 'bg-gray-50 dark:bg-gray-700' : 'bg-white dark:bg-gray-800'
                    } hover:bg-blue-50 dark:hover:bg-blue-900/20 transition-colors`}
                  >
                    <td className="p-3 font-semibold text-gray-800 dark:text-gray-200">{exp.name}</td>
                    <td className="p-3">
                      <span className="px-2 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 rounded font-mono text-xs">
                        {exp.model}
                      </span>
                    </td>
                    <td className="p-3">
                      <span className="px-2 py-1 bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300 rounded font-mono text-xs">
                        {exp.prompt_version}
                      </span>
                    </td>
                    <td className="p-3 text-center">
                      <span className={`text-lg font-bold ${getScoreColor(exp.avg_score)}`}>
                        {(exp.avg_score * 100).toFixed(0)}%
                      </span>
                    </td>
                    <td className="p-3 text-center text-gray-700 dark:text-gray-300">{exp.avg_latency}s</td>
                    <td className="p-3 text-center text-gray-700 dark:text-gray-300">${exp.avg_cost.toFixed(4)}</td>
                    <td className="p-3 text-center">
                      <span className={`text-lg font-bold ${getScoreColor(exp.success_rate)}`}>
                        {(exp.success_rate * 100).toFixed(0)}%
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="grid grid-cols-2 gap-6">
            <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-lg">
              <h4 className="font-bold mb-4 text-gray-800 dark:text-gray-200">质量 vs 成本</h4>
              <div className="space-y-3">
                {experiments.map((exp) => (
                  <div key={exp.name} className="flex items-center gap-3">
                    <div className="w-32 text-sm text-gray-600 dark:text-gray-400 truncate">
                      {exp.name}
                    </div>
                    <div className="flex-1 bg-gray-200 dark:bg-gray-700 rounded-full h-6 relative overflow-hidden">
                      <div
                        className={`h-full bg-gradient-to-r from-${exp.color}-400 to-${exp.color}-600 flex items-center justify-end pr-2`}
                        style={{ width: `${exp.avg_score * 100}%` }}
                      >
                        <span className="text-xs font-bold text-white">{(exp.avg_score * 100).toFixed(0)}%</span>
                      </div>
                    </div>
                    <div className="w-20 text-right text-sm text-gray-600 dark:text-gray-400">
                      ${exp.avg_cost.toFixed(4)}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-lg">
              <h4 className="font-bold mb-4 text-gray-800 dark:text-gray-200">质量 vs 延迟</h4>
              <div className="space-y-3">
                {experiments.map((exp) => (
                  <div key={exp.name} className="flex items-center gap-3">
                    <div className="w-32 text-sm text-gray-600 dark:text-gray-400 truncate">
                      {exp.name}
                    </div>
                    <div className="flex-1 bg-gray-200 dark:bg-gray-700 rounded-full h-6 relative overflow-hidden">
                      <div
                        className={`h-full bg-gradient-to-r from-${exp.color}-400 to-${exp.color}-600 flex items-center justify-end pr-2`}
                        style={{ width: `${exp.avg_score * 100}%` }}
                      >
                        <span className="text-xs font-bold text-white">{(exp.avg_score * 100).toFixed(0)}%</span>
                      </div>
                    </div>
                    <div className="w-16 text-right text-sm text-gray-600 dark:text-gray-400">
                      {exp.avg_latency}s
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="mt-6 p-6 bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-2xl border-l-4 border-green-500 shadow-lg">
        <div className="flex items-start gap-4">
          <div className="flex-shrink-0 w-10 h-10 bg-green-500 rounded-full flex items-center justify-center shadow-lg">
            <span className="text-white text-xl">🎯</span>
          </div>
          <div>
            <h4 className="font-bold text-gray-800 dark:text-gray-100 mb-2">优化建议</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300 leading-relaxed">
              <strong>GPT-4 + Few-Shot</strong> 表现最佳（94% 分数，100% 成功率），但成本较高。
              可考虑混合策略：简单查询使用 <strong>Optimized Prompt (GPT-3.5)</strong>，
              复杂/高价值查询使用 GPT-4，预计可节省 40% 成本同时保持 90%+ 整体质量。
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
