"use client";

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ThumbsUp, ThumbsDown, Star, MessageSquare, TrendingUp, AlertCircle, BarChart2 } from 'lucide-react';

interface FeedbackItem {
  id: string;
  runId: string;
  timestamp: Date;
  rating: 'positive' | 'negative';
  score?: number;
  comment?: string;
  category?: string;
}

const FeedbackDashboard: React.FC = () => {
  const [selectedCategory, setSelectedCategory] = useState<'all' | 'positive' | 'negative'>('all');
  const [showDetails, setShowDetails] = useState<string | null>(null);

  const feedbackData: FeedbackItem[] = [
    {
      id: '1',
      runId: 'run-abc123',
      timestamp: new Date('2024-01-15T10:30:00'),
      rating: 'positive',
      score: 1,
      comment: '答案准确且详细',
      category: 'correctness',
    },
    {
      id: '2',
      runId: 'run-def456',
      timestamp: new Date('2024-01-15T10:32:00'),
      rating: 'positive',
      score: 1,
      comment: '回答很有帮助',
    },
    {
      id: '3',
      runId: 'run-ghi789',
      timestamp: new Date('2024-01-15T10:35:00'),
      rating: 'negative',
      score: 0,
      comment: '答案不完整',
      category: 'completeness',
    },
    {
      id: '4',
      runId: 'run-jkl012',
      timestamp: new Date('2024-01-15T10:40:00'),
      rating: 'positive',
      score: 1,
    },
    {
      id: '5',
      runId: 'run-mno345',
      timestamp: new Date('2024-01-15T10:45:00'),
      rating: 'negative',
      score: 0,
      comment: '事实错误：说巴黎在德国',
      category: 'factual_error',
    },
    {
      id: '6',
      runId: 'run-pqr678',
      timestamp: new Date('2024-01-15T10:50:00'),
      rating: 'positive',
      score: 1,
      comment: '非常清晰',
    },
  ];

  const filteredFeedback = feedbackData.filter(item => {
    if (selectedCategory === 'all') return true;
    return item.rating === selectedCategory;
  });

  const stats = {
    total: feedbackData.length,
    positive: feedbackData.filter(f => f.rating === 'positive').length,
    negative: feedbackData.filter(f => f.rating === 'negative').length,
    satisfactionRate: (feedbackData.filter(f => f.rating === 'positive').length / feedbackData.length * 100).toFixed(1),
  };

  const categoryDistribution = feedbackData
    .filter(f => f.category)
    .reduce((acc, item) => {
      acc[item.category!] = (acc[item.category!] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-green-50 to-blue-50 rounded-xl shadow-lg">
      <div className="mb-6">
        <h3 className="text-2xl font-bold text-gray-800 mb-2 flex items-center gap-2">
          <BarChart2 className="w-6 h-6 text-green-600" />
          用户反馈仪表盘
        </h3>
        <p className="text-gray-600">实时监控用户满意度，发现质量问题</p>
      </div>

      {/* 统计卡片 */}
      <div className="grid grid-cols-4 gap-4 mb-6">
        <div className="p-4 bg-white rounded-lg shadow border border-gray-200">
          <div className="flex items-center gap-2 mb-1">
            <MessageSquare className="w-4 h-4 text-blue-500" />
            <p className="text-sm text-gray-600">总反馈数</p>
          </div>
          <p className="text-3xl font-bold text-gray-800">{stats.total}</p>
        </div>
        <div className="p-4 bg-white rounded-lg shadow border border-gray-200">
          <div className="flex items-center gap-2 mb-1">
            <ThumbsUp className="w-4 h-4 text-green-500" />
            <p className="text-sm text-gray-600">好评</p>
          </div>
          <p className="text-3xl font-bold text-green-600">{stats.positive}</p>
        </div>
        <div className="p-4 bg-white rounded-lg shadow border border-gray-200">
          <div className="flex items-center gap-2 mb-1">
            <ThumbsDown className="w-4 h-4 text-red-500" />
            <p className="text-sm text-gray-600">差评</p>
          </div>
          <p className="text-3xl font-bold text-red-600">{stats.negative}</p>
        </div>
        <div className="p-4 bg-white rounded-lg shadow border border-gray-200">
          <div className="flex items-center gap-2 mb-1">
            <Star className="w-4 h-4 text-yellow-500" />
            <p className="text-sm text-gray-600">满意度</p>
          </div>
          <p className="text-3xl font-bold text-indigo-600">{stats.satisfactionRate}%</p>
        </div>
      </div>

      {/* 过滤器 */}
      <div className="flex gap-2 mb-4">
        <button
          onClick={() => setSelectedCategory('all')}
          className={`px-4 py-2 rounded-lg font-medium transition-colors ${
            selectedCategory === 'all'
              ? 'bg-indigo-600 text-white'
              : 'bg-white text-gray-700 hover:bg-gray-100'
          }`}
        >
          全部 ({feedbackData.length})
        </button>
        <button
          onClick={() => setSelectedCategory('positive')}
          className={`px-4 py-2 rounded-lg font-medium transition-colors ${
            selectedCategory === 'positive'
              ? 'bg-green-600 text-white'
              : 'bg-white text-gray-700 hover:bg-gray-100'
          }`}
        >
          好评 ({stats.positive})
        </button>
        <button
          onClick={() => setSelectedCategory('negative')}
          className={`px-4 py-2 rounded-lg font-medium transition-colors ${
            selectedCategory === 'negative'
              ? 'bg-red-600 text-white'
              : 'bg-white text-gray-700 hover:bg-gray-100'
          }`}
        >
          差评 ({stats.negative})
        </button>
      </div>

      {/* 反馈列表 */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div className="space-y-3">
          <h4 className="font-semibold text-gray-800">反馈流</h4>
          <div className="space-y-2 max-h-[400px] overflow-y-auto">
            <AnimatePresence>
              {filteredFeedback.map((item) => (
                <motion.div
                  key={item.id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 20 }}
                  className={`p-4 rounded-lg border-2 cursor-pointer transition-all ${
                    showDetails === item.id
                      ? 'border-indigo-500 bg-indigo-50'
                      : item.rating === 'positive'
                      ? 'border-green-200 bg-green-50 hover:border-green-400'
                      : 'border-red-200 bg-red-50 hover:border-red-400'
                  }`}
                  onClick={() => setShowDetails(showDetails === item.id ? null : item.id)}
                >
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex items-center gap-2">
                      {item.rating === 'positive' ? (
                        <ThumbsUp className="w-5 h-5 text-green-600" />
                      ) : (
                        <ThumbsDown className="w-5 h-5 text-red-600" />
                      )}
                      <span className="text-sm font-medium text-gray-700">
                        {item.rating === 'positive' ? '好评' : '差评'}
                      </span>
                    </div>
                    <span className="text-xs text-gray-500">
                      {item.timestamp.toLocaleTimeString()}
                    </span>
                  </div>
                  {item.comment && (
                    <p className="text-sm text-gray-700 mb-1">&quot;{item.comment}&quot;</p>
                  )}
                  <div className="flex items-center gap-2 mt-2">
                    <code className="text-xs bg-white px-2 py-1 rounded border border-gray-200">
                      {item.runId}
                    </code>
                    {item.category && (
                      <span className="text-xs bg-indigo-100 text-indigo-700 px-2 py-1 rounded">
                        {item.category}
                      </span>
                    )}
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        </div>

        {/* 问题分类 */}
        <div className="space-y-3">
          <h4 className="font-semibold text-gray-800">问题分类统计</h4>
          <div className="p-4 bg-white rounded-lg shadow border border-gray-200">
            {Object.keys(categoryDistribution).length > 0 ? (
              <div className="space-y-3">
                {Object.entries(categoryDistribution).map(([category, count]) => (
                  <div key={category}>
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-sm text-gray-700 font-medium">
                        {category === 'correctness' && '正确性'}
                        {category === 'completeness' && '完整性'}
                        {category === 'factual_error' && '事实错误'}
                      </span>
                      <span className="text-sm text-gray-600">{count} 次</span>
                    </div>
                    <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-red-500 rounded-full transition-all"
                        style={{ width: `${(count / stats.negative) * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-sm text-gray-500 text-center py-8">暂无分类数据</p>
            )}
          </div>

          {/* 行动建议 */}
          <div className="p-4 bg-yellow-50 rounded-lg border border-yellow-200">
            <div className="flex items-start gap-2">
              <AlertCircle className="w-5 h-5 text-yellow-600 flex-shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-gray-800 mb-2">⚠️ 需要关注</h4>
                <ul className="text-sm text-gray-700 space-y-1">
                  <li>• 事实错误：1 条 - 需检查知识库准确性</li>
                  <li>• 完整性问题：1 条 - 考虑增加提示细节</li>
                  <li>• 满意度 {stats.satisfactionRate}% - 目标 &gt; 90%</li>
                </ul>
              </div>
            </div>
          </div>

          {/* 改进建议 */}
          <div className="p-4 bg-white rounded-lg shadow border border-gray-200">
            <div className="flex items-start gap-2">
              <TrendingUp className="w-5 h-5 text-green-600 flex-shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-gray-800 mb-2">💡 改进建议</h4>
                <ul className="text-sm text-gray-700 space-y-1">
                  <li>1. 将差评样本添加到评估数据集</li>
                  <li>2. 针对事实错误案例改进 RAG 检索</li>
                  <li>3. 提示增加"确保答案完整"指令</li>
                  <li>4. 每周重新评估改进效果</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* 代码示例 */}
      <div className="p-4 bg-white rounded-lg border border-blue-200">
        <h4 className="font-semibold text-gray-800 mb-2">💻 集成代码示例</h4>
        <pre className="text-xs bg-gray-50 p-3 rounded border border-gray-200 overflow-x-auto">
{`from langsmith import Client

client = Client()

# 收集用户反馈
def collect_feedback(run_id: str, thumbs_up: bool, comment: str = ""):
    client.create_feedback(
        run_id=run_id,
        key="user_rating",
        score=1 if thumbs_up else 0,
        comment=comment
    )

# 查询差评样本
low_rated = client.list_runs(
    project_name="production",
    filter='feedback.user_rating.score = 0'
)

# 添加到数据集进行改进
for run in low_rated[:10]:
    client.create_example(
        dataset_id=dataset.id,
        inputs=run.inputs,
        outputs=run.outputs,  # 期望的输出（需人工修正）
    )`}
        </pre>
      </div>
    </div>
  );
};

export default FeedbackDashboard;
