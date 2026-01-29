"use client";

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { DollarSign, TrendingUp, AlertCircle, Lightbulb, PieChart } from 'lucide-react';

const CostAnalysisDashboard: React.FC = () => {
  const [timeRange, setTimeRange] = useState<'day' | 'week' | 'month'>('week');

  const costData = {
    day: {
      total: 12.45,
      byModel: {
        'gpt-4': 8.50,
        'gpt-4-turbo': 2.30,
        'gpt-3.5-turbo': 1.65,
      },
      byFunction: {
        '聊天对话': 4.20,
        '文档摘要': 3.50,
        '代码生成': 2.80,
        '翻译': 1.95,
      },
      trend: 15,
    },
    week: {
      total: 87.20,
      byModel: {
        'gpt-4': 59.50,
        'gpt-4-turbo': 16.10,
        'gpt-3.5-turbo': 11.60,
      },
      byFunction: {
        '聊天对话': 29.40,
        '文档摘要': 24.50,
        '代码生成': 19.60,
        '翻译': 13.70,
      },
      trend: 12,
    },
    month: {
      total: 356.80,
      byModel: {
        'gpt-4': 243.70,
        'gpt-4-turbo': 65.90,
        'gpt-3.5-turbo': 47.20,
      },
      byFunction: {
        '聊天对话': 120.30,
        '文档摘要': 100.20,
        '代码生成': 80.30,
        '翻译': 56.00,
      },
      trend: 8,
    },
  };

  const data = costData[timeRange];

  const recommendations = [
    {
      priority: 'HIGH',
      title: 'GPT-4 使用率过高',
      description: 'GPT-4 占总成本 68.3%，考虑对简单任务降级使用 GPT-3.5 Turbo',
      potential_savings: '$15-20/week',
      color: 'red',
    },
    {
      priority: 'MEDIUM',
      title: '聊天对话平均 Token 数过高',
      description: '平均 1200 tokens/对话，建议优化 Prompt 模板移除冗余指令',
      potential_savings: '$8-12/week',
      color: 'yellow',
    },
    {
      priority: 'LOW',
      title: '缓存命中率较低',
      description: '当前缓存命中率 35%，增加语义缓存可节省重复请求成本',
      potential_savings: '$5-8/week',
      color: 'blue',
    },
  ];

  const CustomPieChart: React.FC<{ data: Record<string, number>; colors: string[] }> = ({ data, colors }) => {
    if (!data || Object.keys(data).length === 0) {
      return (
        <div className="relative w-48 h-48 mx-auto flex items-center justify-center">
          <p className="text-sm text-gray-400">暂无数据</p>
        </div>
      );
    }

    const total = Object.values(data).reduce((sum, val) => sum + val, 0);
    let cumulativePercent = 0;

    return (
      <div className="relative w-48 h-48 mx-auto">
        <svg viewBox="0 0 100 100" className="transform -rotate-90">
          {Object.entries(data).map(([key, value], idx) => {
            const percent = (value / total) * 100;
            const startPercent = cumulativePercent;
            cumulativePercent += percent;

            const startAngle = (startPercent / 100) * 360;
            const endAngle = (cumulativePercent / 100) * 360;

            const x1 = 50 + 45 * Math.cos((startAngle * Math.PI) / 180);
            const y1 = 50 + 45 * Math.sin((startAngle * Math.PI) / 180);
            const x2 = 50 + 45 * Math.cos((endAngle * Math.PI) / 180);
            const y2 = 50 + 45 * Math.sin((endAngle * Math.PI) / 180);

            const largeArcFlag = percent > 50 ? 1 : 0;

            return (
              <path
                key={key}
                d={`M 50 50 L ${x1} ${y1} A 45 45 0 ${largeArcFlag} 1 ${x2} ${y2} Z`}
                fill={colors[idx % colors.length]}
                stroke="white"
                strokeWidth="0.5"
              />
            );
          })}
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-center">
            <p className="text-2xl font-bold text-gray-800">${total.toFixed(2)}</p>
            <p className="text-xs text-gray-500">总成本</p>
          </div>
        </div>
      </div>
    );
  };

  const modelColors = ['#3b82f6', '#8b5cf6', '#10b981'];
  const functionColors = ['#f59e0b', '#ef4444', '#06b6d4', '#ec4899'];

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-green-50 to-emerald-50 rounded-xl shadow-lg">
      <div className="mb-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-2xl font-bold text-gray-800 flex items-center gap-2">
              <DollarSign className="w-6 h-6 text-green-600" />
              成本分析仪表盘
            </h3>
            <p className="text-gray-600">追踪 Token 消耗与成本，发现优化机会</p>
          </div>
        </div>

        {/* 时间范围选择器 */}
        <div className="flex gap-2">
          {(['day', 'week', 'month'] as const).map((range) => (
            <button
              key={range}
              onClick={() => setTimeRange(range)}
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                timeRange === range
                  ? 'bg-green-600 text-white'
                  : 'bg-white text-gray-700 hover:bg-gray-100'
              }`}
            >
              {range === 'day' ? '今天' : range === 'week' ? '本周' : '本月'}
            </button>
          ))}
        </div>
      </div>

      {/* 总成本卡片 */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        <div className="col-span-1 p-6 bg-white rounded-lg shadow border border-gray-200">
          <div className="flex items-center gap-2 mb-2">
            <DollarSign className="w-5 h-5 text-green-500" />
            <span className="text-sm text-gray-600">总成本</span>
          </div>
          <p className="text-4xl font-bold text-gray-800 mb-1">${data.total.toFixed(2)}</p>
          <div className="flex items-center gap-1">
            <TrendingUp className="w-4 h-4 text-red-500" />
            <span className="text-sm text-red-600">+{data.trend}% vs 上周</span>
          </div>
        </div>

        <div className="col-span-2 p-6 bg-white rounded-lg shadow border border-gray-200">
          <h4 className="text-sm font-semibold text-gray-700 mb-4">每日成本趋势</h4>
          <div className="flex items-end gap-1 h-32">
            {[8.2, 9.5, 11.2, 10.8, 12.1, 13.5, 12.45].map((cost, idx) => (
              <div key={idx} className="flex-1 flex flex-col justify-end">
                <div
                  className="bg-green-400 rounded-t hover:bg-green-500 transition-colors cursor-pointer"
                  style={{ height: `${(cost / 15) * 100}%` }}
                  title={`$${cost}`}
                />
                <p className="text-xs text-gray-500 text-center mt-1">
                  {['周一', '周二', '周三', '周四', '周五', '周六', '周日'][idx]}
                </p>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* 按模型和功能拆分 */}
      <div className="grid grid-cols-2 gap-6 mb-6">
        {/* 按模型拆分 */}
        <div className="p-6 bg-white rounded-lg shadow border border-gray-200">
          <h4 className="text-lg font-semibold text-gray-800 mb-4 flex items-center gap-2">
            <PieChart className="w-5 h-5 text-blue-500" />
            按模型拆分
          </h4>
          <CustomPieChart data={data.byModel} colors={modelColors} />
          <div className="mt-4 space-y-2">
            {Object.entries(data.byModel).map(([model, cost], idx) => {
              const percentage = ((cost / data.total) * 100).toFixed(1);
              return (
                <div key={model} className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <div
                      className="w-3 h-3 rounded"
                      style={{ backgroundColor: modelColors[idx % modelColors.length] }}
                    />
                    <span className="text-sm text-gray-700">{model}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-semibold text-gray-800">${cost.toFixed(2)}</span>
                    <span className="text-xs text-gray-500">({percentage}%)</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* 按功能拆分 */}
        <div className="p-6 bg-white rounded-lg shadow border border-gray-200">
          <h4 className="text-lg font-semibold text-gray-800 mb-4 flex items-center gap-2">
            <PieChart className="w-5 h-5 text-orange-500" />
            按功能拆分
          </h4>
          <CustomPieChart data={data.byFunction} colors={functionColors} />
          <div className="mt-4 space-y-2">
            {Object.entries(data.byFunction).map(([func, cost], idx) => {
              const percentage = ((cost / data.total) * 100).toFixed(1);
              return (
                <div key={func} className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <div
                      className="w-3 h-3 rounded"
                      style={{ backgroundColor: functionColors[idx % functionColors.length] }}
                    />
                    <span className="text-sm text-gray-700">{func}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-semibold text-gray-800">${cost.toFixed(2)}</span>
                    <span className="text-xs text-gray-500">({percentage}%)</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* 优化建议 */}
      <div className="p-6 bg-white rounded-lg shadow border border-gray-200">
        <h4 className="text-lg font-semibold text-gray-800 mb-4 flex items-center gap-2">
          <Lightbulb className="w-5 h-5 text-yellow-500" />
          成本优化建议
        </h4>
        <div className="space-y-3">
          {recommendations.map((rec, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: idx * 0.1 }}
              className={`p-4 rounded-lg border-2 ${
                rec.color === 'red'
                  ? 'border-red-200 bg-red-50'
                  : rec.color === 'yellow'
                  ? 'border-yellow-200 bg-yellow-50'
                  : 'border-blue-200 bg-blue-50'
              }`}
            >
              <div className="flex items-start gap-3">
                <AlertCircle className={`w-5 h-5 flex-shrink-0 mt-0.5 ${
                  rec.color === 'red'
                    ? 'text-red-500'
                    : rec.color === 'yellow'
                    ? 'text-yellow-500'
                    : 'text-blue-500'
                }`} />
                <div className="flex-grow">
                  <div className="flex items-center gap-2 mb-1">
                    <span className={`px-2 py-0.5 rounded text-xs font-bold ${
                      rec.color === 'red'
                        ? 'bg-red-200 text-red-800'
                        : rec.color === 'yellow'
                        ? 'bg-yellow-200 text-yellow-800'
                        : 'bg-blue-200 text-blue-800'
                    }`}>
                      {rec.priority}
                    </span>
                    <h5 className="font-semibold text-gray-800">{rec.title}</h5>
                  </div>
                  <p className="text-sm text-gray-700 mb-2">{rec.description}</p>
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-gray-600">💰 潜在节省:</span>
                    <span className="text-sm font-semibold text-green-600">{rec.potential_savings}</span>
                  </div>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Token 消耗统计 */}
      <div className="mt-6 p-6 bg-white rounded-lg shadow border border-gray-200">
        <h4 className="text-lg font-semibold text-gray-800 mb-4">Token 消耗统计</h4>
        <div className="grid grid-cols-4 gap-4">
          <div className="text-center">
            <p className="text-sm text-gray-600 mb-1">Prompt Tokens</p>
            <p className="text-2xl font-bold text-gray-800">1.2M</p>
            <p className="text-xs text-gray-500">$36.00</p>
          </div>
          <div className="text-center">
            <p className="text-sm text-gray-600 mb-1">Completion Tokens</p>
            <p className="text-2xl font-bold text-gray-800">850K</p>
            <p className="text-xs text-gray-500">$51.20</p>
          </div>
          <div className="text-center">
            <p className="text-sm text-gray-600 mb-1">总 Tokens</p>
            <p className="text-2xl font-bold text-gray-800">2.05M</p>
            <p className="text-xs text-gray-500">{timeRange === 'day' ? '今日' : timeRange === 'week' ? '本周' : '本月'}</p>
          </div>
          <div className="text-center">
            <p className="text-sm text-gray-600 mb-1">平均/请求</p>
            <p className="text-2xl font-bold text-gray-800">425</p>
            <p className="text-xs text-gray-500">tokens</p>
          </div>
        </div>
      </div>

      {/* 说明 */}
      <div className="mt-6 p-4 bg-white rounded-lg border border-green-200">
        <h4 className="font-semibold text-gray-800 mb-2">💡 成本优化策略</h4>
        <ul className="text-sm text-gray-600 space-y-1">
          <li>• <strong>模型降级</strong>：简单任务使用 GPT-3.5 可节省 70-90% 成本</li>
          <li>• <strong>Prompt 优化</strong>：移除冗余指令，减少不必要的 Token</li>
          <li>• <strong>缓存机制</strong>：相似问题重用结果，避免重复 API 调用</li>
          <li>• <strong>批处理</strong>：合并多个请求，减少 Overhead</li>
          <li>• <strong>预算告警</strong>：设置每日/每月成本上限，超限自动通知</li>
        </ul>
      </div>
    </div>
  );
};

export default CostAnalysisDashboard;
