"use client";

import React, { useState } from 'react';
import { motion } from 'framer-motion';

type TokenBreakdown = {
  category: string;
  tokens: number;
  cost: number;
  percentage: number;
  color: string;
  details?: string;
};

const promptTokens: TokenBreakdown[] = [
  {
    category: 'System Prompt',
    tokens: 120,
    cost: 0.0036,
    percentage: 14.0,
    color: '#8B5CF6',
    details: 'You are a helpful customer service agent...',
  },
  {
    category: 'User Question',
    tokens: 36,
    cost: 0.00108,
    percentage: 4.2,
    color: '#3B82F6',
    details: 'How do I reset my password?',
  },
  {
    category: 'Retrieved Context',
    tokens: 700,
    cost: 0.021,
    percentage: 81.8,
    color: '#F59E0B',
    details: '10 documents × ~70 tokens each (too many!)',
  },
];

const completionTokens: TokenBreakdown[] = [
  {
    category: 'LLM Output',
    tokens: 378,
    cost: 0.0227,
    percentage: 100,
    color: '#10B981',
    details: 'Generated answer text',
  },
];

export default function TokenUsageBreakdown() {
  const [view, setView] = useState<'breakdown' | 'comparison'>('breakdown');
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);

  const totalPromptTokens = promptTokens.reduce((sum, item) => sum + item.tokens, 0);
  const totalCompletionTokens = completionTokens.reduce(
    (sum, item) => sum + item.tokens,
    0
  );
  const totalTokens = totalPromptTokens + totalCompletionTokens;
  const totalCost = [...promptTokens, ...completionTokens].reduce(
    (sum, item) => sum + item.cost,
    0
  );

  const renderTokenBar = (items: TokenBreakdown[], label: string) => {
    const total = items.reduce((sum, item) => sum + item.tokens, 0);

    return (
      <div className="mb-6">
        <div className="flex items-center justify-between mb-2">
          <div className="font-semibold text-slate-800">{label}</div>
          <div className="text-sm text-slate-600">{total.toLocaleString()} tokens</div>
        </div>

        {/* Stacked Bar */}
        <div className="h-12 bg-slate-100 rounded-lg overflow-hidden flex">
          {items.map((item) => (
            <motion.div
              key={item.category}
              className="cursor-pointer relative group"
              style={{
                width: `${item.percentage}%`,
                backgroundColor: item.color,
              }}
              whileHover={{ opacity: 0.8 }}
              onClick={() =>
                setSelectedCategory(
                  selectedCategory === item.category ? null : item.category
                )
              }
            >
              {/* Tooltip */}
              <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
                <div className="bg-slate-800 text-white text-xs px-3 py-2 rounded shadow-lg whitespace-nowrap">
                  <div className="font-semibold">{item.category}</div>
                  <div>{item.tokens} tokens ({item.percentage.toFixed(1)}%)</div>
                  <div>${item.cost.toFixed(4)}</div>
                </div>
                <div className="w-2 h-2 bg-slate-800 absolute left-1/2 -translate-x-1/2 rotate-45 -bottom-1" />
              </div>

              {/* Label (if enough space) */}
              {item.percentage > 15 && (
                <div className="absolute inset-0 flex items-center justify-center text-xs font-medium text-white">
                  {item.category}
                </div>
              )}
            </motion.div>
          ))}
        </div>

        {/* Legend */}
        <div className="mt-3 space-y-1">
          {items.map((item) => (
            <div
              key={item.category}
              className={`flex items-center gap-3 p-2 rounded cursor-pointer transition-all ${
                selectedCategory === item.category
                  ? 'bg-blue-50 border border-blue-200'
                  : 'hover:bg-slate-50'
              }`}
              onClick={() =>
                setSelectedCategory(
                  selectedCategory === item.category ? null : item.category
                )
              }
            >
              <div
                className="w-4 h-4 rounded"
                style={{ backgroundColor: item.color }}
              />
              <div className="flex-1">
                <div className="text-sm font-medium text-slate-800">
                  {item.category}
                </div>
                {item.details && (
                  <div className="text-xs text-slate-500">{item.details}</div>
                )}
              </div>
              <div className="text-sm font-medium text-slate-700">
                {item.tokens.toLocaleString()}
              </div>
              <div className="text-xs text-slate-500 w-20 text-right">
                {item.percentage.toFixed(1)}%
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-green-50 to-blue-50 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold text-center mb-2 text-slate-800">
        Token Usage Breakdown
      </h3>
      <p className="text-center text-slate-600 mb-6">
        分析 Token 消耗分布，识别成本优化机会
      </p>

      {/* Summary Cards */}
      <div className="grid grid-cols-4 gap-4 mb-6">
        <div className="bg-white rounded-lg p-4 shadow-md">
          <div className="text-xs font-semibold text-slate-500 uppercase mb-1">
            Total Tokens
          </div>
          <div className="text-2xl font-bold text-slate-800">
            {totalTokens.toLocaleString()}
          </div>
        </div>

        <div className="bg-white rounded-lg p-4 shadow-md">
          <div className="text-xs font-semibold text-slate-500 uppercase mb-1">
            Prompt
          </div>
          <div className="text-2xl font-bold text-purple-600">
            {totalPromptTokens.toLocaleString()}
          </div>
        </div>

        <div className="bg-white rounded-lg p-4 shadow-md">
          <div className="text-xs font-semibold text-slate-500 uppercase mb-1">
            Completion
          </div>
          <div className="text-2xl font-bold text-green-600">
            {totalCompletionTokens.toLocaleString()}
          </div>
        </div>

        <div className="bg-white rounded-lg p-4 shadow-md">
          <div className="text-xs font-semibold text-slate-500 uppercase mb-1">
            Est. Cost
          </div>
          <div className="text-2xl font-bold text-blue-600">
            ${totalCost.toFixed(4)}
          </div>
        </div>
      </div>

      {/* View Toggle */}
      <div className="flex justify-center gap-2 mb-6">
        <button
          onClick={() => setView('breakdown')}
          className={`px-4 py-2 rounded-lg font-medium transition-all ${
            view === 'breakdown'
              ? 'bg-blue-600 text-white'
              : 'bg-white text-slate-700 hover:bg-slate-50'
          }`}
        >
          Token Breakdown
        </button>
        <button
          onClick={() => setView('comparison')}
          className={`px-4 py-2 rounded-lg font-medium transition-all ${
            view === 'comparison'
              ? 'bg-blue-600 text-white'
              : 'bg-white text-slate-700 hover:bg-slate-50'
          }`}
        >
          Before/After Comparison
        </button>
      </div>

      {/* Content */}
      <div className="bg-white rounded-lg p-6 shadow-md">
        {view === 'breakdown' ? (
          <>
            {renderTokenBar(promptTokens, 'Prompt Tokens')}
            {renderTokenBar(completionTokens, 'Completion Tokens')}

            {/* Cost Breakdown */}
            <div className="mt-6 pt-6 border-t border-slate-200">
              <div className="font-semibold text-slate-800 mb-3">Cost Breakdown</div>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-slate-600">
                    Prompt ({totalPromptTokens} tokens × $0.03/1K):
                  </span>
                  <span className="font-medium text-slate-800">
                    ${((totalPromptTokens * 0.03) / 1000).toFixed(4)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-600">
                    Completion ({totalCompletionTokens} tokens × $0.06/1K):
                  </span>
                  <span className="font-medium text-slate-800">
                    ${((totalCompletionTokens * 0.06) / 1000).toFixed(4)}
                  </span>
                </div>
                <div className="flex justify-between pt-2 border-t border-slate-200 font-semibold">
                  <span className="text-slate-800">Total Cost:</span>
                  <span className="text-blue-600">${totalCost.toFixed(4)}</span>
                </div>
              </div>
            </div>
          </>
        ) : (
          <div className="space-y-6">
            <div className="grid grid-cols-2 gap-6">
              {/* Before Optimization */}
              <div>
                <h4 className="font-semibold text-lg mb-3 text-slate-800">
                  优化前 (Current)
                </h4>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between p-2 bg-red-50 rounded">
                    <span>Prompt Tokens:</span>
                    <span className="font-medium">856</span>
                  </div>
                  <div className="flex justify-between p-2 bg-slate-50 rounded">
                    <span>Completion Tokens:</span>
                    <span className="font-medium">378</span>
                  </div>
                  <div className="flex justify-between p-2 bg-slate-100 rounded font-semibold">
                    <span>Total:</span>
                    <span>1,234 tokens</span>
                  </div>
                  <div className="flex justify-between p-2 bg-red-100 rounded font-bold">
                    <span>Cost:</span>
                    <span className="text-red-600">$0.0484</span>
                  </div>
                </div>

                <div className="mt-4 p-3 bg-yellow-50 rounded border border-yellow-200">
                  <div className="text-xs font-semibold text-yellow-800 mb-1">
                    ⚠️ Issues
                  </div>
                  <ul className="text-xs text-yellow-700 space-y-1">
                    <li>• 检索了 10 个文档（过多）</li>
                    <li>• Context 占用 81.8% 的 Prompt</li>
                    <li>• 使用 GPT-4（昂贵且慢）</li>
                  </ul>
                </div>
              </div>

              {/* After Optimization */}
              <div>
                <h4 className="font-semibold text-lg mb-3 text-slate-800">
                  优化后 (Optimized)
                </h4>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between p-2 bg-green-50 rounded">
                    <span>Prompt Tokens:</span>
                    <span className="font-medium">256</span>
                  </div>
                  <div className="flex justify-between p-2 bg-slate-50 rounded">
                    <span>Completion Tokens:</span>
                    <span className="font-medium">198</span>
                  </div>
                  <div className="flex justify-between p-2 bg-slate-100 rounded font-semibold">
                    <span>Total:</span>
                    <span>454 tokens</span>
                  </div>
                  <div className="flex justify-between p-2 bg-green-100 rounded font-bold">
                    <span>Cost:</span>
                    <span className="text-green-600">$0.0016</span>
                  </div>
                </div>

                <div className="mt-4 p-3 bg-green-50 rounded border border-green-200">
                  <div className="text-xs font-semibold text-green-800 mb-1">
                    ✅ Improvements
                  </div>
                  <ul className="text-xs text-green-700 space-y-1">
                    <li>• 减少到 3 个文档（k=3）</li>
                    <li>• 简化 System Prompt</li>
                    <li>• 切换到 GPT-3.5-Turbo</li>
                  </ul>
                </div>
              </div>
            </div>

            {/* Savings */}
            <div className="p-4 bg-gradient-to-r from-green-50 to-blue-50 rounded-lg border-2 border-green-200">
              <div className="text-center">
                <div className="text-sm font-semibold text-slate-600 mb-1">
                  总体节省
                </div>
                <div className="text-3xl font-bold text-green-600 mb-2">
                  97% 成本 · 63% Token
                </div>
                <div className="text-xs text-slate-600">
                  每月 1000 次请求可节省 ${((totalCost - 0.0016) * 1000).toFixed(2)}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Optimization Tips */}
      <div className="mt-6 p-4 bg-blue-50 rounded-lg border border-blue-200">
        <div className="font-semibold text-sm text-slate-800 mb-2">
          💡 Token 优化建议
        </div>
        <ul className="text-sm text-slate-700 space-y-1">
          <li>• 减少检索文档数量（k 值）</li>
          <li>• 使用更短的 System Prompt</li>
          <li>• 考虑 GPT-3.5-Turbo 替代 GPT-4（非关键场景）</li>
          <li>• 对常见问题使用缓存</li>
          <li>• 实现智能 Chunk 大小控制</li>
        </ul>
      </div>
    </div>
  );
}
