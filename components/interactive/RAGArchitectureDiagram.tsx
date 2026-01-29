"use client";

import React, { useState } from 'react';
import { Layers, Zap, Network, ArrowRight } from 'lucide-react';

type RAGArchitecture = 'naive' | 'advanced' | 'modular';

const architectureData: Record<RAGArchitecture, {
  title: string;
  color: string;
  steps: { label: string; description: string; }[];
  features: string[];
  pros: string[];
  cons: string[];
}> = {
  naive: {
    title: 'Naive RAG',
    color: 'blue',
    steps: [
      { label: '查询', description: 'User Query' },
      { label: '嵌入', description: 'Embedding' },
      { label: '检索', description: 'Vector Search' },
      { label: '拼接', description: 'Context + Query' },
      { label: '生成', description: 'LLM Generate' }
    ],
    features: ['简单直接', '快速实现', '适合原型'],
    pros: ['实现简单', '延迟低', '易于理解'],
    cons: ['检索精度有限', '上下文冗余', '缺乏优化']
  },
  advanced: {
    title: 'Advanced RAG',
    color: 'purple',
    steps: [
      { label: '查询优化', description: 'Query Rewriting' },
      { label: '混合检索', description: 'BM25 + Vector' },
      { label: '重排序', description: 'Reranking' },
      { label: '压缩', description: 'Compression' },
      { label: '生成', description: 'Generate' }
    ],
    features: ['查询改写', '混合检索', '重排序', '压缩'],
    pros: ['精度高', '上下文优化', '可控性强'],
    cons: ['复杂度高', '延迟增加', '成本上升']
  },
  modular: {
    title: 'Modular RAG',
    color: 'green',
    steps: [
      { label: '多路检索', description: 'Multi-path' },
      { label: '融合', description: 'Fusion' },
      { label: '迭代', description: 'Iterative' },
      { label: 'Self-RAG', description: 'Reflection' },
      { label: '引用', description: 'Citation' }
    ],
    features: ['多阶段', 'Self-RAG', 'Active Retrieval'],
    pros: ['最高精度', '可解释强', '生产就绪'],
    cons: ['复杂', '资源消耗大', '调优难']
  }
};

export default function RAGArchitectureDiagram() {
  const [selected, setSelected] = useState<RAGArchitecture>('naive');
  const data = architectureData[selected];

  const getColorClass = (variant: 'bg' | 'text' | 'border') => {
    const map: Record<string, Record<string, string>> = {
      blue: { bg: 'bg-blue-500', text: 'text-blue-600', border: 'border-blue-500' },
      purple: { bg: 'bg-purple-500', text: 'text-purple-600', border: 'border-purple-500' },
      green: { bg: 'bg-green-500', text: 'text-green-600', border: 'border-green-500' }
    };
    return map[data.color]?.[variant] || '';
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-white rounded-xl border border-slate-200 my-8">
      <h3 className="text-xl font-bold text-slate-800 mb-4">RAG 架构对比</h3>

      {/* 架构选择器 */}
      <div className="flex gap-3 mb-6">
        {(Object.keys(architectureData) as RAGArchitecture[]).map((arch) => (
          <button
            key={arch}
            onClick={() => setSelected(arch)}
            className={`flex-1 px-4 py-3 rounded-lg font-medium transition-colors ${
              selected === arch
                ? `${getColorClass('bg')} text-white`
                : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
            }`}
          >
            {architectureData[arch].title}
          </button>
        ))}
      </div>

      {/* 流程图 */}
      <div className="bg-slate-50 rounded-lg p-6 mb-6">
        <h4 className="font-semibold text-slate-700 mb-4">处理流程</h4>
        <div className="flex items-center justify-between gap-2">
          {data.steps.map((step, idx) => (
            <React.Fragment key={idx}>
              <div className="flex flex-col items-center flex-1">
                <div className={`w-16 h-16 rounded-full ${getColorClass('bg')} bg-opacity-10 flex items-center justify-center mb-2`}>
                  <div className={`w-8 h-8 rounded-full ${getColorClass('bg')} text-white flex items-center justify-center font-bold text-sm`}>
                    {idx + 1}
                  </div>
                </div>
                <div className="text-center">
                  <div className="font-medium text-slate-800 text-sm">{step.label}</div>
                  <div className="text-xs text-slate-500">{step.description}</div>
                </div>
              </div>
              {idx < data.steps.length - 1 && (
                <ArrowRight className={`${getColorClass('text')} flex-shrink-0`} size={20} />
              )}
            </React.Fragment>
          ))}
        </div>
      </div>

      {/* 特性对比 */}
      <div className="grid md:grid-cols-3 gap-4">
        <div className="bg-slate-50 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-3">
            <Layers className={getColorClass('text')} size={18} />
            <h5 className="font-semibold text-slate-800 text-sm">核心特性</h5>
          </div>
          <ul className="space-y-1">
            {data.features.map((f, i) => (
              <li key={i} className="text-sm text-slate-600 flex items-start gap-2">
                <span className={`${getColorClass('text')} mt-1`}>•</span>
                <span>{f}</span>
              </li>
            ))}
          </ul>
        </div>

        <div className="bg-green-50 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-3">
            <Zap className="text-green-600" size={18} />
            <h5 className="font-semibold text-slate-800 text-sm">优势</h5>
          </div>
          <ul className="space-y-1">
            {data.pros.map((p, i) => (
              <li key={i} className="text-sm text-slate-600 flex items-start gap-2">
                <span className="text-green-600 mt-1">✓</span>
                <span>{p}</span>
              </li>
            ))}
          </ul>
        </div>

        <div className="bg-orange-50 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-3">
            <Network className="text-orange-600" size={18} />
            <h5 className="font-semibold text-slate-800 text-sm">局限</h5>
          </div>
          <ul className="space-y-1">
            {data.cons.map((c, i) => (
              <li key={i} className="text-sm text-slate-600 flex items-start gap-2">
                <span className="text-orange-600 mt-1">⚠</span>
                <span>{c}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>

      {/* 对比表 */}
      <div className="mt-6">
        <h4 className="font-semibold text-slate-800 mb-3">架构对比</h4>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-200">
                <th className="text-left py-2 px-3 font-semibold text-slate-700">架构</th>
                <th className="text-center py-2 px-3 font-semibold text-slate-700">精度</th>
                <th className="text-center py-2 px-3 font-semibold text-slate-700">延迟</th>
                <th className="text-center py-2 px-3 font-semibold text-slate-700">成本</th>
                <th className="text-left py-2 px-3 font-semibold text-slate-700">适用场景</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-b border-slate-100">
                <td className="py-2 px-3 font-medium text-blue-600">Naive RAG</td>
                <td className="text-center py-2 px-3">⭐⭐⭐</td>
                <td className="text-center py-2 px-3">低</td>
                <td className="text-center py-2 px-3">低</td>
                <td className="py-2 px-3 text-slate-600">快速原型、简单问答</td>
              </tr>
              <tr className="border-b border-slate-100">
                <td className="py-2 px-3 font-medium text-purple-600">Advanced RAG</td>
                <td className="text-center py-2 px-3">⭐⭐⭐⭐</td>
                <td className="text-center py-2 px-3">中</td>
                <td className="text-center py-2 px-3">中</td>
                <td className="py-2 px-3 text-slate-600">企业应用、专业问答</td>
              </tr>
              <tr>
                <td className="py-2 px-3 font-medium text-green-600">Modular RAG</td>
                <td className="text-center py-2 px-3">⭐⭐⭐⭐⭐</td>
                <td className="text-center py-2 px-3">高</td>
                <td className="text-center py-2 px-3">高</td>
                <td className="py-2 px-3 text-slate-600">生产级系统、复杂场景</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
