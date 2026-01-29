'use client';

import React, { useState, useMemo } from 'react';

type SpanData = {
  id: string;
  name: string;
  type: 'chain' | 'llm' | 'tool' | 'retriever' | 'parser';
  startTime: number;
  endTime: number;
  inputs: Record<string, any>;
  outputs: Record<string, any>;
  metadata?: {
    tokens?: { prompt: number; completion: number; total: number };
    model?: string;
    cost?: number;
  };
  children?: SpanData[];
  status: 'success' | 'error';
  error?: string;
};

export default function LangSmithTraceVisualization() {
  const [selectedSpan, setSelectedSpan] = useState<string | null>(null);
  const [expandedSpans, setExpandedSpans] = useState<Set<string>>(new Set(['root']));

  const traceData: SpanData = useMemo(() => ({
    id: 'root',
    name: 'Customer Support Chain',
    type: 'chain',
    startTime: 0,
    endTime: 2300,
    inputs: { question: '如何重置密码？' },
    outputs: { answer: '点击登录页面的"忘记密码"链接，输入您的注册邮箱...' },
    status: 'success',
    children: [
      {
        id: 'prompt',
        name: 'PromptTemplate.format',
        type: 'chain',
        startTime: 0,
        endTime: 50,
        inputs: { role: '客服专家', question: '如何重置密码？' },
        outputs: {
          messages: [
            { role: 'system', content: '你是专业客服专家。' },
            { role: 'user', content: '如何重置密码？' }
          ]
        },
        status: 'success'
      },
      {
        id: 'llm',
        name: 'ChatOpenAI.invoke',
        type: 'llm',
        startTime: 50,
        endTime: 2100,
        inputs: {
          messages: [
            { role: 'system', content: '你是专业客服专家。' },
            { role: 'user', content: '如何重置密码？' }
          ]
        },
        outputs: {
          content: '点击登录页面的"忘记密码"链接，输入您的注册邮箱...'
        },
        metadata: {
          model: 'gpt-4',
          tokens: { prompt: 28, completion: 156, total: 184 },
          cost: 0.00552
        },
        status: 'success'
      },
      {
        id: 'parser',
        name: 'StrOutputParser.parse',
        type: 'parser',
        startTime: 2100,
        endTime: 2300,
        inputs: { content: '点击登录页面的"忘记密码"链接，输入您的注册邮箱...' },
        outputs: { result: '点击登录页面的"忘记密码"链接，输入您的注册邮箱...' },
        status: 'success'
      }
    ]
  }), []);

  const toggleExpand = (spanId: string) => {
    const newExpanded = new Set(expandedSpans);
    if (newExpanded.has(spanId)) {
      newExpanded.delete(spanId);
    } else {
      newExpanded.add(spanId);
    }
    setExpandedSpans(newExpanded);
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'chain': return { bg: 'bg-blue-100 dark:bg-blue-900/30', border: 'border-blue-500', text: 'text-blue-700 dark:text-blue-300' };
      case 'llm': return { bg: 'bg-purple-100 dark:bg-purple-900/30', border: 'border-purple-500', text: 'text-purple-700 dark:text-purple-300' };
      case 'tool': return { bg: 'bg-green-100 dark:bg-green-900/30', border: 'border-green-500', text: 'text-green-700 dark:text-green-300' };
      case 'retriever': return { bg: 'bg-orange-100 dark:bg-orange-900/30', border: 'border-orange-500', text: 'text-orange-700 dark:text-orange-300' };
      case 'parser': return { bg: 'bg-gray-100 dark:bg-gray-700', border: 'border-gray-500', text: 'text-gray-700 dark:text-gray-300' };
      default: return { bg: 'bg-gray-100', border: 'border-gray-500', text: 'text-gray-700' };
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'chain': return '🔗';
      case 'llm': return '🤖';
      case 'tool': return '🔧';
      case 'retriever': return '🔍';
      case 'parser': return '📝';
      default: return '📦';
    }
  };

  const renderSpan = (span: SpanData, depth: number = 0) => {
    const isExpanded = expandedSpans.has(span.id);
    const isSelected = selectedSpan === span.id;
    const hasChildren = span.children && span.children.length > 0;
    const colors = getTypeColor(span.type);
    const duration = span.endTime - span.startTime;

    return (
      <div key={span.id} className="mb-2">
        <div
          className={`flex items-center gap-3 p-3 rounded-lg cursor-pointer transition-all ${
            isSelected
              ? `${colors.bg} border-2 ${colors.border} shadow-lg scale-105`
              : `bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 hover:shadow-md`
          }`}
          style={{ marginLeft: `${depth * 24}px` }}
          onClick={() => setSelectedSpan(span.id)}
        >
          {hasChildren && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                toggleExpand(span.id);
              }}
              className="flex-shrink-0 w-6 h-6 flex items-center justify-center text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
            >
              {isExpanded ? '▼' : '▶'}
            </button>
          )}
          {!hasChildren && <div className="w-6" />}

          <div className="flex-shrink-0 text-2xl">{getTypeIcon(span.type)}</div>

          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <span className={`font-semibold ${colors.text}`}>{span.name}</span>
              <span className={`text-xs px-2 py-0.5 rounded-full ${colors.bg} ${colors.text} border ${colors.border}`}>
                {span.type}
              </span>
            </div>
            <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
              Duration: {duration}ms
              {span.metadata?.tokens && (
                <span className="ml-3">
                  Tokens: {span.metadata.tokens.total} ({span.metadata.tokens.prompt}+{span.metadata.tokens.completion})
                </span>
              )}
              {span.metadata?.cost && (
                <span className="ml-3">Cost: ${span.metadata.cost.toFixed(5)}</span>
              )}
            </div>
          </div>

          <div className="flex-shrink-0">
            {span.status === 'success' ? (
              <div className="w-8 h-8 bg-green-500 rounded-full flex items-center justify-center shadow-lg">
                <span className="text-white font-bold">✓</span>
              </div>
            ) : (
              <div className="w-8 h-8 bg-red-500 rounded-full flex items-center justify-center shadow-lg">
                <span className="text-white font-bold">✗</span>
              </div>
            )}
          </div>
        </div>

        {hasChildren && isExpanded && (
          <div className="mt-2">
            {span.children!.map((child) => renderSpan(child, depth + 1))}
          </div>
        )}
      </div>
    );
  };

  const selectedSpanData = useMemo(() => {
    const findSpan = (span: SpanData): SpanData | null => {
      if (span.id === selectedSpan) return span;
      if (span.children) {
        for (const child of span.children) {
          const found = findSpan(child);
          if (found) return found;
        }
      }
      return null;
    };
    return selectedSpan ? findSpan(traceData) : null;
  }, [selectedSpan, traceData]);

  return (
    <div className="my-8 p-8 bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-900 rounded-2xl shadow-xl border border-gray-200 dark:border-gray-700">
      <h3 className="text-2xl font-bold mb-2 bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
        LangSmith Trace 可视化
      </h3>
      <p className="text-sm text-gray-600 dark:text-gray-400 mb-6">
        探索 Trace 层级结构，查看每个 Span 的输入输出和性能指标
      </p>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-lg">
            <h4 className="font-bold mb-4 flex items-center gap-2 text-gray-800 dark:text-gray-200">
              <span className="text-xl">🔍</span>
              Trace 树结构
            </h4>
            <div className="space-y-2">
              {renderSpan(traceData)}
            </div>
          </div>

          <div className="mt-6 grid grid-cols-3 gap-4">
            <div className="p-4 bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/30 dark:to-blue-800/30 rounded-xl shadow-md border border-blue-200 dark:border-blue-700">
              <div className="text-3xl font-bold text-blue-600 dark:text-blue-400">2.3s</div>
              <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">总延迟</div>
            </div>
            <div className="p-4 bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-900/30 dark:to-purple-800/30 rounded-xl shadow-md border border-purple-200 dark:border-purple-700">
              <div className="text-3xl font-bold text-purple-600 dark:text-purple-400">184</div>
              <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">总 Token</div>
            </div>
            <div className="p-4 bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/30 dark:to-green-800/30 rounded-xl shadow-md border border-green-200 dark:border-green-700">
              <div className="text-3xl font-bold text-green-600 dark:text-green-400">$0.0055</div>
              <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">总成本</div>
            </div>
          </div>
        </div>

        <div className="lg:col-span-1">
          <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-lg sticky top-4">
            <h4 className="font-bold mb-4 flex items-center gap-2 text-gray-800 dark:text-gray-200">
              <span className="text-xl">📊</span>
              Span 详情
            </h4>
            {selectedSpanData ? (
              <div className="space-y-4">
                <div>
                  <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">名称</div>
                  <div className="font-semibold text-gray-800 dark:text-gray-200">{selectedSpanData.name}</div>
                </div>
                <div>
                  <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">类型</div>
                  <div className="font-mono text-sm text-gray-700 dark:text-gray-300">{selectedSpanData.type}</div>
                </div>
                <div>
                  <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">状态</div>
                  <div className={`inline-block px-3 py-1 rounded-full text-sm font-semibold ${
                    selectedSpanData.status === 'success'
                      ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-300'
                      : 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300'
                  }`}>
                    {selectedSpanData.status === 'success' ? '✓ Success' : '✗ Error'}
                  </div>
                </div>
                <div>
                  <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">输入</div>
                  <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3 font-mono text-xs overflow-x-auto">
                    <pre className="text-gray-700 dark:text-gray-300">{JSON.stringify(selectedSpanData.inputs, null, 2)}</pre>
                  </div>
                </div>
                <div>
                  <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">输出</div>
                  <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3 font-mono text-xs overflow-x-auto">
                    <pre className="text-gray-700 dark:text-gray-300">{JSON.stringify(selectedSpanData.outputs, null, 2)}</pre>
                  </div>
                </div>
                {selectedSpanData.metadata && (
                  <div>
                    <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">元数据</div>
                    <div className="space-y-2">
                      {selectedSpanData.metadata.model && (
                        <div className="flex justify-between items-center p-2 bg-blue-50 dark:bg-blue-900/20 rounded">
                          <span className="text-xs text-gray-600 dark:text-gray-400">模型</span>
                          <span className="font-mono text-xs text-blue-600 dark:text-blue-400">{selectedSpanData.metadata.model}</span>
                        </div>
                      )}
                      {selectedSpanData.metadata.tokens && (
                        <>
                          <div className="flex justify-between items-center p-2 bg-purple-50 dark:bg-purple-900/20 rounded">
                            <span className="text-xs text-gray-600 dark:text-gray-400">Prompt Tokens</span>
                            <span className="font-mono text-xs text-purple-600 dark:text-purple-400">{selectedSpanData.metadata.tokens.prompt}</span>
                          </div>
                          <div className="flex justify-between items-center p-2 bg-purple-50 dark:bg-purple-900/20 rounded">
                            <span className="text-xs text-gray-600 dark:text-gray-400">Completion Tokens</span>
                            <span className="font-mono text-xs text-purple-600 dark:text-purple-400">{selectedSpanData.metadata.tokens.completion}</span>
                          </div>
                        </>
                      )}
                      {selectedSpanData.metadata.cost && (
                        <div className="flex justify-between items-center p-2 bg-green-50 dark:bg-green-900/20 rounded">
                          <span className="text-xs text-gray-600 dark:text-gray-400">成本</span>
                          <span className="font-mono text-xs text-green-600 dark:text-green-400">${selectedSpanData.metadata.cost.toFixed(5)}</span>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center py-8 text-gray-400 dark:text-gray-500">
                <div className="text-4xl mb-2">👈</div>
                <div className="text-sm">点击左侧 Span 查看详情</div>
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="mt-6 p-6 bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-2xl border-l-4 border-blue-500 shadow-lg">
        <div className="flex items-start gap-4">
          <div className="flex-shrink-0 w-10 h-10 bg-blue-500 rounded-full flex items-center justify-center shadow-lg">
            <span className="text-white text-xl">💡</span>
          </div>
          <div>
            <h4 className="font-bold text-gray-800 dark:text-gray-100 mb-2">Trace 分析洞察</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300 leading-relaxed">
              LLM 调用占总延迟的 <strong>91%</strong> (2.1s/2.3s)，是主要性能瓶颈。
              可考虑：1) 使用缓存减少重复调用；2) 切换到更快的模型（GPT-3.5）；3) 添加流式输出改善用户体验。
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
