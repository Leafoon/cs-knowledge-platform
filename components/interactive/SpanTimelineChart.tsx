"use client";

import React, { useState } from 'react';
import { motion } from 'framer-motion';

type Span = {
  id: string;
  name: string;
  type: 'chain' | 'llm' | 'tool' | 'retriever' | 'embedding';
  startTime: number;
  endTime: number;
  parent?: string;
  color: string;
};

const spans: Span[] = [
  {
    id: 'root',
    name: 'RetrievalQA Chain',
    type: 'chain',
    startTime: 0,
    endTime: 3200,
    color: '#8B5CF6',
  },
  {
    id: 'retriever',
    name: 'VectorStoreRetriever',
    type: 'retriever',
    startTime: 0,
    endTime: 1200,
    parent: 'root',
    color: '#F59E0B',
  },
  {
    id: 'embedding',
    name: 'OpenAIEmbeddings',
    type: 'embedding',
    startTime: 100,
    endTime: 600,
    parent: 'retriever',
    color: '#EF4444',
  },
  {
    id: 'search',
    name: 'Chroma Search',
    type: 'tool',
    startTime: 600,
    endTime: 1200,
    parent: 'retriever',
    color: '#10B981',
  },
  {
    id: 'llm',
    name: 'ChatOpenAI (GPT-4)',
    type: 'llm',
    startTime: 1200,
    endTime: 3000,
    parent: 'root',
    color: '#3B82F6',
  },
  {
    id: 'api',
    name: 'API Wait',
    type: 'llm',
    startTime: 1300,
    endTime: 2900,
    parent: 'llm',
    color: '#60A5FA',
  },
  {
    id: 'parser',
    name: 'StrOutputParser',
    type: 'tool',
    startTime: 3000,
    endTime: 3200,
    parent: 'root',
    color: '#10B981',
  },
];

export default function SpanTimelineChart() {
  const [selectedSpan, setSelectedSpan] = useState<string | null>(null);
  const [hoveredSpan, setHoveredSpan] = useState<string | null>(null);

  const totalDuration = 3200;
  const pixelsPerMs = 0.2; // 缩放比例

  // 计算层级
  const getLevel = (span: Span): number => {
    if (!span.parent) return 0;
    const parent = spans.find((s) => s.id === span.parent);
    return parent ? getLevel(parent) + 1 : 0;
  };

  // 按层级排序
  const sortedSpans = [...spans].sort((a, b) => {
    const levelA = getLevel(a);
    const levelB = getLevel(b);
    if (levelA !== levelB) return levelA - levelB;
    return a.startTime - b.startTime;
  });

  const getSpanById = (id: string) => spans.find((s) => s.id === id);
  const activeSpan = selectedSpan || hoveredSpan;
  const activeSpanData = activeSpan ? getSpanById(activeSpan) : null;

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-purple-50 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold text-center mb-2 text-slate-800">
        Span Timeline Chart
      </h3>
      <p className="text-center text-slate-600 mb-6">
        时间线视图：展示每个组件的执行时间与嵌套关系
      </p>

      {/* Timeline */}
      <div className="bg-white rounded-lg p-6 shadow-md mb-6">
        {/* Time Axis */}
        <div className="relative mb-4" style={{ height: '40px' }}>
          <div className="absolute inset-0 flex items-end">
            {[0, 500, 1000, 1500, 2000, 2500, 3000].map((time) => (
              <div
                key={time}
                className="absolute"
                style={{ left: `${(time / totalDuration) * 100}%` }}
              >
                <div className="w-px h-2 bg-slate-300" />
                <div className="text-xs text-slate-500 -translate-x-1/2 mt-1">
                  {time}ms
                </div>
              </div>
            ))}
          </div>
          <div className="absolute bottom-0 left-0 right-0 h-px bg-slate-300" />
        </div>

        {/* Spans */}
        <div className="space-y-2">
          {sortedSpans.map((span) => {
            const duration = span.endTime - span.startTime;
            const leftPercent = (span.startTime / totalDuration) * 100;
            const widthPercent = (duration / totalDuration) * 100;
            const level = getLevel(span);
            const isActive = activeSpan === span.id;

            return (
              <motion.div
                key={span.id}
                className="relative"
                style={{
                  marginLeft: `${level * 30}px`,
                  height: '40px',
                }}
                onMouseEnter={() => setHoveredSpan(span.id)}
                onMouseLeave={() => setHoveredSpan(null)}
                onClick={() => setSelectedSpan(span.id === selectedSpan ? null : span.id)}
              >
                {/* Span Label */}
                <div className="text-xs font-medium text-slate-700 mb-1 truncate">
                  {span.name}
                </div>

                {/* Span Bar */}
                <div className="relative h-6">
                  <motion.div
                    className={`absolute h-full rounded cursor-pointer transition-all ${
                      isActive ? 'ring-2 ring-blue-500 ring-offset-2' : ''
                    }`}
                    style={{
                      left: `${leftPercent}%`,
                      width: `${widthPercent}%`,
                      backgroundColor: span.color,
                      opacity: isActive ? 1 : 0.8,
                    }}
                    whileHover={{ scale: 1.05, opacity: 1 }}
                    animate={{ opacity: isActive ? 1 : 0.8 }}
                  >
                    {/* Duration Label */}
                    {widthPercent > 10 && (
                      <div className="absolute inset-0 flex items-center justify-center text-xs font-medium text-white">
                        {duration}ms
                      </div>
                    )}
                  </motion.div>
                </div>
              </motion.div>
            );
          })}
        </div>
      </div>

      {/* Statistics */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        <div className="bg-white rounded-lg p-4 shadow-md">
          <div className="text-xs font-semibold text-slate-500 uppercase mb-1">
            Total Duration
          </div>
          <div className="text-2xl font-bold text-slate-800">{totalDuration}ms</div>
          <div className="text-xs text-slate-500 mt-1">3.2 seconds</div>
        </div>

        <div className="bg-white rounded-lg p-4 shadow-md">
          <div className="text-xs font-semibold text-slate-500 uppercase mb-1">
            Slowest Component
          </div>
          <div className="text-lg font-bold text-slate-800">LLM Call</div>
          <div className="text-xs text-slate-500 mt-1">
            1800ms (56.3% of total)
          </div>
        </div>

        <div className="bg-white rounded-lg p-4 shadow-md">
          <div className="text-xs font-semibold text-slate-500 uppercase mb-1">
            Total Spans
          </div>
          <div className="text-2xl font-bold text-slate-800">{spans.length}</div>
          <div className="text-xs text-slate-500 mt-1">Including nested</div>
        </div>
      </div>

      {/* Active Span Details */}
      {activeSpanData && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white rounded-lg p-4 shadow-md border-2 border-blue-200"
        >
          <div className="flex items-center gap-3 mb-3">
            <div
              className="w-4 h-4 rounded"
              style={{ backgroundColor: activeSpanData.color }}
            />
            <div className="font-semibold text-lg text-slate-800">
              {activeSpanData.name}
            </div>
            <div className="text-xs text-slate-500 uppercase">
              {activeSpanData.type}
            </div>
          </div>

          <div className="grid grid-cols-4 gap-4 text-sm">
            <div>
              <div className="text-xs text-slate-500 mb-1">Start Time</div>
              <div className="font-medium text-slate-800">{activeSpanData.startTime}ms</div>
            </div>
            <div>
              <div className="text-xs text-slate-500 mb-1">End Time</div>
              <div className="font-medium text-slate-800">{activeSpanData.endTime}ms</div>
            </div>
            <div>
              <div className="text-xs text-slate-500 mb-1">Duration</div>
              <div className="font-medium text-slate-800">
                {activeSpanData.endTime - activeSpanData.startTime}ms
              </div>
            </div>
            <div>
              <div className="text-xs text-slate-500 mb-1">% of Total</div>
              <div className="font-medium text-slate-800">
                {(
                  ((activeSpanData.endTime - activeSpanData.startTime) / totalDuration) *
                  100
                ).toFixed(1)}
                %
              </div>
            </div>
          </div>
        </motion.div>
      )}

      {/* Performance Insights */}
      <div className="mt-6 p-4 bg-yellow-50 rounded-lg border border-yellow-200">
        <div className="font-semibold text-sm text-slate-800 mb-2">
          ⚡ Performance Insights
        </div>
        <ul className="text-sm text-slate-700 space-y-1">
          <li>• LLM Call 占用 56.3% 执行时间 (1800ms / 3200ms)</li>
          <li>• Retriever 占用 37.5% 执行时间 (1200ms / 3200ms)</li>
          <li>
            • 优化建议：考虑使用 Streaming 或 GPT-3.5-Turbo 减少 LLM 延迟
          </li>
        </ul>
      </div>

      {/* Instructions */}
      <div className="mt-4 p-4 bg-blue-50 rounded-lg border border-blue-200">
        <div className="text-sm text-slate-700">
          <span className="font-semibold">使用说明：</span>
          悬停或点击时间条查看详细信息。缩进表示嵌套关系，颜色区分组件类型。
        </div>
      </div>
    </div>
  );
}
