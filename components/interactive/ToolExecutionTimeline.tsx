"use client";

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Clock, CheckCircle2, XCircle, AlertTriangle, Play, RotateCcw } from 'lucide-react';

interface ExecutionEvent {
  id: string;
  timestamp: number;
  type: 'start' | 'tool_call' | 'tool_result' | 'error' | 'retry' | 'success';
  tool?: string;
  message: string;
  data?: any;
}

const SCENARIOS = {
  success: {
    name: '成功执行',
    description: '所有工具调用顺利完成',
    events: [
      { type: 'start', tool: undefined, message: '开始处理用户请求', data: { query: 'What is 15 + 23?' } },
      { type: 'tool_call', tool: 'calculator', message: '调用 calculator 工具', data: { expression: '15 + 23' } },
      { type: 'tool_result', tool: 'calculator', message: '工具执行成功', data: { result: 38 } },
      { type: 'success', tool: undefined, message: '生成最终响应', data: { response: '15 + 23 equals 38' } }
    ]
  },
  error_recovery: {
    name: '错误恢复',
    description: '首次调用失败，重试后成功',
    events: [
      { type: 'start', tool: undefined, message: '开始处理用户请求', data: { query: 'Get weather in Tokyo' } },
      { type: 'tool_call', tool: 'get_weather', message: '调用 get_weather 工具', data: { city: 'Tokyo' } },
      { type: 'error', tool: 'get_weather', message: 'API 超时错误', data: { error: 'Request timeout' } },
      { type: 'retry', tool: 'get_weather', message: '重试工具调用', data: { attempt: 2 } },
      { type: 'tool_result', tool: 'get_weather', message: '重试成功', data: { result: 'Sunny, 18°C' } },
      { type: 'success', tool: undefined, message: '生成最终响应', data: { response: 'The weather in Tokyo is sunny, 18°C' } }
    ]
  },
  parallel: {
    name: '并行调用',
    description: '同时调用多个工具',
    events: [
      { type: 'start', tool: undefined, message: '开始处理用户请求', data: { query: 'Weather in Beijing and time in Shanghai' } },
      { type: 'tool_call', tool: 'get_weather', message: '调用 get_weather 工具', data: { city: 'Beijing' } },
      { type: 'tool_call', tool: 'get_time', message: '并行调用 get_time 工具', data: { timezone: 'Asia/Shanghai' } },
      { type: 'tool_result', tool: 'get_weather', message: 'get_weather 返回结果', data: { result: 'Sunny, 15°C' } },
      { type: 'tool_result', tool: 'get_time', message: 'get_time 返回结果', data: { result: '14:30' } },
      { type: 'success', tool: undefined, message: '生成最终响应', data: { response: 'Beijing: Sunny, 15°C. Shanghai time: 14:30' } }
    ]
  },
  timeout: {
    name: '超时处理',
    description: '工具执行超时并fallback',
    events: [
      { type: 'start', tool: undefined, message: '开始处理用户请求', data: { query: 'Search complex database' } },
      { type: 'tool_call', tool: 'search_database', message: '调用 search_database 工具', data: { query: 'complex query' } },
      { type: 'error', tool: 'search_database', message: '执行超时（5秒）', data: { error: 'Timeout after 5s' } },
      { type: 'tool_call', tool: 'search_cache', message: 'Fallback 到缓存搜索', data: { query: 'complex query' } },
      { type: 'tool_result', tool: 'search_cache', message: '从缓存返回结果', data: { result: 'Cached results' } },
      { type: 'success', tool: undefined, message: '使用缓存数据响应', data: { response: 'Found results from cache' } }
    ]
  }
};

export default function ToolExecutionTimeline() {
  const [selectedScenario, setSelectedScenario] = useState<keyof typeof SCENARIOS>('success');
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentEventIndex, setCurrentEventIndex] = useState(-1);
  const [executionEvents, setExecutionEvents] = useState<ExecutionEvent[]>([]);

  const scenario = SCENARIOS[selectedScenario];

  const playScenario = () => {
    setIsPlaying(true);
    setCurrentEventIndex(-1);
    setExecutionEvents([]);

    let eventIndex = 0;
    const startTime = Date.now();

    const interval = setInterval(() => {
      if (eventIndex >= scenario.events.length) {
        clearInterval(interval);
        setIsPlaying(false);
        return;
      }

      const event = scenario.events[eventIndex];
      const newEvent: ExecutionEvent = {
        id: `event-${eventIndex}`,
        timestamp: Date.now() - startTime,
        type: event.type as any,
        tool: event.tool,
        message: event.message,
        data: event.data
      };

      setExecutionEvents(prev => [...prev, newEvent]);
      setCurrentEventIndex(eventIndex);
      eventIndex++;
    }, 1000);
  };

  const resetTimeline = () => {
    setExecutionEvents([]);
    setCurrentEventIndex(-1);
    setIsPlaying(false);
  };

  const getEventIcon = (type: string) => {
    switch (type) {
      case 'start':
        return <Play className="w-5 h-5 text-blue-500" />;
      case 'tool_call':
        return <Clock className="w-5 h-5 text-purple-500 animate-spin" />;
      case 'tool_result':
        return <CheckCircle2 className="w-5 h-5 text-green-500" />;
      case 'error':
        return <XCircle className="w-5 h-5 text-red-500" />;
      case 'retry':
        return <RotateCcw className="w-5 h-5 text-orange-500" />;
      case 'success':
        return <CheckCircle2 className="w-5 h-5 text-green-600" />;
      default:
        return <AlertTriangle className="w-5 h-5 text-slate-400" />;
    }
  };

  const getEventColor = (type: string) => {
    switch (type) {
      case 'start':
        return 'bg-blue-50 border-blue-200';
      case 'tool_call':
        return 'bg-purple-50 border-purple-200';
      case 'tool_result':
        return 'bg-green-50 border-green-200';
      case 'error':
        return 'bg-red-50 border-red-200';
      case 'retry':
        return 'bg-orange-50 border-orange-200';
      case 'success':
        return 'bg-green-100 border-green-300';
      default:
        return 'bg-slate-50 border-slate-200';
    }
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-purple-50 rounded-xl border border-slate-200">
      <div className="mb-6">
        <h3 className="text-2xl font-bold text-slate-800 mb-2">Tool Execution Timeline</h3>
        <p className="text-slate-600">可视化工具调用时间线，包含错误处理与重试</p>
      </div>

      {/* Scenario Selection */}
      <div className="mb-6 p-4 bg-white rounded-lg border border-slate-200">
        <label className="block text-sm font-semibold text-slate-700 mb-3">
          选择场景：
        </label>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {(Object.keys(SCENARIOS) as Array<keyof typeof SCENARIOS>).map(key => (
            <button
              key={key}
              onClick={() => { setSelectedScenario(key); resetTimeline(); }}
              disabled={isPlaying}
              className={`p-3 rounded-lg border-2 transition-all text-left ${
                selectedScenario === key
                  ? 'bg-purple-50 border-purple-300 shadow-md'
                  : 'bg-white border-slate-200 hover:border-slate-300'
              } disabled:opacity-50 disabled:cursor-not-allowed`}
            >
              <div className="font-semibold text-sm mb-1">{SCENARIOS[key].name}</div>
              <div className="text-xs text-slate-600">{SCENARIOS[key].description}</div>
            </button>
          ))}
        </div>
      </div>

      {/* Timeline */}
      <div className="mb-6 bg-white rounded-lg border border-slate-200 p-6">
        <div className="flex items-center justify-between mb-6">
          <h4 className="font-semibold text-slate-800">执行时间线</h4>
          <div className="flex gap-3">
            <button
              onClick={playScenario}
              disabled={isPlaying}
              className="px-4 py-2 bg-purple-500 text-white rounded-lg hover:bg-purple-600 disabled:bg-slate-300 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
            >
              <Play className={`w-4 h-4 ${isPlaying ? 'animate-pulse' : ''}`} />
              {isPlaying ? '执行中...' : '开始执行'}
            </button>
            <button
              onClick={resetTimeline}
              disabled={isPlaying}
              className="px-4 py-2 bg-slate-200 text-slate-700 rounded-lg hover:bg-slate-300 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              重置
            </button>
          </div>
        </div>

        {executionEvents.length === 0 && (
          <div className="text-center py-12 text-slate-400">
            点击"开始执行"查看工具调用时间线
          </div>
        )}

        <div className="relative">
          {/* Timeline Line */}
          {executionEvents.length > 0 && (
            <div className="absolute left-6 top-0 bottom-0 w-0.5 bg-slate-200" />
          )}

          {/* Events */}
          <div className="space-y-4">
            <AnimatePresence>
              {executionEvents.map((event, idx) => (
                <motion.div
                  key={event.id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.3 }}
                  className="relative"
                >
                  {/* Timeline Dot */}
                  <div className="absolute left-6 top-4 w-3 h-3 rounded-full bg-white border-2 border-purple-500 -translate-x-1/2 z-10" />

                  {/* Event Card */}
                  <div className={`ml-14 p-4 rounded-lg border-2 ${getEventColor(event.type)}`}>
                    <div className="flex items-start gap-3">
                      <div className="flex-shrink-0 mt-0.5">
                        {getEventIcon(event.type)}
                      </div>

                      <div className="flex-1">
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center gap-2">
                            <span className="font-semibold text-slate-800">{event.message}</span>
                            {event.tool && (
                              <span className="px-2 py-0.5 bg-slate-100 text-slate-700 rounded text-xs font-mono">
                                {event.tool}
                              </span>
                            )}
                          </div>
                          <span className="text-xs text-slate-500">
                            +{(event.timestamp / 1000).toFixed(1)}s
                          </span>
                        </div>

                        {event.data && (
                          <pre className="bg-white p-2 rounded text-xs font-mono overflow-x-auto border border-slate-200">
                            {JSON.stringify(event.data, null, 2)}
                          </pre>
                        )}
                      </div>
                    </div>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        </div>
      </div>

      {/* Stats */}
      {executionEvents.length > 0 && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-white rounded-lg border border-slate-200 p-4 text-center">
            <div className="text-3xl font-bold text-purple-600">{executionEvents.length}</div>
            <div className="text-sm text-slate-600 mt-1">总事件数</div>
          </div>
          <div className="bg-white rounded-lg border border-slate-200 p-4 text-center">
            <div className="text-3xl font-bold text-blue-600">
              {executionEvents.filter(e => e.type === 'tool_call').length}
            </div>
            <div className="text-sm text-slate-600 mt-1">工具调用</div>
          </div>
          <div className="bg-white rounded-lg border border-slate-200 p-4 text-center">
            <div className="text-3xl font-bold text-red-600">
              {executionEvents.filter(e => e.type === 'error').length}
            </div>
            <div className="text-sm text-slate-600 mt-1">错误次数</div>
          </div>
          <div className="bg-white rounded-lg border border-slate-200 p-4 text-center">
            <div className="text-3xl font-bold text-green-600">
              {executionEvents.length > 0 ? ((executionEvents[executionEvents.length - 1].timestamp / 1000).toFixed(1)) : '0'}s
            </div>
            <div className="text-sm text-slate-600 mt-1">总耗时</div>
          </div>
        </div>
      )}

      {/* Code Example */}
      <div className="mt-6 p-4 bg-slate-900 text-slate-100 rounded-lg">
        <h4 className="font-semibold mb-3">错误处理代码示例</h4>
        <pre className="text-xs font-mono overflow-x-auto">
{`from langchain_core.tools import tool
import time

@tool
def api_call_with_retry(endpoint: str, max_retries: int = 3) -> str:
    """Call API with automatic retry."""
    for attempt in range(max_retries):
        try:
            # 模拟 API 调用
            result = call_external_api(endpoint)
            return result
        except TimeoutError:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} failed, retrying...")
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                return "Error: Max retries exceeded"

# 使用 Fallback
from langchain_core.runnables import RunnableLambda

primary_tool = api_call_with_retry
fallback_tool = search_cache

chain = primary_tool.with_fallbacks([fallback_tool])`}
        </pre>
      </div>
    </div>
  );
}
