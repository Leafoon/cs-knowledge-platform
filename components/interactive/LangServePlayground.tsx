'use client';

import React, { useState, useMemo } from 'react';

type RequestExample = {
  id: string;
  name: string;
  endpoint: string;
  method: string;
  payload: any;
};

export default function LangServePlayground() {
  const [selectedExample, setSelectedExample] = useState<string>('invoke');
  const [isLoading, setIsLoading] = useState(false);
  const [response, setResponse] = useState<string | null>(null);

  const examples: RequestExample[] = useMemo(() => [
    {
      id: 'invoke',
      name: '同步调用 (/invoke)',
      endpoint: '/translate/invoke',
      method: 'POST',
      payload: {
        input: {
          language: '法语',
          text: 'Hello, how are you?'
        }
      }
    },
    {
      id: 'stream',
      name: '流式输出 (/stream)',
      endpoint: '/translate/stream',
      method: 'POST',
      payload: {
        input: {
          language: '中文',
          text: 'LangServe makes deployment easy.'
        }
      }
    },
    {
      id: 'batch',
      name: '批处理 (/batch)',
      endpoint: '/translate/batch',
      method: 'POST',
      payload: {
        inputs: [
          { language: '西班牙语', text: 'Good morning' },
          { language: '日语', text: 'Thank you' },
          { language: '德语', text: 'Goodbye' }
        ]
      }
    }
  ], []);

  const currentExample = useMemo(
    () => examples.find(ex => ex.id === selectedExample)!,
    [selectedExample, examples]
  );

  const handleExecute = () => {
    setIsLoading(true);
    setResponse(null);

    setTimeout(() => {
      if (selectedExample === 'invoke') {
        setResponse(JSON.stringify({
          output: 'Bonjour, comment allez-vous?',
          metadata: {
            run_id: 'run_abc123...',
            tokens: { prompt: 15, completion: 8, total: 23 },
            latency_ms: 1250
          }
        }, null, 2));
      } else if (selectedExample === 'stream') {
        setResponse('data: LangServe\n\ndata: 使得\n\ndata: 部署\n\ndata: 变得\n\ndata: 简单\n\ndata: 。\n\nevent: end\ndata: null');
      } else {
        setResponse(JSON.stringify({
          outputs: [
            'Buenos días',
            'ありがとう',
            'Auf Wiedersehen'
          ]
        }, null, 2));
      }
      setIsLoading(false);
    }, 1500);
  };

  return (
    <div className="my-8 p-8 bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-900 rounded-2xl shadow-xl border border-gray-200 dark:border-gray-700">
      <h3 className="text-2xl font-bold mb-2 bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
        LangServe Playground
      </h3>
      <p className="text-sm text-gray-600 dark:text-gray-400 mb-6">
        交互式测试 LangServe API 端点，探索不同的调用模式
      </p>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="space-y-6">
          <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-lg">
            <h4 className="font-bold mb-4 text-gray-800 dark:text-gray-200 flex items-center gap-2">
              <span className="text-xl">🎯</span>
              选择端点
            </h4>
            <div className="space-y-2">
              {examples.map((example) => (
                <button
                  key={example.id}
                  onClick={() => setSelectedExample(example.id)}
                  className={`w-full p-4 rounded-xl text-left transition-all border-2 ${
                    selectedExample === example.id
                      ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20 shadow-lg'
                      : 'border-gray-200 dark:border-gray-600 bg-gray-50 dark:bg-gray-700 hover:shadow-md'
                  }`}
                >
                  <div className="font-semibold text-gray-800 dark:text-gray-200 mb-1">
                    {example.name}
                  </div>
                  <div className="flex items-center gap-2">
                    <span className={`px-2 py-0.5 rounded text-xs font-bold ${
                      example.method === 'POST' 
                        ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-300'
                        : 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300'
                    }`}>
                      {example.method}
                    </span>
                    <span className="text-xs text-gray-500 dark:text-gray-400 font-mono">
                      {example.endpoint}
                    </span>
                  </div>
                </button>
              ))}
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-lg">
            <h4 className="font-bold mb-4 text-gray-800 dark:text-gray-200 flex items-center gap-2">
              <span className="text-xl">📝</span>
              请求体
            </h4>
            <div className="bg-gray-900 rounded-lg p-4 font-mono text-xs overflow-x-auto">
              <pre className="text-green-400">{JSON.stringify(currentExample.payload, null, 2)}</pre>
            </div>
          </div>

          <button
            onClick={handleExecute}
            disabled={isLoading}
            className="w-full px-6 py-4 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-xl font-bold text-lg hover:shadow-xl hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 transition-all shadow-lg"
          >
            {isLoading ? (
              <span className="flex items-center justify-center gap-2">
                <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                执行中...
              </span>
            ) : (
              '▶ 执行请求'
            )}
          </button>
        </div>

        <div className="space-y-6">
          <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-lg">
            <h4 className="font-bold mb-4 text-gray-800 dark:text-gray-200 flex items-center gap-2">
              <span className="text-xl">📤</span>
              响应结果
            </h4>
            {response ? (
              <div className="bg-gray-900 rounded-lg p-4 font-mono text-xs overflow-x-auto max-h-96 overflow-y-auto">
                <pre className="text-cyan-400">{response}</pre>
              </div>
            ) : (
              <div className="text-center py-12 text-gray-400 dark:text-gray-500">
                <div className="text-5xl mb-3">📭</div>
                <div className="text-sm">点击&quot;执行请求&quot;查看响应</div>
              </div>
            )}
          </div>

          <div className="bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-2xl p-6 border-l-4 border-green-500 shadow-lg">
            <div className="flex items-start gap-3">
              <div className="flex-shrink-0 w-10 h-10 bg-green-500 rounded-full flex items-center justify-center shadow-lg">
                <span className="text-white text-xl">💡</span>
              </div>
              <div>
                <h4 className="font-bold text-gray-800 dark:text-gray-100 mb-2">端点说明</h4>
                <div className="text-sm text-gray-700 dark:text-gray-300 space-y-2">
                  {selectedExample === 'invoke' && (
                    <p><strong>/invoke</strong>：同步调用，等待完整结果返回。适用于短文本处理。</p>
                  )}
                  {selectedExample === 'stream' && (
                    <p><strong>/stream</strong>：流式输出（SSE），实时返回生成内容。适用于长文本生成、聊天场景。</p>
                  )}
                  {selectedExample === 'batch' && (
                    <p><strong>/batch</strong>：批量处理多个请求，提高吞吐量。适用于离线任务、数据处理。</p>
                  )}
                </div>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-3 gap-3">
            <div className="p-4 bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/30 dark:to-blue-800/30 rounded-xl shadow-md text-center border border-blue-200 dark:border-blue-700">
              <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">1.25s</div>
              <div className="text-xs text-gray-600 dark:text-gray-400 mt-1">延迟</div>
            </div>
            <div className="p-4 bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-900/30 dark:to-purple-800/30 rounded-xl shadow-md text-center border border-purple-200 dark:border-purple-700">
              <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">23</div>
              <div className="text-xs text-gray-600 dark:text-gray-400 mt-1">Tokens</div>
            </div>
            <div className="p-4 bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/30 dark:to-green-800/30 rounded-xl shadow-md text-center border border-green-200 dark:border-green-700">
              <div className="text-2xl font-bold text-green-600 dark:text-green-400">$0.0003</div>
              <div className="text-xs text-gray-600 dark:text-gray-400 mt-1">成本</div>
            </div>
          </div>
        </div>
      </div>

      <div className="mt-6 grid grid-cols-3 gap-4">
        <div className="p-4 bg-white dark:bg-gray-800 rounded-xl shadow-md">
          <div className="flex items-center gap-3 mb-2">
            <div className="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center">
              <span className="text-white text-sm">🔗</span>
            </div>
            <div className="font-semibold text-gray-800 dark:text-gray-200 text-sm">自动路由</div>
          </div>
          <div className="text-xs text-gray-600 dark:text-gray-400">
            add_routes() 自动生成所有端点
          </div>
        </div>
        <div className="p-4 bg-white dark:bg-gray-800 rounded-xl shadow-md">
          <div className="flex items-center gap-3 mb-2">
            <div className="w-8 h-8 bg-purple-500 rounded-full flex items-center justify-center">
              <span className="text-white text-sm">📄</span>
            </div>
            <div className="font-semibold text-gray-800 dark:text-gray-200 text-sm">OpenAPI</div>
          </div>
          <div className="text-xs text-gray-600 dark:text-gray-400">
            自动生成 API 文档和 Schema
          </div>
        </div>
        <div className="p-4 bg-white dark:bg-gray-800 rounded-xl shadow-md">
          <div className="flex items-center gap-3 mb-2">
            <div className="w-8 h-8 bg-green-500 rounded-full flex items-center justify-center">
              <span className="text-white text-sm">⚡</span>
            </div>
            <div className="font-semibold text-gray-800 dark:text-gray-200 text-sm">FastAPI</div>
          </div>
          <div className="text-xs text-gray-600 dark:text-gray-400">
            基于 FastAPI 高性能框架
          </div>
        </div>
      </div>
    </div>
  );
}
