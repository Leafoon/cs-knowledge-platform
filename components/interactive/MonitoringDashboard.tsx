'use client';

import React, { useState, useMemo } from 'react';

type MetricData = {
  timestamp: string;
  requests: number;
  errors: number;
  avg_latency: number;
  p95_latency: number;
  total_tokens: number;
  total_cost: number;
};

type AlertConfig = {
  id: string;
  name: string;
  condition: string;
  threshold: number;
  status: 'active' | 'triggered' | 'disabled';
  triggered_at?: string;
};

export default function MonitoringDashboard() {
  const [selectedTimeRange, setSelectedTimeRange] = useState<'1h' | '24h' | '7d'>('24h');
  const [selectedMetric, setSelectedMetric] = useState<'latency' | 'errors' | 'cost'>('latency');

  const metricsData: MetricData[] = useMemo(() => {
    const baseData: MetricData[] = [];
    const now = Date.now();
    const interval = selectedTimeRange === '1h' ? 5 * 60 * 1000 :
      selectedTimeRange === '24h' ? 60 * 60 * 1000 :
        24 * 60 * 60 * 1000;
    const points = 12;

    for (let i = points - 1; i >= 0; i--) {
      const timestamp = new Date(now - i * interval);
      const hour = timestamp.getHours();
      const isPeakHour = hour >= 9 && hour <= 17;

      baseData.push({
        timestamp: timestamp.toISOString(),
        requests: isPeakHour ? 150 + Math.random() * 50 : 50 + Math.random() * 30,
        errors: Math.random() * 10,
        avg_latency: 1.2 + Math.random() * 0.8,
        p95_latency: 2.5 + Math.random() * 1.5,
        total_tokens: isPeakHour ? 25000 + Math.random() * 5000 : 8000 + Math.random() * 2000,
        total_cost: isPeakHour ? 0.8 + Math.random() * 0.3 : 0.3 + Math.random() * 0.2
      });
    }
    return baseData;
  }, [selectedTimeRange]);

  const alerts: AlertConfig[] = useMemo(() => [
    {
      id: 'alert_1',
      name: '高错误率警报',
      condition: 'error_rate > 5%',
      threshold: 5,
      status: 'active'
    },
    {
      id: 'alert_2',
      name: 'P95 延迟异常',
      condition: 'p95_latency > 5s',
      threshold: 5,
      status: 'active'
    },
    {
      id: 'alert_3',
      name: '每日成本超限',
      condition: 'daily_cost > $50',
      threshold: 50,
      status: 'triggered',
      triggered_at: '2024-01-15 14:32:15'
    },
    {
      id: 'alert_4',
      name: 'Token 消耗激增',
      condition: 'hourly_tokens > 100k',
      threshold: 100000,
      status: 'disabled'
    }
  ], []);

  const currentStats = useMemo(() => {
    const latest = metricsData[metricsData.length - 1];
    const totalRequests = metricsData.reduce((sum, d) => sum + d.requests, 0);
    const totalErrors = metricsData.reduce((sum, d) => sum + d.errors, 0);
    const errorRate = (totalErrors / totalRequests) * 100;
    const avgLatency = metricsData.reduce((sum, d) => sum + d.avg_latency, 0) / metricsData.length;
    const totalCost = metricsData.reduce((sum, d) => sum + d.total_cost, 0);

    return {
      currentRequests: latest.requests,
      errorRate,
      avgLatency,
      p95Latency: latest.p95_latency,
      totalCost,
      totalTokens: metricsData.reduce((sum, d) => sum + d.total_tokens, 0)
    };
  }, [metricsData]);

  const chartData = useMemo(() => {
    switch (selectedMetric) {
      case 'latency':
        return metricsData.map(d => ({ x: d.timestamp, y: d.avg_latency }));
      case 'errors':
        return metricsData.map(d => ({ x: d.timestamp, y: (d.errors / d.requests) * 100 }));
      case 'cost':
        return metricsData.map(d => ({ x: d.timestamp, y: d.total_cost }));
    }
  }, [selectedMetric, metricsData]);

  const maxValue = useMemo(() => Math.max(...chartData.map(d => d.y)), [chartData]);

  return (
    <div className="my-8 p-8 bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-900 rounded-2xl shadow-xl border border-gray-200 dark:border-gray-700">
      <h3 className="text-2xl font-bold mb-2 bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
        生产监控仪表板
      </h3>
      <p className="text-sm text-gray-600 dark:text-gray-400 mb-6">
        实时监控 LLM 应用的性能、成本、错误率，及时发现和解决问题
      </p>

      <div className="flex gap-3 mb-6">
        <button
          onClick={() => setSelectedTimeRange('1h')}
          className={`px-4 py-2 rounded-lg font-semibold transition-all ${selectedTimeRange === '1h'
              ? 'bg-blue-500 text-white shadow-lg'
              : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 shadow-md hover:shadow-lg'
            }`}
        >
          1 小时
        </button>
        <button
          onClick={() => setSelectedTimeRange('24h')}
          className={`px-4 py-2 rounded-lg font-semibold transition-all ${selectedTimeRange === '24h'
              ? 'bg-blue-500 text-white shadow-lg'
              : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 shadow-md hover:shadow-lg'
            }`}
        >
          24 小时
        </button>
        <button
          onClick={() => setSelectedTimeRange('7d')}
          className={`px-4 py-2 rounded-lg font-semibold transition-all ${selectedTimeRange === '7d'
              ? 'bg-blue-500 text-white shadow-lg'
              : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 shadow-md hover:shadow-lg'
            }`}
        >
          7 天
        </button>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <div className="p-5 bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/30 dark:to-blue-800/30 rounded-xl shadow-md border border-blue-200 dark:border-blue-700">
          <div className="flex items-center gap-3 mb-2">
            <div className="w-10 h-10 bg-blue-500 rounded-full flex items-center justify-center shadow-lg">
              <span className="text-white text-lg">📊</span>
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-400">当前请求数</div>
          </div>
          <div className="text-3xl font-bold text-blue-600 dark:text-blue-400">
            {currentStats.currentRequests.toFixed(0)}
          </div>
          <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">RPM</div>
        </div>

        <div className="p-5 bg-gradient-to-br from-red-50 to-red-100 dark:from-red-900/30 dark:to-red-800/30 rounded-xl shadow-md border border-red-200 dark:border-red-700">
          <div className="flex items-center gap-3 mb-2">
            <div className="w-10 h-10 bg-red-500 rounded-full flex items-center justify-center shadow-lg">
              <span className="text-white text-lg">⚠️</span>
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-400">错误率</div>
          </div>
          <div className="text-3xl font-bold text-red-600 dark:text-red-400">
            {currentStats.errorRate.toFixed(2)}%
          </div>
          <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
            {currentStats.errorRate < 5 ? '✅ 正常' : '🔴 超限'}
          </div>
        </div>

        <div className="p-5 bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-900/30 dark:to-purple-800/30 rounded-xl shadow-md border border-purple-200 dark:border-purple-700">
          <div className="flex items-center gap-3 mb-2">
            <div className="w-10 h-10 bg-purple-500 rounded-full flex items-center justify-center shadow-lg">
              <span className="text-white text-lg">⏱️</span>
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-400">P95 延迟</div>
          </div>
          <div className="text-3xl font-bold text-purple-600 dark:text-purple-400">
            {currentStats.p95Latency.toFixed(1)}s
          </div>
          <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
            平均: {currentStats.avgLatency.toFixed(1)}s
          </div>
        </div>

        <div className="p-5 bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/30 dark:to-green-800/30 rounded-xl shadow-md border border-green-200 dark:border-green-700">
          <div className="flex items-center gap-3 mb-2">
            <div className="w-10 h-10 bg-green-500 rounded-full flex items-center justify-center shadow-lg">
              <span className="text-white text-lg">💰</span>
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-400">总成本</div>
          </div>
          <div className="text-3xl font-bold text-green-600 dark:text-green-400">
            ${currentStats.totalCost.toFixed(2)}
          </div>
          <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
            {(currentStats.totalTokens / 1000).toFixed(0)}K tokens
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-lg">
          <div className="flex items-center justify-between mb-4">
            <h4 className="font-bold text-gray-800 dark:text-gray-200">趋势图表</h4>
            <div className="flex gap-2">
              <button
                onClick={() => setSelectedMetric('latency')}
                className={`px-3 py-1 rounded-lg text-sm font-semibold transition-all ${selectedMetric === 'latency'
                    ? 'bg-purple-500 text-white'
                    : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
                  }`}
              >
                延迟
              </button>
              <button
                onClick={() => setSelectedMetric('errors')}
                className={`px-3 py-1 rounded-lg text-sm font-semibold transition-all ${selectedMetric === 'errors'
                    ? 'bg-red-500 text-white'
                    : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
                  }`}
              >
                错误率
              </button>
              <button
                onClick={() => setSelectedMetric('cost')}
                className={`px-3 py-1 rounded-lg text-sm font-semibold transition-all ${selectedMetric === 'cost'
                    ? 'bg-green-500 text-white'
                    : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
                  }`}
              >
                成本
              </button>
            </div>
          </div>

          <div className="bg-gray-50 dark:bg-gray-700 rounded-xl p-4" style={{ height: '300px' }}>
            <svg width="100%" height="100%" viewBox="0 0 800 280" preserveAspectRatio="none">
              <defs>
                <linearGradient id="chartGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                  <stop offset="0%" style={{ stopColor: selectedMetric === 'latency' ? '#a855f7' : selectedMetric === 'errors' ? '#ef4444' : '#10b981', stopOpacity: 0.5 }} />
                  <stop offset="100%" style={{ stopColor: selectedMetric === 'latency' ? '#a855f7' : selectedMetric === 'errors' ? '#ef4444' : '#10b981', stopOpacity: 0.1 }} />
                </linearGradient>
              </defs>

              <line x1="0" y1="260" x2="800" y2="260" stroke="#d1d5db" strokeWidth="1" />

              {chartData.map((_, idx) => (
                <line
                  key={`grid-${idx}`}
                  x1={(800 / (chartData.length - 1)) * idx}
                  y1="0"
                  x2={(800 / (chartData.length - 1)) * idx}
                  y2="260"
                  stroke="#e5e7eb"
                  strokeWidth="1"
                  strokeDasharray="4,4"
                  opacity="0.3"
                />
              ))}

              <polyline
                points={chartData.map((d, idx) => {
                  const x = (800 / (chartData.length - 1)) * idx;
                  const y = 260 - ((d.y / maxValue) * 240);
                  return `${x},${y}`;
                }).join(' ')}
                fill="none"
                stroke={selectedMetric === 'latency' ? '#a855f7' : selectedMetric === 'errors' ? '#ef4444' : '#10b981'}
                strokeWidth="3"
                strokeLinejoin="round"
              />

              <polygon
                points={`0,260 ${chartData.map((d, idx) => {
                  const x = (800 / (chartData.length - 1)) * idx;
                  const y = 260 - ((d.y / maxValue) * 240);
                  return `${x},${y}`;
                }).join(' ')} 800,260`}
                fill="url(#chartGradient)"
              />

              {chartData.map((d, idx) => {
                const x = (800 / (chartData.length - 1)) * idx;
                const y = 260 - ((d.y / maxValue) * 240);
                return (
                  <circle
                    key={`point-${idx}`}
                    cx={x}
                    cy={y}
                    r="4"
                    fill="white"
                    stroke={selectedMetric === 'latency' ? '#a855f7' : selectedMetric === 'errors' ? '#ef4444' : '#10b981'}
                    strokeWidth="2"
                  />
                );
              })}
            </svg>
          </div>
        </div>

        <div className="lg:col-span-1 bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-lg">
          <h4 className="font-bold mb-4 text-gray-800 dark:text-gray-200 flex items-center gap-2">
            <span className="text-xl">🔔</span>
            告警配置
          </h4>
          <div className="space-y-3">
            {alerts.map((alert) => (
              <div
                key={alert.id}
                className={`p-3 rounded-lg border-2 ${alert.status === 'triggered'
                    ? 'border-red-500 bg-red-50 dark:bg-red-900/20'
                    : alert.status === 'active'
                      ? 'border-green-500 bg-green-50 dark:bg-green-900/20'
                      : 'border-gray-300 dark:border-gray-600 bg-gray-50 dark:bg-gray-700'
                  }`}
              >
                <div className="flex items-center justify-between mb-2">
                  <div className="font-semibold text-sm text-gray-800 dark:text-gray-200">
                    {alert.name}
                  </div>
                  <div className={`px-2 py-0.5 rounded-full text-xs font-bold ${alert.status === 'triggered'
                      ? 'bg-red-500 text-white'
                      : alert.status === 'active'
                        ? 'bg-green-500 text-white'
                        : 'bg-gray-400 text-white'
                    }`}>
                    {alert.status === 'triggered' ? '🔴 触发' : alert.status === 'active' ? '✅ 活跃' : '⏸️ 禁用'}
                  </div>
                </div>
                <div className="text-xs text-gray-600 dark:text-gray-400 mb-1">
                  {alert.condition}
                </div>
                {alert.triggered_at && (
                  <div className="text-xs text-red-600 dark:text-red-400 font-semibold">
                    触发时间: {alert.triggered_at}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="mt-6 p-6 bg-gradient-to-r from-orange-50 to-yellow-50 dark:from-orange-900/20 dark:to-yellow-900/20 rounded-2xl border-l-4 border-orange-500 shadow-lg">
        <div className="flex items-start gap-4">
          <div className="flex-shrink-0 w-10 h-10 bg-orange-500 rounded-full flex items-center justify-center shadow-lg">
            <span className="text-white text-xl">📈</span>
          </div>
          <div>
            <h4 className="font-bold text-gray-800 dark:text-gray-100 mb-2">监控洞察</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300 leading-relaxed">
              当前系统运行稳定，错误率 <strong>{currentStats.errorRate.toFixed(2)}%</strong> 低于 5% 阈值。
              P95 延迟 <strong>{currentStats.p95Latency.toFixed(1)}s</strong> 略高，建议检查慢查询并优化。
              每日成本警报已触发，需评估是否因流量增长或模型选择导致。
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
