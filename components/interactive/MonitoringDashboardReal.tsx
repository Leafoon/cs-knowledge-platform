"use client";

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Activity, AlertTriangle, Clock, DollarSign, TrendingUp, TrendingDown, Zap } from 'lucide-react';

interface Metric {
  value: number;
  trend: 'up' | 'down' | 'stable';
  change: number;
}

const MonitoringDashboard: React.FC = () => {
  const [timeRange, setTimeRange] = useState<'1h' | '24h' | '7d'>('1h');
  const [isLive, setIsLive] = useState(false);
  
  const [metrics, setMetrics] = useState({
    requestsPerMin: { value: 45, trend: 'up' as const, change: 12 },
    successRate: { value: 98.5, trend: 'stable' as const, change: 0.2 },
    errorRate: { value: 1.5, trend: 'down' as const, change: -0.3 },
    avgLatency: { value: 850, trend: 'down' as const, change: -15 },
    p95Latency: { value: 1850, trend: 'stable' as const, change: 5 },
    p99Latency: { value: 3200, trend: 'up' as const, change: 8 },
    tokenCost: { value: 12.45, trend: 'up' as const, change: 18 },
  });

  // 模拟实时更新
  useEffect(() => {
    if (!isLive) return;

    const interval = setInterval(() => {
      setMetrics(prev => ({
        requestsPerMin: {
          ...prev.requestsPerMin,
          value: Math.max(20, prev.requestsPerMin.value + (Math.random() - 0.5) * 10),
          change: (Math.random() - 0.5) * 20,
        },
        successRate: {
          ...prev.successRate,
          value: Math.min(100, Math.max(95, prev.successRate.value + (Math.random() - 0.5) * 0.5)),
          change: (Math.random() - 0.5),
        },
        errorRate: {
          ...prev.errorRate,
          value: Math.max(0, Math.min(5, prev.errorRate.value + (Math.random() - 0.5) * 0.5)),
          change: (Math.random() - 0.5),
        },
        avgLatency: {
          ...prev.avgLatency,
          value: Math.max(500, prev.avgLatency.value + (Math.random() - 0.5) * 100),
          change: (Math.random() - 0.5) * 30,
        },
        p95Latency: {
          ...prev.p95Latency,
          value: Math.max(1000, prev.p95Latency.value + (Math.random() - 0.5) * 200),
          change: (Math.random() - 0.5) * 50,
        },
        p99Latency: {
          ...prev.p99Latency,
          value: Math.max(2000, prev.p99Latency.value + (Math.random() - 0.5) * 300),
          change: (Math.random() - 0.5) * 100,
        },
        tokenCost: {
          ...prev.tokenCost,
          value: prev.tokenCost.value + Math.random() * 0.5,
          change: Math.random() * 10,
        },
      }));
    }, 2000);

    return () => clearInterval(interval);
  }, [isLive]);

  const MetricCard: React.FC<{
    title: string;
    metric: Metric;
    unit: string;
    icon: React.ReactNode;
    threshold?: { warning: number; critical: number };
    inverse?: boolean; // true 表示值越低越好
  }> = ({ title, metric, unit, icon, threshold, inverse = false }) => {
    const getStatus = () => {
      if (!threshold) return 'normal';
      if (inverse) {
        if (metric.value > threshold.critical) return 'critical';
        if (metric.value > threshold.warning) return 'warning';
      } else {
        if (metric.value < threshold.critical) return 'critical';
        if (metric.value < threshold.warning) return 'warning';
      }
      return 'normal';
    };

    const status = getStatus();
    const statusColors = {
      normal: 'border-green-200 bg-green-50',
      warning: 'border-yellow-200 bg-yellow-50',
      critical: 'border-red-200 bg-red-50',
    };

    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className={`p-4 rounded-lg border-2 ${statusColors[status]} transition-all`}
      >
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            {icon}
            <span className="text-sm text-gray-600">{title}</span>
          </div>
          {status !== 'normal' && (
            <AlertTriangle className={`w-4 h-4 ${status === 'critical' ? 'text-red-500' : 'text-yellow-500'}`} />
          )}
        </div>
        <div className="flex items-baseline gap-2">
          <span className="text-3xl font-bold text-gray-800">
            {typeof metric.value === 'number' ? metric.value.toFixed(unit === '$' ? 2 : unit === '%' ? 1 : 0) : metric.value}
          </span>
          <span className="text-sm text-gray-500">{unit}</span>
        </div>
        <div className="flex items-center gap-1 mt-2">
          {metric.trend === 'up' ? (
            <TrendingUp className="w-4 h-4 text-green-500" />
          ) : metric.trend === 'down' ? (
            <TrendingDown className="w-4 h-4 text-red-500" />
          ) : (
            <span className="w-4 h-4" />
          )}
          <span className={`text-xs ${metric.change > 0 ? 'text-green-600' : metric.change < 0 ? 'text-red-600' : 'text-gray-500'}`}>
            {metric.change > 0 ? '+' : ''}{metric.change.toFixed(1)}% vs {timeRange === '1h' ? '上小时' : timeRange === '24h' ? '昨天' : '上周'}
          </span>
        </div>
      </motion.div>
    );
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-blue-50 to-purple-50 rounded-xl shadow-lg">
      <div className="mb-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-2xl font-bold text-gray-800 flex items-center gap-2">
              <Activity className="w-6 h-6 text-blue-600" />
              LangSmith 监控仪表盘
            </h3>
            <p className="text-gray-600">实时追踪应用性能、可靠性与成本</p>
          </div>
          <button
            onClick={() => setIsLive(!isLive)}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              isLive
                ? 'bg-red-500 text-white hover:bg-red-600'
                : 'bg-green-500 text-white hover:bg-green-600'
            }`}
          >
            {isLive ? '⏸️ 暂停实时' : '▶️ 开启实时'}
          </button>
        </div>

        {/* 时间范围选择器 */}
        <div className="flex gap-2">
          {(['1h', '24h', '7d'] as const).map((range) => (
            <button
              key={range}
              onClick={() => setTimeRange(range)}
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                timeRange === range
                  ? 'bg-indigo-600 text-white'
                  : 'bg-white text-gray-700 hover:bg-gray-100'
              }`}
            >
              {range === '1h' ? '最近 1 小时' : range === '24h' ? '最近 24 小时' : '最近 7 天'}
            </button>
          ))}
        </div>
      </div>

      {/* 指标卡片 */}
      <div className="grid grid-cols-4 gap-4 mb-6">
        <MetricCard
          title="请求量/分钟"
          metric={metrics.requestsPerMin}
          unit="req/min"
          icon={<Zap className="w-4 h-4 text-blue-500" />}
        />
        <MetricCard
          title="成功率"
          metric={metrics.successRate}
          unit="%"
          icon={<Activity className="w-4 h-4 text-green-500" />}
          threshold={{ warning: 98, critical: 95 }}
        />
        <MetricCard
          title="错误率"
          metric={metrics.errorRate}
          unit="%"
          icon={<AlertTriangle className="w-4 h-4 text-red-500" />}
          threshold={{ warning: 2, critical: 5 }}
          inverse
        />
        <MetricCard
          title="Token 成本"
          metric={metrics.tokenCost}
          unit="$/h"
          icon={<DollarSign className="w-4 h-4 text-yellow-500" />}
        />
      </div>

      <div className="grid grid-cols-3 gap-4 mb-6">
        <MetricCard
          title="平均延迟"
          metric={metrics.avgLatency}
          unit="ms"
          icon={<Clock className="w-4 h-4 text-purple-500" />}
          threshold={{ warning: 2000, critical: 5000 }}
          inverse
        />
        <MetricCard
          title="P95 延迟"
          metric={metrics.p95Latency}
          unit="ms"
          icon={<Clock className="w-4 h-4 text-indigo-500" />}
          threshold={{ warning: 3000, critical: 5000 }}
          inverse
        />
        <MetricCard
          title="P99 延迟"
          metric={metrics.p99Latency}
          unit="ms"
          icon={<Clock className="w-4 h-4 text-pink-500" />}
          threshold={{ warning: 5000, critical: 10000 }}
          inverse
        />
      </div>

      {/* 延迟分布图（模拟） */}
      <div className="p-6 bg-white rounded-lg shadow border border-gray-200 mb-6">
        <h4 className="text-lg font-semibold text-gray-800 mb-4">延迟分布（最近 {timeRange}）</h4>
        <div className="flex items-end gap-1 h-40">
          {[100, 150, 200, 300, 500, 800, 1200, 1500, 1800, 2000, 2500, 3000, 3500, 4000, 5000].map((latency, idx) => {
            const height = Math.max(10, Math.random() * 100);
            const isHighlight = latency === metrics.avgLatency.value || latency === metrics.p95Latency.value || latency === metrics.p99Latency.value;
            return (
              <div
                key={idx}
                className="flex-1 relative group"
                title={`${latency}ms`}
              >
                <div
                  className={`w-full rounded-t transition-all ${
                    isHighlight ? 'bg-indigo-500' : 'bg-blue-300 hover:bg-blue-400'
                  }`}
                  style={{ height: `${height}%` }}
                />
                <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 hidden group-hover:block bg-gray-800 text-white text-xs px-2 py-1 rounded whitespace-nowrap">
                  {latency}ms
                </div>
              </div>
            );
          })}
        </div>
        <div className="flex justify-between mt-2 text-xs text-gray-500">
          <span>0ms</span>
          <span>5000ms</span>
        </div>
      </div>

      {/* 告警信息 */}
      {(metrics.errorRate.value > 2 || metrics.p99Latency.value > 5000) && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="p-4 bg-red-50 border-2 border-red-200 rounded-lg"
        >
          <div className="flex items-start gap-2">
            <AlertTriangle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
            <div>
              <h4 className="font-semibold text-red-800 mb-1">⚠️ 检测到异常</h4>
              <ul className="text-sm text-red-700 space-y-1">
                {metrics.errorRate.value > 2 && (
                  <li>• 错误率 {metrics.errorRate.value.toFixed(1)}% 超过阈值（2%）</li>
                )}
                {metrics.p99Latency.value > 5000 && (
                  <li>• P99 延迟 {metrics.p99Latency.value.toFixed(0)}ms 超过阈值（5000ms）</li>
                )}
              </ul>
              <p className="text-xs text-red-600 mt-2">已通过 Slack 和邮件发送告警通知</p>
            </div>
          </div>
        </motion.div>
      )}

      {/* 说明 */}
      <div className="mt-6 p-4 bg-white rounded-lg border border-blue-200">
        <h4 className="font-semibold text-gray-800 mb-2">💡 监控指标说明</h4>
        <ul className="text-sm text-gray-600 space-y-1">
          <li>• <strong>成功率</strong>：应保持在 99% 以上，低于 95% 为严重告警</li>
          <li>• <strong>P95/P99 延迟</strong>：反映 95%/99% 用户的体验，比平均值更重要</li>
          <li>• <strong>Token 成本</strong>：实时追踪 API 调用费用，及时发现异常消耗</li>
          <li>• <strong>实时模式</strong>：每 2 秒刷新一次数据（演示模式模拟随机波动）</li>
        </ul>
      </div>
    </div>
  );
};

export default MonitoringDashboard;
