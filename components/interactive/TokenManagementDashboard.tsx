"use client";

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Activity, TrendingUp, DollarSign, AlertTriangle } from 'lucide-react';

interface TokenUsage {
  sessionId: string;
  tokensUsed: number;
  cost: number;
  timestamp: number;
}

interface DailyStats {
  total_tokens: number;
  total_cost: number;
  avg_tokens_per_session: number;
  sessions_count: number;
}

export default function TokenManagementDashboard() {
  const [currentTokens, setCurrentTokens] = useState(3500);
  const [maxTokens] = useState(4000);
  const [dailyUsage, setDailyUsage] = useState<TokenUsage[]>([
    { sessionId: 'alice', tokensUsed: 2500, cost: 0.05, timestamp: Date.now() - 3600000 },
    { sessionId: 'bob', tokensUsed: 1800, cost: 0.036, timestamp: Date.now() - 1800000 },
    { sessionId: 'charlie', tokensUsed: 3200, cost: 0.064, timestamp: Date.now() - 900000 }
  ]);

  const dailyStats: DailyStats = {
    total_tokens: dailyUsage.reduce((sum, u) => sum + u.tokensUsed, 0),
    total_cost: dailyUsage.reduce((sum, u) => sum + u.cost, 0),
    avg_tokens_per_session: dailyUsage.length > 0 ? 
      dailyUsage.reduce((sum, u) => sum + u.tokensUsed, 0) / dailyUsage.length : 0,
    sessions_count: dailyUsage.length
  };

  const usage_percentage = (currentTokens / maxTokens) * 100;
  const cost_per_1k = 0.02; // $0.02 per 1K tokens

  const getUsageColor = (percentage: number) => {
    if (percentage >= 90) return 'text-red-600';
    if (percentage >= 70) return 'text-yellow-600';
    return 'text-green-600';
  };

  const getUsageLevel = (percentage: number) => {
    if (percentage >= 90) return '危险';
    if (percentage >= 70) return '警告';
    return '正常';
  };

  return (
    <div className="w-full max-w-7xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-blue-50 rounded-xl border border-slate-200">
      <div className="mb-6">
        <h3 className="text-2xl font-bold text-slate-800 mb-2">
          Token Management Dashboard
        </h3>
        <p className="text-slate-600">
          实时监控 Token 使用情况与成本分析
        </p>
      </div>

      {/* Current Session Stats */}
      <div className="grid md:grid-cols-4 gap-4 mb-6">
        <div className="bg-white rounded-lg border border-slate-200 p-4">
          <div className="flex items-center justify-between mb-2">
            <Activity className="w-5 h-5 text-blue-500" />
            <span className="text-xs text-slate-500">当前会话</span>
          </div>
          <div className="text-2xl font-bold text-slate-800">
            {currentTokens.toLocaleString()}
          </div>
          <div className="text-xs text-slate-500 mt-1">
            / {maxTokens.toLocaleString()} tokens
          </div>
        </div>

        <div className="bg-white rounded-lg border border-slate-200 p-4">
          <div className="flex items-center justify-between mb-2">
            <TrendingUp className="w-5 h-5 text-green-500" />
            <span className="text-xs text-slate-500">使用率</span>
          </div>
          <div className={`text-2xl font-bold ${getUsageColor(usage_percentage)}`}>
            {usage_percentage.toFixed(1)}%
          </div>
          <div className="text-xs text-slate-500 mt-1">
            {getUsageLevel(usage_percentage)}
          </div>
        </div>

        <div className="bg-white rounded-lg border border-slate-200 p-4">
          <div className="flex items-center justify-between mb-2">
            <DollarSign className="w-5 h-5 text-purple-500" />
            <span className="text-xs text-slate-500">今日成本</span>
          </div>
          <div className="text-2xl font-bold text-slate-800">
            ${dailyStats.total_cost.toFixed(3)}
          </div>
          <div className="text-xs text-slate-500 mt-1">
            {dailyStats.total_tokens.toLocaleString()} tokens
          </div>
        </div>

        <div className="bg-white rounded-lg border border-slate-200 p-4">
          <div className="flex items-center justify-between mb-2">
            <AlertTriangle className="w-5 h-5 text-orange-500" />
            <span className="text-xs text-slate-500">会话数</span>
          </div>
          <div className="text-2xl font-bold text-slate-800">
            {dailyStats.sessions_count}
          </div>
          <div className="text-xs text-slate-500 mt-1">
            平均 {Math.round(dailyStats.avg_tokens_per_session)} tokens
          </div>
        </div>
      </div>

      {/* Token Usage Bar */}
      <div className="bg-white rounded-lg border border-slate-200 p-6 mb-6">
        <h4 className="font-semibold text-slate-800 mb-4">Token 使用进度</h4>
        
        <div className="relative">
          <div className="h-8 bg-slate-100 rounded-full overflow-hidden">
            <motion.div
              className={`h-full ${
                usage_percentage >= 90 ? 'bg-red-500' :
                usage_percentage >= 70 ? 'bg-yellow-500' :
                'bg-green-500'
              }`}
              initial={{ width: 0 }}
              animate={{ width: `${usage_percentage}%` }}
              transition={{ duration: 1 }}
            />
          </div>
          
          {/* Warning Line */}
          <div className="absolute top-0 h-8 border-l-2 border-dashed border-yellow-500" style={{ left: '70%' }}>
            <div className="absolute -top-6 left-0 transform -translate-x-1/2 text-xs text-yellow-600">
              警告线 (70%)
            </div>
          </div>

          {/* Danger Line */}
          <div className="absolute top-0 h-8 border-l-2 border-dashed border-red-500" style={{ left: '90%' }}>
            <div className="absolute -top-6 left-0 transform -translate-x-1/2 text-xs text-red-600">
              危险线 (90%)
            </div>
          </div>
        </div>

        <div className="mt-4 grid grid-cols-3 gap-4 text-center">
          <div>
            <div className="text-sm text-slate-600">已使用</div>
            <div className="font-semibold text-slate-800">
              {currentTokens.toLocaleString()} tokens
            </div>
          </div>
          <div>
            <div className="text-sm text-slate-600">剩余</div>
            <div className="font-semibold text-slate-800">
              {(maxTokens - currentTokens).toLocaleString()} tokens
            </div>
          </div>
          <div>
            <div className="text-sm text-slate-600">预估成本</div>
            <div className="font-semibold text-slate-800">
              ${(currentTokens / 1000 * cost_per_1k).toFixed(3)}
            </div>
          </div>
        </div>
      </div>

      {/* Top Sessions */}
      <div className="bg-white rounded-lg border border-slate-200 p-6">
        <h4 className="font-semibold text-slate-800 mb-4">Top Token 使用会话</h4>
        
        <div className="space-y-3">
          {dailyUsage
            .sort((a, b) => b.tokensUsed - a.tokensUsed)
            .map((usage, idx) => {
              const percentage = (usage.tokensUsed / dailyStats.total_tokens) * 100;
              
              return (
                <div key={usage.sessionId} className="flex items-center gap-4">
                  <div className="w-8 h-8 rounded-full bg-blue-100 text-blue-600 flex items-center justify-center font-bold text-sm">
                    {idx + 1}
                  </div>
                  
                  <div className="flex-1">
                    <div className="flex justify-between items-center mb-1">
                      <span className="font-medium text-slate-800">
                        {usage.sessionId}
                      </span>
                      <span className="text-sm text-slate-600">
                        {usage.tokensUsed.toLocaleString()} tokens
                      </span>
                    </div>
                    
                    <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
                      <motion.div
                        className="h-full bg-blue-500"
                        initial={{ width: 0 }}
                        animate={{ width: `${percentage}%` }}
                        transition={{ duration: 0.5, delay: idx * 0.1 }}
                      />
                    </div>
                  </div>
                  
                  <div className="text-right">
                    <div className="text-sm font-semibold text-slate-800">
                      ${usage.cost.toFixed(3)}
                    </div>
                    <div className="text-xs text-slate-500">
                      {percentage.toFixed(1)}%
                    </div>
                  </div>
                </div>
              );
            })}
        </div>
      </div>

      {/* Code Example */}
      <div className="mt-6 p-4 bg-slate-900 text-slate-100 rounded-lg">
        <h4 className="font-semibold mb-3">Token 监控代码</h4>
        <pre className="text-xs font-mono overflow-x-auto">
{`class TokenUsageMonitor:
    def record_usage(self, session_id: str, tokens_used: int, cost: float):
        """记录 Token 使用情况"""
        timestamp = datetime.now()
        key_today = f"token_usage:{timestamp.strftime('%Y-%m-%d')}"
        
        # 记录使用量
        self.redis.hincrby(key_today, session_id, tokens_used)
        
        # 记录成本
        cost_key = f"token_cost:{timestamp.strftime('%Y-%m-%d')}"
        self.redis.hincrbyfloat(cost_key, session_id, cost)
        
        # 设置7天过期
        self.redis.expire(key_today, 7 * 24 * 3600)
    
    def check_quota(self, session_id: str, quota: int) -> bool:
        """检查会话是否超出配额"""
        today = datetime.now().strftime('%Y-%m-%d')
        key = f"token_usage:{today}"
        used = int(self.redis.hget(key, session_id) or 0)
        return used < quota

# 使用
monitor = TokenUsageMonitor(redis_client)
monitor.record_usage("alice", tokens_used=1500, cost=0.03)

if not monitor.check_quota("alice", quota=10000):
    print("警告：用户已超出每日配额！")`}
        </pre>
      </div>
    </div>
  );
}
