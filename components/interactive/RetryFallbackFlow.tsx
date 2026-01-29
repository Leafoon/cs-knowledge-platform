'use client';

import React, { useState, useEffect } from 'react';

type RetryStrategy = 'exponential' | 'fixed' | 'linear';
type RequestStatus = 'idle' | 'pending' | 'retry' | 'fallback' | 'success' | 'failed';

interface RetryAttempt {
  attempt: number;
  status: RequestStatus;
  provider: string;
  delay: number;
  timestamp: number;
}

const RetryFallbackFlow: React.FC = () => {
  const [strategy, setStrategy] = useState<RetryStrategy>('exponential');
  const [maxRetries, setMaxRetries] = useState(3);
  const [isRunning, setIsRunning] = useState(false);
  const [attempts, setAttempts] = useState<RetryAttempt[]>([]);
  const [currentStatus, setCurrentStatus] = useState<RequestStatus>('idle');
  const [useFallback, setUseFallback] = useState(true);

  const calculateDelay = (attempt: number): number => {
    switch (strategy) {
      case 'exponential':
        return Math.min(Math.pow(2, attempt) * 1000, 60000); // 2^n秒，最多60秒
      case 'linear':
        return attempt * 2000; // 2n秒
      case 'fixed':
        return 2000; // 固定2秒
      default:
        return 1000;
    }
  };

  const simulateRequest = async () => {
    setIsRunning(true);
    setAttempts([]);
    setCurrentStatus('pending');

    let currentAttempt = 0;
    const newAttempts: RetryAttempt[] = [];

    // 主Provider重试
    for (let i = 0; i <= maxRetries; i++) {
      const delay = i === 0 ? 0 : calculateDelay(i - 1);
      
      if (i > 0) {
        await new Promise(resolve => setTimeout(resolve, delay));
      }

      const attemptData: RetryAttempt = {
        attempt: i + 1,
        status: 'pending',
        provider: 'GPT-4 (Primary)',
        delay: delay,
        timestamp: Date.now(),
      };

      setAttempts(prev => [...prev, attemptData]);
      setCurrentStatus('pending');

      // 模拟请求（70%失败率用于演示）
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      const success = Math.random() > 0.7;
      
      if (success) {
        attemptData.status = 'success';
        setAttempts(prev => 
          prev.map((a, idx) => idx === prev.length - 1 ? { ...a, status: 'success' } : a)
        );
        setCurrentStatus('success');
        setIsRunning(false);
        return;
      } else {
        attemptData.status = i === maxRetries ? 'failed' : 'retry';
        setAttempts(prev => 
          prev.map((a, idx) => idx === prev.length - 1 ? { ...a, status: i === maxRetries ? 'failed' : 'retry' } : a)
        );
        setCurrentStatus('retry');
      }

      currentAttempt = i;
    }

    // Fallback 链
    if (useFallback) {
      setCurrentStatus('fallback');
      await new Promise(resolve => setTimeout(resolve, 500));

      // Fallback 1: GPT-3.5
      const fallback1: RetryAttempt = {
        attempt: currentAttempt + 2,
        status: 'pending',
        provider: 'GPT-3.5 (Fallback 1)',
        delay: 0,
        timestamp: Date.now(),
      };
      setAttempts(prev => [...prev, fallback1]);
      
      await new Promise(resolve => setTimeout(resolve, 800));
      
      const fallback1Success = Math.random() > 0.3;
      
      if (fallback1Success) {
        setAttempts(prev => 
          prev.map((a, idx) => idx === prev.length - 1 ? { ...a, status: 'success' } : a)
        );
        setCurrentStatus('success');
        setIsRunning(false);
        return;
      } else {
        setAttempts(prev => 
          prev.map((a, idx) => idx === prev.length - 1 ? { ...a, status: 'retry' } : a)
        );
      }

      await new Promise(resolve => setTimeout(resolve, 500));

      // Fallback 2: Claude
      const fallback2: RetryAttempt = {
        attempt: currentAttempt + 3,
        status: 'pending',
        provider: 'Claude (Fallback 2)',
        delay: 0,
        timestamp: Date.now(),
      };
      setAttempts(prev => [...prev, fallback2]);
      
      await new Promise(resolve => setTimeout(resolve, 800));
      
      setAttempts(prev => 
        prev.map((a, idx) => idx === prev.length - 1 ? { ...a, status: 'success' } : a)
      );
      setCurrentStatus('success');
    } else {
      setCurrentStatus('failed');
    }

    setIsRunning(false);
  };

  const getStatusColor = (status: RequestStatus): string => {
    switch (status) {
      case 'pending':
        return 'bg-blue-500';
      case 'retry':
        return 'bg-yellow-500';
      case 'fallback':
        return 'bg-purple-500';
      case 'success':
        return 'bg-green-500';
      case 'failed':
        return 'bg-red-500';
      default:
        return 'bg-gray-300';
    }
  };

  const getStatusIcon = (status: RequestStatus): string => {
    switch (status) {
      case 'pending':
        return '⏳';
      case 'retry':
        return '🔄';
      case 'fallback':
        return '🔀';
      case 'success':
        return '✅';
      case 'failed':
        return '❌';
      default:
        return '⚪';
    }
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold text-gray-800 mb-6">重试与回退流程可视化</h3>

      {/* 配置面板 */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <div className="bg-white rounded-lg p-4 shadow">
          <label className="block text-sm font-medium text-gray-700 mb-2">重试策略</label>
          <select
            value={strategy}
            onChange={(e) => setStrategy(e.target.value as RetryStrategy)}
            disabled={isRunning}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="exponential">指数退避 (2^n)</option>
            <option value="linear">线性退避 (2n)</option>
            <option value="fixed">固定延迟 (2s)</option>
          </select>
        </div>

        <div className="bg-white rounded-lg p-4 shadow">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            最大重试次数: {maxRetries}
          </label>
          <input
            type="range"
            min="1"
            max="5"
            value={maxRetries}
            onChange={(e) => setMaxRetries(Number(e.target.value))}
            disabled={isRunning}
            className="w-full"
          />
        </div>

        <div className="bg-white rounded-lg p-4 shadow">
          <label className="flex items-center space-x-2">
            <input
              type="checkbox"
              checked={useFallback}
              onChange={(e) => setUseFallback(e.target.checked)}
              disabled={isRunning}
              className="w-4 h-4 text-blue-600"
            />
            <span className="text-sm font-medium text-gray-700">启用 Fallback 链</span>
          </label>
          <p className="text-xs text-gray-500 mt-2">
            失败后切换到备用Provider
          </p>
        </div>
      </div>

      {/* 执行按钮 */}
      <div className="flex justify-center mb-6">
        <button
          onClick={simulateRequest}
          disabled={isRunning}
          className={`px-6 py-3 rounded-lg font-semibold text-white shadow-lg transition-all ${
            isRunning
              ? 'bg-gray-400 cursor-not-allowed'
              : 'bg-gradient-to-r from-blue-500 to-indigo-600 hover:from-blue-600 hover:to-indigo-700'
          }`}
        >
          {isRunning ? '执行中...' : '开始模拟请求'}
        </button>
      </div>

      {/* 执行流程 */}
      {attempts.length > 0 && (
        <div className="bg-white rounded-lg p-6 shadow-lg">
          <h4 className="text-lg font-semibold text-gray-800 mb-4">执行流程</h4>
          
          <div className="space-y-3">
            {attempts.map((attempt, index) => (
              <div
                key={index}
                className="flex items-center space-x-4 p-4 bg-gray-50 rounded-lg border-l-4 transition-all"
                style={{
                  borderLeftColor: attempt.status === 'success' ? '#10b981' :
                                  attempt.status === 'failed' ? '#ef4444' :
                                  attempt.status === 'retry' ? '#f59e0b' :
                                  attempt.status === 'pending' ? '#3b82f6' : '#6b7280'
                }}
              >
                <div className="flex-shrink-0">
                  <span className="text-2xl">{getStatusIcon(attempt.status)}</span>
                </div>

                <div className="flex-1">
                  <div className="flex items-center justify-between">
                    <div>
                      <span className="font-semibold text-gray-800">
                        尝试 #{attempt.attempt}
                      </span>
                      <span className="ml-2 text-sm text-gray-600">
                        {attempt.provider}
                      </span>
                    </div>
                    
                    {attempt.delay > 0 && (
                      <span className="text-sm text-gray-500">
                        等待 {(attempt.delay / 1000).toFixed(1)}s
                      </span>
                    )}
                  </div>

                  <div className="mt-1 flex items-center space-x-2">
                    <span className={`inline-flex items-center px-2 py-1 rounded text-xs font-medium text-white ${getStatusColor(attempt.status)}`}>
                      {attempt.status === 'pending' && '请求中'}
                      {attempt.status === 'retry' && '失败-重试'}
                      {attempt.status === 'success' && '成功'}
                      {attempt.status === 'failed' && '失败'}
                    </span>
                  </div>
                </div>

                {attempt.status === 'pending' && (
                  <div className="flex-shrink-0">
                    <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-500"></div>
                  </div>
                )}
              </div>
            ))}
          </div>

          {/* 总结 */}
          {!isRunning && (
            <div className="mt-6 p-4 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-700">
                    最终状态: 
                    <span className={`ml-2 font-semibold ${
                      currentStatus === 'success' ? 'text-green-600' : 'text-red-600'
                    }`}>
                      {currentStatus === 'success' ? '成功 ✓' : '失败 ✗'}
                    </span>
                  </p>
                  <p className="text-xs text-gray-500 mt-1">
                    总尝试次数: {attempts.length} | 
                    成功Provider: {attempts.find(a => a.status === 'success')?.provider || 'N/A'}
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* 策略说明 */}
      <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-white rounded-lg p-4 shadow">
          <h5 className="font-semibold text-gray-800 mb-2">📈 指数退避</h5>
          <p className="text-sm text-gray-600">
            延迟: 1s, 2s, 4s, 8s, 16s...
            <br />适合临时故障，快速恢复
          </p>
        </div>

        <div className="bg-white rounded-lg p-4 shadow">
          <h5 className="font-semibold text-gray-800 mb-2">📊 线性退避</h5>
          <p className="text-sm text-gray-600">
            延迟: 2s, 4s, 6s, 8s, 10s...
            <br />适合稳定负载，均匀重试
          </p>
        </div>

        <div className="bg-white rounded-lg p-4 shadow">
          <h5 className="font-semibold text-gray-800 mb-2">🔄 Fallback</h5>
          <p className="text-sm text-gray-600">
            主Provider失败后自动切换
            <br />GPT-4 → GPT-3.5 → Claude
          </p>
        </div>
      </div>
    </div>
  );
};

export default RetryFallbackFlow;
