'use client';
import { useState } from 'react';

interface FlowStep {
  name: string;
  icon: string;
  description: string;
  timing: string;
  status: 'done' | 'active' | 'pending';
}

export function JITCompilationFlow() {
  const [steps, setSteps] = useState<FlowStep[]>([
    { name: 'Python调用', icon: '🐍', description: '用户代码调用TileLang算子', timing: '~1μs', status: 'done' },
    { name: '缓存查找', icon: '🔍', description: '根据算子参数计算哈希，查询缓存', timing: '~10μs', status: 'done' },
    { name: '缓存命中?', icon: '❓', description: '检查是否已有编译后的二进制', timing: '-', status: 'active' },
    { name: 'IR生成', icon: '📝', description: '生成TileLang中间表示', timing: '~100μs', status: 'pending' },
    { name: '优化Pass', icon: '⚙️', description: '执行各种优化变换', timing: '~500μs', status: 'pending' },
    { name: '代码生成', icon: '💻', description: '生成目标平台机器码', timing: '~200μs', status: 'pending' },
    { name: '缓存写入', icon: '💾', description: '将编译结果存入缓存', timing: '~50μs', status: 'pending' },
    { name: 'Kernel启动', icon: '🚀', description: '在GPU/NPU上执行计算', timing: '~10μs', status: 'pending' },
  ]);

  const [hitCache, setHitCache] = useState(false);

  const advanceStep = () => {
    setSteps(prev => {
      const next = [...prev];
      const activeIdx = next.findIndex(s => s.status === 'active');
      if (activeIdx === -1) return prev;

      if (activeIdx === 2) {
        next[activeIdx] = { ...next[activeIdx], status: 'done' };
        if (hitCache) {
          next[activeIdx + 5] = { ...next[activeIdx + 5], status: 'active' };
        } else {
          next[activeIdx + 1] = { ...next[activeIdx + 1], status: 'active' };
        }
      } else {
        next[activeIdx] = { ...next[activeIdx], status: 'done' };
        const nextIdx = activeIdx + 1;
        if (nextIdx < next.length) next[nextIdx] = { ...next[nextIdx], status: 'active' };
      }
      return next;
    });
  };

  const reset = () => {
    setSteps(prev => prev.map((s, i) => ({
      ...s,
      status: i === 0 ? 'done' : i === 1 ? 'done' : i === 2 ? 'active' : 'pending',
    })));
  };

  return (
    <div className="bg-gray-900 rounded-xl p-6 text-white">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-bold text-cyan-400">JIT编译流程</h2>
        <div className="flex gap-2">
          <label className="flex items-center gap-2 text-sm text-gray-400">
            <input type="checkbox" checked={hitCache} onChange={e => setHitCache(e.target.checked)}
              className="rounded bg-gray-700" />缓存命中
          </label>
          <button onClick={advanceStep} disabled={steps.every(s => s.status !== 'active')}
            className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 rounded-lg text-sm disabled:opacity-50">
            ▶ 下一步
          </button>
          <button onClick={reset} className="px-3 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg text-sm">
            🔄 重置
          </button>
        </div>
      </div>

      {/* Flow diagram */}
      <div className="space-y-2">
        {steps.map((step, i) => {
          const skipped = hitCache && i >= 3 && i <= 6;
          return (
            <div key={i} className={`flex items-center gap-4 p-3 rounded-lg transition-all ${
              step.status === 'active' ? 'bg-cyan-900/20 border border-cyan-500' :
              step.status === 'done' ? 'bg-gray-800/50' :
              skipped ? 'bg-gray-800/20 opacity-30' :
              'bg-gray-800/30'
            }`}>
              <div className={`w-10 h-10 rounded-full flex items-center justify-center text-lg ${
                step.status === 'done' ? 'bg-green-800' :
                step.status === 'active' ? 'bg-cyan-600 animate-pulse' :
                'bg-gray-700'
              }`}>
                {step.icon}
              </div>
              <div className="flex-1">
                <div className={`font-medium text-sm ${
                  step.status === 'active' ? 'text-cyan-300' :
                  step.status === 'done' ? 'text-green-300' :
                  skipped ? 'text-gray-600' : 'text-gray-400'
                }`}>
                  {step.name}
                  {skipped && <span className="text-xs text-gray-600 ml-2">(跳过)</span>}
                </div>
                <div className="text-xs text-gray-500">{step.description}</div>
              </div>
              <span className="text-xs text-gray-500 font-mono">{step.timing}</span>
              <span className={`w-3 h-3 rounded-full ${
                step.status === 'done' ? 'bg-green-500' :
                step.status === 'active' ? 'bg-cyan-500 animate-pulse' :
                'bg-gray-600'
              }`} />
            </div>
          );
        })}
      </div>

      <div className="mt-4 p-3 bg-gray-800 rounded-lg text-xs text-gray-400">
        {hitCache ? '缓存命中模式: 跳过编译直接执行，总耗时 ~11μs' : '首次编译模式: 需要完整编译流程，总耗时 ~861μs'}
      </div>
    </div>
  );
}
