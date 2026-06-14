'use client';
import { useState, useEffect } from 'react';

const components = [
  {
    name: '搜索空间定义',
    icon: '📐',
    color: 'border-cyan-500 bg-cyan-900/20',
    textColor: 'text-cyan-300',
    description: '定义所有可调参数及其范围',
    params: ['Tile大小 (M, N, K)', 'Pipeline阶段数', '线程配置', '数据布局', '循环展开因子'],
  },
  {
    name: '代价模型',
    icon: '📊',
    color: 'border-blue-500 bg-blue-900/20',
    textColor: 'text-blue-300',
    description: '预测给定配置的性能',
    params: ['算力利用率', '内存带宽利用', '缓存命中率', '指令级并行度', '寄存器压力'],
  },
  {
    name: '搜索策略',
    icon: '🔎',
    color: 'border-purple-500 bg-purple-900/20',
    textColor: 'text-purple-300',
    description: '在搜索空间中高效探索',
    params: ['随机搜索', '贝叶斯优化', '进化算法', '网格搜索', '学习型搜索'],
  },
  {
    name: '最优配置',
    icon: '🏆',
    color: 'border-green-500 bg-green-900/20',
    textColor: 'text-green-300',
    description: '收敛后的最佳参数组合',
    params: ['tile_size = [64, 64, 32]', 'pipeline_stages = 3', 'threads = [16, 16]', 'layout = NHWC', 'unroll_factor = 4'],
  },
];

export function AutoScheduleFramework() {
  const [activeComponent, setActiveComponent] = useState<number | null>(null);
  const [iteration, setIteration] = useState(0);
  const [bestPerf, setBestPerf] = useState(0);
  const [isRunning, setIsRunning] = useState(false);

  const history: number[] = [];
  for (let i = 0; i <= iteration; i++) {
    const base = 50 + Math.random() * 20;
    const improvement = Math.min(i * 3, 40);
    history.push(Math.min(base + improvement + Math.random() * 10, 95));
  }

  const runSearch = () => {
    if (isRunning) { setIsRunning(false); return; }
    setIsRunning(true);
    setIteration(0);
    setBestPerf(0);
    const timer = setInterval(() => {
      setIteration(prev => {
        if (prev >= 20) { setIsRunning(false); clearInterval(timer); return prev; }
        const perf = 50 + Math.min(prev * 3, 40) + Math.random() * 10;
        setBestPerf(b => Math.max(b, perf));
        return prev + 1;
      });
    }, 200);
  };

  return (
    <div className="bg-gray-900 rounded-xl p-6 text-white">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-bold text-cyan-400">自动调度框架</h2>
        <button onClick={runSearch}
          className={`px-4 py-2 rounded-lg text-sm transition-all ${isRunning ? 'bg-red-600' : 'bg-cyan-600 hover:bg-cyan-500'}`}>
          {isRunning ? '⏹ 停止' : '▶ 运行搜索'}
        </button>
      </div>

      {/* Framework diagram */}
      <div className="grid grid-cols-4 gap-3 mb-6">
        {components.map((c, i) => (
          <div key={i}
            onClick={() => setActiveComponent(activeComponent === i ? null : i)}
            className={`border-2 rounded-xl p-4 cursor-pointer transition-all ${
              activeComponent === i ? c.color + ' ring-2 ring-white/20' : 'border-gray-700 hover:border-gray-500'
            }`}>
            <div className="text-center mb-2">
              <div className="text-2xl mb-1">{c.icon}</div>
              <div className={`font-medium text-sm ${c.textColor}`}>{c.name}</div>
            </div>
            <div className="text-xs text-gray-400 text-center">{c.description}</div>
            {activeComponent === i && (
              <div className="mt-3 space-y-1">
                {c.params.map((p, j) => (
                  <div key={j} className="text-[10px] text-gray-300 bg-gray-800/50 rounded px-2 py-1">▸ {p}</div>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Search progress */}
      <div className="grid grid-cols-2 gap-6">
        <div>
          <div className="text-sm text-gray-400 mb-2">搜索进度</div>
          <div className="bg-gray-800 rounded-lg p-3">
            <div className="flex items-center justify-between text-sm mb-2">
              <span>迭代: {iteration}/20</span>
              <span className="text-green-400">最优: {bestPerf.toFixed(1)}%</span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-3 overflow-hidden">
              <div className="h-full bg-gradient-to-r from-cyan-600 to-green-500 rounded-full transition-all"
                style={{ width: `${(iteration / 20) * 100}%` }} />
            </div>
          </div>
        </div>
        <div>
          <div className="text-sm text-gray-400 mb-2">搜索策略选择</div>
          <div className="grid grid-cols-2 gap-2">
            {['随机搜索', '贝叶斯优化', '进化算法', '网格搜索'].map((s, i) => (
              <div key={i} className={`text-xs p-2 rounded-lg border ${
                i === 1 ? 'border-cyan-500 bg-cyan-900/20 text-cyan-300' : 'border-gray-700 text-gray-400'
              }`}>{s}</div>
            ))}
          </div>
        </div>
      </div>

      {/* Best config found */}
      {iteration > 0 && (
        <div className="mt-4 p-3 bg-gray-800 rounded-lg">
          <div className="text-xs text-gray-400">当前最优配置</div>
          <div className="text-sm text-green-300 font-mono mt-1">
            tile=[64,64,32] pipeline=3 threads=[16,16] layout=NHWC unroll=4 → {bestPerf.toFixed(1)}% 利用率
          </div>
        </div>
      )}
    </div>
  );
}
