'use client';
import { useState } from 'react';

interface Parameter {
  name: string;
  category: string;
  options: (string | number)[];
  selectedIdx: number;
  description: string;
}

const defaultParams: Parameter[] = [
  { name: 'Tile M', category: 'Tile尺寸', options: [16, 32, 64, 128, 256], selectedIdx: 2, description: '矩阵A的行分块大小' },
  { name: 'Tile N', category: 'Tile尺寸', options: [16, 32, 64, 128, 256], selectedIdx: 2, description: '矩阵B的列分块大小' },
  { name: 'Tile K', category: 'Tile尺寸', options: [8, 16, 32, 64], selectedIdx: 1, description: '内积维度分块大小' },
  { name: 'Pipeline阶段', category: '流水线', options: [1, 2, 3, 4], selectedIdx: 1, description: '数据预取的流水线深度' },
  { name: 'Block线程X', category: '线程配置', options: [8, 16, 32], selectedIdx: 1, description: 'Block中X方向线程数' },
  { name: 'Block线程Y', category: '线程配置', options: [1, 2, 4, 8, 16], selectedIdx: 3, description: 'Block中Y方向线程数' },
  { name: '循环展开', category: '优化', options: [1, 2, 4, 8], selectedIdx: 1, description: '内层循环展开因子' },
  { name: '数据布局', category: '内存', options: ['NCHW', 'NHWC', 'NC4HW4'], selectedIdx: 1, description: '输入数据内存布局' },
];

export function SearchSpaceExplorer() {
  const [params, setParams] = useState(defaultParams);
  const [selectedCategory, setSelectedCategory] = useState('全部');

  const categories = ['全部', ...new Set(params.map(p => p.category))];
  const filtered = selectedCategory === '全部' ? params : params.filter(p => p.category === selectedCategory);

  const totalCombinations = params.reduce((s, p) => s * p.options.length, 1);

  const updateParam = (paramIdx: number, optionIdx: number) => {
    setParams(prev => prev.map((p, i) => i === paramIdx ? { ...p, selectedIdx: optionIdx } : p));
  };

  const getEstPerf = () => {
    const tileM = params[0].options[params[0].selectedIdx] as number;
    const tileN = params[1].options[params[1].selectedIdx] as number;
    const pipeStages = params[3].options[params[3].selectedIdx] as number;
    const base = Math.min(tileM, tileN) / 256 * 50 + pipeStages * 10;
    return Math.min(base + 30 + Math.random() * 10, 95).toFixed(1);
  };

  return (
    <div className="bg-gray-900 rounded-xl p-6 text-white">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-bold text-cyan-400">参数搜索空间探索器</h2>
        <div className="text-sm text-gray-400">搜索空间: {totalCombinations.toLocaleString()} 种组合</div>
      </div>

      {/* Category filter */}
      <div className="flex gap-2 mb-4">
        {categories.map(c => (
          <button key={c} onClick={() => setSelectedCategory(c)}
            className={`px-3 py-1 rounded-full text-xs transition-all ${
              selectedCategory === c ? 'bg-cyan-600' : 'bg-gray-700 hover:bg-gray-600'
            }`}>{c}</button>
        ))}
      </div>

      {/* Parameters */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        {filtered.map((param, fi) => {
          const paramIdx = params.indexOf(param);
          return (
            <div key={fi} className="bg-gray-800 rounded-lg p-3">
              <div className="flex items-center justify-between mb-2">
                <div>
                  <span className="text-sm font-medium text-gray-200">{param.name}</span>
                  <span className="text-xs text-gray-500 ml-2">{param.category}</span>
                </div>
                <span className="text-sm font-mono text-cyan-300">{String(param.options[param.selectedIdx])}</span>
              </div>
              <div className="text-xs text-gray-500 mb-2">{param.description}</div>
              <div className="flex gap-1">
                {param.options.map((opt, oi) => (
                  <button key={oi} onClick={() => updateParam(paramIdx, oi)}
                    className={`px-2 py-1 rounded text-xs transition-all ${
                      oi === param.selectedIdx ? 'bg-cyan-600 text-white' : 'bg-gray-700 text-gray-400 hover:bg-gray-600'
                    }`}>
                    {String(opt)}
                  </button>
                ))}
              </div>
            </div>
          );
        })}
      </div>

      {/* Current configuration summary */}
      <div className="bg-gray-800 rounded-lg p-4">
        <div className="text-sm text-gray-400 mb-2">当前配置</div>
        <div className="grid grid-cols-4 gap-2 text-xs font-mono">
          {params.map((p, i) => (
            <div key={i} className="bg-gray-700/50 rounded px-2 py-1">
              <span className="text-gray-500">{p.name}:</span>{' '}
              <span className="text-cyan-300">{String(p.options[p.selectedIdx])}</span>
            </div>
          ))}
        </div>
        <div className="mt-3 flex items-center gap-4">
          <div className="text-sm text-gray-400">预估性能:</div>
          <div className="flex-1 bg-gray-700 rounded-full h-4 overflow-hidden">
            <div className="h-full bg-gradient-to-r from-cyan-600 to-green-500 rounded-full transition-all"
              style={{ width: `${getEstPerf()}%` }} />
          </div>
          <span className="text-sm text-green-400 font-mono">{getEstPerf()}%</span>
        </div>
      </div>
    </div>
  );
}
