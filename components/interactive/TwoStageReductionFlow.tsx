'use client';

import { useState } from 'react';

export function TwoStageReductionFlow() {
  const [stage, setStage] = useState(0);

  return (
    <div className="p-6 bg-white rounded-xl shadow-lg">
      <h2 className="text-xl font-bold mb-4 text-gray-800">两阶段归约流程</h2>
      
      <div className="flex gap-2 mb-6">
        <button
          onClick={() => setStage(0)}
          className={`px-4 py-2 rounded-lg transition-colors ${
            stage === 0 ? 'bg-blue-500 text-white' : 'bg-gray-100'
          }`}
        >
          阶段1: Tile内归约
        </button>
        <button
          onClick={() => setStage(1)}
          className={`px-4 py-2 rounded-lg transition-colors ${
            stage === 1 ? 'bg-green-500 text-white' : 'bg-gray-100'
          }`}
        >
          阶段2: 跨Tile归约
        </button>
      </div>

      {stage === 0 ? (
        <div className="space-y-4">
          <div className="text-center mb-4">
            <span className="px-3 py-1 bg-blue-100 rounded-full text-blue-700 text-sm">
              每个线程块处理一个Tile
            </span>
          </div>
          
          <div className="grid grid-cols-4 gap-2">
            {[0, 1, 2, 3].map((tile) => (
              <div key={tile} className="p-3 bg-blue-50 rounded-lg">
                <div className="text-xs text-blue-500 mb-2">Tile {tile}</div>
                <div className="space-y-1">
                  {[0, 1, 2, 3].map((i) => (
                    <div key={i} className="text-xs font-mono text-gray-600">
                      {tile * 16 + i * 4}~{tile * 16 + i * 4 + 3}
                    </div>
                  ))}
                </div>
                <div className="mt-2 text-center text-sm font-bold text-blue-700">
                  → partial_{tile}
                </div>
              </div>
            ))}
          </div>
        </div>
      ) : (
        <div className="space-y-4">
          <div className="text-center mb-4">
            <span className="px-3 py-1 bg-green-100 rounded-full text-green-700 text-sm">
              一个线程块归约所有partial结果
            </span>
          </div>
          
          <div className="flex items-center justify-center gap-4">
            {['partial_0', 'partial_1', 'partial_2', 'partial_3'].map((p) => (
              <div key={p} className="px-3 py-2 bg-green-100 rounded-lg text-sm font-mono text-green-700">
                {p}
              </div>
            ))}
          </div>
          
          <div className="text-center text-2xl">↓</div>
          
          <div className="flex justify-center">
            <div className="px-6 py-3 bg-green-200 rounded-lg text-lg font-bold text-green-800">
              最终结果
            </div>
          </div>
        </div>
      )}

      <div className="mt-6 p-3 bg-yellow-50 rounded-lg text-sm text-yellow-700">
        💡 两阶段归约通过分层处理，既利用了并行性又减少了同步开销
      </div>
    </div>
  );
}