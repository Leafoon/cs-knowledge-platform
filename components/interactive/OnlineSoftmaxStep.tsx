'use client';

import { useState } from 'react';

const initialData = [
  { block: 0, values: [1.0, 2.0, 3.0], max: 3.0, sum: 12.0 },
  { block: 1, values: [4.0, 1.0, 2.0], max: 4.0, sum: 45.0 },
  { block: 2, values: [2.0, 5.0, 1.0], max: 5.0, sum: 147.0 },
];

const rawScores = [
  [1, 2, 3], [4, 1, 2], [2, 5, 1],
];

const fullSoftmax = () => {
  const flat = rawScores.flat();
  const m = Math.max(...flat);
  const exps = flat.map(x => Math.exp(x - m));
  const s = exps.reduce((a, b) => a + b, 0);
  return flat.map(e => e / s);
};

export default function OnlineSoftmaxStep() {
  const [step, setStep] = useState(0);

  const fullResult = fullSoftmax();
  const currentMax = initialData[step].max;
  const currentSum = initialData[step].sum;
  const runningExpSum = rawScores.slice(0, step + 1).flat().reduce((acc, v) => acc + Math.exp(v - currentMax), 0);

  return (
    <div className="p-6 bg-gray-900 rounded-xl text-white">
      <h2 className="text-xl font-bold mb-4">在线 Softmax 逐步可视化</h2>

      <div className="flex gap-2 mb-4">
        {rawScores.map((_, i) => (
          <button key={i} onClick={() => setStep(i)}
            className={`px-4 py-2 rounded-lg text-sm font-bold transition-all ${
              step === i ? 'bg-purple-600 text-white' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
            }`}>
            Block {i}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-3 gap-4 mb-4">
        {rawScores.map((block, i) => (
          <div key={i} className={`rounded-lg p-3 border transition-all ${
            i <= step ? 'border-purple-500 bg-purple-500/10' : 'border-gray-700 bg-gray-800/50 opacity-40'
          }`}>
            <div className="text-xs text-gray-400 mb-2">Block {i}</div>
            <div className="flex gap-1 mb-2">
              {block.map((v, j) => (
                <div key={j} className="bg-gray-700 rounded px-2 py-1 font-mono text-xs text-center flex-1">
                  {v}
                </div>
              ))}
            </div>
            {i <= step && (
              <div className="space-y-1 text-[10px]">
                <div className="flex justify-between">
                  <span className="text-gray-400">Running Max:</span>
                  <span className="text-yellow-400 font-mono">{initialData[i].max.toFixed(1)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Scaled Sum:</span>
                  <span className="text-blue-400 font-mono">{runningExpSum.toFixed(2)}</span>
                </div>
                {i === step && (
                  <div className="bg-black/30 rounded p-1.5 mt-1">
                    <span className="text-gray-500">Δm = </span>
                    <span className="text-orange-400">{i === 0 ? '0' : `${initialData[i].max.toFixed(1)} - ${initialData[i-1].max.toFixed(1)} = ${(initialData[i].max - initialData[i-1].max).toFixed(1)}`}</span>
                  </div>
                )}
              </div>
            )}
          </div>
        ))}
      </div>

      <div className="bg-gray-800 rounded-lg p-4 text-xs space-y-2">
        <div className="font-bold text-purple-400">在线 Softmax 公式</div>
        <div className="font-mono bg-black/40 rounded p-2 text-green-400">
          m_new = max(m_old, m_block)<br/>
          d_new = d_old × e^(m_old - m_new) + Σ e^(x_i - m_new)
        </div>
        <div className="text-gray-400">
          核心：通过 Δm 修正旧的 sum，无需重新计算所有 exp，单遍扫描完成 softmax。
        </div>
      </div>
    </div>
  );
}
