'use client';
import { useState } from 'react';

interface Comparison {
  name: string;
  beforeCode: string;
  afterCode: string;
  speedup: string;
  description: string;
}

const comparisons: Comparison[] = [
  {
    name: '算子融合',
    beforeCode: `for (int i = 0; i < N; i++) {\n  temp[i] = a[i] + b[i];\n}\nfor (int i = 0; i < N; i++) {\n  out[i] = temp[i] * 2;\n}`,
    afterCode: `for (int i = 0; i < N; i++) {\n  out[i] = (a[i] + b[i]) * 2;\n}`,
    speedup: '1.8x',
    description: '消除中间变量temp，减少一次全局内存读写'
  },
  {
    name: '循环展开',
    beforeCode: `for (int i = 0; i < 1024; i++) {\n  sum += data[i];\n}`,
    afterCode: `for (int i = 0; i < 1024; i += 4) {\n  sum += data[i] + data[i+1]\n       + data[i+2] + data[i+3];\n}`,
    speedup: '2.3x',
    description: '减少循环次数4倍，提高指令级并行度'
  },
  {
    name: '内存合并',
    beforeCode: `// 线程i访问第i行\ntid = threadIdx.x;\nfor (int j = 0; j < N; j++) {\n  val = mat[tid * N + j];\n}`,
    afterCode: `// 线程i访问第i列\ntid = threadIdx.x;\nfor (int j = 0; j < N; j++) {\n  val = mat[j * N + tid];\n}`,
    speedup: '3.1x',
    description: '使相邻线程访问连续内存，实现合并访问'
  },
  {
    name: '常量传播',
    beforeCode: `const int TILE = 16;\nfor (int i = 0; i < M; i++) {\n  for (int j = 0; j < N; j++) {\n    c[i][j] = a[i][j] + b[i][j];\n  }\n}`,
    afterCode: `// 编译期计算 M * N\nfor (int idx = 0; idx < MN; idx++) {\n  c[idx] = a[idx] + b[idx];\n}`,
    speedup: '1.5x',
    description: '消除二维索引计算，循环展平'
  },
];

export function OptimizationPassComparison() {
  const [selected, setSelected] = useState(0);
  const [showDiff, setShowDiff] = useState(false);

  return (
    <div className="bg-gray-900 rounded-xl p-6 text-white">
      <h2 className="text-xl font-bold text-cyan-400 mb-4">优化Pass前后对比</h2>

      <div className="flex gap-2 mb-4 flex-wrap">
        {comparisons.map((c, i) => (
          <button key={i} onClick={() => { setSelected(i); setShowDiff(false); }}
            className={`px-3 py-2 rounded-lg text-sm transition-all ${
              selected === i ? 'bg-cyan-600 text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}>
            {c.name}
            <span className="ml-2 text-green-400 font-mono text-xs">{c.speedup}</span>
          </button>
        ))}
      </div>

      <div className="p-3 bg-gray-800 rounded-lg text-sm text-gray-300 mb-4">
        {comparisons[selected].description}
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="rounded-lg overflow-hidden border border-red-800/50">
          <div className="bg-red-900/30 px-3 py-1.5 text-xs font-medium text-red-300 flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-red-400" />优化前
          </div>
          <pre className="p-3 bg-gray-950 text-sm font-mono text-gray-300 overflow-x-auto whitespace-pre">{comparisons[selected].beforeCode}</pre>
        </div>
        <div className="rounded-lg overflow-hidden border border-green-800/50">
          <div className="bg-green-900/30 px-3 py-1.5 text-xs font-medium text-green-300 flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-green-400" />优化后
          </div>
          <pre className="p-3 bg-gray-950 text-sm font-mono text-gray-300 overflow-x-auto whitespace-pre">{comparisons[selected].afterCode}</pre>
        </div>
      </div>

      <div className="mt-4 flex items-center gap-4">
        <div className="flex items-center gap-2">
          <span className="text-gray-500 text-sm">性能提升:</span>
          <span className="text-green-400 font-bold text-lg">{comparisons[selected].speedup}</span>
        </div>
        <div className="flex-1 bg-gray-800 rounded-full h-3 overflow-hidden">
          <div className="h-full bg-gradient-to-r from-green-600 to-green-400 rounded-full transition-all duration-700"
            style={{ width: `${(parseFloat(comparisons[selected].speedup) / 4) * 100}%` }} />
        </div>
      </div>
    </div>
  );
}
