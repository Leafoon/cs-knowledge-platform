'use client';
import { useState } from 'react';

const loweringStages = [
  {
    stage: 'TileLang IR',
    level: 0,
    code: `@schedule\ndef matmul(A: Tensor, B: Tensor, C: Tensor):\n    for i in tile(M, 16):\n        for j in tile(N, 16):\n            for k in tile(K, 16):\n                C[i,j] += A[i,k] * B[k,j]`,
    description: '高层TileLang IR，描述算子语义',
  },
  {
    stage: '中间表示',
    level: 1,
    code: `// Tiling后的IR\ntile_loop i in [0, M, 16]:\n  tile_loop j in [0, N, 16]:\n    local_tile A[i:i+16, k:k+16]\n    local_tile B[k:k+16, j:j+16]\n    cube_compute C_local += A_local * B_local\n    write_back C[i:i+16, j:j+16] = C_local`,
    description: '显式分块和数据搬运指令',
  },
  {
    stage: 'Ascend C代码',
    level: 2,
    code: `// Ascend C编程语言\nvoid Compute(Tensor& A, Tensor& B, Tensor& C) {\n  DataCopy(l1A, gmA, {16, 16});\n  DataCopy(l1B, gmB, {16, 16});\n  pipe_barrier<PIPE_V>();\n  Cube(C_local, A_local, B_local);\n  pipe_barrier<PIPE_C>();\n  DataCopy(gmC, C_local, {16, 16});\n}`,
    description: '使用Ascend C API编写硬件原语',
  },
  {
    stage: 'NPU指令',
    level: 3,
    code: `// BiSheng编译后的NPU指令\n// L1加载指令\nLD.ARCX r0, [GM_A], 512\nLD.ARCX r1, [GM_B], 512\n// Cube计算指令\nMATMUL r2, r0, r1, 16, 16, 16\n// 数据搬运指令\nST.ARCX [GM_C], r2, 512`,
    description: '目标NPU硬件指令',
  },
];

export function AscendCLoweringFlow() {
  const [selectedStage, setSelectedStage] = useState(0);

  return (
    <div className="bg-gray-900 rounded-xl p-6 text-white">
      <h2 className="text-xl font-bold text-orange-400 mb-6">昇腾多级Lowering流程</h2>

      {/* Stage selector */}
      <div className="flex gap-2 mb-6">
        {loweringStages.map((s, i) => (
          <button key={i} onClick={() => setSelectedStage(i)}
            className={`px-4 py-2 rounded-lg text-sm transition-all ${
              selectedStage === i ? 'bg-orange-600' : 'bg-gray-700 hover:bg-gray-600'
            }`}>{s.stage}</button>
        ))}
      </div>

      {/* Lowering visualization */}
      <div className="space-y-4">
        {loweringStages.map((s, i) => (
          <div key={i} className="relative">
            <div className={`flex items-start gap-4 p-4 rounded-xl border-2 transition-all ${
              i === selectedStage ? 'border-orange-500 bg-orange-900/10' : 'border-gray-700 hover:border-gray-500'
            }`}
              onClick={() => setSelectedStage(i)}>
              <div className="flex-shrink-0">
                <div className={`w-10 h-10 rounded-full flex items-center justify-center font-bold ${
                  i === selectedStage ? 'bg-orange-600' : 'bg-gray-700'
                }`}>{s.level}</div>
              </div>
              <div className="flex-1">
                <div className="flex items-center justify-between mb-2">
                  <div className={`font-bold ${i === selectedStage ? 'text-orange-300' : 'text-gray-300'}`}>
                    {s.stage}
                  </div>
                  <div className="text-xs text-gray-500">{s.description}</div>
                </div>
                <pre className="text-xs font-mono text-gray-300 bg-gray-950 rounded-lg p-3 overflow-x-auto whitespace-pre">{s.code}</pre>
              </div>
            </div>
            {i < loweringStages.length - 1 && (
              <div className="flex justify-center py-2">
                <div className="text-gray-600 text-xs flex items-center gap-1">
                  <span>⬇</span>
                  <span>{i === 0 ? 'Tiling' : i === 1 ? 'API映射' : '指令生成'}</span>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>

      <div className="mt-4 p-3 bg-gray-800 rounded-lg text-xs text-gray-400">
        每一级Lowering将高层语义逐步降低为更接近硬件的表示，最终生成NPU可执行的机器指令。
      </div>
    </div>
  );
}
