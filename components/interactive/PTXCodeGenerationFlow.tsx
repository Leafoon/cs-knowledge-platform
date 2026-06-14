'use client';
import { useState } from 'react';

const transformations = [
  {
    stage: 'IR优化',
    input: 'for (i=0; i<M; i++)\n  for (j=0; j<N; j++)\n    C[i][j] = A[i][k]*B[k][j]',
    output: '// 循环分块 + 向量化\nfor (ii=0; ii<M; ii+=16)\n  for (jj=0; jj<N; jj+=16)\n    vC = vA * vB  // 128-bit ops',
    description: '循环分块、向量化、寄存器分配优化',
  },
  {
    stage: '寄存器分配',
    input: 'vC = vA * vB\nC[i][j] = vC\nA[i][k] = load(...)',
    output: 'SASS:\n  LD.E.128 vA, [addrA]\n  LD.E.128 vB, [addrB]\n  HMMA.16816 vC, vA, vB\n  ST.E.128 [addrC], vC',
    description: '将虚拟寄存器映射到物理寄存器，指令调度',
  },
  {
    stage: '指令选择',
    input: '// 抽象矩阵乘法\nfragment c = mma(a, b)',
    output: '// PTX MMA指令\nmma.sync.aligned.m16n8k16\n  .f32.f16.f16.f32\n  {r0,r1,r2,r3},\n  {r4-r7}, {r8-r11},\n  {r0,r1,r2,r3}',
    description: '将抽象操作映射到具体PTX/SASS指令',
  },
  {
    stage: '指令调度',
    input: 'LD.E vA\nLD.E vB\nHMMA vC\nST.E vC',
    output: 'LD.E vA0\nLD.E vB0\nLD.E vA1  // 流水线化\nLD.E vB1\nHMMA vC0\nHMMA vC1  // ILP最大化',
    description: '重排指令顺序以最大化指令级并行度',
  },
];

export function PTXCodeGenerationFlow() {
  const [selectedStage, setSelectedStage] = useState(0);

  return (
    <div className="bg-gray-900 rounded-xl p-6 text-white">
      <h2 className="text-xl font-bold text-cyan-400 mb-6">PTX代码生成流程</h2>

      {/* Stage selector */}
      <div className="flex gap-2 mb-6">
        {transformations.map((t, i) => (
          <button key={i} onClick={() => setSelectedStage(i)}
            className={`px-4 py-2 rounded-lg text-sm transition-all ${
              selectedStage === i ? 'bg-cyan-600' : 'bg-gray-700 hover:bg-gray-600'
            }`}>
            {t.stage}
          </button>
        ))}
      </div>

      {/* Transformation display */}
      <div className="grid grid-cols-2 gap-6">
        <div>
          <div className="text-sm text-gray-400 mb-2">输入 (Before)</div>
          <div className="border border-red-800/50 rounded-lg overflow-hidden">
            <div className="bg-red-900/20 px-3 py-1.5 text-xs text-red-300 flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-red-400" />{transformations[selectedStage].stage} 前
            </div>
            <pre className="p-4 bg-gray-950 text-sm font-mono text-gray-300 whitespace-pre overflow-x-auto">{transformations[selectedStage].input}</pre>
          </div>
        </div>
        <div>
          <div className="text-sm text-gray-400 mb-2">输出 (After)</div>
          <div className="border border-green-800/50 rounded-lg overflow-hidden">
            <div className="bg-green-900/20 px-3 py-1.5 text-xs text-green-300 flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-green-400" />{transformations[selectedStage].stage} 后
            </div>
            <pre className="p-4 bg-gray-950 text-sm font-mono text-gray-300 whitespace-pre overflow-x-auto">{transformations[selectedStage].output}</pre>
          </div>
        </div>
      </div>

      <div className="mt-4 p-3 bg-gray-800 rounded-lg text-sm text-gray-300">
        {transformations[selectedStage].description}
      </div>

      {/* Full pipeline */}
      <div className="mt-6 border border-gray-700 rounded-lg p-4">
        <div className="text-sm text-gray-400 mb-3">完整编译管线</div>
        <div className="flex items-center gap-2 text-xs">
          {['TileLang', '优化IR', 'LLVM IR', 'PTX', 'SASS', 'Binary'].map((s, i) => (
            <div key={i} className="flex items-center gap-2">
              <div className={`px-3 py-1.5 rounded ${
                i <= selectedStage + 1 ? 'bg-cyan-800 text-cyan-200' : 'bg-gray-800 text-gray-400'
              }`}>{s}</div>
              {i < 5 && <span className="text-gray-600">→</span>}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
