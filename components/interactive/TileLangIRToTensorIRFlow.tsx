'use client';

import { useState } from 'react';

const stages = [
  {
    label: 'TileLang Python',
    lang: 'python',
    color: '#F59E0B',
    code: `@T.prim_func
def matmul(A, B, C):
  for i, j, k in T.grid(M, N, K):
    with T.block("compute"):
      C[i,j] += A[i,k] * B[k,j]`,
  },
  {
    label: 'Tile IR',
    lang: 'tile-ir',
    color: '#8B5CF6',
    code: `@T.prim_func
def main(A: T.Buffer(...), B: T.Buffer(...)):
  for bx, by in T.grid(MB, NB):
    with T.block("block"):
      acc = T.alloc_buffer(...)
      for ko in T.serial(KB):
        A_tile = T.read(A, bx, ko)
        B_tile = T.read(B, ko, by)
        acc += A_tile @ B_tile
      T.write(acc, C, bx, by)`,
  },
  {
    label: 'TensorIR',
    lang: 'tensor-ir',
    color: '#10B981',
    code: `@T.prim_func
def main(A: T.handle, B: T.handle, C: T.handle):
  A_buf = T.match_buffer(A, (M, K))
  B_buf = T.match_buffer(B, (K, N))
  C_buf = T.match_buffer(C, (M, N))
  for bx, by, ko in T.grid(mb, nb, kb):
    with T.block("compute"):
      A_reg = T.alloc_buffer(...)
      B_reg = T.alloc_buffer(...)
      acc = T.alloc_buffer(...)
      for ki in T.serial(bk):
        A_reg[...] = A_buf[bx*bk+ki, ...]
        B_reg[...] = B_buf[..., by*bk+ki]
        for i, j in T.serial(bm, bn):
          acc[i,j] += A_reg[i,ki]*B_reg[ki,j]`,
  },
];

export default function TileLangIRToTensorIRFlow() {
  const [activeStage, setActiveStage] = useState(0);

  return (
    <div className="p-6 bg-gray-900 rounded-xl text-white">
      <h2 className="text-xl font-bold mb-4">TileLang → IR → TensorIR 编译流程</h2>

      <div className="flex items-center gap-2 mb-6">
        {stages.map((s, i) => (
          <div key={i} className="flex items-center">
            <button
              className={`px-3 py-2 rounded-lg text-sm font-bold transition-all ${
                activeStage === i ? 'text-white' : 'bg-gray-800 text-gray-500'
              }`}
              style={activeStage === i ? { backgroundColor: s.color + '30', border: `2px solid ${s.color}`, color: s.color } : {}}
              onClick={() => setActiveStage(i)}>
              {s.label}
            </button>
            {i < stages.length - 1 && (
              <div className="text-gray-600 mx-1">→</div>
            )}
          </div>
        ))}
      </div>

      <div className="bg-black rounded-lg p-4 font-mono text-xs overflow-x-auto">
        <div className="flex items-center gap-2 mb-3">
          <div className="w-3 h-3 rounded-full" style={{ backgroundColor: stages[activeStage].color }} />
          <span className="font-bold" style={{ color: stages[activeStage].color }}>
            {stages[activeStage].label}
          </span>
        </div>
        <pre className="leading-5" style={{ color: stages[activeStage].color + 'CC' }}>
          {stages[activeStage].code}
        </pre>
      </div>

      <div className="mt-4 grid grid-cols-3 gap-3 text-xs">
        <div className="bg-gray-800 rounded p-3">
          <div className="font-bold text-yellow-400 mb-1">TileLang</div>
          <div className="text-gray-400">高层 Python DSL，描述计算意图</div>
          <div className="text-gray-500 mt-1">用户友好，易调试</div>
        </div>
        <div className="bg-gray-800 rounded p-3">
          <div className="font-bold text-purple-400 mb-1">Tile IR</div>
          <div className="text-gray-400">中间表示，显式分块和内存层次</div>
          <div className="text-gray-500 mt-1">自动分块决策</div>
        </div>
        <div className="bg-gray-800 rounded p-3">
          <div className="font-bold text-green-400 mb-1">TensorIR</div>
          <div className="text-gray-400">TVM 低级 IR，可直接生成代码</div>
          <div className="text-gray-500 mt-1">GPU/CPU 后端适配</div>
        </div>
      </div>
    </div>
  );
}
