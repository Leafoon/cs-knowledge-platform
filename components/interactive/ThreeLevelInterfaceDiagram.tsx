'use client';

import React, { useState } from 'react';

interface Level {
  name: string;
  color: string;
  complexity: number;
  description: string;
  codeExample: string;
}

export function ThreeLevelInterfaceDiagram() {
  const [activeLevel, setActiveLevel] = useState(0);

  const levels: Level[] = [
    {
      name: 'Beginner',
      color: '#10B981',
      complexity: 1,
      description: '最简单的接口，自动处理内存管理和调度',
      codeExample: `@tilelang.autotvm
def matmul(M, N, K):
    A = te.placeholder((M, K), name='A')
    B = te.placeholder((K, N), name='B')
    C = te.compute((M, N), lambda i, j:
        te.sum(A[i, k] * B[k, j], axis=k))
    return [A, B, C]`
    },
    {
      name: 'Developer',
      color: '#F59E0B',
      complexity: 2,
      description: '中级接口，支持自定义tiling和内存布局',
      codeExample: `@T.prim_func
def gemm(
    A: T.Buffer((M, K), "float16"),
    B: T.Buffer((K, N), "float16"),
    C: T.Buffer((M, N), "float32"),
):
    T.func_with_sch(lambda:
        for bx, by in T.grid(M//BM, N//BN):
            for k in range(0, K, BK):
                A_shared = T.alloc_shared((BM, BK), "float16")
                B_shared = T.alloc_shared((BK, BN), "float16")
                for ti, tj in T.grid(BM, BN):
                    C[by*BN+tj, bx*BM+ti] += A_shared[ti, k] * B_shared[k, tj]
    )`
    },
    {
      name: 'Expert',
      color: '#EF4444',
      complexity: 3,
      description: '专家级接口，完全控制所有优化细节',
      codeExample: `@T.prim_func
def gemm_optimized(
    A: T.Buffer((M, K), "float16"),
    B: T.Buffer((K, N), "float16"),
    C: T.Buffer((M, N), "float32"),
):
    for bx, by in T.grid(M//BM, N//BN):
        A_local = T.alloc_fragment((BM, BK), "float16")
        B_local = T.alloc_fragment((BK, BN), "float16")
        C_local = T.alloc_fragment((BM, BN), "float32")
        
        T.tvm_fill_fragment(C_local, 0.0)
        
        for k in range(0, K, BK):
            T.tvm_load_matrix(A, [bx*BM, k], A_local)
            T.tvm_load_matrix(B, [k, by*BN], B_local)
            T.tvm_gemm(A_local, B_local, C_local)
        
        T.tvm_store_matrix(C_local, C, [bx*BM, by*BN])`
    },
  ];

  return (
    <div className="p-6 bg-gray-900 rounded-xl">
      <h2 className="text-2xl font-bold text-white mb-6">三级接口层次结构</h2>
      
      <div className="flex gap-4 mb-6">
        {levels.map((level, i) => (
          <button
            key={i}
            onClick={() => setActiveLevel(i)}
            className={`flex-1 p-4 rounded-lg border-2 transition-all ${
              activeLevel === i
                ? 'border-current bg-opacity-20'
                : 'border-gray-700 hover:border-gray-600'
            }`}
            style={{ borderColor: activeLevel === i ? level.color : undefined }}
          >
            <div className="text-center">
              <div className="text-lg font-bold" style={{ color: level.color }}>{level.name}</div>
              <div className="text-gray-400 text-sm mt-1">复杂度: {'★'.repeat(level.complexity)}</div>
            </div>
          </button>
        ))}
      </div>
      
      <div className="bg-gray-800 rounded-lg p-6">
        <div className="flex items-center mb-4">
          <div className="w-3 h-3 rounded-full mr-3" style={{ backgroundColor: levels[activeLevel].color }} />
          <h3 className="text-xl text-white font-bold">{levels[activeLevel].name} Interface</h3>
        </div>
        <p className="text-gray-300 mb-4">{levels[activeLevel].description}</p>
        
        <div className="bg-gray-900 rounded-lg p-4 overflow-x-auto">
          <pre className="text-sm text-green-400 font-mono whitespace-pre-wrap">
            {levels[activeLevel].codeExample}
          </pre>
        </div>
      </div>
      
      <div className="mt-6 flex items-center justify-center gap-2">
        <span className="text-gray-400">简单</span>
        <div className="w-64 h-2 bg-gray-700 rounded-full overflow-hidden">
          <div
            className="h-full transition-all duration-300"
            style={{
              width: `${((activeLevel + 1) / 3) * 100}%`,
              background: `linear-gradient(to right, #10B981, #F59E0B, #EF4444)`,
            }}
          />
        </div>
        <span className="text-gray-400">复杂</span>
      </div>
    </div>
  );
}
