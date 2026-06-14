'use client';

import React, { useState } from 'react';

interface CodeVersion {
  level: string;
  color: string;
  code: string;
  lines: number;
}

export function ThreeLevelInterfaceComparison() {
  const [activeTab, setActiveTab] = useState(0);

  const versions: CodeVersion[] = [
    {
      level: 'Beginner',
      color: '#10B981',
      lines: 8,
      code: `@tilelang.autotvm
def matmul(M, N, K):
    A = te.placeholder((M, K), name='A')
    B = te.placeholder((K, N), name='B')
    C = te.compute((M, N), lambda i, j:
        te.sum(A[i, k] * B[k, j], axis=k))
    return [A, B, C]`,
    },
    {
      level: 'Developer',
      color: '#F59E0B',
      lines: 25,
      code: `@T.prim_func
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
    )`,
    },
    {
      level: 'Expert',
      color: '#EF4444',
      lines: 35,
      code: `@T.prim_func
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
        
        T.tvm_store_matrix(C_local, C, [bx*BM, by*BN])`,
    },
  ];

  return (
    <div className="p-6 bg-gray-900 rounded-xl">
      <h2 className="text-2xl font-bold text-white mb-6">三级接口代码对比</h2>
      
      <div className="flex gap-2 mb-4">
        {versions.map((v, i) => (
          <button
            key={i}
            onClick={() => setActiveTab(i)}
            className={`px-4 py-2 rounded-lg font-bold transition-all ${
              activeTab === i
                ? 'text-white'
                : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
            }`}
            style={{
              backgroundColor: activeTab === i ? v.color : undefined,
            }}
          >
            {v.level} ({v.lines}行)
          </button>
        ))}
      </div>
      
      <div className="bg-gray-800 rounded-lg overflow-hidden">
        {/* Editor header */}
        <div className="flex items-center gap-2 px-4 py-2 bg-gray-900 border-b border-gray-700">
          <div className="w-3 h-3 rounded-full bg-red-500" />
          <div className="w-3 h-3 rounded-full bg-yellow-500" />
          <div className="w-3 h-3 rounded-full bg-green-500" />
          <span className="ml-4 text-gray-400 text-sm font-mono">
            gemm_{versions[activeTab].level.toLowerCase()}.py
          </span>
        </div>
        
        {/* Code display */}
        <div className="p-4 overflow-x-auto">
          <pre className="text-sm font-mono">
            {versions[activeTab].code.split('\n').map((line, i) => (
              <div key={i} className="flex">
                <span className="text-gray-500 w-8 text-right mr-4 select-none">
                  {i + 1}
                </span>
                <span className="text-green-400">{line}</span>
              </div>
            ))}
          </pre>
        </div>
      </div>
      
      <div className="mt-4 grid grid-cols-3 gap-4 text-center">
        {versions.map((v, i) => (
          <div
            key={i}
            className={`p-3 rounded-lg border ${
              activeTab === i ? 'border-current' : 'border-gray-700'
            }`}
            style={{ borderColor: activeTab === i ? v.color : undefined }}
          >
            <div className="text-white font-bold">{v.level}</div>
            <div className="text-gray-400 text-sm">{v.lines} 行代码</div>
          </div>
        ))}
      </div>
    </div>
  );
}
