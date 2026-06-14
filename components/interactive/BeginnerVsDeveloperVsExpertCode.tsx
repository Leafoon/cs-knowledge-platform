'use client';

import React, { useState } from 'react';

interface CodeVersion {
  level: string;
  color: string;
  code: string;
  features: string[];
}

export function BeginnerVsDeveloperVsExpertCode() {
  const [selectedLevel, setSelectedLevel] = useState<number>(0);

  const versions: CodeVersion[] = [
    {
      level: 'Beginner',
      color: '#10B981',
      features: ['自动内存管理', '简单API', '快速原型'],
      code: `# Beginner Level - 最简单的GEMM实现
import tilelang
from tilelang import te

@tilelang.autotvm
def gemm(M, N, K):
    # 定义输入输出
    A = te.placeholder((M, K), name='A')
    B = te.placeholder((K, N), name='B')
    
    # 定义计算
    k = te.reduce_axis((0, K), name='k')
    C = te.compute((M, N), lambda i, j:
        te.sum(A[i, k] * B[k, j], axis=k),
        name='C')
    
    return [A, B, C]

# 编译并运行
func = tilelang.build(gemm(M=1024, N=1024, K=1024))`,
    },
    {
      level: 'Developer',
      color: '#F59E0B',
      features: ['自定义Tiling', 'Shared Memory', '循环优化'],
      code: `# Developer Level - 带Tiling的GEMM
import tilelang
from tilelang import T

@T.prim_func
def gemm_tiled(
    A: T.Buffer((M, K), "float16"),
    B: T.Buffer((K, N), "float16"),
    C: T.Buffer((M, N), "float32"),
):
    # 定义Block大小
    BM, BN, BK = 128, 128, 32
    
    T.func_with_sch(lambda:
        # 外层循环: 按Block遍历
        for bx, by in T.grid(M//BM, N//BN):
            # 内层循环: K维度分块
            for k in range(0, K, BK):
                # 分配Shared Memory
                A_shared = T.alloc_shared((BM, BK), "float16")
                B_shared = T.alloc_shared((BK, BN), "float16")
                
                # 计算当前块
                for ti, tj in T.grid(BM, BN):
                    C[by*BN+tj, bx*BM+ti] += A_shared[ti, k] * B_shared[k, tj]
    )`,
    },
    {
      level: 'Expert',
      color: '#EF4444',
      features: ['完全控制', 'Tensor Core', '极致优化'],
      code: `# Expert Level - 手动优化的GEMM
import tilelang
from tilelang import T

@T.prim_func
def gemm_expert(
    A: T.Buffer((M, K), "float16"),
    B: T.Buffer((K, N), "float16"),
    C: T.Buffer((M, N), "float32"),
):
    BM, BN, BK = 128, 128, 32
    warp_size = 32
    
    for bx, by in T.grid(M//BM, N//BN):
        # 分配Fragment (寄存器)
        A_local = T.alloc_fragment((BM, BK), "float16")
        B_local = T.alloc_fragment((BK, BN), "float16")
        C_local = T.alloc_fragment((BM, BN), "float32")
        
        # 初始化累加器
        T.tvm_fill_fragment(C_local, 0.0)
        
        # 主循环: K维度
        for k in range(0, K, BK):
            # 加载数据到Fragment
            T.tvm_load_matrix(A, [bx*BM, k], A_local)
            T.tvm_load_matrix(B, [k, by*BN], B_local)
            
            # 使用Tensor Core计算
            T.tvm_gemm(A_local, B_local, C_local)
        
        # 写回结果
        T.tvm_store_matrix(C_local, C, [bx*BM, by*BN])`,
    },
  ];

  return (
    <div className="p-6 bg-gray-900 rounded-xl">
      <h2 className="text-2xl font-bold text-white mb-6">代码编辑器风格展示</h2>
      
      <div className="flex gap-4 mb-6">
        {versions.map((v, i) => (
          <button
            key={i}
            onClick={() => setSelectedLevel(i)}
            className={`flex-1 p-4 rounded-lg border-2 transition-all ${
              selectedLevel === i ? 'scale-105' : 'opacity-70 hover:opacity-100'
            }`}
            style={{
              borderColor: v.color,
              backgroundColor: selectedLevel === i ? `${v.color}20` : 'transparent',
            }}
          >
            <div className="text-white font-bold">{v.level}</div>
            <div className="text-gray-400 text-sm mt-1">{v.features.join(' • ')}</div>
          </button>
        ))}
      </div>
      
      {/* Code editor */}
      <div className="bg-gray-800 rounded-lg overflow-hidden">
        <div className="flex items-center px-4 py-2 bg-gray-900 border-b border-gray-700">
          <div className="flex gap-2">
            <div className="w-3 h-3 rounded-full bg-red-500" />
            <div className="w-3 h-3 rounded-full bg-yellow-500" />
            <div className="w-3 h-3 rounded-full bg-green-500" />
          </div>
          <div className="ml-4 text-gray-400 text-sm font-mono">
            gemm_{versions[selectedLevel].level.toLowerCase()}.py
          </div>
        </div>
        
        <div className="p-4 overflow-x-auto max-h-96 overflow-y-auto">
          <pre className="text-sm font-mono">
            {versions[selectedLevel].code.split('\n').map((line, i) => (
              <div key={i} className="flex hover:bg-gray-700">
                <span className="text-gray-500 w-10 text-right mr-4 select-none">
                  {i + 1}
                </span>
                <span className="text-green-400">{line}</span>
              </div>
            ))}
          </pre>
        </div>
      </div>
      
      <div className="mt-4 text-center text-gray-400 text-sm">
        点击切换不同抽象级别的代码实现
      </div>
    </div>
  );
}
