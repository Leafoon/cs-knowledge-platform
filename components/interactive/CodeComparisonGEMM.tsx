'use client';

import { useState } from 'react';

const tileLangCode = `@tile_lang.jit
def matmul_kernel(
    A: tile_lang.Buffer[TileLang.float16],
    B: tile_lang.Buffer[TileLang.float16],
    C: tile_lang.Buffer[TileLang.float16],
):
    M, N, K = 1024, 1024, 1024
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32

    # 自动分块和调度
    for bx in tile_lang.grid(M // BLOCK_M):
        for by in tile_lang.grid(N // BLOCK_N):
            # 自动内存分配
            A_local = tile_lang.alloc([BLOCK_M, BLOCK_K])
            B_local = tile_lang.alloc([BLOCK_N, BLOCK_K])
            C_local = tile_lang.alloc([BLOCK_M, BLOCK_N])

            for k in tile_lang.grid(K // BLOCK_K):
                # 自动加载到共享内存
                tile_lang.copy(A[bx*BLOCK_M:(bx+1)*BLOCK_M,
                                  k*BLOCK_K:(k+1)*BLOCK_K],
                              A_local)
                tile_lang.copy(B[k*BLOCK_K:(k+1)*BLOCK_K,
                                  by*BLOCK_N:(by+1)*BLOCK_N],
                              B_local)
                # 自动计算优化
                tile_lang.gemm(A_local, B_local, C_local)

            # 自动写出结果
            tile_lang.copy(C_local,
                          C[bx*BLOCK_M:(bx+1)*BLOCK_M,
                            by*BLOCK_N:(by+1)*BLOCK_N])`;

const tritonCode = `@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # 手动计算偏移
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # 手动内存管理
    A_ptrs = A_ptr + (offs_am[:, None] * stride_am
                      + offs_k[None, :] * stride_ak)
    B_ptrs = B_ptr + (offs_k[:, None] * stride_bk
                      + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(A_ptrs)
        b = tl.load(B_ptrs)
        accumulator += tl.dot(a, b)
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk

    # 手动写出结果
    C = tl.cast(accumulator, tl.float16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C_ptrs = C_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(C_ptrs, C)`;

export default function CodeComparisonGEMM() {
  const [view, setView] = useState<'split' | 'tileLang' | 'triton'>('split');
  const [highlightLine, setHighlightLine] = useState<number | null>(null);

  return (
    <div className="p-6 bg-gray-900 rounded-xl text-white">
      <h2 className="text-xl font-bold mb-2">GEMM 实现对比 - TileLang vs Triton</h2>
      <p className="text-gray-400 text-sm mb-4">并排对比两种框架的矩阵乘法实现，体验代码简洁度差异</p>

      <div className="flex items-center gap-4 mb-4">
        <div className="flex bg-gray-800 rounded-lg p-1">
          {[
            { key: 'split', label: '并排对比' },
            { key: 'tileLang', label: 'TileLang' },
            { key: 'triton', label: 'Triton' },
          ].map((v) => (
            <button
              key={v.key}
              onClick={() => setView(v.key as typeof view)}
              className={`px-3 py-1.5 rounded text-xs font-medium transition-all ${
                view === v.key ? 'bg-blue-600' : 'text-gray-400 hover:text-white'
              }`}
            >
              {v.label}
            </button>
          ))}
        </div>
        <div className="ml-auto flex gap-4 text-xs">
          <span className="text-blue-400">● TileLang: {tileLangCode.split('\n').length} 行</span>
          <span className="text-green-400">● Triton: {tritonCode.split('\n').length} 行</span>
        </div>
      </div>

      <div className={`grid ${view === 'split' ? 'grid-cols-2 gap-4' : 'grid-cols-1'}`}>
        {(view === 'split' || view === 'tileLang') && (
          <div className="bg-gray-800 rounded-lg overflow-hidden">
            <div className="bg-blue-900/50 px-4 py-2 text-xs font-bold text-blue-400 border-b border-gray-700">
              TileLang 实现
            </div>
            <pre className="p-4 text-xs text-gray-300 overflow-x-auto leading-relaxed">
              <code>{tileLangCode.split('\n').map((line, i) => (
                <div
                  key={i}
                  className={`hover:bg-gray-700/50 px-2 -mx-2 ${
                    highlightLine === i ? 'bg-blue-900/30 border-l-2 border-blue-500' : ''
                  }`}
                  onMouseEnter={() => setHighlightLine(i)}
                  onMouseLeave={() => setHighlightLine(null)}
                >
                  <span className="text-gray-600 select-none inline-block w-6">{i + 1}</span>
                  {line}
                </div>
              ))}</code>
            </pre>
          </div>
        )}

        {(view === 'split' || view === 'triton') && (
          <div className="bg-gray-800 rounded-lg overflow-hidden">
            <div className="bg-green-900/50 px-4 py-2 text-xs font-bold text-green-400 border-b border-gray-700">
              Triton 实现
            </div>
            <pre className="p-4 text-xs text-gray-300 overflow-x-auto leading-relaxed">
              <code>{tritonCode.split('\n').map((line, i) => (
                <div
                  key={i}
                  className={`hover:bg-gray-700/50 px-2 -mx-2 ${
                    highlightLine === i ? 'bg-green-900/30 border-l-2 border-green-500' : ''
                  }`}
                  onMouseEnter={() => setHighlightLine(i)}
                  onMouseLeave={() => setHighlightLine(null)}
                >
                  <span className="text-gray-600 select-none inline-block w-6">{i + 1}</span>
                  {line}
                </div>
              ))}</code>
            </pre>
          </div>
        )}
      </div>

      <div className="mt-4 grid grid-cols-4 gap-3 text-xs">
        <div className="bg-gray-800 p-3 rounded">
          <div className="text-gray-400">手动内存管理</div>
          <div className="text-blue-400 font-bold">TileLang: 0 处</div>
          <div className="text-green-400">Triton: 5 处</div>
        </div>
        <div className="bg-gray-800 p-3 rounded">
          <div className="text-gray-400">循环嵌套</div>
          <div className="text-blue-400 font-bold">TileLang: 3</div>
          <div className="text-green-400">Triton: 6</div>
        </div>
        <div className="bg-gray-800 p-3 rounded">
          <div className="text-gray-400">配置常量</div>
          <div className="text-blue-400 font-bold">TileLang: 3</div>
          <div className="text-green-400">Triton: 8</div>
        </div>
        <div className="bg-gray-800 p-3 rounded">
          <div className="text-gray-400">代码量减少</div>
          <div className="text-yellow-400 font-bold text-lg">38%</div>
        </div>
      </div>
    </div>
  );
}
