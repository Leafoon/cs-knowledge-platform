'use client';

import { useState } from 'react';

const irExamples = [
  {
    name: 'TileLang IR',
    color: 'blue',
    code: `// TileLang TIR 示例
@T.prim_func
def matmul(A: T.Buffer[(1024, 1024), "float16"],
           B: T.Buffer[(1024, 1024), "float16"],
           C: T.Buffer[(1024, 1024), "float16"]):
    # 自动分块原语
    for bx in T.thread_binding(8, "blockIdx.x"):
        for by in T.thread_binding(8, "blockIdx.y"):
            A_local = T.alloc_buffer([128, 32], "float16", scope="shared")
            B_local = T.alloc_buffer([32, 128], "float16", scope="shared")
            C_local = T.alloc_buffer([128, 128], "float16", scope="local")

            for k in T.serial(32):
                # 自动同步和缓存
                T.copy(A[bx*128:(bx+1)*128, k*32:(k+1)*32], A_local)
                T.copy(B[k*32:(k+1)*32, by*128:(by+1)*128], B_local)
                T.gemm(A_local, B_local, C_local)

            T.copy(C_local, C[bx*128:(bx+1)*128, by*128:(by+1)*128])`,
    features: ['高层抽象', '自动分块', '内存管理'],
  },
  {
    name: 'Triton IR',
    color: 'green',
    code: `# Triton IR 示例
def matmul_kernel(A_ptr, B_ptr, C_ptr, M, N, K):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, 128)
    num_pid_n = tl.cdiv(N, 128)

    # 手动计算偏移
    offs_m = (pid // num_pid_n) * 128 + tl.arange(0, 128)
    offs_n = (pid % num_pid_n) * 128 + tl.arange(0, 128)
    offs_k = tl.arange(0, 32)

    # 手动内存管理
    A_ptrs = A_ptr + offs_m[:, None] * K + offs_k[None, :]
    B_ptrs = B_ptr + offs_k[:, None] * N + offs_n[None, :]

    acc = tl.zeros((128, 128), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, 32)):
        a = tl.load(A_ptrs, mask=offs_k[None, :] < K - k * 32)
        b = tl.load(B_ptrs, mask=offs_k[:, None] < K - k * 32)
        acc += tl.dot(a, b)
        A_ptrs += 32
        B_ptrs += 32 * N

    tl.store(C_ptr + offs_m[:, None] * N + offs_n[None, :],
             acc.to(tl.float16))`,
    features: ['Python DSL', '手动调度', 'Block 操作'],
  },
  {
    name: 'TVM TIR',
    color: 'purple',
    code: `# TVM TIR 示例
@T.prim_func
def matmul(A: T.Buffer[(1024, 1024), "float16"],
           B: T.Buffer[(1024, 1024), "float16"],
           C: T.Buffer[(1024, 1024), "float16"]):
    # 手动分块
    for bx in T.thread_binding(8, "blockIdx.x"):
        for by in T.thread_binding(8, "blockIdx.y"):
            for k in T.serial(32):
                for i in T.serial(128):
                    for j in T.vectorize(128):
                        C[bx*128+i, by*128+j] += (
                            A[bx*128+i, k*32+j] *
                            B[k*32+j, by*128+j]
                        )

# 调度原语
s = te.create_schedule(C.op)
bx, by, k, i, j = s[C].op.axis
s[C].bind(bx, "blockIdx.x")
s[C].bind(by, "blockIdx.y")
s[C].unroll(k)
s[C].vectorize(j)`,
    features: ['Schedule API', '自动调优', '多后端'],
  },
];

export default function IRDesignComparison() {
  const [selectedIR, setSelectedIR] = useState<number>(0);
  const [lineHover, setLineHover] = useState<number | null>(null);

  const currentIR = irExamples[selectedIR];

  return (
    <div className="p-6 bg-gray-900 rounded-xl text-white">
      <h2 className="text-xl font-bold mb-2">IR 设计对比</h2>
      <p className="text-gray-400 text-sm mb-4">对比 TileLang IR、Triton IR 和 TVM TIR 的设计哲学</p>

      <div className="flex gap-2 mb-6">
        {irExamples.map((ir, idx) => (
          <button
            key={idx}
            onClick={() => setSelectedIR(idx)}
            className={`px-4 py-2 rounded text-sm font-medium transition-all ${
              selectedIR === idx ? `bg-${ir.color}-600` : 'bg-gray-800 text-gray-400'
            }`}
          >
            {ir.name}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-3 gap-4 mb-6">
        {irExamples.map((ir, idx) => (
          <div
            key={idx}
            className={`bg-gray-800 rounded-lg p-3 cursor-pointer transition-all ${
              selectedIR === idx ? 'ring-2 ring-blue-500' : 'hover:bg-gray-750'
            }`}
            onClick={() => setSelectedIR(idx)}
          >
            <h3 className={`text-sm font-bold text-${ir.color}-400 mb-2`}>{ir.name}</h3>
            <div className="space-y-1">
              {ir.features.map((feat, fidx) => (
                <div key={fidx} className="flex items-center gap-2 text-xs text-gray-400">
                  <span className={`text-${ir.color}-400`}>●</span>
                  {feat}
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      <div className="bg-gray-800 rounded-lg overflow-hidden">
        <div className={`bg-${currentIR.color}-900/50 px-4 py-2 border-b border-gray-700`}>
          <span className={`text-sm font-bold text-${currentIR.color}-400`}>{currentIR.name}</span>
          <span className="text-xs text-gray-400 ml-2">示例代码</span>
        </div>
        <pre className="p-4 text-xs text-gray-300 overflow-x-auto leading-relaxed">
          {currentIR.code.split('\n').map((line, i) => (
            <div
              key={i}
              className={`px-2 -mx-2 ${lineHover === i ? `bg-${currentIR.color}-900/30` : ''}`}
              onMouseEnter={() => setLineHover(i)}
              onMouseLeave={() => setLineHover(null)}
            >
              <span className="text-gray-600 select-none inline-block w-6">{i + 1}</span>
              {line}
            </div>
          ))}
        </pre>
      </div>

      <div className="mt-4 grid grid-cols-3 gap-4 text-xs">
        <div className="bg-gray-800 p-3 rounded">
          <div className="text-blue-400 font-bold mb-1">TileLang</div>
          <div className="text-gray-400">高级抽象，自动优化</div>
        </div>
        <div className="bg-gray-800 p-3 rounded">
          <div className="text-green-400 font-bold mb-1">Triton</div>
          <div className="text-gray-400">Python 友好，手动控制</div>
        </div>
        <div className="bg-gray-800 p-3 rounded">
          <div className="text-purple-400 font-bold mb-1">TVM</div>
          <div className="text-gray-400">通用框架，自动调优</div>
        </div>
      </div>
    </div>
  );
}
