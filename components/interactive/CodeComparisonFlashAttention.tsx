'use client';

import { useState } from 'react';

const tileLangFA = `@tile_lang.jit
def flash_attention_kernel(
    Q: tile_lang.Buffer[TileLang.float16],
    K: tile_lang.Buffer[TileLang.float16],
    V: tile_lang.Buffer[TileLang.float16],
    O: tile_lang.Buffer[TileLang.float16],
):
    BLOCK_Q = 128
    BLOCK_KV = 64
    HEAD_DIM = 128

    # 自动处理 Online Softmax
    for i in tile_lang.grid(Q.shape[0]):
        Q_local = tile_lang.alloc([BLOCK_Q, HEAD_DIM])
        O_local = tile_lang.alloc([BLOCK_Q, HEAD_DIM])
        m_local = tile_lang.alloc([BLOCK_Q], init=-float('inf'))
        l_local = tile_lang.alloc([BLOCK_Q], init=0.0)

        for j in tile_lang.grid(K.shape[0] // BLOCK_KV):
            K_local = tile_lang.alloc([BLOCK_KV, HEAD_DIM])
            V_local = tile_lang.alloc([BLOCK_KV, HEAD_DIM])

            tile_lang.copy(Q[i*BLOCK_Q:(i+1)*BLOCK_Q], Q_local)
            tile_lang.copy(K[j*BLOCK_KV:(j+1)*BLOCK_KV], K_local)
            tile_lang.copy(V[j*BLOCK_KV:(j+1)*BLOCK_KV], V_local)

            # 自动 Flash Attention 算法
            tile_lang.flash_attn(Q_local, K_local, V_local,
                                O_local, m_local, l_local)

        tile_lang.copy(O_local, O[i*BLOCK_Q:(i+1)*BLOCK_Q])`;

const tritonFA = `@triton.jit
def flash_attention_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_kh, stride_kd,
    stride_vb, stride_vh, stride_vd,
    stride_ob, stride_oh, stride_od,
    B, H, N, D: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    pid = tl.program_id(0)
    head_idx = pid % H
    batch_idx = pid // H

    # 手动管理所有指针偏移
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_idx * stride_qb + head_idx * stride_qh,
        shape=(N, D), strides=(stride_qd, 1),
        block_shape=(BLOCK_Q, D), order=(1, 0))

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_idx * stride_kb + head_idx * stride_kh,
        shape=(N, D), strides=(stride_kd, 1),
        block_shape=(BLOCK_KV, D), order=(1, 0))

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_idx * stride_vb + head_idx * stride_vh,
        shape=(N, D), strides=(stride_vd, 1),
        block_shape=(BLOCK_KV, D), order=(1, 0))

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_idx * stride_ob + head_idx * stride_oh,
        shape=(N, D), strides=(stride_od, 1),
        block_shape=(BLOCK_Q, D), order=(1, 0))

    # 手动实现 Online Softmax
    m_i = tl.zeros([BLOCK_Q], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_Q], dtype=tl.float32)
    acc = tl.zeros([BLOCK_Q, D], dtype=tl.float32)

    for k in range(0, N, BLOCK_KV):
        q = tl.load(Q_block_ptr)
        k_block = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)

        qk = tl.dot(q, tl.trans(k_block))
        qk = qk * 1.0 / tl.math.sqrt(D)

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(qk - m_new[:, None])

        l_new = alpha * l_i + tl.sum(beta, axis=1)
        acc = acc * alpha[:, None] + tl.dot(beta.to(tl.float16), v)

        m_i = m_new
        l_i = l_new

        K_block_ptr = tl.advance(K_block_ptr, [BLOCK_KV, 0])
        V_block_ptr = tl.advance(V_block_ptr, [BLOCK_KV, 0])

    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(tl.float16))`;

export default function CodeComparisonFlashAttention() {
  const [view, setView] = useState<'split' | 'tileLang' | 'triton'>('split');
  const [lineHover, setLineHover] = useState<number | null>(null);

  return (
    <div className="p-6 bg-gray-900 rounded-xl text-white">
      <h2 className="text-xl font-bold mb-2">FlashAttention 实现对比</h2>
      <p className="text-gray-400 text-sm mb-4">对比 FlashAttention 的实现复杂度 - 自动 vs 手动 Online Softmax</p>

      <div className="flex items-center gap-4 mb-4">
        <div className="flex bg-gray-800 rounded-lg p-1">
          {[
            { key: 'split', label: '并排' },
            { key: 'tileLang', label: 'TileLang' },
            { key: 'triton', label: 'Triton' },
          ].map((v) => (
            <button
              key={v.key}
              onClick={() => setView(v.key as typeof view)}
              className={`px-3 py-1.5 rounded text-xs font-medium transition-all ${
                view === v.key ? 'bg-blue-600' : 'text-gray-400'
              }`}
            >
              {v.label}
            </button>
          ))}
        </div>
        <div className="ml-auto flex gap-4 text-xs">
          <span className="text-blue-400">TileLang: {tileLangFA.split('\n').length} 行</span>
          <span className="text-green-400">Triton: {tritonFA.split('\n').length} 行</span>
          <span className="text-yellow-400">减少 {Math.round((1 - tileLangFA.split('\n').length / tritonFA.split('\n').length) * 100)}%</span>
        </div>
      </div>

      <div className={`grid ${view === 'split' ? 'grid-cols-2 gap-4' : 'grid-cols-1'}`}>
        {(view === 'split' || view === 'tileLang') && (
          <div className="bg-gray-800 rounded-lg overflow-hidden">
            <div className="bg-blue-900/50 px-4 py-2 text-xs font-bold text-blue-400 border-b border-gray-700 flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-blue-400" />
              TileLang - 自动 Flash Attention
            </div>
            <pre className="p-4 text-xs text-gray-300 overflow-x-auto leading-relaxed">
              {tileLangFA.split('\n').map((line, i) => (
                <div
                  key={i}
                  className={`px-2 -mx-2 ${lineHover === i ? 'bg-blue-900/30' : ''}`}
                  onMouseEnter={() => setLineHover(i)}
                  onMouseLeave={() => setLineHover(null)}
                >
                  <span className="text-gray-600 select-none inline-block w-6">{i + 1}</span>
                  {line}
                </div>
              ))}
            </pre>
          </div>
        )}

        {(view === 'split' || view === 'triton') && (
          <div className="bg-gray-800 rounded-lg overflow-hidden">
            <div className="bg-green-900/50 px-4 py-2 text-xs font-bold text-green-400 border-b border-gray-700 flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-green-400" />
              Triton - 手动 Online Softmax
            </div>
            <pre className="p-4 text-xs text-gray-300 overflow-x-auto leading-relaxed">
              {tritonFA.split('\n').map((line, i) => (
                <div
                  key={i}
                  className={`px-2 -mx-2 ${lineHover === i ? 'bg-green-900/30' : ''}`}
                  onMouseEnter={() => setLineHover(i)}
                  onMouseLeave={() => setLineHover(null)}
                >
                  <span className="text-gray-600 select-none inline-block w-6">{i + 1}</span>
                  {line}
                </div>
              ))}
            </pre>
          </div>
        )}
      </div>

      <div className="mt-4 bg-gray-800 rounded-lg p-4">
        <h3 className="text-sm font-bold mb-3 text-gray-300">关键差异分析</h3>
        <div className="grid grid-cols-3 gap-4 text-xs">
          <div>
            <div className="text-gray-400 mb-1">Online Softmax 实现</div>
            <div className="text-blue-400">TileLang: 1 行 (tile_lang.flash_attn)</div>
            <div className="text-green-400">Triton: 15+ 行手动实现</div>
          </div>
          <div>
            <div className="text-gray-400 mb-1">内存管理</div>
            <div className="text-blue-400">TileLang: 自动分配/释放</div>
            <div className="text-green-400">Triton: 3 个 block_ptr 手动管理</div>
          </div>
          <div>
            <div className="text-gray-400 mb-1">数值稳定性</div>
            <div className="text-blue-400">TileLang: 内置处理</div>
            <div className="text-green-400">Triton: 需手动实现 m/l 跟踪</div>
          </div>
        </div>
      </div>
    </div>
  );
}
