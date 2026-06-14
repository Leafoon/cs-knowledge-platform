'use client';

import { useState } from 'react';

const cudaCode = `// FlashMLA CUDA 实现 (~500行)
__global__ void flash_mla_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ KV,
    half* __restrict__ O,
    const int* __restrict__ cu_seqlens,
    const int max_seqlen,
    const int num_heads,
    const int head_dim,
    const int kv_head_dim
) {
    // 线程块索引计算
    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int tile_idx = blockIdx.z;

    const int seqlen = cu_seqlens[batch_idx + 1] - cu_seqlens[batch_idx];
    const int tile_size = 64;

    // 共享内存声明
    __shared__ half Q_shared[TILE_M][HEAD_DIM];
    __shared__ half K_shared[TILE_N][KV_HEAD_DIM];
    __shared__ half V_shared[TILE_N][KV_HEAD_DIM];
    __shared__ float s_shared[TILE_M][TILE_N];

    // 外层循环 - 分块处理序列
    for (int kv_start = 0; kv_start < seqlen; kv_start += tile_size) {
        // 加载 Q 到共享内存
        for (int i = threadIdx.x; i < TILE_M * HEAD_DIM; i += blockDim.x) {
            int row = i / HEAD_DIM;
            int col = i % HEAD_DIM;
            if (row < TILE_M && kv_start + row < seqlen) {
                Q_shared[row][col] = Q[...];
            }
        }

        // 加载 K 到共享内存
        for (int i = threadIdx.x; i < TILE_N * KV_HEAD_DIM; i += blockDim.x) {
            int row = i / KV_HEAD_DIM;
            int col = i % KV_HEAD_DIM;
            if (row < TILE_N && kv_start + row < seqlen) {
                K_shared[row][col] = KV[...];
            }
        }

        __syncthreads();

        // Scaled Dot Product
        for (int i = threadIdx.x; i < TILE_M * TILE_N; i += blockDim.x) {
            int q_idx = i / TILE_N;
            int k_idx = i % TILE_N;
            float sum = 0.0f;
            for (int d = 0; d < HEAD_DIM; d++) {
                sum += __half2float(Q_shared[q_idx][d]) *
                       __half2float(K_shared[k_idx][d]);
            }
            s_shared[q_idx][k_idx] = sum / sqrtf(HEAD_DIM);
        }

        __syncthreads();

        // Online Softmax 更新
        for (int row = threadIdx.x; row < TILE_M; row += blockDim.x) {
            float row_max = -INFINITY;
            for (int col = 0; col < TILE_N; col++) {
                row_max = fmaxf(row_max, s_shared[row][col]);
            }
            // ... softmax 计算 ...
        }

        __syncthreads();

        // 加载 V 并计算输出
        for (int i = threadIdx.x; i < TILE_N * KV_HEAD_DIM; i += blockDim.x) {
            int row = i / KV_HEAD_DIM;
            int col = i % KV_HEAD_DIM;
            if (row < TILE_N && kv_start + row < seqlen) {
                V_shared[row][col] = KV[...];
            }
        }

        __syncthreads();

        // 累加输出
        for (int i = threadIdx.x; i < TILE_M * KV_HEAD_DIM; i += blockDim.x) {
            int row = i / KV_HEAD_DIM;
            int col = i % KV_HEAD_DIM;
            float sum = 0.0f;
            for (int k = 0; k < TILE_N; k++) {
                sum += s_shared[row][k] *
                       __half2float(V_shared[k][col]);
            }
            // ... 写回结果 ...
        }
    }
}`;

const tileLangCode = `# FlashMLA TileLang 实现 (~50行)
@tile_lang.jit
def flash_mla_kernel(
    Q: tile_lang.Buffer[TileLang.float16],
    KV: tile_lang.Buffer[TileLang.float16],
    O: tile_lang.Buffer[TileLang.float16],
    cu_seqlens: tile_lang.Buffer[TileLang.int32],
):
    # 自动处理 batch 和 head 维度
    batch_idx = tile_lang blockIdx.x
    head_idx = tile_lang blockIdx.y

    seqlen = cu_seqlens[batch_idx + 1] - cu_seqlens[batch_idx]

    # TileLang 自动内存管理
    Q_local = tile_lang.alloc([128, 128], scope="shared")
    K_local = tile_lang.alloc([64, 128], scope="shared")
    V_local = tile_lang.alloc([64, 128], scope="shared")
    O_local = tile_lang.alloc([128, 128], scope="local")

    # 自动 Online Softmax 状态
    m_local = tile_lang.alloc([128], init=-float('inf'))
    l_local = tile_lang.alloc([128], init=0.0)

    # 自动分块循环
    for kv_start in tile_lang.grid(seqlen, step=64):
        # 自动加载和同步
        tile_lang.copy(Q[...], Q_local)
        tile_lang.copy(KV[kv_start:kv_start+64], K_local)
        tile_lang.copy(KV[kv_start:kv_start+64], V_local)

        # 自动 Flash Attention 计算
        tile_lang.flash_attn(
            Q_local, K_local, V_local,
            O_local, m_local, l_local,
            scale=1.0 / sqrt(128)
        )

    # 自动写出最终结果
    tile_lang.copy(O_local, O[...])`;

export default function FlashMLACompressionRatio() {
  const [view, setView] = useState<'split' | 'cuda' | 'tileLang'>('split');

  const cudaLines = cudaCode.split('\n').length;
  const tileLangLines = tileLangCode.split('\n').length;
  const compressionRatio = (cudaLines / tileLangLines).toFixed(1);

  return (
    <div className="p-6 bg-gray-900 rounded-xl text-white">
      <h2 className="text-xl font-bold mb-2">FlashMLA 代码压缩比</h2>
      <p className="text-gray-400 text-sm mb-4">从 500 行 CUDA 到 50 行 TileLang 的代码压缩</p>

      <div className="flex items-center gap-4 mb-4">
        <div className="flex bg-gray-800 rounded-lg p-1">
          {[
            { key: 'split', label: '并排对比' },
            { key: 'cuda', label: 'CUDA' },
            { key: 'tileLang', label: 'TileLang' },
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

        <div className="ml-auto flex items-center gap-6">
          <div className="text-center">
            <div className="text-2xl font-bold text-red-400">{cudaLines}</div>
            <div className="text-xs text-gray-400">CUDA 行数</div>
          </div>
          <div className="text-3xl text-gray-600">→</div>
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-400">{tileLangLines}</div>
            <div className="text-xs text-gray-400">TileLang 行数</div>
          </div>
          <div className="text-center bg-yellow-500/20 px-3 py-1 rounded">
            <div className="text-2xl font-bold text-yellow-400">{compressionRatio}x</div>
            <div className="text-xs text-gray-400">压缩比</div>
          </div>
        </div>
      </div>

      <div className={`grid ${view === 'split' ? 'grid-cols-2 gap-4' : 'grid-cols-1'}`}>
        {(view === 'split' || view === 'cuda') && (
          <div className="bg-gray-800 rounded-lg overflow-hidden">
            <div className="bg-red-900/50 px-4 py-2 text-xs font-bold text-red-400 border-b border-gray-700 flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-red-400" />
              CUDA 实现 - {cudaLines} 行
            </div>
            <pre className="p-4 text-[10px] text-gray-300 overflow-x-auto leading-relaxed max-h-80 overflow-y-auto">
              {cudaCode.split('\n').map((line, i) => (
                <div key={i} className="hover:bg-gray-700/50 px-2 -mx-2">
                  <span className="text-gray-600 select-none inline-block w-8">{i + 1}</span>
                  {line}
                </div>
              ))}
            </pre>
          </div>
        )}

        {(view === 'split' || view === 'tileLang') && (
          <div className="bg-gray-800 rounded-lg overflow-hidden">
            <div className="bg-blue-900/50 px-4 py-2 text-xs font-bold text-blue-400 border-b border-gray-700 flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-blue-400" />
              TileLang 实现 - {tileLangLines} 行
            </div>
            <pre className="p-4 text-[10px] text-gray-300 overflow-x-auto leading-relaxed max-h-80 overflow-y-auto">
              {tileLangCode.split('\n').map((line, i) => (
                <div key={i} className="hover:bg-gray-700/50 px-2 -mx-2">
                  <span className="text-gray-600 select-none inline-block w-8">{i + 1}</span>
                  {line}
                </div>
              ))}
            </pre>
          </div>
        )}
      </div>

      <div className="mt-4 grid grid-cols-4 gap-4 text-xs">
        <div className="bg-gray-800 p-3 rounded">
          <div className="text-gray-400">内存管理</div>
          <div className="text-red-400">CUDA: 手动分配</div>
          <div className="text-blue-400">TileLang: 自动</div>
        </div>
        <div className="bg-gray-800 p-3 rounded">
          <div className="text-gray-400">同步机制</div>
          <div className="text-red-400">CUDA: 手动调用</div>
          <div className="text-blue-400">TileLang: 隐式</div>
        </div>
        <div className="bg-gray-800 p-3 rounded">
          <div className="text-gray-400">Softmax</div>
          <div className="text-red-400">CUDA: 30+ 行</div>
          <div className="text-blue-400">TileLang: 1 行</div>
        </div>
        <div className="bg-gray-800 p-3 rounded">
          <div className="text-gray-400">可维护性</div>
          <div className="text-red-400">CUDA: 困难</div>
          <div className="text-blue-400">TileLang: 容易</div>
        </div>
      </div>
    </div>
  );
}
