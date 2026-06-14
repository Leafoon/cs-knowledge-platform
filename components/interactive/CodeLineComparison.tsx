'use client';

import { useState } from 'react';

const cudaLines = [
  { n: 1, t: '// FlashMLA CUDA - 500+ lines kernel', c: '#6B7280' },
  { n: 2, t: '__global__ void flash_mla_kernel(', c: '#60A5FA' },
  { n: 3, t: '  half* __restrict__ Q,', c: '#F9A8D4' },
  { n: 4, t: '  half* __restrict__ cKV,', c: '#F9A8D4' },
  { n: 5, t: '  half* __restrict__ O,', c: '#F9A8D4' },
  { n: 6, t: '  int batch_size, int seq_len) {', c: '#F9A8D4' },
  { n: 7, t: '  // Register declarations...', c: '#6B7280' },
  { n: 8, t: '  extern __shared__ half smem[];', c: '#FBBF24' },
  { n: 9, t: '  const int tid = threadIdx.x;', c: '#FFF' },
  { n: 10, t: '  const int bid = blockIdx.x;', c: '#FFF' },
  { n: 11, t: '  // Compute tile indices...', c: '#6B7280' },
  { n: 12, t: '  // Load Q tile to registers...', c: '#6B7280' },
  { n: 13, t: '  for (int ko = 0; ko < N_BLOCKS; ko++) {', c: '#FB923C' },
  { n: 14, t: '    // Load cKV block...', c: '#6B7280' },
  { n: 15, t: '    // Low-rank decode K, V...', c: '#6B7280' },
  { n: 16, t: '    // Online softmax update...', c: '#6B7280' },
  { n: 17, t: '    // S = Q @ K^T ...', c: '#6B7280' },
  { n: 18, t: '    // P = softmax(S) ...', c: '#6B7280' },
  { n: 19, t: '    // O += P @ V ...', c: '#6B7280' },
  { n: 20, t: '  }', c: '#FFF' },
  { n: 21, t: '  // Write output...', c: '#6B7280' },
  { n: 22, t: '}', c: '#FFF' },
];

const tileLangLines = [
  { n: 1, t: '# TileLang FlashMLA - ~50 lines', c: '#6B7280' },
  { n: 2, t: '@T.prim_func', c: '#C084FC' },
  { n: 3, t: 'def flash_mla(', c: '#60A5FA' },
  { n: 4, t: '  Q: T.Buffer((B, S, H, D), "float16"),', c: '#F9A8D4' },
  { n: 5, t: '  cKV: T.Buffer((B, S, H, DC), "float16"),', c: '#F9A8D4' },
  { n: 6, t: '  O: T.Buffer((B, S, H, D), "float16"),', c: '#F9A8D4' },
  { n: 7, t: '):', c: '#FFF' },
  { n: 8, t: '  with T.block("main"):', c: '#FFF' },
  { n: 9, t: '    acc = T.alloc_buffer((BD, DV), "float32")', c: '#34D399' },
  { n: 10, t: '    m_prev = T.alloc_buffer((BD,), "float32")', c: '#34D399' },
  { n: 11, t: '    l_prev = T.alloc_buffer((BD,), "float32")', c: '#34D399' },
  { n: 12, t: '    for ko in T.serial(NB):', c: '#FB923C' },
  { n: 13, t: '      K = decode_K(cKV[b, ko])', c: '#10B981' },
  { n: 14, t: '      V = decode_V(cKV[b, ko])', c: '#10B981' },
  { n: 15, t: '      S = T.call_extern("mma", Q, K)', c: '#10B981' },
  { n: 16, t: '      m, l, acc = online_softmax(', c: '#8B5CF6' },
  { n: 17, t: '        S, m_prev, l_prev, acc)', c: '#8B5CF6' },
  { n: 18, t: '    O[b, s] = acc / l', c: '#F59E0B' },
];

export default function CodeLineComparison() {
  const [side, setSide] = useState<'cuda' | 'tilelang'>('cuda');

  const lines = side === 'cuda' ? cudaLines : tileLangLines;

  return (
    <div className="p-6 bg-gray-900 rounded-xl text-white">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-bold">代码行数对比</h2>
        <div className="flex gap-2">
          <button onClick={() => setSide('cuda')}
            className={`px-3 py-1 rounded text-sm ${side === 'cuda' ? 'bg-red-600' : 'bg-gray-700'}`}>
            CUDA ({cudaLines.length} 行)
          </button>
          <button onClick={() => setSide('tilelang')}
            className={`px-3 py-1 rounded text-sm ${side === 'tilelang' ? 'bg-green-600' : 'bg-gray-700'}`}>
            TileLang ({tileLangLines.length} 行)
          </button>
        </div>
      </div>

      <div className="flex gap-4">
        {/* CUDA side */}
        <div className="flex-1 bg-black rounded-lg p-3 font-mono text-xs overflow-x-auto" style={{ opacity: side === 'cuda' ? 1 : 0.3 }}>
          <div className="text-red-400 text-[10px] mb-2 font-bold">{'// CUDA FlashMLA - ' + cudaLines.length + '+ 行 (核心部分)'}</div>
          {cudaLines.map(l => (
            <div key={l.n} className="flex gap-2 leading-5">
              <span className="text-gray-600 w-5 text-right select-none">{l.n}</span>
              <span style={{ color: l.c }}>{l.t}</span>
            </div>
          ))}
          <div className="text-gray-600 text-[10px] mt-2">{'// ... 省略约 480 行'}</div>
        </div>

        {/* TileLang side */}
        <div className="flex-1 bg-black rounded-lg p-3 font-mono text-xs overflow-x-auto" style={{ opacity: side === 'tilelang' ? 1 : 0.3 }}>
          <div className="text-green-400 text-[10px] mb-2 font-bold"># TileLang FlashMLA - {tileLangLines.length} 行 (完整)</div>
          {tileLangLines.map(l => (
            <div key={l.n} className="flex gap-2 leading-5">
              <span className="text-gray-600 w-5 text-right select-none">{l.n}</span>
              <span style={{ color: l.c }}>{l.t}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="mt-4 bg-gray-800 rounded-lg p-3 text-xs flex items-center gap-4">
        <div className="text-center">
          <div className="text-red-400 font-bold text-lg">500+</div>
          <div className="text-gray-400">CUDA 行数</div>
        </div>
        <div className="text-2xl text-gray-600">→</div>
        <div className="text-center">
          <div className="text-green-400 font-bold text-lg">50</div>
          <div className="text-gray-400">TileLang 行数</div>
        </div>
        <div className="text-center ml-4">
          <div className="text-purple-400 font-bold text-lg">10×</div>
          <div className="text-gray-400">代码量减少</div>
        </div>
        <div className="text-xs text-gray-400 ml-4">
          性能相同，TileLang 自动生成优化的底层代码
        </div>
      </div>
    </div>
  );
}
