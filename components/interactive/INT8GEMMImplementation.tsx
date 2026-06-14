'use client';

import { useState } from 'react';

const codeLines = [
  { num: 1, text: '// INT8 GEMM with Per-Channel Dequantization', color: '#6B7280', type: 'comment' },
  { num: 2, text: 'template<int BM, int BN, int BK>', color: '#C084FC', type: 'template' },
  { num: 3, text: '__global__ void int8_gemm_kernel(', color: '#60A5FA', type: 'func' },
  { num: 4, text: '    int8_t* A, int8_t* B,', color: '#F9A8D4', type: 'param' },
  { num: 5, text: '    float* scale_a, float* scale_b,', color: '#F9A8D4', type: 'param' },
  { num: 6, text: '    float* C, int M, int N, int K)', color: '#F9A8D4', type: 'param' },
  { num: 7, text: '{', color: '#FFF', type: 'brace' },
  { num: 8, text: '  __shared__ int8_t sA[BM][BK], sB[BK][BN];', color: '#FBBF24', type: 'shared' },
  { num: 9, text: '  int8_t frag_a[16], frag_b[16];', color: '#34D399', type: 'reg' },
  { num: 10, text: '  int32_t acc[8] = {0};', color: '#F472B6', type: 'acc' },
  { num: 11, text: '', color: '#FFF', type: 'empty' },
  { num: 12, text: '  for (int k = 0; k < K; k += BK) {', color: '#FB923C', type: 'loop' },
  { num: 13, text: '    // Load tiles to shared memory', color: '#6B7280', type: 'comment' },
  { num: 14, text: '    cp.async(sA, A + ..., 16);', color: '#60A5FA', type: 'async' },
  { num: 15, text: '    cp.async(sB, B + ..., 16);', color: '#60A5FA', type: 'async' },
  { num: 16, text: '    __syncthreads();', color: '#FB923C', type: 'sync' },
  { num: 17, text: '', color: '#FFF', type: 'empty' },
  { num: 18, text: '    // INT8 MMA on Tensor Cores', color: '#6B7280', type: 'comment' },
  { num: 19, text: '    mma.sync.int8(sA, sB, acc);', color: '#10B981', type: 'mma' },
  { num: 20, text: '    __syncthreads();', color: '#FB923C', type: 'sync' },
  { num: 21, text: '  }', color: '#FFF', type: 'brace' },
  { num: 22, text: '', color: '#FFF', type: 'empty' },
  { num: 23, text: '  // Per-channel dequantization', color: '#6B7280', type: 'comment' },
  { num: 24, text: '  float scale = scale_a[row] * scale_b[col];', color: '#F59E0B', type: 'dequant' },
  { num: 25, text: '  C[row*N+col] = (float)acc[i] * scale;', color: '#F59E0B', type: 'dequant' },
  { num: 26, text: '}', color: '#FFF', type: 'brace' },
];

const typeColors: Record<string, string> = {
  comment: '#6B7280', template: '#C084FC', func: '#60A5FA', param: '#F9A8D4',
  brace: '#FFF', shared: '#FBBF24', reg: '#34D399', acc: '#F472B6', loop: '#FB923C',
  sync: '#FB923C', async: '#60A5FA', mma: '#10B981', dequant: '#F59E0B', empty: '#FFF',
};

export default function INT8GEMMImplementation() {
  const [hoveredLine, setHoveredLine] = useState<number | null>(null);

  const annotations: Record<number, string> = {
    8: '共享内存用于存储 INT8 tile，大小 = BM×BK + BK×BN 字节',
    10: '累加器必须使用 INT32 防止 INT8 溢出',
    14: 'cp.async 拷贝不占用寄存器带宽',
    19: 'Tensor Core 的 INT8 MMA：4 个 8×8 外积累加',
    24: 'Per-channel 反量化：每个输出通道独立 scale',
  };

  return (
    <div className="p-6 bg-gray-900 rounded-xl text-white">
      <h2 className="text-xl font-bold mb-2">INT8 GEMM 实现（Per-Channel 反量化）</h2>
      <p className="text-sm text-gray-400 mb-4">Tensor Core INT8 矩阵乘 + 逐通道反量化到 FP32</p>

      <div className="flex gap-4">
        <div className="bg-black rounded-lg p-4 font-mono text-xs leading-5 overflow-x-auto flex-1">
          {codeLines.map(l => (
            <div key={l.num}
              className={`flex gap-3 px-2 rounded cursor-pointer transition-colors ${
                hoveredLine === l.num ? 'bg-gray-800' : ''
              }`}
              onMouseEnter={() => setHoveredLine(l.num)}
              onMouseLeave={() => setHoveredLine(null)}>
              <span className="text-gray-600 w-6 text-right select-none">{l.num}</span>
              <span style={{ color: typeColors[l.type] || '#FFF' }}>{l.text}</span>
            </div>
          ))}
        </div>

        <div className="w-64 flex-shrink-0">
          {hoveredLine && annotations[hoveredLine] ? (
            <div className="bg-blue-900/30 border border-blue-700 rounded-lg p-3 text-sm">
              <div className="text-xs text-blue-400 mb-1">行 {hoveredLine} 说明</div>
              <div className="text-gray-300">{annotations[hoveredLine]}</div>
            </div>
          ) : (
            <div className="bg-gray-800 rounded-lg p-3 text-xs text-gray-500">
              鼠标悬停代码行查看说明
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
