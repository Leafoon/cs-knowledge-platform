'use client';

import React, { useState } from 'react';
import { motion } from 'framer-motion';

const vectorModes = [
  {
    id: 'scalar',
    name: '标量',
    width: 1,
    icon: '🔢',
    color: 'from-gray-500 to-gray-600',
    borderColor: 'border-gray-500/30',
    instruction: 'ADDSS xmm0, xmm1',
    arch: 'x86 Baseline',
    description: '逐元素处理，每次运算一个数据',
    ops: 1,
    throughput: 1,
  },
  {
    id: 'sse',
    name: 'SSE (128-bit)',
    width: 4,
    icon: '▶️',
    color: 'from-blue-500 to-indigo-600',
    borderColor: 'border-blue-500/30',
    instruction: 'ADDPS xmm0, xmm1',
    arch: 'SSE / NEON',
    description: '128位SIMD，同时处理4个float32',
    ops: 4,
    throughput: 4,
  },
  {
    id: 'avx256',
    name: 'AVX (256-bit)',
    width: 8,
    icon: '⏩',
    color: 'from-indigo-500 to-purple-600',
    borderColor: 'border-indigo-500/30',
    instruction: 'VADDPS ymm0, ymm1, ymm2',
    arch: 'AVX / AVX2',
    description: '256位SIMD，同时处理8个float32',
    ops: 8,
    throughput: 8,
  },
  {
    id: 'avx512',
    name: 'AVX-512 (512-bit)',
    width: 16,
    icon: '⏭️',
    color: 'from-purple-500 to-violet-600',
    borderColor: 'border-purple-500/30',
    instruction: 'VADDPS zmm0, zmm1, zmm2',
    arch: 'AVX-512',
    description: '512位SIMD，同时处理16个float32',
    ops: 16,
    throughput: 16,
  },
];

const tvmCodeExamples: Record<string, string> = {
  scalar: `// 标量操作
for (int i = 0; i < n; i++) {
  C[i] = A[i] + B[i];
}`,
  sse: `// TVM 向量化 (width=4)
for (int i = 0; i < n; i += 4) {
  x128 va = load_128(&A[i]);
  x128 vb = load_128(&B[i]);
  x128 vc = add_f32x4(va, vb);
  store_128(&C[i], vc);
}`,
  avx256: `// TVM 向量化 (width=8)
for (int i = 0; i < n; i += 8) {
  y256 va = load_256(&A[i]);
  y256 vb = load_256(&B[i]);
  y256 vc = add_f32x8(va, vb);
  store_256(&C[i], vc);
}`,
  avx512: `// TVM 向量化 (width=16)
for (int i = 0; i < n; i += 16) {
  z512 va = load_512(&A[i]);
  z512 vb = load_512(&B[i]);
  z512 vc = add_f32x16(va, vb);
  store_512(&C[i], vc);
}`,
};

export default function VectorizationExplorer() {
  const [selected, setSelected] = useState(0);
  const [arraySize] = useState(1024);
  const mode = vectorModes[selected];

  const loopIterations = Math.ceil(arraySize / mode.width);

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gray-900 rounded-2xl">
      <h2 className="text-2xl font-bold text-center mb-2 bg-gradient-to-r from-indigo-400 via-purple-400 to-blue-400 bg-clip-text text-transparent">
        向量化探索器
      </h2>
      <p className="text-gray-400 text-center text-sm mb-6">
        从标量到 AVX-512 / NEON 的向量化演进
      </p>

      <div className="flex gap-3 justify-center mb-8">
        {vectorModes.map((m, i) => (
          <button
            key={m.id}
            onClick={() => setSelected(i)}
            className={`flex flex-col items-center px-4 py-3 rounded-xl transition-all ${
              i === selected
                ? `bg-gradient-to-br ${m.color} text-white shadow-lg`
                : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
            }`}
          >
            <span className="text-lg mb-1">{m.icon}</span>
            <span className="text-xs font-medium">{m.name}</span>
          </button>
        ))}
      </div>

      <div className="bg-gray-800/40 rounded-xl border border-gray-700 p-6 mb-6">
        <div className="text-center text-sm text-gray-300 mb-4">
          向量宽度对比（处理 {arraySize} 个 float32 元素）
        </div>

        <div className="flex justify-center mb-4">
          <div className="flex gap-1">
            {Array.from({ length: Math.min(mode.width, 16) }).map((_, i) => (
              <motion.div
                key={`${mode.id}-${i}`}
                initial={{ scale: 0, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ delay: i * 0.05 }}
                className={`w-8 h-8 rounded bg-gradient-to-br ${mode.color} flex items-center justify-center text-white text-xs font-mono`}
              >
                {i}
              </motion.div>
            ))}
            {mode.width > 16 && (
              <div className="w-8 h-8 rounded bg-gray-700 flex items-center justify-center text-gray-400 text-xs">
                +{mode.width - 16}
              </div>
            )}
          </div>
        </div>

        <div className="text-center text-xs text-gray-500">
          一条指令同时处理 {mode.width} 个元素 · {mode.instruction}
        </div>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-6">
        {[
          { label: '向量宽度', value: `${mode.width * 32}-bit`, color: 'text-blue-400' },
          { label: 'float32并行数', value: mode.width, color: 'text-purple-400' },
          { label: '循环迭代次数', value: loopIterations, color: 'text-indigo-400' },
          { label: '指令集', value: mode.arch, color: 'text-emerald-400' },
        ].map((item) => (
          <motion.div
            key={item.label}
            layout
            className="bg-gray-800/60 rounded-lg p-3 border border-gray-700 text-center"
          >
            <div className={`text-lg font-bold font-mono ${item.color}`}>{item.value}</div>
            <div className="text-xs text-gray-500">{item.label}</div>
          </motion.div>
        ))}
      </div>

      <div className="bg-gray-950 rounded-xl border border-gray-700 overflow-hidden">
        <div className="flex items-center gap-2 px-4 py-2 bg-gray-800/80 border-b border-gray-700">
          <div className="flex gap-1.5">
            <div className="w-3 h-3 rounded-full bg-red-500/70" />
            <div className="w-3 h-3 rounded-full bg-yellow-500/70" />
            <div className="w-3 h-3 rounded-full bg-green-500/70" />
          </div>
          <span className="text-xs text-gray-500 ml-2">vectorize.{mode.id}.c</span>
        </div>
        <pre className="p-4 text-sm font-mono text-gray-300 overflow-x-auto">
          {tvmCodeExamples[mode.id]}
        </pre>
      </div>

      <div className="mt-4 bg-gray-800/40 rounded-xl border border-gray-700 p-4">
        <div className="text-xs text-gray-500 mb-2">TVM 调度代码</div>
        <code className="text-sm font-mono text-indigo-300">
          sch[C].vectorize(xo)  # vectorize width = {mode.width}
        </code>
      </div>

      <div className="mt-6 bg-gray-800/40 rounded-xl border border-gray-700 p-4">
        <div className="text-sm text-gray-300 mb-3">理论加速比（相对标量）</div>
        <div className="space-y-2">
          {vectorModes.map((m, i) => (
            <div key={m.id} className="flex items-center gap-3">
              <span className="text-xs text-gray-400 w-28">{m.name}</span>
              <div className="flex-1 bg-gray-900 rounded-full h-4 overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${(m.width / vectorModes[vectorModes.length - 1].width) * 100}%` }}
                  transition={{ delay: i * 0.1, duration: 0.5 }}
                  className={`h-full rounded-full ${
                    i === selected
                      ? `bg-gradient-to-r ${m.color}`
                      : 'bg-gray-600'
                  }`}
                />
              </div>
              <span className="text-xs font-mono text-gray-500 w-10">{m.width}x</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
