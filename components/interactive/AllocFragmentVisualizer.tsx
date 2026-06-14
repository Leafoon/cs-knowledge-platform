'use client';

import React, { useState } from 'react';

interface FragmentElement {
  row: number;
  col: number;
  thread: number;
}

export function AllocFragmentVisualizer() {
  const [selectedFragment, setSelectedFragment] = useState<'A' | 'B' | 'C'>('A');

  const fragments = {
    A: { name: 'A Fragment', color: '#3B82F6', rows: 4, cols: 8, threads: 32 },
    B: { name: 'B Fragment', color: '#10B981', rows: 8, cols: 4, threads: 32 },
    C: { name: 'C Fragment', color: '#F59E0B', rows: 4, cols: 4, threads: 32 },
  };

  const currentFragment = fragments[selectedFragment];

  const generateElements = (): FragmentElement[] => {
    const elements: FragmentElement[] = [];
    const { rows, cols, threads } = currentFragment;
    
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        elements.push({
          row: r,
          col: c,
          thread: (r * cols + c) % threads,
        });
      }
    }
    return elements;
  };

  const elements = generateElements();

  return (
    <div className="p-6 bg-gray-900 rounded-xl">
      <h2 className="text-2xl font-bold text-white mb-6">T.alloc_fragment 可视化</h2>
      
      {/* Fragment selector */}
      <div className="flex gap-4 mb-6">
        {(Object.keys(fragments) as Array<'A' | 'B' | 'C'>).map((key) => (
          <button
            key={key}
            onClick={() => setSelectedFragment(key)}
            className={`px-4 py-2 rounded-lg font-bold transition-all ${
              selectedFragment === key
                ? 'text-white'
                : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
            }`}
            style={{
              backgroundColor: selectedFragment === key ? fragments[key].color : undefined,
            }}
          >
            {fragments[key].name}
          </button>
        ))}
      </div>
      
      <div className="grid grid-cols-2 gap-6">
        {/* Fragment visualization */}
        <div className="bg-gray-800 rounded-lg p-4">
          <h3 className="text-white font-bold mb-4">{currentFragment.name} Layout</h3>
          
          <div className="inline-grid gap-1" style={{ gridTemplateColumns: `repeat(${currentFragment.cols}, 1fr)` }}>
            {elements.map((elem, i) => (
              <div
                key={i}
                className="w-12 h-12 flex flex-col items-center justify-center rounded text-xs font-mono"
                style={{
                  backgroundColor: currentFragment.color,
                  opacity: 0.3 + (elem.thread / currentFragment.threads) * 0.7,
                }}
              >
                <span className="text-white font-bold">T{elem.thread}</span>
                <span className="text-gray-200 text-[10px]">({elem.row},{elem.col})</span>
              </div>
            ))}
          </div>
          
          <div className="mt-4 text-gray-400 text-sm">
            每个元素由一个线程处理
          </div>
        </div>
        
        {/* Code example */}
        <div className="bg-gray-800 rounded-lg p-4">
          <h3 className="text-white font-bold mb-4">代码示例</h3>
          
          <pre className="text-sm font-mono text-green-400 overflow-x-auto">
{`# 分配 Fragment (寄存器)
${selectedFragment}_local = T.alloc_fragment(
    (${currentFragment.rows}, ${currentFragment.cols}),  # 形状
    "float16"  # 数据类型
)

# Fragment 特点:
# 1. 存储在寄存器中
# 2. 线程私有
# 3. 用于 Tensor Core 计算

# 使用示例:
T.tvm_fill_fragment(
    ${selectedFragment}_local, 
    0.0  # 初始值
)

# 加载数据
T.tvm_load_matrix(
    ${selectedFragment.toUpperCase()},  # Global Memory
    [row_idx, col_idx],
    ${selectedFragment}_local  # Fragment
)

# Tensor Core 计算
T.tvm_gemm(
    A_local,  # Fragment A
    B_local,  # Fragment B
    C_local   # Fragment C (累加)
)`}
          </pre>
        </div>
      </div>
      
      {/* Fragment comparison */}
      <div className="mt-6 grid grid-cols-3 gap-4">
        {Object.entries(fragments).map(([key, frag]) => (
          <div
            key={key}
            className={`p-4 rounded-lg border-2 transition-all ${
              selectedFragment === key ? 'scale-105' : ''
            }`}
            style={{ borderColor: frag.color }}
          >
            <h4 className="text-white font-bold mb-2">{frag.name}</h4>
            <div className="text-gray-300 text-sm space-y-1">
              <p>形状: {frag.rows}×{frag.cols}</p>
              <p>元素数: {frag.rows * frag.cols}</p>
              <p>线程数: {frag.threads}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
