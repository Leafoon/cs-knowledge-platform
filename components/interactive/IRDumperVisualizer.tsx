'use client';

import { useState } from 'react';

const irLines = [
  { line: 1, text: '@tvm.script.ir_module', color: '#C084FC' },
  { line: 2, text: 'class Module:', color: '#C084FC' },
  { line: 3, text: '  @T.prim_func', color: '#F59E0B' },
  { line: 4, text: '  def matmul(', color: '#60A5FA' },
  { line: 5, text: '    A: T.handle,', color: '#F9A8D4' },
  { line: 6, text: '    B: T.handle,', color: '#F9A8D4' },
  { line: 7, text: '    C: T.handle', color: '#F9A8D4' },
  { line: 8, text: ':', color: '#FFF' },
  { line: 9, text: '  ) -> None:', color: '#60A5FA' },
  { line: 10, text: '    A = T.match_buffer(A, (1024, 1024))', color: '#34D399' },
  { line: 11, text: '    B = T.match_buffer(B, (1024, 1024))', color: '#34D399' },
  { line: 12, text: '    C = T.match_buffer(C, (1024, 1024))', color: '#34D399' },
  { line: 13, text: '    for bx in T.thread_binding(32, "blockIdx.x"):', color: '#FB923C' },
  { line: 14, text: '      for by in T.thread_binding(32, "blockIdx.y"):', color: '#FB923C' },
  { line: 15, text: '        for ko in T.serial(32):', color: '#FB923C' },
  { line: 16, text: '          with T.block("compute"):', color: '#F59E0B' },
  { line: 17, text: '            A_reg = T.alloc_buffer((32, 32), "float16")', color: '#EF4444' },
  { line: 18, text: '            B_reg = T.alloc_buffer((32, 32), "float16")', color: '#EF4444' },
  { line: 19, text: '            acc = T.alloc_buffer((32, 32), "float32")', color: '#EF4444' },
  { line: 20, text: '            for ki in T.serial(32):', color: '#FB923C' },
  { line: 21, text: '              for ii, jj in T.serial(32, 32):', color: '#FB923C' },
  { line: 22, text: '                acc[ii,jj] += A_reg[ii,ki]*B_reg[ki,jj]', color: '#60A5FA' },
];

const highlights: Record<number, string> = {
  3: 'T.prim_func 装饰器标记 TensorIR 原语函数',
  10: 'T.handle 为动态形状提供灵活性',
  13: 'thread_binding 绑定到 CUDA blockIdx',
  16: 'T.block 定义计算单元的约束',
  17: 'T.alloc_buffer 在指定内存层级分配',
};

export default function IRDumperVisualizer() {
  const [hoveredLine, setHoveredLine] = useState<number | null>(null);
  const [filter, setFilter] = useState<string>('all');

  const filterTypes: Record<string, (l: typeof irLines[0]) => boolean> = {
    all: () => true,
    buffer: l => l.text.includes('match_buffer') || l.text.includes('alloc_buffer'),
    loop: l => l.text.includes('for '),
    block: l => l.text.includes('T.block'),
  };

  return (
    <div className="p-6 bg-gray-900 rounded-xl text-white">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-bold">IR Dump 可视化</h2>
        <div className="flex gap-1">
          {['all', 'buffer', 'loop', 'block'].map(f => (
            <button key={f} onClick={() => setFilter(f)}
              className={`px-2 py-1 rounded text-[10px] ${filter === f ? 'bg-blue-600' : 'bg-gray-700'}`}>
              {f}
            </button>
          ))}
        </div>
      </div>

      <div className="flex gap-4">
        <div className="flex-1 bg-black rounded-lg p-3 overflow-x-auto">
          {irLines.map(l => {
            const visible = filterTypes[filter]?.(l) ?? true;
            const isHovered = hoveredLine === l.line;
            return (
              <div key={l.line}
                className={`flex gap-3 font-mono text-xs leading-6 px-2 rounded transition-all ${
                  isHovered ? 'bg-gray-800' : ''
                } ${!visible ? 'opacity-20' : ''}`}
                onMouseEnter={() => setHoveredLine(l.line)}
                onMouseLeave={() => setHoveredLine(null)}>
                <span className="text-gray-600 w-6 text-right select-none">{l.line}</span>
                <span style={{ color: l.color }}>{l.text}</span>
              </div>
            );
          })}
        </div>

        <div className="w-56">
          {hoveredLine && highlights[hoveredLine] ? (
            <div className="bg-blue-900/30 border border-blue-700 rounded-lg p-3 text-xs">
              <div className="text-blue-400 font-bold mb-1">行 {hoveredLine}</div>
              <div className="text-gray-300">{highlights[hoveredLine]}</div>
            </div>
          ) : (
            <div className="space-y-2 text-xs">
              <div className="bg-gray-800 rounded p-2">
                <div className="text-gray-400">IR 统计</div>
                <div className="text-white mt-1">
                  函数: 1 · Block: 1 · 循环: 6<br/>
                  Buffer: 5 · 运算: 1
                </div>
              </div>
              <div className="bg-gray-800 rounded p-2">
                <div className="text-gray-400">语法高亮</div>
                <div className="space-y-1 mt-1">
                  {[
                    ['装饰器', '#F59E0B'], ['函数名', '#60A5FA'],
                    ['参数', '#F9A8D4'], ['Buffer', '#34D399'],
                    ['循环', '#FB923C'], ['分配', '#EF4444'],
                  ].map(([label, color]) => (
                    <div key={label} className="flex items-center gap-2">
                      <div className="w-2 h-2 rounded-full" style={{ backgroundColor: color as string }} />
                      <span className="text-gray-400">{label}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
