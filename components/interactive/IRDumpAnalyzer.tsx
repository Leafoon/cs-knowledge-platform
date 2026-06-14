'use client';

import { useState } from 'react';

const irDump = `@primfn(A: Pointer[float16], B: Pointer[float16], C: Pointer[float16])
  // 问题 1: 未使用的变量
  buffer_A = allocate([1024, 1024], float16)
  buffer_B = allocate([1024, 1024], float16)

  // 问题 2: 内存对齐问题
  for (bx, 0, 8):  # blockIdx.x
    for (by, 0, 8):  # blockIdx.y
      A_local = allocate([128, 32], float16, scope="shared")
      B_local = allocate([32, 128], float16, scope="shared")

      // 问题 3: 缺少同步
      for (k, 0, 32):
        copy(A[bx*128 + threadIdx.y, k*32 + threadIdx.x], A_local[threadIdx.y, threadIdx.x])
        copy(B[k*32 + threadIdx.y, by*128 + threadIdx.x], B_local[threadIdx.y, threadIdx.x])
        // 缺少 __syncthreads()

      // 问题 4: 循环展开不当
      for (i, 0, 128):
        for (j, 0, 128):
          C[bx*128 + i, by*128 + j] += A_local[i, j] * B_local[j, i]  # 转置错误

      // 问题 5: 数据类型不匹配
      C_local = allocate([128, 128], float32)  # 应该是 float16
      for (i, 0, 128):
        for (j, 0, 128):
          C_local[i, j] = cast(A_local[i, j], float32) * cast(B_local[i, j], float32)

      copy(C_local, C[bx*128:(bx+1)*128, by*128:(by+1)*128])`;

const issues = [
  { line: 3, type: 'warning', message: '未使用的变量 buffer_A, buffer_B', suggestion: '删除未使用的变量分配' },
  { line: 11, type: 'error', message: '缺少同步原语', suggestion: '在共享内存写入后添加 tile_lang.sync()' },
  { line: 16, type: 'error', message: '矩阵乘法转置错误', suggestion: '检查 B_local 的索引顺序' },
  { line: 20, type: 'warning', message: '数据类型不匹配', suggestion: '统一使用 float16 或添加显式转换' },
  { line: 8, type: 'info', message: '可以考虑循环展开优化', suggestion: '添加 tile_lang.unroll(4)' },
];

export default function IRDumpAnalyzer() {
  const [selectedIssue, setSelectedIssue] = useState<number | null>(null);
  const [hoveredLine, setHoveredLine] = useState<number | null>(null);

  const getLineColor = (lineNum: number) => {
    const issue = issues.find(i => i.line === lineNum);
    if (!issue) return '';
    if (selectedIssue !== null && issues[selectedIssue].line === lineNum) {
      return issue.type === 'error' ? 'bg-red-900/40 border-l-2 border-red-500' :
             issue.type === 'warning' ? 'bg-yellow-900/40 border-l-2 border-yellow-500' :
             'bg-blue-900/40 border-l-2 border-blue-500';
    }
    if (hoveredLine === lineNum) {
      return 'bg-gray-700/50';
    }
    return '';
  };

  return (
    <div className="p-6 bg-gray-900 rounded-xl text-white">
      <h2 className="text-xl font-bold mb-2">IR Dump 分析器</h2>
      <p className="text-gray-400 text-sm mb-4">查看 IR 代码并高亮常见问题</p>

      <div className="grid grid-cols-3 gap-2 mb-4">
        {['error', 'warning', 'info'].map((type) => {
          const count = issues.filter(i => i.type === type).length;
          return (
            <div key={type} className={`flex items-center justify-between p-2 rounded ${
              type === 'error' ? 'bg-red-900/30' : type === 'warning' ? 'bg-yellow-900/30' : 'bg-blue-900/30'
            }`}>
              <span className="text-xs text-gray-400 capitalize">{type}</span>
              <span className={`text-sm font-bold ${
                type === 'error' ? 'text-red-400' : type === 'warning' ? 'text-yellow-400' : 'text-blue-400'
              }`}>{count}</span>
            </div>
          );
        })}
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="bg-gray-800 rounded-lg overflow-hidden">
          <div className="bg-gray-700 px-4 py-2 text-xs font-bold text-gray-300 border-b border-gray-600">
            IR Dump 代码
          </div>
          <pre className="p-4 text-[11px] text-gray-300 overflow-x-auto leading-relaxed max-h-96 overflow-y-auto">
            {irDump.split('\n').map((line, i) => {
              const lineNum = i + 1;
              const issue = issues.find(iss => iss.line === lineNum);

              return (
                <div
                  key={i}
                  className={`px-2 -mx-2 cursor-pointer transition-all ${getLineColor(lineNum)}`}
                  onMouseEnter={() => setHoveredLine(lineNum)}
                  onMouseLeave={() => setHoveredLine(null)}
                  onClick={() => {
                    if (issue) {
                      setSelectedIssue(issues.indexOf(issue));
                    }
                  }}
                >
                  <span className="text-gray-600 select-none inline-block w-8">{lineNum}</span>
                  {line}
                  {issue && (
                    <span className={`ml-2 text-[9px] px-1 rounded ${
                      issue.type === 'error' ? 'bg-red-800 text-red-300' :
                      issue.type === 'warning' ? 'bg-yellow-800 text-yellow-300' :
                      'bg-blue-800 text-blue-300'
                    }`}>
                      {issue.type === 'error' ? '✗' : issue.type === 'warning' ? '⚠' : 'ℹ'}
                    </span>
                  )}
                </div>
              );
            })}
          </pre>
        </div>

        <div className="space-y-3">
          <div className="bg-gray-800 rounded-lg p-4">
            <h3 className="text-sm font-bold text-gray-300 mb-3">检测到的问题</h3>
            <div className="space-y-2">
              {issues.map((issue, idx) => (
                <div
                  key={idx}
                  className={`p-3 rounded-lg cursor-pointer transition-all ${
                    selectedIssue === idx ? 'bg-gray-700 ring-1 ring-blue-500' : 'bg-gray-700/50 hover:bg-gray-700'
                  }`}
                  onClick={() => setSelectedIssue(selectedIssue === idx ? null : idx)}
                >
                  <div className="flex items-center gap-2 mb-1">
                    <span className={`w-2 h-2 rounded-full ${
                      issue.type === 'error' ? 'bg-red-500' :
                      issue.type === 'warning' ? 'bg-yellow-500' : 'bg-blue-500'
                    }`} />
                    <span className="text-xs text-gray-500">行 {issue.line}</span>
                    <span className={`text-[10px] px-1 rounded ${
                      issue.type === 'error' ? 'bg-red-900/50 text-red-400' :
                      issue.type === 'warning' ? 'bg-yellow-900/50 text-yellow-400' :
                      'bg-blue-900/50 text-blue-400'
                    }`}>
                      {issue.type}
                    </span>
                  </div>
                  <div className="text-sm text-gray-300">{issue.message}</div>
                  {selectedIssue === idx && (
                    <div className="mt-2 text-xs text-gray-400 bg-gray-800 rounded p-2">
                      <span className="text-green-400">建议:</span> {issue.suggestion}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
