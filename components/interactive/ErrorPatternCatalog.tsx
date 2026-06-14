'use client';

import { useState } from 'react';

const errorPatterns = [
  {
    id: 1,
    category: '内存',
    symptom: 'CUDA error: an illegal memory access was encountered',
    cause: '数组越界或未对齐的内存访问',
    solution: '检查边界条件，确保内存对齐',
    code: `# 错误示例
for i in range(1024):
    C[i] = A[i] * B[i]  # 当 i >= min(A.size, B.size) 时越界

# 正确示例
for i in range(min(A.size, B.size)):
    C[i] = A[i] * B[i]`,
    severity: 'high',
  },
  {
    id: 2,
    category: '同步',
    symptom: 'Race condition detected in shared memory',
    cause: '缺少线程同步导致数据竞争',
    solution: '在共享内存操作后添加同步',
    code: `# 错误示例
A_local[threadIdx.x] = A[global_idx]
B_local[threadIdx.x] = B[global_idx]  # 可能其他线程还在写入

# 正确示例
A_local[threadIdx.x] = A[global_idx]
tile_lang.sync()  # 确保所有线程完成写入
B_local[threadIdx.x] = B[global_idx]`,
    severity: 'high',
  },
  {
    id: 3,
    category: '类型',
    symptom: 'RuntimeError: expected scalar type Half but found Float',
    cause: '数据类型不匹配',
    solution: '统一数据类型或添加显式转换',
    code: `# 错误示例
A = tile_lang.alloc([128, 128], "float32")
B = tile_lang.alloc([128, 128], "float16")
C = tile_lang.gemm(A, B)  # 类型不匹配

# 正确示例
A = tile_lang.alloc([128, 128], "float16")
B = tile_lang.alloc([128, 128], "float16")
C = tile_lang.gemm(A, B)`,
    severity: 'medium',
  },
  {
    id: 4,
    category: '性能',
    symptom: 'Low GPU utilization (<30%)',
    cause: 'Tile 大小不合适或并行度不足',
    solution: '调整 Tile 大小，增加并行度',
    code: `# 错误示例 - 过小的 Tile
BLOCK_M = 16
BLOCK_N = 16
BLOCK_K = 16

# 正确示例 - 合适的 Tile
BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 32`,
    severity: 'medium',
  },
  {
    id: 5,
    category: '形状',
    symptom: 'ValueError: shape mismatch for broadcast',
    cause: '张量形状不兼容',
    solution: '检查张量形状，确保可广播',
    code: `# 错误示例
A = tile_lang.alloc([1024, 512])
B = tile_lang.alloc([512, 256])
C = A + B  # 形状不匹配

# 正确示例
A = tile_lang.alloc([1024, 512])
B = tile_lang.alloc([1024, 512])
C = A + B  # 形状匹配`,
    severity: 'medium',
  },
];

const categories = ['全部', '内存', '同步', '类型', '性能', '形状'];

export default function ErrorPatternCatalog() {
  const [selectedCategory, setSelectedCategory] = useState('全部');
  const [expandedError, setExpandedError] = useState<number | null>(null);

  const filteredErrors = selectedCategory === '全部'
    ? errorPatterns
    : errorPatterns.filter(e => e.category === selectedCategory);

  return (
    <div className="p-6 bg-gray-900 rounded-xl text-white">
      <h2 className="text-xl font-bold mb-2">错误模式目录</h2>
      <p className="text-gray-400 text-sm mb-4">常见错误的症状、原因和解决方案</p>

      <div className="flex gap-2 mb-6">
        {categories.map((cat) => (
          <button
            key={cat}
            onClick={() => setSelectedCategory(cat)}
            className={`px-3 py-1.5 rounded text-xs font-medium transition-all ${
              selectedCategory === cat ? 'bg-blue-600' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
            }`}
          >
            {cat}
          </button>
        ))}
      </div>

      <div className="space-y-4">
        {filteredErrors.map((error) => (
          <div
            key={error.id}
            className={`bg-gray-800 rounded-lg overflow-hidden transition-all ${
              expandedError === error.id ? 'ring-1 ring-blue-500' : ''
            }`}
          >
            <div
              className="p-4 cursor-pointer hover:bg-gray-750"
              onClick={() => setExpandedError(expandedError === error.id ? null : error.id)}
            >
              <div className="flex items-center gap-3 mb-2">
                <span className={`w-2 h-2 rounded-full ${
                  error.severity === 'high' ? 'bg-red-500' : 'bg-yellow-500'
                }`} />
                <span className="text-xs text-gray-500 px-2 py-0.5 bg-gray-700 rounded">
                  {error.category}
                </span>
                <span className="text-sm font-mono text-red-400">{error.symptom}</span>
              </div>
              <div className="text-xs text-gray-400">
                <span className="text-gray-500">原因:</span> {error.cause}
              </div>
            </div>

            {expandedError === error.id && (
              <div className="border-t border-gray-700 p-4">
                <div className="mb-4">
                  <h4 className="text-xs font-bold text-green-400 mb-2">解决方案</h4>
                  <p className="text-sm text-gray-300">{error.solution}</p>
                </div>
                <div>
                  <h4 className="text-xs font-bold text-blue-400 mb-2">代码示例</h4>
                  <pre className="bg-gray-900 rounded p-3 text-xs text-gray-300 overflow-x-auto">
                    {error.code}
                  </pre>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>

      <div className="mt-6 bg-gray-800 rounded-lg p-4">
        <h3 className="text-sm font-bold text-gray-300 mb-3">统计信息</h3>
        <div className="grid grid-cols-5 gap-4 text-center">
          {categories.slice(1).map((cat) => {
            const count = errorPatterns.filter(e => e.category === cat).length;
            return (
              <div key={cat}>
                <div className="text-lg font-bold text-white">{count}</div>
                <div className="text-xs text-gray-400">{cat}</div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
