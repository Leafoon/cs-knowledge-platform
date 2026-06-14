'use client';

import { useState } from 'react';

const pipeline = [
  { label: 'Relax Graph', color: '#3B82F6', desc: '计算图输入' },
  { label: '图优化', color: '#8B5CF6', desc: '常量折叠、死代码消除' },
  { label: '算子融合', color: '#F59E0B', desc: '自动融合连续算子' },
  { label: '代码生成', color: '#10B981', desc: 'TensorIR → LLVM/CUDA' },
  { label: '运行时执行', color: '#EF4444', desc: 'Relax VM 调度' },
];

const fusionGroups = [
  { ops: ['add', 'relu', 'add'], fused: 'fused_add_relu_add', saving: '60% 内存' },
  { ops: ['matmul', 'add'], fused: 'fused_matmul_bias', saving: '40% 延迟' },
];

export default function TVMRelaxPipelineFlow() {
  const [stage, setStage] = useState(0);

  return (
    <div className="p-6 bg-gray-900 rounded-xl text-white">
      <h2 className="text-xl font-bold mb-2">TVM Relax 端到端流程</h2>
      <p className="text-sm text-gray-400 mb-4">Relax 图优化 → 算子融合 → 代码生成全流程</p>

      {/* Pipeline */}
      <div className="flex items-center gap-1 mb-6 overflow-x-auto pb-2">
        {pipeline.map((p, i) => (
          <div key={i} className="flex items-center">
            <button
              className={`flex flex-col items-center min-w-[90px] py-2 px-3 rounded-lg transition-all ${
                i <= stage ? 'opacity-100' : 'opacity-40'
              }`}
              style={{
                backgroundColor: i === stage ? `${p.color}20` : 'transparent',
                border: `2px solid ${i === stage ? p.color : 'transparent'}`,
              }}
              onClick={() => setStage(i)}>
              <div className="w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold mb-1"
                style={{ backgroundColor: `${p.color}30`, color: p.color }}>
                {i + 1}
              </div>
              <span className="text-[10px] font-bold" style={{ color: p.color }}>{p.label}</span>
            </button>
            {i < pipeline.length - 1 && (
              <div className={`w-6 h-0.5 ${i < stage ? 'bg-blue-500' : 'bg-gray-700'}`} />
            )}
          </div>
        ))}
      </div>

      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="bg-gray-800 rounded-lg p-4">
          <div className="text-sm font-bold mb-2" style={{ color: pipeline[stage].color }}>
            Stage {stage + 1}: {pipeline[stage].label}
          </div>
          <div className="text-xs text-gray-400 mb-3">{pipeline[stage].desc}</div>

          {stage === 2 && fusionGroups.map((fg, i) => (
            <div key={i} className="mb-2 p-2 bg-black/30 rounded text-xs">
              <div className="flex items-center gap-2 mb-1">
                {fg.ops.map((op, j) => (
                  <span key={j}>
                    <span className="text-gray-300">{op}</span>
                    {j < fg.ops.length - 1 && <span className="text-gray-500 mx-1">→</span>}
                  </span>
                ))}
                <span className="text-gray-500 mx-1">→</span>
                <span className="text-green-400 font-bold">{fg.fused}</span>
              </div>
              <span className="text-yellow-400">节省: {fg.saving}</span>
            </div>
          ))}

          {stage === 0 && (
            <div className="bg-black/30 rounded p-2 font-mono text-[10px] text-gray-400">
              <div className="text-blue-400">@relax.function</div>
              <div>def main(x, w, b):</div>
              <div>  y = relax.nn.linear(x, w, b)</div>
              <div>  z = relax.nn.relu(y)</div>
              <div>  return z</div>
            </div>
          )}

          {stage === 3 && (
            <div className="bg-black/30 rounded p-2 font-mono text-[10px] text-gray-400">
              <div className="text-green-400">{'// Generated CUDA kernel'}</div>
              <div>__global__ void fused_matmul_bias_relu(</div>
              <div>  half* x, half* w, float* b, half* out)</div>
              <div>{'{'} {'// Tensor Core MMA + fused ops'} {'}'}</div>
            </div>
          )}
        </div>

        <div className="space-y-2 text-xs">
          <div className="bg-gray-800 rounded p-3">
            <div className="font-bold text-purple-400 mb-1">Relax 优化 Pass</div>
            <ul className="space-y-1 text-gray-400">
              <li>• <span className="text-blue-400">常量折叠</span>：编译期计算确定表达式</li>
              <li>• <span className="text-blue-400">死代码消除</span>：移除未使用节点</li>
              <li>• <span className="text-blue-400">内存规划</span>：Buffer 复用减少分配</li>
              <li>• <span className="text-blue-400">算子融合</span>：减少内核启动和内存往返</li>
            </ul>
          </div>
          <div className="bg-gray-800 rounded p-3">
            <div className="font-bold text-green-400 mb-1">后端支持</div>
            <ul className="space-y-1 text-gray-400">
              <li>• <span className="text-green-400">LLVM</span>：CPU / 通用 GPU</li>
              <li>• <span className="text-green-400">CUDA</span>：NVIDIA GPU</li>
              <li>• <span className="text-green-400">HIP</span>：AMD GPU</li>
              <li>• <span className="text-green-400">Metal</span>：Apple GPU</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
