'use client';

import { useState } from 'react';

const steps = [
  {
    phase: '定义',
    steps: [
      { label: '实现函数', desc: '编写 C++/CUDA 计算内核' },
      { label: '创建包装', desc: '生成 PackedFunc 接口' },
    ],
  },
  {
    phase: '注册',
    steps: [
      { label: '注册 Dispatch', desc: 'TVM_REGISTER_GLOBAL("my.func")' },
      { label: '设备分发', desc: '注册 CPU/CUDA/ROCm 版本' },
    ],
  },
  {
    phase: '调用',
    steps: [
      { label: 'Python 绑定', desc: 'tvm.get_global_func("my.func")' },
      { label: 'Relax 融合', desc: '在编译图中自动融合' },
    ],
  },
];

const codeSnippets = [
  `// C++ 算子实现
TVM_REGISTER_GLOBAL("my.matmul")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  DLTensor* A = args[0];
  DLTensor* B = args[1];
  DLTensor* C = args[2];
  // 调用 CUDA kernel
  launch_matmul(A, B, C);
});`,
  `# Python 调用
import tvm

# 获取注册函数
my_matmul = tvm.get_global_func("my.matmul")

# 在 Relax 图中使用
@relax.function
def main(x, w):
    return tvm.call_packed("my.matmul", x, w)`,
  `# 编译和执行
mod = tvm.compile(IRModule, target="cuda")
vm = relax.VirtualMachine(mod, device)
result = vm["main"](x, w)`,
];

export default function OperatorRegistrationDiagram() {
  const [activePhase, setActivePhase] = useState(0);

  return (
    <div className="p-6 bg-gray-900 rounded-xl text-white">
      <h2 className="text-xl font-bold mb-4">算子注册与分发流程</h2>

      <div className="flex gap-3 mb-6">
        {steps.map((s, i) => (
          <button key={i} onClick={() => setActivePhase(i)}
            className={`flex-1 p-3 rounded-lg border-2 transition-all ${
              activePhase === i ? 'opacity-100' : 'opacity-50'
            }`} style={{
              borderColor: i === 0 ? '#3B82F6' : i === 1 ? '#F59E0B' : '#10B981',
              backgroundColor: activePhase === i ? `${i === 0 ? '#3B82F6' : i === 1 ? '#F59E0B' : '#10B981'}15` : 'transparent',
            }}>
            <div className="text-sm font-bold" style={{ color: i === 0 ? '#3B82F6' : i === 1 ? '#F59E0B' : '#10B981' }}>
              {s.phase}
            </div>
            <div className="mt-2 space-y-1">
              {s.steps.map((step, j) => (
                <div key={j} className="text-xs text-gray-400">
                  {j + 1}. {step.label}
                </div>
              ))}
            </div>
          </button>
        ))}
      </div>

      <div className="flex gap-4">
        <div className="flex-1 bg-black rounded-lg p-3 font-mono text-xs overflow-x-auto">
          <pre className="leading-5 text-gray-300">{codeSnippets[activePhase]}</pre>
        </div>

        <div className="w-56 space-y-2 text-xs">
          <div className="bg-gray-800 rounded p-3">
            <div className="font-bold text-white mb-2">流程详解</div>
            {steps[activePhase].steps.map((s, i) => (
              <div key={i} className="mb-2">
                <div className="text-blue-400 font-bold">{s.label}</div>
                <div className="text-gray-400 mt-0.5">{s.desc}</div>
              </div>
            ))}
          </div>
          <div className="bg-gray-800 rounded p-3">
            <div className="font-bold text-yellow-400 mb-1">关键概念</div>
            <div className="text-gray-400">
              PackedFunc 是 TVM 的统一函数接口，支持任意参数类型，跨语言调用。
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
