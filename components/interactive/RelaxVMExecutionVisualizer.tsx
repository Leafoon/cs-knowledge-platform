'use client';

import { useState } from 'react';

const instructions = [
  { op: 'call_packed', target: 'matmul', args: ['A', 'B', 'C'], color: '#3B82F6' },
  { op: 'call_packed', target: 'add_bias', args: ['C', 'bias'], color: '#8B5CF6' },
  { op: 'call_packed', target: 'relu', args: ['C'], color: '#10B981' },
  { op: 'call_packed', target: 'store', args: ['out', 'C'], color: '#F59E0B' },
];

const dispatchTargets = {
  matmul: { cpu: 'dgemm_', cuda: 'cublasGemmEx', hip: 'rocblas_gemm', color: '#3B82F6' },
  add_bias: { cpu: 'avx512_add', cuda: 'fused_add_kernel', hip: 'fused_add_kernel', color: '#8B5CF6' },
  relu: { cpu: 'avx512_relu', cuda: 'elementwise_relu', hip: 'elementwise_relu', color: '#10B981' },
  store: { cpu: 'memcpy', cuda: 'cudaMemcpy', hip: 'hipMemcpy', color: '#F59E0B' },
};

const devices = ['cpu', 'cuda', 'hip'];

export default function RelaxVMExecutionVisualizer() {
  const [currentPC, setCurrentPC] = useState(0);
  const [targetDevice, setTargetDevice] = useState<'cpu' | 'cuda' | 'hip'>('cuda');
  const [running, setRunning] = useState(false);

  const runVM = () => {
    setRunning(true);
    let pc = 0;
    const iv = setInterval(() => {
      setCurrentPC(pc);
      pc++;
      if (pc >= instructions.length) {
        clearInterval(iv);
        setTimeout(() => { setRunning(false); setCurrentPC(0); }, 800);
      }
    }, 700);
  };

  return (
    <div className="p-6 bg-gray-900 rounded-xl text-white">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-bold">Relax VM 执行模型</h2>
        <div className="flex items-center gap-2">
          <div className="flex gap-1">
            {devices.map(d => (
              <button key={d} onClick={() => setTargetDevice(d as any)}
                className={`px-2 py-1 rounded text-[10px] uppercase ${targetDevice === d ? 'bg-blue-600' : 'bg-gray-700'}`}>
                {d}
              </button>
            ))}
          </div>
          <button onClick={runVM} disabled={running}
            className="px-3 py-1 bg-green-600 rounded text-sm hover:bg-green-500 disabled:opacity-50">
            {running ? '执行中...' : '运行 VM'}
          </button>
        </div>
      </div>

      <div className="flex gap-4 mb-4">
        {/* Instruction stream */}
        <div className="flex-1">
          <div className="text-xs text-gray-400 mb-2">指令流</div>
          <div className="space-y-1">
            {instructions.map((inst, i) => {
              const isCurrent = i === currentPC;
              const isExecuted = i < currentPC;
              return (
                <div key={i} className={`flex items-center gap-2 p-2 rounded font-mono text-xs transition-all ${
                  isCurrent ? 'bg-blue-900/30 border border-blue-500' : isExecuted ? 'opacity-50' : 'opacity-70'
                }`}>
                  <span className="text-gray-600 w-5">{i}</span>
                  <span style={{ color: inst.color }}>{inst.op}</span>
                  <span className="text-gray-400">→</span>
                  <span className="text-white font-bold">{inst.target}</span>
                  <span className="text-gray-500">({inst.args.join(', ')})</span>
                  {isExecuted && <span className="text-green-400 ml-auto">✓</span>}
                </div>
              );
            })}
          </div>
        </div>

        {/* Dispatch table */}
        <div className="w-64">
          <div className="text-xs text-gray-400 mb-2">设备分发 ({targetDevice})</div>
          <div className="space-y-1">
            {instructions.map((inst, i) => {
              const dispatch = dispatchTargets[inst.target as keyof typeof dispatchTargets];
              const isCurrent = i === currentPC;
              return (
                <div key={i} className={`flex items-center gap-2 p-2 rounded text-xs ${
                  isCurrent ? 'bg-green-900/30 border border-green-500' : 'bg-gray-800'
                }`}>
                  <span className="text-gray-400 w-16">{inst.target}</span>
                  <span className="text-green-400 font-mono">{dispatch[targetDevice]}</span>
                </div>
              );
            })}
          </div>

          <div className="mt-3 bg-gray-800 rounded p-2 text-[10px] text-gray-400">
            VM 根据目标设备自动选择最优实现
          </div>
        </div>
      </div>

      <div className="bg-gray-800 rounded-lg p-3 text-xs text-gray-300">
        <b className="text-white">Relax VM 执行模型：</b>
        字节码解释器按顺序执行 call_packed 指令，每个指令通过 PackedFunc 分发到目标设备的算子实现，
        支持 CPU/GPU 异步执行和内存管理。
      </div>
    </div>
  );
}
