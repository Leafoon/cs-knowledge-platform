'use client';
import { useState } from 'react';

interface Instruction {
  tilelangOp: string;
  ptxInstr: string;
  shape: string;
  dtype: string;
  threads: number;
  flopsPerCycle: number;
  notes: string;
}

const instructions: Instruction[] = [
  { tilelangOp: 'matmul_fp16', ptxInstr: 'mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32', shape: '16×8×16', dtype: 'FP16→FP32', threads: 32, flopsPerCycle: 256, notes: '最常用，支持累加' },
  { tilelangOp: 'matmul_bf16', ptxInstr: 'mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32', shape: '16×8×16', dtype: 'BF16→FP32', threads: 32, flopsPerCycle: 256, notes: 'BF16精度，训练推荐' },
  { tilelangOp: 'matmul_int8', ptxInstr: 'mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32', shape: '16×8×16', dtype: 'INT8→INT32', threads: 32, flopsPerCycle: 512, notes: '量化推理，吞吐翻倍' },
  { tilelangOp: 'load_matrix', ptxInstr: 'ldmatrix.sync.aligned.m8n8.x4.shared.b16', shape: '8×8×4', dtype: 'B16', threads: 32, flopsPerCycle: 0, notes: '从共享内存加载到寄存器' },
  { tilelangOp: 'matmul_tf32', ptxInstr: 'mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32', shape: '16×8×8', dtype: 'TF32→FP32', threads: 32, flopsPerCycle: 128, notes: 'TF32格式，FP32精度近似' },
  { tilelangOp: 'matmul_fp64', ptxInstr: 'mma.sync.aligned.m16n8k4.row.col.f64.f64.f64.f64', shape: '16×8×4', dtype: 'FP64→FP64', threads: 32, flopsPerCycle: 32, notes: '双精度，科学计算' },
];

export function TensorCoreInstructionMap() {
  const [selectedRow, setSelectedRow] = useState<number | null>(null);
  const [filterDtype, setFilterDtype] = useState('全部');

  const dtypes = ['全部', ...new Set(instructions.map(i => i.dtype))];
  const filtered = filterDtype === '全部' ? instructions : instructions.filter(i => i.dtype === filterDtype);

  return (
    <div className="bg-gray-900 rounded-xl p-6 text-white">
      <h2 className="text-xl font-bold text-cyan-400 mb-4">Tensor Core指令映射表</h2>

      <div className="flex gap-2 mb-4">
        {dtypes.map(d => (
          <button key={d} onClick={() => setFilterDtype(d)}
            className={`px-3 py-1 rounded-full text-xs transition-all ${
              filterDtype === d ? 'bg-cyan-600' : 'bg-gray-700 hover:bg-gray-600'
            }`}>{d}</button>
        ))}
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-700">
              <th className="text-left py-2 px-2 text-gray-400 text-xs">TileLang操作</th>
              <th className="text-left py-2 px-2 text-gray-400 text-xs">PTX指令</th>
              <th className="text-center py-2 px-2 text-gray-400 text-xs">Shape</th>
              <th className="text-center py-2 px-2 text-gray-400 text-xs">数据类型</th>
              <th className="text-center py-2 px-2 text-gray-400 text-xs">线程数</th>
              <th className="text-center py-2 px-2 text-gray-400 text-xs">FLOP/周期</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((inst, i) => (
              <tr key={i}
                className={`border-b border-gray-800 cursor-pointer transition-colors ${
                  selectedRow === i ? 'bg-gray-800' : 'hover:bg-gray-800/50'
                }`}
                onClick={() => setSelectedRow(selectedRow === i ? null : i)}>
                <td className="py-2 px-2 font-mono text-cyan-300 text-xs">{inst.tilelangOp}</td>
                <td className="py-2 px-2">
                  <code className="text-[10px] text-green-400 font-mono break-all">{inst.ptxInstr}</code>
                </td>
                <td className="py-2 px-2 text-center text-xs text-gray-300">{inst.shape}</td>
                <td className="py-2 px-2 text-center text-xs text-gray-300">{inst.dtype}</td>
                <td className="py-2 px-2 text-center text-xs text-gray-300">{inst.threads}</td>
                <td className="py-2 px-2 text-center">
                  <span className="text-xs font-mono text-yellow-400">{inst.flopsPerCycle}</span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {selectedRow !== null && (
        <div className="mt-4 p-4 bg-gray-800 rounded-lg">
          <div className="text-sm font-medium text-cyan-300">{instructions[selectedRow].tilelangOp}</div>
          <div className="text-xs text-gray-400 mt-1">{instructions[selectedRow].notes}</div>
          <div className="mt-2">
            <code className="text-[11px] text-green-400 font-mono break-all">{instructions[selectedRow].ptxInstr}</code>
          </div>
        </div>
      )}
    </div>
  );
}
