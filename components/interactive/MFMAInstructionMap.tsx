'use client';
import { useState } from 'react';

interface MFMAInstruction {
  name: string;
  shape: string;
  dtype: string;
  threads: number;
  flopsPerCycle: number;
  vgprUsage: number;
  description: string;
}

const mfmaInstructions: MFMAInstruction[] = [
  { name: 'v_mfma_f32_32x32x8f16', shape: '32×32×8', dtype: 'FP16→FP32', threads: 64, flopsPerCycle: 8192, vgprUsage: 32, description: '最大Tile，适合大矩阵乘法' },
  { name: 'v_mfma_f32_16x16x16f16', shape: '16×16×16', dtype: 'FP16→FP32', threads: 64, flopsPerCycle: 8192, vgprUsage: 12, description: '平衡型，适合中等矩阵' },
  { name: 'v_mfma_f32_4x4x4f32', shape: '4×4×4', dtype: 'FP32→FP32', threads: 64, flopsPerCycle: 512, vgprUsage: 4, description: '小Tile，适合FP32矩阵' },
  { name: 'v_mfma_f32_32x32x4f32', shape: '32×32×4', dtype: 'FP32→FP32', threads: 64, flopsPerCycle: 8192, vgprUsage: 32, description: 'FP32大Tile矩阵乘法' },
  { name: 'v_mfma_f32_16x16x8f16', shape: '16×16×8', dtype: 'FP16→FP32', threads: 64, flopsPerCycle: 4096, vgprUsage: 12, description: '中等Tile FP16矩阵' },
  { name: 'v_mfma_i32_32x32x8i8', shape: '32×32×8', dtype: 'INT8→INT32', threads: 64, flopsPerCycle: 16384, vgprUsage: 32, description: 'INT8量化推理，吞吐翻倍' },
  { name: 'v_mfma_f32_32x32x2bf16', shape: '32×32×2', dtype: 'BF16→FP32', threads: 64, flopsPerCycle: 4096, vgprUsage: 32, description: 'BF16训练格式' },
  { name: 'v_mfma_f32_16x16x4bf16', shape: '16×16×4', dtype: 'BF16→FP32', threads: 64, flopsPerCycle: 4096, vgprUsage: 12, description: 'BF16中等Tile' },
];

export function MFMAInstructionMap() {
  const [selectedRow, setSelectedRow] = useState<number | null>(null);
  const [filterDtype, setFilterDtype] = useState('全部');

  const dtypes = ['全部', ...new Set(mfmaInstructions.map(i => i.dtype))];
  const filtered = filterDtype === '全部' ? mfmaInstructions : mfmaInstructions.filter(i => i.dtype === filterDtype);

  return (
    <div className="bg-gray-900 rounded-xl p-6 text-white">
      <h2 className="text-xl font-bold text-red-400 mb-4">AMD MFMA指令映射表</h2>

      <div className="flex gap-2 mb-4">
        {dtypes.map(d => (
          <button key={d} onClick={() => setFilterDtype(d)}
            className={`px-3 py-1 rounded-full text-xs transition-all ${
              filterDtype === d ? 'bg-red-600' : 'bg-gray-700 hover:bg-gray-600'
            }`}>{d}</button>
        ))}
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-700">
              <th className="text-left py-2 px-2 text-gray-400 text-xs">指令名称</th>
              <th className="text-center py-2 px-2 text-gray-400 text-xs">Shape</th>
              <th className="text-center py-2 px-2 text-gray-400 text-xs">数据类型</th>
              <th className="text-center py-2 px-2 text-gray-400 text-xs">线程数</th>
              <th className="text-center py-2 px-2 text-gray-400 text-xs">FLOP/周期</th>
              <th className="text-center py-2 px-2 text-gray-400 text-xs">VGPR</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((inst, i) => (
              <tr key={i}
                className={`border-b border-gray-800 cursor-pointer transition-colors ${
                  selectedRow === i ? 'bg-gray-800' : 'hover:bg-gray-800/50'
                }`}
                onClick={() => setSelectedRow(selectedRow === i ? null : i)}>
                <td className="py-2 px-2 font-mono text-red-300 text-xs">{inst.name}</td>
                <td className="py-2 px-2 text-center text-xs text-gray-300">{inst.shape}</td>
                <td className="py-2 px-2 text-center text-xs text-gray-300">{inst.dtype}</td>
                <td className="py-2 px-2 text-center text-xs text-gray-300">{inst.threads}</td>
                <td className="py-2 px-2 text-center">
                  <span className="text-xs font-mono text-yellow-400">{inst.flopsPerCycle.toLocaleString()}</span>
                </td>
                <td className="py-2 px-2 text-center text-xs text-gray-300">{inst.vgprUsage}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {selectedRow !== null && (
        <div className="mt-4 p-4 bg-gray-800 rounded-lg">
          <div className="text-sm font-medium text-red-300">{mfmaInstructions[selectedRow].name}</div>
          <div className="text-xs text-gray-400 mt-1">{mfmaInstructions[selectedRow].description}</div>
          <div className="mt-2 flex gap-4 text-xs text-gray-500">
            <span>FLOP效率: {((mfmaInstructions[selectedRow].flopsPerCycle / 64) / 64 * 100).toFixed(1)}%</span>
            <span>VGPR压力: {mfmaInstructions[selectedRow].vgprUsage > 20 ? '高' : mfmaInstructions[selectedRow].vgprUsage > 10 ? '中' : '低'}</span>
          </div>
        </div>
      )}
    </div>
  );
}
