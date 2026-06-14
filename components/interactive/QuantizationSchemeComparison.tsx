'use client';

import { useState } from 'react';

const schemes = [
  { format: 'FP32', bits: 32, range: '±3.4×10³⁸', speed: '1×', mem: '4B', use: '训练基准', precision: '100%' },
  { format: 'FP16', bits: 16, range: '±65504', speed: '2×', mem: '2B', use: '混合精度训练', precision: '99.9%' },
  { format: 'BF16', bits: 16, range: '±3.4×10³⁸', speed: '2×', mem: '2B', use: '大模型训练', precision: '99.8%' },
  { format: 'INT8', bits: 8, range: '-128~127', speed: '4×', mem: '1B', use: '推理加速', precision: '99.5%' },
  { format: 'FP8', bits: 8, range: '±448', speed: '4×', mem: '1B', use: 'H100训练', precision: '99.7%' },
  { format: 'INT4', bits: 4, range: '-8~7', speed: '8×', mem: '0.5B', use: '模型量化', precision: '98.5%' },
  { format: 'NF4', bits: 4, range: '归一化', speed: '8×', mem: '0.5B', use: 'QLoRA', precision: '98.8%' },
];

const cols = ['格式', '位宽', '数值范围', '相对速度', '内存/参数', '应用场景', '精度保持'];

export default function QuantizationSchemeComparison() {
  const [selectedRow, setSelectedRow] = useState<number | null>(null);

  return (
    <div className="p-6 bg-gray-900 rounded-xl text-white">
      <h2 className="text-xl font-bold mb-4">量化方案对比</h2>

      <div className="overflow-x-auto mb-4">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-700">
              {cols.map((col, i) => (
                <th key={i} className="px-3 py-2 text-left text-gray-400 font-medium">{col}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {schemes.map((s, i) => (
              <tr key={i}
                className={`border-b border-gray-800 cursor-pointer transition-colors ${
                  selectedRow === i ? 'bg-blue-900/30' : 'hover:bg-gray-800/50'
                }`}
                onClick={() => setSelectedRow(selectedRow === i ? null : i)}>
                <td className="px-3 py-2 font-mono font-bold text-blue-400">{s.format}</td>
                <td className="px-3 py-2">{s.bits}bit</td>
                <td className="px-3 py-2 font-mono text-xs">{s.range}</td>
                <td className="px-3 py-2">
                  <span className="text-green-400 font-bold">{s.speed}</span>
                </td>
                <td className="px-3 py-2">{s.mem}</td>
                <td className="px-3 py-2 text-gray-300">{s.use}</td>
                <td className="px-3 py-2">
                  <div className="flex items-center gap-2">
                    <div className="w-16 bg-gray-700 rounded-full h-1.5">
                      <div className="h-1.5 rounded-full bg-green-500"
                        style={{ width: s.precision }} />
                    </div>
                    <span className="text-xs">{s.precision}</span>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {selectedRow !== null && (
        <div className="bg-gray-800 rounded-lg p-4 text-sm space-y-2">
          <div className="font-bold text-blue-400">{schemes[selectedRow].format} 详解</div>
          <div className="grid grid-cols-3 gap-4 text-xs">
            <div>
              <span className="text-gray-400">内存节省：</span>
              <span className="text-green-400 font-bold">{(32 / schemes[selectedRow].bits).toFixed(0)}×</span>
            </div>
            <div>
              <span className="text-gray-400">量化误差：</span>
              <span className="text-yellow-400">{(100 - parseFloat(schemes[selectedRow].precision)).toFixed(1)}%</span>
            </div>
            <div>
              <span className="text-gray-400">位宽效率：</span>
              <span className="text-purple-400">{(parseFloat(schemes[selectedRow].speed) / schemes[selectedRow].bits * 8).toFixed(1)} TOPS/B</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
