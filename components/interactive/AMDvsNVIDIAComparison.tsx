'use client';
import { useState } from 'react';

interface Spec {
  category: string;
  amd: string;
  nvidia: string;
  amdScore: number;
  nvidiaScore: number;
}

const specs: Spec[] = [
  { category: 'GPU型号', amd: 'MI300X', nvidia: 'H100 SXM', amdScore: 0, nvidiaScore: 0 },
  { category: '显存容量', amd: '192 GB HBM3', nvidia: '80 GB HBM3', amdScore: 3, nvidiaScore: 1 },
  { category: '显存带宽', amd: '5.3 TB/s', nvidia: '3.35 TB/s', amdScore: 3, nvidiaScore: 2 },
  { category: 'FP16算力', amd: '1307 TFLOPS', nvidia: '989 TFLOPS', amdScore: 3, nvidiaScore: 2 },
  { category: 'FP32算力', amd: '817 TFLOPS', nvidia: '67 TFLOPS', amdScore: 3, nvidiaScore: 1 },
  { category: 'INT8算力', amd: '2614 TOPS', nvidia: '1979 TOPS', amdScore: 3, nvidiaScore: 2 },
  { category: '制程', amd: '5nm + 6nm', nvidia: '4nm', amdScore: 2, nvidiaScore: 3 },
  { category: 'TDP功耗', amd: '750W', nvidia: '700W', amdScore: 2, nvidiaScore: 3 },
  { category: '互联', amd: 'Infinity Fabric', nvidia: 'NVLink 4.0', amdScore: 3, nvidiaScore: 3 },
  { category: '软件生态', amd: 'ROCm / HIP', nvidia: 'CUDA / cuDNN', amdScore: 2, nvidiaScore: 3 },
];

export function AMDvsNVIDIAComparison() {
  const [hoveredRow, setHoveredRow] = useState<number | null>(null);
  const [showOnlyDiff, setShowOnlyDiff] = useState(false);

  const filtered = showOnlyDiff ? specs.filter(s => s.amdScore !== s.nvidiaScore && s.amdScore > 0) : specs;

  const amdTotal = specs.reduce((s, sp) => s + sp.amdScore, 0);
  const nvidiaTotal = specs.reduce((s, sp) => s + sp.nvidiaScore, 0);

  return (
    <div className="bg-gray-900 rounded-xl p-6 text-white">
      <h2 className="text-xl font-bold text-cyan-400 mb-4">AMD vs NVIDIA 硬件对比</h2>

      <div className="flex items-center justify-between mb-4">
        <div className="flex gap-4 text-sm">
          <span className="text-red-400">AMD MI300X: {amdTotal}分</span>
          <span className="text-green-400">NVIDIA H100: {nvidiaTotal}分</span>
        </div>
        <button onClick={() => setShowOnlyDiff(!showOnlyDiff)}
          className="px-3 py-1 rounded text-xs bg-gray-700 hover:bg-gray-600">
          {showOnlyDiff ? '显示全部' : '仅显示差异'}
        </button>
      </div>

      {/* Score bars */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div>
          <div className="flex items-center justify-between text-xs text-gray-400 mb-1">
            <span>AMD MI300X</span><span>{amdTotal}/{specs.reduce((s, sp) => s + Math.max(sp.amdScore, sp.nvidiaScore), 0)}</span>
          </div>
          <div className="bg-gray-800 rounded-full h-4 overflow-hidden">
            <div className="h-full bg-gradient-to-r from-red-700 to-red-500 rounded-full transition-all"
              style={{ width: `${(amdTotal / (amdTotal + nvidiaTotal)) * 100}%` }} />
          </div>
        </div>
        <div>
          <div className="flex items-center justify-between text-xs text-gray-400 mb-1">
            <span>NVIDIA H100</span><span>{nvidiaTotal}/{specs.reduce((s, sp) => s + Math.max(sp.amdScore, sp.nvidiaScore), 0)}</span>
          </div>
          <div className="bg-gray-800 rounded-full h-4 overflow-hidden">
            <div className="h-full bg-gradient-to-r from-green-700 to-green-500 rounded-full transition-all"
              style={{ width: `${(nvidiaTotal / (amdTotal + nvidiaTotal)) * 100}%` }} />
          </div>
        </div>
      </div>

      {/* Spec table */}
      <div className="space-y-1">
        {filtered.map((s, i) => (
          <div key={i}
            className={`grid grid-cols-3 gap-4 p-2 rounded text-sm transition-colors ${
              hoveredRow === i ? 'bg-gray-800' : 'hover:bg-gray-800/50'
            }`}
            onMouseEnter={() => setHoveredRow(i)}
            onMouseLeave={() => setHoveredRow(null)}>
            <div className="text-gray-300">{s.category}</div>
            <div className={`text-center font-mono ${
              s.amdScore > s.nvidiaScore ? 'text-red-400 font-bold' : 'text-gray-400'
            }`}>{s.amd}</div>
            <div className={`text-center font-mono ${
              s.nvidiaScore > s.amdScore ? 'text-green-400 font-bold' : 'text-gray-400'
            }`}>{s.nvidia}</div>
          </div>
        ))}
      </div>

      <div className="mt-4 grid grid-cols-2 gap-4 text-xs text-gray-500">
        <div>优势: 大显存、高带宽、FP32算力</div>
        <div>优势: 成熟生态、先进制程、低功耗</div>
      </div>
    </div>
  );
}
