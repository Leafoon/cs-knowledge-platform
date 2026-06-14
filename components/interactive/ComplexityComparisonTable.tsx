'use client';

import { useState } from 'react';

const COMPLEXITIES = [
  { label: 'O(1)', fn: (_n: number) => 1, color: '#22c55e', example: '哈希表查找' },
  { label: 'O(log n)', fn: (n: number) => Math.ceil(Math.log2(n)), color: '#3b82f6', example: '二分搜索' },
  { label: 'O(√n)', fn: (n: number) => Math.ceil(Math.sqrt(n)), color: '#8b5cf6', example: '质数检测' },
  { label: 'O(n)', fn: (n: number) => n, color: '#a855f7', example: '线性扫描' },
  { label: 'O(n log n)', fn: (n: number) => Math.ceil(n * Math.log2(n)), color: '#f59e0b', example: '归并排序' },
  { label: 'O(n²)', fn: (n: number) => n * n, color: '#f97316', example: '冒泡排序' },
  { label: 'O(n³)', fn: (n: number) => n * n * n, color: '#ef4444', example: '矩阵乘法' },
  { label: 'O(2ⁿ)', fn: (n: number) => Math.pow(2, n), color: '#dc2626', example: '子集枚举' },
  { label: 'O(n!)', fn: (n: number) => {
    if (n > 20) return Infinity;
    let f = 1; for (let i = 2; i <= n; i++) f *= i; return f;
  }, color: '#7f1d1d', example: '全排列' },
];

const NS = [1, 5, 10, 20, 50, 100, 1000];

// Assume 10^9 operations per second
const CPU_OPS = 1e9;

function fmt(v: number): string {
  if (!isFinite(v) || v > 1e60) return '∞';
  if (v >= 1e18) return '>10^18';
  if (v >= 1e15) return `${(v / 1e15).toPrecision(2)}P`;
  if (v >= 1e12) return `${(v / 1e12).toPrecision(2)}T`;
  if (v >= 1e9) return `${(v / 1e9).toPrecision(2)}B`;
  if (v >= 1e6) return `${(v / 1e6).toPrecision(2)}M`;
  if (v >= 1e3) return `${(v / 1e3).toPrecision(2)}K`;
  return v.toFixed(0);
}

function fmtTime(ops: number): string {
  const sec = ops / CPU_OPS;
  if (!isFinite(sec) || sec > 1e40) return '宇宙年龄+';
  if (sec < 1e-9) return '<1 ns';
  if (sec < 1e-6) return `${(sec * 1e9).toPrecision(2)} ns`;
  if (sec < 1e-3) return `${(sec * 1e6).toPrecision(2)} μs`;
  if (sec < 1) return `${(sec * 1e3).toPrecision(2)} ms`;
  if (sec < 60) return `${sec.toPrecision(2)} s`;
  if (sec < 3600) return `${(sec / 60).toPrecision(2)} min`;
  if (sec < 86400) return `${(sec / 3600).toPrecision(2)} h`;
  if (sec < 3.15e7) return `${(sec / 86400).toPrecision(2)} d`;
  if (sec < 3.15e9) return `${(sec / 3.15e7).toPrecision(2)} yr`;
  return `${(sec / 3.15e7).toExponential(1)} yr`;
}

function colorCell(v: number, maxV: number): string {
  if (!isFinite(v) || v > 1e15) return 'rgba(220,38,38,0.3)';
  const ratio = Math.log(1 + v) / Math.log(1 + maxV);
  const r = Math.round(34 + ratio * (220 - 34));
  const g = Math.round(197 - ratio * (197 - 38));
  const b = Math.round(94 - ratio * (94 - 38));
  return `rgba(${r},${g},${b},0.25)`;
}

export default function ComplexityComparisonTable() {
  const [mode, setMode] = useState<'ops' | 'time'>('ops');
  const [highlightRow, setHighlightRow] = useState<number | null>(null);
  const [highlightCol, setHighlightCol] = useState<number | null>(null);

  return (
    <div className="rounded-xl border border-border-subtle bg-bg-secondary p-4 space-y-4">
      <div className="flex items-center justify-between flex-wrap gap-2">
        <h3 className="text-base font-semibold text-text-primary">📊 复杂度实际操作数对比</h3>
        <div className="flex gap-2">
          <button
            onClick={() => setMode('ops')}
            className={`px-3 py-1 rounded-full text-xs transition-colors ${mode === 'ops' ? 'bg-blue-500 text-white' : 'border border-border-subtle text-text-secondary hover:text-blue-400'}`}
          >
            操作次数
          </button>
          <button
            onClick={() => setMode('time')}
            className={`px-3 py-1 rounded-full text-xs transition-colors ${mode === 'time' ? 'bg-blue-500 text-white' : 'border border-border-subtle text-text-secondary hover:text-blue-400'}`}
          >
            估算时间（10⁹ ops/s）
          </button>
        </div>
      </div>

      <div className="overflow-x-auto rounded-lg border border-border-subtle">
        <table className="text-xs w-full border-collapse">
          <thead>
            <tr className="bg-bg-tertiary">
              <th className="px-3 py-2 text-left text-text-tertiary font-medium whitespace-nowrap border-b border-border-subtle">复杂度</th>
              <th className="px-3 py-2 text-left text-text-tertiary font-medium whitespace-nowrap border-b border-border-subtle">典型场景</th>
              {NS.map((n, ci) => (
                <th
                  key={n}
                  className={`px-3 py-2 text-center font-mono text-text-tertiary border-b border-l border-border-subtle cursor-pointer transition-colors whitespace-nowrap ${highlightCol === ci ? 'bg-blue-500/20 text-blue-400' : ''}`}
                  onClick={() => setHighlightCol(highlightCol === ci ? null : ci)}
                >
                  n={n}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {COMPLEXITIES.map((c, ri) => {
              const vals = NS.map(n => c.fn(n));
              const maxVal = Math.max(...vals.filter(isFinite));
              return (
                <tr
                  key={c.label}
                  className={`border-b border-border-subtle cursor-pointer transition-colors ${highlightRow === ri ? 'bg-bg-tertiary' : ''}`}
                  onMouseEnter={() => setHighlightRow(ri)}
                  onMouseLeave={() => setHighlightRow(null)}
                >
                  <td className="px-3 py-2 font-mono font-semibold whitespace-nowrap" style={{ color: c.color }}>{c.label}</td>
                  <td className="px-3 py-2 text-text-tertiary whitespace-nowrap">{c.example}</td>
                  {vals.map((v, ci) => (
                    <td
                      key={ci}
                      className={`px-3 py-2 text-center font-mono border-l border-border-subtle transition-colors ${highlightCol === ci ? 'ring-1 ring-inset ring-blue-400/40' : ''}`}
                      style={{ backgroundColor: colorCell(v, maxVal) }}
                    >
                      {mode === 'ops' ? fmt(v) : fmtTime(v)}
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <p className="text-xs text-text-tertiary">
        💡 点击列头可高亮对比同一 n 值下各算法的差异。悬停行查看某算法在不同 n 下的增长。
        时间估算基于假设：每秒 10⁹ 次基本操作（实际依硬件和操作类型而异）。
      </p>
    </div>
  );
}
