'use client';
import { useState } from 'react';

interface CacheEntry {
  hash: string;
  kernelName: string;
  binarySize: string;
  compileTime: string;
  hitCount: number;
  lastUsed: string;
  status: 'hit' | 'miss';
}

const cacheEntries: CacheEntry[] = [
  { hash: 'a3f7b2c1', kernelName: 'matmul_1024x1024x1024_fp16', binarySize: '2.4 MB', compileTime: '1.2s', hitCount: 47, lastUsed: '2分钟前', status: 'hit' },
  { hash: 'e8d4c6a9', kernelName: 'conv2d_256x56x56_64', binarySize: '1.8 MB', compileTime: '0.9s', hitCount: 23, lastUsed: '5分钟前', status: 'hit' },
  { hash: 'f1b9e3d7', kernelName: 'attention_512x512x64', binarySize: '3.1 MB', compileTime: '2.1s', hitCount: 12, lastUsed: '1分钟前', status: 'hit' },
  { hash: '7c2a4e8f', kernelName: 'softmax_1024x1024', binarySize: '0.5 MB', compileTime: '0.3s', hitCount: 89, lastUsed: '30秒前', status: 'hit' },
  { hash: '9d1f5b3e', kernelName: 'layernorm_2048x768', binarySize: '0.8 MB', compileTime: '0.5s', hitCount: 5, lastUsed: '10分钟前', status: 'hit' },
];

const pendingCompilations = [
  { hash: 'b4e8c2a1', kernelName: 'matmul_2048x2048x2048_bf16', status: 'compiling' },
  { hash: 'd7f1e9b3', kernelName: 'flash_attention_1024x1024', status: 'queued' },
];

export function CompilationCacheVisualizer() {
  const [selectedEntry, setSelectedEntry] = useState<number | null>(null);
  const [showPending, setShowPending] = useState(false);

  const totalHits = cacheEntries.reduce((s, e) => s + e.hitCount, 0);
  const hitRate = ((totalHits / (totalHits + 15)) * 100).toFixed(1);

  return (
    <div className="bg-gray-900 rounded-xl p-6 text-white">
      <h2 className="text-xl font-bold text-cyan-400 mb-4">编译缓存可视化</h2>

      {/* Stats */}
      <div className="grid grid-cols-4 gap-4 mb-6">
        <div className="bg-gray-800 rounded-lg p-3 text-center">
          <div className="text-2xl font-bold text-green-400">{cacheEntries.length}</div>
          <div className="text-xs text-gray-400">缓存条目</div>
        </div>
        <div className="bg-gray-800 rounded-lg p-3 text-center">
          <div className="text-2xl font-bold text-cyan-400">{totalHits}</div>
          <div className="text-xs text-gray-400">总命中次数</div>
        </div>
        <div className="bg-gray-800 rounded-lg p-3 text-center">
          <div className="text-2xl font-bold text-yellow-400">{hitRate}%</div>
          <div className="text-xs text-gray-400">命中率</div>
        </div>
        <div className="bg-gray-800 rounded-lg p-3 text-center">
          <div className="text-2xl font-bold text-purple-400">8.6 MB</div>
          <div className="text-xs text-gray-400">缓存大小</div>
        </div>
      </div>

      {/* Cache hash visualization */}
      <div className="mb-6">
        <div className="text-sm text-gray-400 mb-2">缓存哈希索引</div>
        <div className="flex gap-1 flex-wrap">
          {cacheEntries.map((entry, i) => (
            <div key={i}
              onClick={() => setSelectedEntry(selectedEntry === i ? null : i)}
              className={`px-2 py-1 rounded text-xs font-mono cursor-pointer transition-all ${
                selectedEntry === i ? 'bg-green-600 text-white' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
              }`}>
              {entry.hash}
            </div>
          ))}
          {pendingCompilations.map((p, i) => (
            <div key={i}
              className={`px-2 py-1 rounded text-xs font-mono ${
                p.status === 'compiling' ? 'bg-yellow-600/50 text-yellow-300 animate-pulse' : 'bg-gray-700 text-gray-500'
              }`}>
              {p.hash}
            </div>
          ))}
        </div>
      </div>

      {/* Cache entries table */}
      <div className="space-y-1 mb-4">
        <div className="grid grid-cols-5 gap-2 text-xs text-gray-500 px-2 py-1">
          <span>Hash</span><span>Kernel</span><span>大小</span><span>命中</span><span>最近使用</span>
        </div>
        {cacheEntries.map((entry, i) => (
          <div key={i}
            onClick={() => setSelectedEntry(selectedEntry === i ? null : i)}
            className={`grid grid-cols-5 gap-2 text-sm p-2 rounded cursor-pointer transition-colors ${
              selectedEntry === i ? 'bg-gray-800' : 'hover:bg-gray-800/50'
            }`}>
            <span className="font-mono text-green-400 text-xs">{entry.hash}</span>
            <span className="text-gray-300 text-xs truncate">{entry.kernelName}</span>
            <span className="text-gray-400 text-xs">{entry.binarySize}</span>
            <span className="text-yellow-400 text-xs font-mono">{entry.hitCount}</span>
            <span className="text-gray-500 text-xs">{entry.lastUsed}</span>
          </div>
        ))}
      </div>

      {/* Pending compilations */}
      <div className="border border-gray-700 rounded-lg p-3">
        <div className="text-xs text-gray-400 mb-2">编译队列</div>
        {pendingCompilations.map((p, i) => (
          <div key={i} className="flex items-center gap-2 text-xs py-1">
            <span className={`w-2 h-2 rounded-full ${p.status === 'compiling' ? 'bg-yellow-500 animate-pulse' : 'bg-gray-600'}`} />
            <span className="font-mono text-gray-400">{p.hash}</span>
            <span className="text-gray-500">{p.kernelName}</span>
            <span className={`ml-auto ${p.status === 'compiling' ? 'text-yellow-400' : 'text-gray-600'}`}>
              {p.status === 'compiling' ? '编译中...' : '排队中'}
            </span>
          </div>
        ))}
      </div>

      {selectedEntry !== null && (
        <div className="mt-4 p-4 bg-gray-800 rounded-lg">
          <div className="text-sm font-medium text-green-300">{cacheEntries[selectedEntry].kernelName}</div>
          <div className="text-xs text-gray-400 mt-1">
            Hash: {cacheEntries[selectedEntry].hash} · 二进制: {cacheEntries[selectedEntry].binarySize} · 编译耗时: {cacheEntries[selectedEntry].compileTime}
          </div>
        </div>
      )}
    </div>
  );
}
