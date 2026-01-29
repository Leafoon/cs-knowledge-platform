"use client";

import React, { useState, useMemo } from 'react';
import { Scissors, FileText, Code2 } from 'lucide-react';

type SplitterType = 'recursive' | 'character' | 'token' | 'markdown';

const sampleText = `# LangChain Introduction

LangChain is a framework for developing applications powered by language models.

## Core Concepts

The main value props of LangChain are:
1. Components: abstractions for working with language models
2. Off-the-shelf chains: assembly of components for certain tasks

## Memory Systems

LangChain provides several memory implementations.`;

export default function TextSplittingVisualizer() {
  const [selectedType, setSelectedType] = useState<SplitterType>('recursive');
  const [chunkSize, setChunkSize] = useState(100);
  const [chunkOverlap, setChunkOverlap] = useState(20);

  const chunks = useMemo(() => {
    const splitText = (text: string, size: number, overlap: number): string[] => {
      if (selectedType === 'markdown') {
        return text.split(/(?=^#)/m).filter(s => s.trim());
      }
      
      const chunks: string[] = [];
      let start = 0;
      
      while (start < text.length) {
        const end = Math.min(start + size, text.length);
        const chunk = text.slice(start, end);
        if (chunk.trim()) chunks.push(chunk);
        start = end - overlap;
      }
      
      return chunks.slice(0, 10);
    };
    
    return splitText(sampleText, chunkSize, chunkOverlap);
  }, [selectedType, chunkSize, chunkOverlap]);

  const configs: Record<SplitterType, { name: string; icon: any; color: string }> = {
    recursive: { name: 'RecursiveCharacterTextSplitter', icon: Scissors, color: 'blue' },
    character: { name: 'CharacterTextSplitter', icon: Code2, color: 'purple' },
    token: { name: 'TokenTextSplitter', icon: FileText, color: 'green' },
    markdown: { name: 'MarkdownHeaderTextSplitter', icon: FileText, color: 'orange' }
  };

  const config = configs[selectedType];
  const Icon = config.icon;

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-white rounded-xl border border-slate-200 my-8">
      <h3 className="text-xl font-bold text-slate-800 mb-4">文本分割策略对比</h3>

      {/* 策略选择 */}
      <div className="grid grid-cols-4 gap-3 mb-6">
        {(Object.keys(configs) as SplitterType[]).map((type) => {
          const cfg = configs[type];
          const Ico = cfg.icon;
          return (
            <button
              key={type}
              onClick={() => setSelectedType(type)}
              className={`p-3 rounded-lg border-2 transition-all ${
                selectedType === type
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-slate-200 hover:border-slate-300'
              }`}
            >
              <Ico className={`w-5 h-5 mx-auto mb-1 ${selectedType === type ? 'text-blue-600' : 'text-slate-400'}`} />
              <div className={`text-xs text-center ${selectedType === type ? 'text-blue-700 font-medium' : 'text-slate-600'}`}>
                {type.charAt(0).toUpperCase() + type.slice(1)}
              </div>
            </button>
          );
        })}
      </div>

      {/* 参数配置 */}
      <div className="bg-slate-50 rounded-lg p-4 mb-6">
        <h4 className="font-semibold text-slate-800 mb-3 text-sm">{config.name}</h4>
        {selectedType !== 'markdown' && (
          <div className="grid md:grid-cols-2 gap-4">
            <div>
              <label className="text-sm text-slate-600 mb-2 block">
                Chunk Size: {chunkSize}
              </label>
              <input
                type="range"
                min="50"
                max="200"
                value={chunkSize}
                onChange={(e) => setChunkSize(Number(e.target.value))}
                className="w-full"
              />
            </div>
            <div>
              <label className="text-sm text-slate-600 mb-2 block">
                Chunk Overlap: {chunkOverlap}
              </label>
              <input
                type="range"
                min="0"
                max="50"
                value={chunkOverlap}
                onChange={(e) => setChunkOverlap(Number(e.target.value))}
                className="w-full"
              />
            </div>
          </div>
        )}
      </div>

      {/* 分割结果 */}
      <div className="mb-6">
        <div className="flex justify-between items-center mb-3">
          <h4 className="font-semibold text-slate-800 text-sm">分割结果</h4>
          <span className="text-sm text-slate-600">共 <span className="font-bold text-blue-600">{chunks.length}</span> 个 chunks</span>
        </div>
        <div className="space-y-2 max-h-80 overflow-y-auto">
          {chunks.map((chunk, idx) => (
            <div
              key={idx}
              className="p-3 rounded-lg border-l-4 border-blue-500 bg-slate-50"
            >
              <div className="flex justify-between items-center mb-2">
                <span className="text-xs font-semibold text-blue-600">
                  Chunk {idx + 1}
                </span>
                <span className="text-xs text-slate-500">{chunk.length} chars</span>
              </div>
              <div className="text-sm text-slate-700 font-mono whitespace-pre-wrap">
                {chunk}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* 策略对比表 */}
      <div>
        <h4 className="font-semibold text-slate-800 mb-3 text-sm">策略对比</h4>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-200">
                <th className="text-left py-2 px-3 font-semibold text-slate-700">分割器</th>
                <th className="text-center py-2 px-3 font-semibold text-slate-700">精度</th>
                <th className="text-center py-2 px-3 font-semibold text-slate-700">性能</th>
                <th className="text-left py-2 px-3 font-semibold text-slate-700">适用场景</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-b border-slate-100">
                <td className="py-2 px-3 font-medium text-blue-600">Recursive</td>
                <td className="text-center py-2 px-3">⭐⭐⭐⭐⭐</td>
                <td className="text-center py-2 px-3">快</td>
                <td className="py-2 px-3 text-slate-600">通用文本（推荐）</td>
              </tr>
              <tr className="border-b border-slate-100">
                <td className="py-2 px-3 font-medium text-purple-600">Character</td>
                <td className="text-center py-2 px-3">⭐⭐⭐</td>
                <td className="text-center py-2 px-3">快</td>
                <td className="py-2 px-3 text-slate-600">简单分割</td>
              </tr>
              <tr className="border-b border-slate-100">
                <td className="py-2 px-3 font-medium text-green-600">Token</td>
                <td className="text-center py-2 px-3">⭐⭐⭐⭐⭐</td>
                <td className="text-center py-2 px-3">中</td>
                <td className="py-2 px-3 text-slate-600">精确 token 控制</td>
              </tr>
              <tr>
                <td className="py-2 px-3 font-medium text-orange-600">Markdown</td>
                <td className="text-center py-2 px-3">⭐⭐⭐⭐</td>
                <td className="text-center py-2 px-3">快</td>
                <td className="py-2 px-3 text-slate-600">Markdown 文档</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
