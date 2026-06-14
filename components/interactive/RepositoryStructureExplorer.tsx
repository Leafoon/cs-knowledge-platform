'use client';

import React, { useState } from 'react';

interface FileNode {
  name: string;
  type: 'file' | 'directory';
  description?: string;
  children?: FileNode[];
}

export function RepositoryStructureExplorer() {
  const [expandedDirs, setExpandedDirs] = useState<Set<string>>(new Set(['tile-lang']));

  const repoStructure: FileNode = {
    name: 'tile-lang',
    type: 'directory',
    children: [
      {
        name: 'src',
        type: 'directory',
        description: '核心源代码目录',
        children: [
          { name: 'tilelang/__init__.py', type: 'file', description: '包初始化文件' },
          { name: 'tilelang/compiler.py', type: 'file', description: '编译器核心逻辑' },
          { name: 'tilelang/autotvm.py', type: 'file', description: '自动调优模块' },
          { name: 'tilelang/ir.py', type: 'file', description: 'IR定义和转换' },
        ],
      },
      {
        name: 'include',
        type: 'directory',
        description: 'C++头文件',
        children: [
          { name: 'tilelang/tir.h', type: 'file', description: 'TIR相关定义' },
          { name: 'tilelang/codegen.h', type: 'file', description: '代码生成器接口' },
        ],
      },
      {
        name: 'tests',
        type: 'directory',
        description: '测试用例',
        children: [
          { name: 'test_gemm.py', type: 'file', description: 'GEMM测试' },
          { name: 'test_conv2d.py', type: 'file', description: '卷积测试' },
          { name: 'test_attention.py', type: 'file', description: '注意力机制测试' },
        ],
      },
      {
        name: 'examples',
        type: 'directory',
        description: '示例代码',
        children: [
          { name: 'gemm_basic.py', type: 'file', description: '基础GEMM示例' },
          { name: 'gemm_optimized.py', type: 'file', description: '优化GEMM示例' },
          { name: 'flash_attention.py', type: 'file', description: 'Flash Attention示例' },
        ],
      },
      { name: 'CMakeLists.txt', type: 'file', description: 'CMake构建配置' },
      { name: 'setup.py', type: 'file', description: 'Python包安装配置' },
      { name: 'README.md', type: 'file', description: '项目说明文档' },
    ],
  };

  const toggleDir = (path: string) => {
    const newExpanded = new Set(expandedDirs);
    if (newExpanded.has(path)) {
      newExpanded.delete(path);
    } else {
      newExpanded.add(path);
    }
    setExpandedDirs(newExpanded);
  };

  const renderNode = (node: FileNode, path: string, depth: number = 0) => {
    const fullPath = `${path}/${node.name}`;
    const isExpanded = expandedDirs.has(fullPath);

    return (
      <div key={fullPath} style={{ paddingLeft: depth * 20 }}>
        <div
          className={`flex items-center gap-2 py-1 px-2 rounded hover:bg-gray-800 cursor-pointer ${
            node.type === 'directory' ? 'text-yellow-400' : 'text-gray-300'
          }`}
          onClick={() => node.type === 'directory' && toggleDir(fullPath)}
        >
          {node.type === 'directory' ? (
            <span className="text-xs">{isExpanded ? '▼' : '▶'}</span>
          ) : (
            <span className="text-xs text-gray-500">📄</span>
          )}
          <span className="font-mono text-sm">{node.name}</span>
          {node.description && (
            <span className="text-gray-500 text-xs ml-2">- {node.description}</span>
          )}
        </div>
        
        {node.type === 'directory' && isExpanded && node.children && (
          <div>
            {node.children.map((child) => renderNode(child, fullPath, depth + 1))}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="p-6 bg-gray-900 rounded-xl">
      <h2 className="text-2xl font-bold text-white mb-6">TileLang 仓库结构</h2>
      
      <div className="bg-gray-800 rounded-lg p-4 font-mono">
        {renderNode(repoStructure, '')}
      </div>
      
      <div className="mt-4 grid grid-cols-2 gap-4 text-sm">
        <div className="flex items-center gap-2">
          <span className="text-yellow-400">▶</span>
          <span className="text-gray-300">目录 (可展开)</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-gray-500">📄</span>
          <span className="text-gray-300">文件</span>
        </div>
      </div>
    </div>
  );
}
