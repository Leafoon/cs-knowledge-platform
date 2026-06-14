'use client';

import { useState } from 'react';

const treeData = {
  name: 'Root',
  type: 'PrimFunc',
  children: [
    {
      name: 'for bx',
      type: 'ThreadBlock',
      children: [
        {
          name: 'for ko',
          type: 'SharedMemory',
          children: [
            {
              name: 'block A',
              type: 'Load',
              children: [
                { name: 'ld.shared', type: 'Inst', children: [] },
                { name: 'bar.sync', type: 'Sync', children: [] },
              ],
            },
            {
              name: 'block B',
              type: 'Compute',
              children: [
                { name: 'mma.sync', type: 'Inst', children: [] },
                { name: 'bar.sync', type: 'Sync', children: [] },
              ],
            },
            {
              name: 'block C',
              type: 'Store',
              children: [
                { name: 'st.shared', type: 'Inst', children: [] },
              ],
            },
          ],
        },
        {
          name: 'for ki',
          type: 'Register',
          children: [
            { name: 'load reg', type: 'Inst', children: [] },
            { name: 'compute', type: 'Inst', children: [] },
          ],
        },
      ],
    },
  ],
};

const typeColors: Record<string, string> = {
  PrimFunc: '#6B7280', ThreadBlock: '#3B82F6', SharedMemory: '#F59E0B',
  Register: '#8B5CF6', Load: '#10B981', Compute: '#EF4444', Store: '#EC4899',
  Inst: '#9CA3AF', Sync: '#FB923C',
};

function TreeNode({ node, depth = 0 }: { node: typeof treeData; depth?: number }) {
  const [expanded, setExpanded] = useState(depth < 2);
  const hasChildren = node.children.length > 0;

  return (
    <div style={{ marginLeft: depth > 0 ? '20px' : '0' }}>
      <div
        className="flex items-center gap-2 py-1 px-2 rounded cursor-pointer hover:bg-gray-800/50 transition-colors"
        onClick={() => hasChildren && setExpanded(!expanded)}>
        {hasChildren && (
          <span className="text-gray-500 text-xs w-3">{expanded ? '▼' : '▶'}</span>
        )}
        {!hasChildren && <span className="w-3" />}
        <div className="w-2 h-2 rounded-full" style={{ backgroundColor: typeColors[node.type] }} />
        <span className="text-sm font-mono" style={{ color: typeColors[node.type] }}>{node.name}</span>
        <span className="text-[10px] text-gray-500">{node.type}</span>
      </div>
      {expanded && node.children.map((child, i) => (
        <TreeNode key={i} node={child} depth={depth + 1} />
      ))}
    </div>
  );
}

export default function BlockLoopStructure() {
  return (
    <div className="p-6 bg-gray-900 rounded-xl text-white">
      <h2 className="text-xl font-bold mb-2">TensorIR Block/Loop 层次结构</h2>
      <p className="text-sm text-gray-400 mb-4">GEMM 内核的嵌套 Block-Loop 组织</p>

      <div className="flex gap-4">
        <div className="flex-1 bg-black rounded-lg p-3">
          <TreeNode node={treeData} />
        </div>

        <div className="w-52 space-y-3 text-xs">
          <div className="bg-gray-800 rounded p-3">
            <div className="font-bold text-white mb-2">层级说明</div>
            {Object.entries(typeColors).map(([type, color]) => (
              <div key={type} className="flex items-center gap-2 py-0.5">
                <div className="w-2 h-2 rounded-full" style={{ backgroundColor: color }} />
                <span className="text-gray-400">{type}</span>
              </div>
            ))}
          </div>
          <div className="bg-gray-800 rounded p-3">
            <div className="font-bold text-blue-400 mb-1">Block 语义</div>
            <div className="text-gray-400">
              Block = 一个独立计算单元<br/>
              包含访问模式、约束和计算逻辑
            </div>
          </div>
          <div className="bg-gray-800 rounded p-3">
            <div className="font-bold text-purple-400 mb-1">Loop 语义</div>
            <div className="text-gray-400">
              Loop = 循环遍历空间<br/>
              可并行化、向量化、展开
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
