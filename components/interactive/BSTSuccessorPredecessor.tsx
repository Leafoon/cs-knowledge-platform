'use client';
import React, { useState, useMemo } from 'react';

// ══════════════════════════════════════════════════════
//  BSTSuccessorPredecessor — 前驱 / 后继 路径可视化
// ══════════════════════════════════════════════════════

interface TNode { key: number; id: number; parent: TNode | null; left: TNode | null; right: TNode | null; }

function buildBST(keys: number[]): { root: TNode | null; nodeMap: Map<number, TNode> } {
  const nodeMap = new Map<number, TNode>();
  let root: TNode | null = null;
  for (const k of keys) {
    const nd: TNode = { key: k, id: k, parent: null, left: null, right: null };
    nodeMap.set(k, nd);
    if (!root) { root = nd; continue; }
    let curr = root;
    while (true) {
      if (k < curr.key) {
        if (!curr.left)  { curr.left  = nd; nd.parent = curr; break; }
        curr = curr.left;
      } else {
        if (!curr.right) { curr.right = nd; nd.parent = curr; break; }
        curr = curr.right;
      }
    }
  }
  return { root, nodeMap };
}

function treeMinimum(n: TNode): TNode { while (n.left) n = n.left; return n; }
function treeMaximum(n: TNode): TNode { while (n.right) n = n.right; return n; }

interface SuccResult { node: TNode; path: number[]; case: 1 | 2 }
function successor(n: TNode): SuccResult | null {
  if (n.right) {
    const path: number[] = [n.id];
    let x = n.right; path.push(x.id);
    while (x.left) { x = x.left; path.push(x.id); }
    return { node: x, path, case: 1 };
  }
  const path: number[] = [n.id];
  let x = n, y = n.parent;
  while (y && x === y.right) { path.push(y.id); x = y; y = y.parent; }
  if (!y) return null;
  path.push(y.id);
  return { node: y, path, case: 2 };
}

interface PredResult { node: TNode; path: number[]; case: 1 | 2 }
function predecessor(n: TNode): PredResult | null {
  if (n.left) {
    const path: number[] = [n.id];
    let x = n.left; path.push(x.id);
    while (x.right) { x = x.right; path.push(x.id); }
    return { node: x, path, case: 1 };
  }
  const path: number[] = [n.id];
  let x = n, y = n.parent;
  while (y && x === y.left) { path.push(y.id); x = y; y = y.parent; }
  if (!y) return null;
  path.push(y.id);
  return { node: y, path, case: 2 };
}

function layout(root: TNode | null, W: number): Map<number, { x: number; y: number }> {
  const pos = new Map<number, { x: number; y: number }>();
  function assign(n: TNode | null, d: number, lo: number, hi: number) {
    if (!n) return;
    pos.set(n.id, { x: (lo + hi) / 2, y: 24 + d * 48 });
    assign(n.left,  d + 1, lo, (lo + hi) / 2);
    assign(n.right, d + 1, (lo + hi) / 2, hi);
  }
  assign(root, 0, 0, W);
  return pos;
}

function collectEdges(n: TNode | null): [number, number][] {
  if (!n) return [];
  const e: [number, number][] = [];
  if (n.left)  { e.push([n.id, n.left.id]);  e.push(...collectEdges(n.left)); }
  if (n.right) { e.push([n.id, n.right.id]); e.push(...collectEdges(n.right)); }
  return e;
}
function collectNodes(n: TNode | null): TNode[] {
  return n ? [n, ...collectNodes(n.left), ...collectNodes(n.right)] : [];
}

const DEFAULT_KEYS = [20, 10, 30, 5, 15, 25, 35, 3, 8, 27, 33];

export default function BSTSuccessorPredecessor() {
  const [selectedKey, setSelectedKey] = useState<number | null>(15);
  const [showMode, setShowMode]       = useState<'both' | 'succ' | 'pred'>('both');

  const { root, nodeMap } = useMemo(() => buildBST(DEFAULT_KEYS), []);

  const W = 440, H = 290;
  const pos   = useMemo(() => layout(root, W), [root]);
  const edges = useMemo(() => collectEdges(root), [root]);
  const nodes = useMemo(() => collectNodes(root), [root]);

  const selNode  = selectedKey !== null ? nodeMap.get(selectedKey) ?? null : null;
  const succRes  = selNode ? successor(selNode)   : null;
  const predRes  = selNode ? predecessor(selNode) : null;

  const succPath = new Set(succRes?.path ?? []);
  const predPath = new Set(predRes?.path ?? []);

  function nodeColor(id: number) {
    if (id === selectedKey)        return '#facc15';
    if (showMode !== 'pred' && succRes && id === succRes.node.id) return '#10b981';
    if (showMode !== 'succ' && predRes && id === predRes.node.id)  return '#a855f7';
    if (showMode !== 'pred' && succPath.has(id) && id !== selectedKey) return '#10b98166';
    if (showMode !== 'succ' && predPath.has(id) && id !== selectedKey)  return '#a855f766';
    return null;
  }

  function edgeColor(a: number, b: number) {
    if (showMode !== 'pred' && succPath.has(a) && succPath.has(b)) return '#10b981';
    if (showMode !== 'succ' && predPath.has(a) && predPath.has(b))  return '#a855f7';
    return null;
  }

  return (
    <div className="rounded-2xl border border-slate-200 dark:border-zinc-800 bg-slate-50 dark:bg-zinc-950 overflow-hidden text-slate-900 dark:text-white">
      {/* Header */}
      <div className="px-6 py-5 bg-white dark:bg-zinc-900 border-b border-slate-200 dark:border-zinc-800">
        <h3 className="text-xl font-bold text-sky-600 dark:text-sky-400">BST 前驱 & 后继</h3>
        <p className="text-sm text-slate-500 dark:text-zinc-400 mt-1">点击节点查看其前驱（紫色）与后继（绿色）的查找路径与算法情形</p>
      </div>

      <div className="p-6 space-y-5">
        {/* Mode switcher */}
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-sm text-slate-500 dark:text-zinc-400 font-medium">显示模式：</span>
          <div className="flex bg-slate-100 dark:bg-zinc-800 rounded-xl p-1 gap-1">
            {([['both','前驱 + 后继'],['succ','仅后继'],['pred','仅前驱']] as const).map(([k, label]) => (
              <button key={k} onClick={() => setShowMode(k)}
                className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                  showMode === k
                    ? k === 'succ' ? 'bg-emerald-600 text-white shadow-sm'
                    : k === 'pred' ? 'bg-purple-600 text-white shadow-sm'
                    : 'bg-sky-600 text-white shadow-sm'
                    : 'text-slate-500 dark:text-zinc-400 hover:text-slate-700 dark:hover:text-zinc-200'
                }`}>
                {label}
              </button>
            ))}
          </div>
        </div>

        {/* SVG Tree — always dark canvas */}
        <div className="rounded-xl overflow-hidden border border-slate-300 dark:border-zinc-800 bg-slate-900 dark:bg-zinc-950">
          <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ display: 'block' }}>
            {edges.map(([a, b], i) => {
              const pa = pos.get(a), pb = pos.get(b);
              if (!pa || !pb) return null;
              const ec = edgeColor(a, b);
              return (
                <line key={i} x1={pa.x} y1={pa.y} x2={pb.x} y2={pb.y}
                  stroke={ec ?? '#374151'} strokeWidth={ec ? 2.5 : 1.5}
                  strokeLinecap="round" />
              );
            })}
            {nodes.map(nd => {
              const p = pos.get(nd.id);
              if (!p) return null;
              const nc = nodeColor(nd.id);
              const isSelected = nd.id === selectedKey;
              return (
                <g key={nd.id} className="cursor-pointer" onClick={() => setSelectedKey(nd.key)}>
                  <circle cx={p.x} cy={p.y} r={20}
                    fill={nc ? nc + '30' : '#1f2937'}
                    stroke={nc ?? '#374151'}
                    strokeWidth={isSelected ? 3 : 1.8} />
                  <text x={p.x} y={p.y + 5} textAnchor="middle" fontSize="12" fontWeight="bold"
                    fill={nc ?? '#6b7280'}>{nd.key}</text>
                </g>
              );
            })}
          </svg>
        </div>

        {/* Legend */}
        <div className="flex flex-wrap gap-4 text-sm">
          <span className="flex items-center gap-1.5">
            <span className="w-3.5 h-3.5 rounded-full inline-block" style={{background:'#facc15'}}></span>
            <span className="text-slate-600 dark:text-zinc-400">选中节点</span>
          </span>
          {showMode !== 'pred' && (
            <span className="flex items-center gap-1.5">
              <span className="w-3.5 h-3.5 rounded-full inline-block" style={{background:'#10b981'}}></span>
              <span className="text-slate-600 dark:text-zinc-400">后继 / 路径</span>
            </span>
          )}
          {showMode !== 'succ' && (
            <span className="flex items-center gap-1.5">
              <span className="w-3.5 h-3.5 rounded-full inline-block" style={{background:'#a855f7'}}></span>
              <span className="text-slate-600 dark:text-zinc-400">前驱 / 路径</span>
            </span>
          )}
        </div>

        {/* Detail cards */}
        {selNode && (
          <div className="grid md:grid-cols-2 gap-4">
            {/* Successor */}
            {showMode !== 'pred' && (
              <div className="bg-white dark:bg-zinc-900 border border-emerald-200 dark:border-emerald-900 rounded-xl p-5">
                <div className="flex items-center gap-2 mb-3">
                  <div className="w-2.5 h-2.5 rounded-full bg-emerald-500"></div>
                  <span className="font-bold text-emerald-600 dark:text-emerald-400">后继（Successor）</span>
                </div>
                {succRes ? (
                  <div className="space-y-2">
                    <div className="text-sm text-slate-700 dark:text-zinc-300">
                      节点 <span className="font-bold text-amber-500">{selNode.key}</span> 的后继 ={' '}
                      <span className="font-bold text-emerald-600 dark:text-emerald-400 text-base">{succRes.node.key}</span>
                    </div>
                    <div className={`text-xs px-3 py-2 rounded-lg ${
                      succRes.case === 1
                        ? 'bg-emerald-50 dark:bg-emerald-950/30 text-emerald-700 dark:text-emerald-400'
                        : 'bg-slate-50 dark:bg-zinc-800 text-slate-600 dark:text-zinc-400'
                    }`}>
                      {succRes.case === 1
                        ? '情形 1：有右子树 → MINIMUM(right)'
                        : '情形 2：无右子树 → 向上走至首个左祖先'}
                    </div>
                    <div className="text-xs text-slate-500 dark:text-zinc-500 font-mono">
                      路径：{succRes.path.map(id => nodeMap.get(id)?.key ?? '?').join(' → ')}
                    </div>
                  </div>
                ) : (
                  <div className="text-sm text-slate-500 dark:text-zinc-500">无后继（最大元素）</div>
                )}
              </div>
            )}

            {/* Predecessor */}
            {showMode !== 'succ' && (
              <div className="bg-white dark:bg-zinc-900 border border-purple-200 dark:border-purple-900 rounded-xl p-5">
                <div className="flex items-center gap-2 mb-3">
                  <div className="w-2.5 h-2.5 rounded-full bg-purple-500"></div>
                  <span className="font-bold text-purple-600 dark:text-purple-400">前驱（Predecessor）</span>
                </div>
                {predRes ? (
                  <div className="space-y-2">
                    <div className="text-sm text-slate-700 dark:text-zinc-300">
                      节点 <span className="font-bold text-amber-500">{selNode.key}</span> 的前驱 ={' '}
                      <span className="font-bold text-purple-600 dark:text-purple-400 text-base">{predRes.node.key}</span>
                    </div>
                    <div className={`text-xs px-3 py-2 rounded-lg ${
                      predRes.case === 1
                        ? 'bg-purple-50 dark:bg-purple-950/30 text-purple-700 dark:text-purple-400'
                        : 'bg-slate-50 dark:bg-zinc-800 text-slate-600 dark:text-zinc-400'
                    }`}>
                      {predRes.case === 1
                        ? '情形 1：有左子树 → MAXIMUM(left)'
                        : '情形 2：无左子树 → 向上走至首个右祖先'}
                    </div>
                    <div className="text-xs text-slate-500 dark:text-zinc-500 font-mono">
                      路径：{predRes.path.map(id => nodeMap.get(id)?.key ?? '?').join(' → ')}
                    </div>
                  </div>
                ) : (
                  <div className="text-sm text-slate-500 dark:text-zinc-500">无前驱（最小元素）</div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Pseudocode panel */}
        <div className="rounded-xl overflow-hidden border border-slate-200 dark:border-zinc-700">
          <div className="flex items-center gap-3 px-5 py-3 bg-slate-100 dark:bg-zinc-900 border-b border-slate-200 dark:border-zinc-700">
            <div className="flex gap-1.5">
              <span className="w-3 h-3 rounded-full bg-red-400/80" />
              <span className="w-3 h-3 rounded-full bg-amber-400/80" />
              <span className="w-3 h-3 rounded-full bg-emerald-400/80" />
            </div>
            <span className="text-xs font-mono text-slate-500 dark:text-zinc-400">CLRS 12.2 — Successor / Predecessor</span>
          </div>
          <div className="bg-slate-950 dark:bg-zinc-950 p-5 grid md:grid-cols-2 gap-6 font-mono text-xs">
            <div className="space-y-0.5">
              <div className="text-emerald-400 font-bold mb-2">TREE-SUCCESSOR(x):</div>
              <div className="text-slate-400">if x.right ≠ NIL</div>
              <div className="text-slate-400 pl-4">return MINIMUM(x.right) <span className="text-slate-600">{'// case 1'}</span></div>
              <div className="text-slate-400">y = x.parent</div>
              <div className="text-slate-400">while y ≠ NIL and x == y.right</div>
              <div className="text-slate-400 pl-4">x = y; y = y.parent <span className="text-slate-600">{'// case 2'}</span></div>
              <div className="text-slate-400">return y</div>
            </div>
            <div className="space-y-0.5">
              <div className="text-purple-400 font-bold mb-2">TREE-PREDECESSOR(x):</div>
              <div className="text-slate-400">if x.left ≠ NIL</div>
              <div className="text-slate-400 pl-4">return MAXIMUM(x.left) <span className="text-slate-600">{'// case 1'}</span></div>
              <div className="text-slate-400">y = x.parent</div>
              <div className="text-slate-400">while y ≠ NIL and x == y.left</div>
              <div className="text-slate-400 pl-4">x = y; y = y.parent <span className="text-slate-600">{'// case 2'}</span></div>
              <div className="text-slate-400">return y</div>
            </div>
            <div className="md:col-span-2 text-slate-600 pt-1 border-t border-slate-800">
              时间复杂度：O(h)，最坏 O(n)，平衡 BST 下 O(log n)
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
