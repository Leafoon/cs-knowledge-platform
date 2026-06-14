'use client';
import React, { useState } from 'react';

// ══════════════════════════════════════════════════════
//  BSTOperationVisualizer — BST 搜索/插入/删除 动画  (light/dark redesign)
// ══════════════════════════════════════════════════════

interface TNode { key: number; id: number; left: TNode | null; right: TNode | null; }

let _id = 0;
function mk(key: number): TNode { return { key, id: _id++, left: null, right: null }; }

// 纯操作（不可变风格）────────────────────────────
function bstInsert(root: TNode | null, key: number): { root: TNode; path: number[] } {
  const path: number[] = [];
  const newNode = mk(key);

  if (!root) return { root: newNode, path: [] };

  // Find insertion point
  let curr: TNode = root;
  const parents: { node: TNode; dir: 'left' | 'right' }[] = [];
  while (true) {
    path.push(curr.id);
    if (key < curr.key) {
      if (!curr.left)  { curr.left  = newNode; break; }
      parents.push({ node: curr, dir: 'left' }); curr = curr.left;
    } else if (key > curr.key) {
      if (!curr.right) { curr.right = newNode; break; }
      parents.push({ node: curr, dir: 'right' }); curr = curr.right;
    } else break; // already exists
  }
  path.push(newNode.id);
  return { root, path };
}

function bstSearch(root: TNode | null, key: number): { found: TNode | null; path: number[] } {
  const path: number[] = [];
  let curr = root;
  while (curr) {
    path.push(curr.id);
    if (key === curr.key) return { found: curr, path };
    curr = key < curr.key ? curr.left : curr.right;
  }
  return { found: null, path };
}

function bstMinNode(n: TNode): TNode {
  while (n.left) n = n.left;
  return n;
}

function cloneTree(n: TNode | null): TNode | null {
  if (!n) return null;
  return { ...n, left: cloneTree(n.left), right: cloneTree(n.right) };
}

function bstDelete(root: TNode | null, key: number): { root: TNode | null; path: number[]; caseNum: 0|1|2|3 } {
  const path: number[] = [];
  let caseNum: 0|1|2|3 = 0;

  function del(n: TNode | null, k: number): TNode | null {
    if (!n) return null;
    path.push(n.id);
    if (k < n.key) { n.left  = del(n.left, k);  return n; }
    if (k > n.key) { n.right = del(n.right, k); return n; }
    // Found
    if (!n.left && !n.right) { caseNum = 1; return null; }
    if (!n.left)  { caseNum = 2; return n.right; }
    if (!n.right) { caseNum = 2; return n.left; }
    caseNum = 3;
    const succ = bstMinNode(n.right);
    n.key = succ.key;
    n.right = del(n.right, succ.key);
    return n;
  }

  const newRoot = del(cloneTree(root), key);
  return { root: newRoot, path, caseNum };
}

// ── 布局 ──────────────────────────────────────────────
interface Pos { x: number; y: number; }
function layout(root: TNode | null): Map<number, Pos> {
  const pos = new Map<number, Pos>();
  function assign(n: TNode | null, d: number, lo: number, hi: number) {
    if (!n) return;
    pos.set(n.id, { x: (lo + hi) / 2, y: 32 + d * 56 });
    assign(n.left, d + 1, lo, (lo + hi) / 2);
    assign(n.right, d + 1, (lo + hi) / 2, hi);
  }
  assign(root, 0, 0, 440);
  return pos;
}

function allNodes(n: TNode | null): TNode[] {
  return n ? [n, ...allNodes(n.left), ...allNodes(n.right)] : [];
}
function allEdges(n: TNode | null): [number, number][] {
  if (!n) return [];
  const e: [number, number][] = [];
  if (n.left)  { e.push([n.id, n.left.id]);  e.push(...allEdges(n.left)); }
  if (n.right) { e.push([n.id, n.right.id]); e.push(...allEdges(n.right)); }
  return e;
}

function inorderKeys(n: TNode | null): number[] {
  if (!n) return [];
  return [...inorderKeys(n.left), n.key, ...inorderKeys(n.right)];
}

// ── 初始树 ────────────────────────────────────────────
function makeInitTree(): TNode {
  _id = 0;
  const root = mk(15);
  const keys = [6, 18, 3, 7, 17, 20, 2, 4, 13, 9];
  let r: TNode | null = root;
  for (const k of keys) { bstInsert(r, k); }
  return root;
}

const CASE_DESC: Record<number, string> = {
  0: '',
  1: '情形 1：叶节点，直接删除',
  2: '情形 2：单子节点，子节点上移',
  3: '情形 3：双子节点，用中序后继替换',
};

const OP_CONFIG = {
  search: { label: '搜索', icon: '⌕', color: '#3b82f6' },
  insert: { label: '插入', icon: '+', color: '#10b981' },
  delete: { label: '删除', icon: '✕', color: '#ef4444' },
} as const;

export default function BSTOperationVisualizer() {
  const [tree, setTree]       = useState<TNode | null>(() => makeInitTree());
  const [op, setOp]           = useState<'search' | 'insert' | 'delete'>('search');
  const [inputKey, setInputKey] = useState('');
  const [result, setResult]   = useState<{
    path: number[]; found: boolean | null; msg: string; caseNum?: 0|1|2|3
  } | null>(null);

  const pos     = layout(tree);
  const pathSet = new Set(result?.path ?? []);
  const cfg     = OP_CONFIG[op];
  const sorted  = inorderKeys(tree);

  const handleOp = () => {
    const k = parseInt(inputKey.trim());
    if (isNaN(k)) return;
    setResult(null);

    if (op === 'search') {
      const { found, path } = bstSearch(tree, k);
      setResult({ path, found: found !== null, msg: found ? `找到节点 ${k}` : `未找到节点 ${k}` });
    } else if (op === 'insert') {
      if (bstSearch(tree, k).found) {
        setResult({ path: bstSearch(tree, k).path, found: null, msg: `键 ${k} 已存在` });
        return;
      }
      const { root: newRoot, path } = bstInsert(cloneTree(tree), k);
      setTree(newRoot);
      setResult({ path, found: null, msg: `成功插入 ${k}，路径深度 ${path.length - 1}` });
    } else {
      const { found: fnd } = bstSearch(tree, k);
      if (!fnd) { setResult({ path: [], found: false, msg: `键 ${k} 不存在` }); return; }
      const { root: newRoot, path, caseNum } = bstDelete(cloneTree(tree), k);
      setTree(newRoot);
      setResult({ path, found: null, caseNum, msg: `已删除节点 ${k}` });
    }
  };

  const resetTree = () => { _id = 0; setTree(makeInitTree()); setResult(null); setInputKey(''); };

  return (
    <div className="rounded-2xl border border-slate-200 dark:border-zinc-800 bg-slate-50 dark:bg-zinc-950 overflow-hidden text-slate-900 dark:text-white">
      {/* Header */}
      <div className="px-6 py-5 bg-white dark:bg-zinc-900 border-b border-slate-200 dark:border-zinc-800 flex items-start justify-between">
        <div>
          <h3 className="text-xl font-bold text-sky-600 dark:text-sky-400">二叉搜索树操作演示</h3>
          <p className="text-sm text-slate-500 dark:text-zinc-400 mt-1">交互式演示 BST 搜索、插入与删除，高亮显示完整访问路径</p>
        </div>
        <button onClick={resetTree}
          className="mt-1 px-3 py-1.5 rounded-lg text-sm bg-slate-100 dark:bg-zinc-800 hover:bg-slate-200 dark:hover:bg-zinc-700 text-slate-500 dark:text-zinc-400 transition-colors border border-slate-200 dark:border-zinc-700">
          ↺ 重置
        </button>
      </div>

      <div className="p-6 space-y-5">
        {/* Controls */}
        <div className="flex flex-wrap items-center gap-3">
          {/* Op tabs */}
          <div className="flex bg-slate-100 dark:bg-zinc-800 rounded-xl p-1 gap-1">
            {(['search', 'insert', 'delete'] as const).map(o => {
              const c = OP_CONFIG[o];
              return (
                <button key={o} onClick={() => { setOp(o); setResult(null); }}
                  className={`px-4 py-2 rounded-lg text-sm font-semibold transition-all ${op === o ? 'text-white shadow-sm' : 'text-slate-500 dark:text-zinc-400 hover:text-slate-700 dark:hover:text-zinc-200'}`}
                  style={op === o ? { backgroundColor: c.color } : {}}>
                  {c.icon} {c.label}
                </button>
              );
            })}
          </div>
          {/* Input */}
          <input type="number" value={inputKey}
            onChange={e => setInputKey(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleOp()}
            placeholder="键值"
            className="px-4 py-2 rounded-xl bg-white dark:bg-zinc-800 border border-slate-300 dark:border-zinc-600 text-sm w-28 focus:outline-none focus:ring-2 focus:ring-sky-400 dark:focus:ring-sky-600 text-slate-800 dark:text-zinc-100 placeholder-slate-400" />
          <button onClick={handleOp}
            className="px-5 py-2 rounded-xl text-sm font-semibold text-white transition-all hover:opacity-90 shadow-sm"
            style={{ backgroundColor: cfg.color }}>
            执行
          </button>
        </div>

        {/* Result banner */}
        {result && (
          <div className={`px-4 py-3 rounded-xl border text-sm font-medium transition-all ${
            result.found === false
              ? 'bg-red-50 dark:bg-red-950/30 border-red-200 dark:border-red-800 text-red-700 dark:text-red-400'
              : result.found === true
              ? 'bg-blue-50 dark:bg-blue-950/30 border-blue-200 dark:border-blue-800 text-blue-700 dark:text-blue-400'
              : 'bg-slate-100 dark:bg-zinc-800 border-slate-200 dark:border-zinc-700 text-slate-700 dark:text-zinc-200'
          }`}>
            <span>{result.msg}</span>
            {result.caseNum && result.caseNum > 0 && (
              <span className="ml-3 text-xs opacity-70 font-normal">{CASE_DESC[result.caseNum]}</span>
            )}
            {result.path.length > 0 && (
              <div className="mt-1 text-xs opacity-60 font-normal">访问节点数：{result.path.length}</div>
            )}
          </div>
        )}

        {/* SVG Tree */}
        <div className="rounded-xl overflow-hidden border border-slate-300 dark:border-zinc-800 bg-slate-900 dark:bg-zinc-950">
          <svg width="100%" viewBox="0 0 440 320" style={{ display: 'block' }}>
            {allEdges(tree).map(([a, b], i) => {
              const pa = pos.get(a), pb = pos.get(b);
              if (!pa || !pb) return null;
              const onPath = pathSet.has(a) && pathSet.has(b);
              return (
                <line key={i} x1={pa.x} y1={pa.y} x2={pb.x} y2={pb.y}
                  stroke={onPath ? cfg.color : '#374151'} strokeWidth={onPath ? 2.5 : 1.5}
                  strokeLinecap="round" />
              );
            })}
            {allNodes(tree).map(n => {
              const p = pos.get(n.id);
              if (!p) return null;
              const onPath = pathSet.has(n.id);
              const isLast = result?.path.length ? n.id === result.path[result.path.length - 1] : false;
              const fill   = isLast ? cfg.color : onPath ? cfg.color + '35' : '#1f2937';
              const stroke = onPath ? cfg.color : '#374151';
              return (
                <g key={n.id}>
                  <circle cx={p.x} cy={p.y} r={20} fill={fill} stroke={stroke} strokeWidth={isLast ? 2.5 : 1.5} />
                  <text x={p.x} y={p.y + 5} textAnchor="middle"
                    fontSize={n.key >= 100 ? '10' : '12'} fontWeight="bold"
                    fill={onPath ? '#f9fafb' : '#9ca3af'}>{n.key}</text>
                </g>
              );
            })}
          </svg>
        </div>

        {/* Inorder */}
        <div className="bg-white dark:bg-zinc-900 border border-slate-200 dark:border-zinc-700 rounded-xl px-5 py-4">
          <div className="text-xs font-semibold text-slate-500 dark:text-zinc-400 uppercase tracking-wide mb-3">中序遍历（有序输出验证）</div>
          <div className="flex flex-wrap gap-1.5">
            {sorted.map((k, i) => (
              <span key={i}
                className="px-2.5 py-1 rounded-md text-sm font-mono font-semibold bg-slate-100 dark:bg-zinc-800 border border-slate-200 dark:border-zinc-700 text-slate-700 dark:text-zinc-300">
                {k}
              </span>
            ))}
          </div>
        </div>

        {/* Complexity cards */}
        <div className="grid grid-cols-3 gap-3">
          {[
            { label: 'SEARCH', wc: 'O(h)', avg: 'O(log n)', color: '#3b82f6', border: 'border-blue-200 dark:border-blue-900' },
            { label: 'INSERT', wc: 'O(h)', avg: 'O(log n)', color: '#10b981', border: 'border-emerald-200 dark:border-emerald-900' },
            { label: 'DELETE', wc: 'O(h)', avg: 'O(log n)', color: '#ef4444', border: 'border-red-200 dark:border-red-900' },
          ].map(r => (
            <div key={r.label} className={`bg-white dark:bg-zinc-900 border ${r.border} rounded-xl p-4 text-center`}>
              <div className="text-sm font-bold font-mono" style={{ color: r.color }}>{r.label}</div>
              <div className="mt-2.5 space-y-1.5">
                <div className="text-xs text-slate-500 dark:text-zinc-400">
                  最坏 <span className="font-mono font-semibold text-slate-700 dark:text-zinc-200">{r.wc}</span>
                </div>
                <div className="text-xs text-slate-500 dark:text-zinc-400">
                  随机 <span className="font-mono font-semibold text-slate-700 dark:text-zinc-200">{r.avg}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
