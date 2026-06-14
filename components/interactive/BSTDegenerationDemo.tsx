'use client';
import React, { useState, useMemo } from 'react';

// ══════════════════════════════════════════════════════
//  BSTDegenerationDemo — 有序 vs 随机 插入对比
// ══════════════════════════════════════════════════════

interface TNode { key: number; id: number; left: TNode | null; right: TNode | null; }
let _uid = 0;

function mkNode(key: number): TNode { return { key, id: _uid++, left: null, right: null }; }

function insertBST(root: TNode | null, key: number): TNode {
  const n = mkNode(key);
  if (!root) return n;
  let curr = root;
  while (true) {
    if (key < curr.key) {
      if (!curr.left) { curr.left = n; break; }
      curr = curr.left;
    } else {
      if (!curr.right) { curr.right = n; break; }
      curr = curr.right;
    }
  }
  return root;
}

function buildFromSeq(seq: number[]): TNode | null {
  _uid = 0;
  let root: TNode | null = null;
  for (const k of seq) root = insertBST(root, k);
  return root;
}

function treeHeight(n: TNode | null): number {
  if (!n) return -1;
  return 1 + Math.max(treeHeight(n.left), treeHeight(n.right));
}

function treeSize(n: TNode | null): number {
  return n ? 1 + treeSize(n.left) + treeSize(n.right) : 0;
}

// 中值插入法（最平衡）
function midInsertSeq(lo: number, hi: number): number[] {
  if (lo > hi) return [];
  const mid = Math.floor((lo + hi) / 2);
  return [mid, ...midInsertSeq(lo, mid - 1), ...midInsertSeq(mid + 1, hi)];
}

interface Pos { x: number; y: number; }
function layout(root: TNode | null, W: number): Map<number, Pos> {
  const pos = new Map<number, Pos>();
  function assign(n: TNode | null, d: number, lo: number, hi: number) {
    if (!n) return;
    pos.set(n.id, { x: (lo + hi) / 2, y: 20 + d * 42 });
    assign(n.left, d + 1, lo, (lo + hi) / 2);
    assign(n.right, d + 1, (lo + hi) / 2, hi);
  }
  assign(root, 0, 0, W);
  return pos;
}

function allEdges(n: TNode | null): [number, number][] {
  if (!n) return [];
  const e: [number, number][] = [];
  if (n.left)  { e.push([n.id, n.left.id]);  e.push(...allEdges(n.left)); }
  if (n.right) { e.push([n.id, n.right.id]); e.push(...allEdges(n.right)); }
  return e;
}
function allNodes(n: TNode | null): TNode[] {
  return n ? [n, ...allNodes(n.left), ...allNodes(n.right)] : [];
}

function shuffle<T>(arr: T[]): T[] {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

const N = 10;
const ORDERED_ASC  = Array.from({ length: N }, (_, i) => i + 1);
const ORDERED_DESC = [...ORDERED_ASC].reverse();
const MID_SEQ      = midInsertSeq(1, N);

type Mode = 'asc' | 'desc' | 'random' | 'mid';
const MODES: { key: Mode; label: string; color: string }[] = [
  { key: 'asc',    label: '升序插入',   color: '#ef4444' },
  { key: 'desc',   label: '降序插入',   color: '#f97316' },
  { key: 'random', label: '随机插入',   color: '#10b981' },
  { key: 'mid',    label: '中值插入',   color: '#3b82f6' },
];

export default function BSTDegenerationDemo() {
  const [mode, setMode]       = useState<Mode>('asc');
  const [randomSeed, setSeed] = useState(42);

  const seq = useMemo(() => {
    if (mode === 'asc')    return ORDERED_ASC;
    if (mode === 'desc')   return ORDERED_DESC;
    if (mode === 'mid')    return MID_SEQ;
    return shuffle(ORDERED_ASC);
  }, [mode, randomSeed]);

  const tree = useMemo(() => buildFromSeq(seq), [seq]);
  const h    = treeHeight(tree);
  const n    = treeSize(tree);
  const modeInfo = MODES.find(m => m.key === mode)!;

  const W = 360, H = Math.max(200, (h + 1) * 44 + 40);
  const pos   = layout(tree, W);
  const edges = allEdges(tree);
  const nodes = allNodes(tree);

  const quality =
    h <= Math.ceil(Math.log2(n + 1)) + 1 ? { label: '接近最优', cls: 'text-emerald-600 dark:text-emerald-400', bg: 'bg-emerald-50 dark:bg-emerald-950/30' } :
    h <= 2 * Math.log2(n + 1)            ? { label: '尚可',     cls: 'text-amber-600 dark:text-amber-400',   bg: 'bg-amber-50 dark:bg-amber-950/30'   } :
                                            { label: '严重退化', cls: 'text-red-600 dark:text-red-400',       bg: 'bg-red-50 dark:bg-red-950/30'        };

  return (
    <div className="rounded-2xl border border-slate-200 dark:border-zinc-800 bg-slate-50 dark:bg-zinc-950 overflow-hidden text-slate-900 dark:text-white">
      {/* Header */}
      <div className="px-6 py-5 bg-white dark:bg-zinc-900 border-b border-slate-200 dark:border-zinc-800">
        <h3 className="text-xl font-bold text-sky-600 dark:text-sky-400">BST 退化演示（n = {N}）</h3>
        <p className="text-sm text-slate-500 dark:text-zinc-400 mt-1">不同插入顺序对 BST 高度的影响——高度越大，所有操作越慢</p>
      </div>

      <div className="p-6 space-y-5">
        {/* Mode selector */}
        <div className="flex flex-wrap gap-2">
          {MODES.map(m => (
            <button key={m.key}
              onClick={() => { setMode(m.key as Mode); if (m.key === 'random') setSeed(s => s + 1); }}
              className={`px-4 py-2 rounded-xl border-2 text-sm font-semibold transition-all ${
                mode === m.key ? 'text-white shadow-sm' : 'bg-white dark:bg-zinc-900 border-slate-200 dark:border-zinc-700 text-slate-600 dark:text-zinc-300 hover:border-slate-300 dark:hover:border-zinc-600'
              }`}
              style={mode === m.key ? { backgroundColor: m.color, borderColor: m.color } : {}}>
              {m.label}
              {m.key === 'random' && <span className="ml-1.5 text-xs opacity-80">（重新随机）</span>}
            </button>
          ))}
        </div>

        {/* Sequence display */}
        <div className="bg-white dark:bg-zinc-900 border border-slate-200 dark:border-zinc-700 rounded-xl px-5 py-3.5">
          <div className="text-xs font-semibold text-slate-500 dark:text-zinc-400 uppercase tracking-wide mb-2">插入顺序</div>
          <div className="flex flex-wrap gap-1.5">
            {seq.map((k, i) => (
              <span key={i} className="px-2.5 py-1 rounded-md text-sm font-mono font-semibold border"
                style={{ color: modeInfo.color, borderColor: modeInfo.color + '50', backgroundColor: modeInfo.color + '15' }}>
                {k}
              </span>
            ))}
          </div>
        </div>

        {/* Stat cards */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <div className="bg-white dark:bg-zinc-900 border border-slate-200 dark:border-zinc-700 rounded-xl p-4 text-center">
            <div className="text-xs text-slate-500 dark:text-zinc-400 font-medium mb-1">节点数 n</div>
            <div className="text-3xl font-bold text-slate-800 dark:text-zinc-100">{n}</div>
          </div>
          <div className="bg-white dark:bg-zinc-900 border border-slate-200 dark:border-zinc-700 rounded-xl p-4 text-center">
            <div className="text-xs text-slate-500 dark:text-zinc-400 font-medium mb-1">实际高度 h</div>
            <div className="text-3xl font-bold" style={{ color: modeInfo.color }}>{h}</div>
          </div>
          <div className="bg-white dark:bg-zinc-900 border border-slate-200 dark:border-zinc-700 rounded-xl p-4 text-center">
            <div className="text-xs text-slate-500 dark:text-zinc-400 font-medium mb-1">⌊log₂ n⌋</div>
            <div className="text-3xl font-bold text-slate-700 dark:text-zinc-200">{Math.floor(Math.log2(n))}</div>
          </div>
          <div className={`${quality.bg} border border-slate-200 dark:border-transparent rounded-xl p-4 text-center`}>
            <div className="text-xs text-slate-500 dark:text-zinc-400 font-medium mb-1">质量评估</div>
            <div className={`text-lg font-bold ${quality.cls}`}>{quality.label}</div>
          </div>
        </div>

        {/* SVG Tree — always dark canvas */}
        <div className="rounded-xl overflow-hidden border border-slate-300 dark:border-zinc-800 bg-slate-900 dark:bg-zinc-950">
          <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ display: 'block', minHeight: 120, maxHeight: 420 }}>
            {edges.map(([a, b], i) => {
              const pa = pos.get(a), pb = pos.get(b);
              if (!pa || !pb) return null;
              return <line key={i} x1={pa.x} y1={pa.y} x2={pb.x} y2={pb.y}
                stroke={modeInfo.color + '55'} strokeWidth="1.5" />;
            })}
            {nodes.map(nd => {
              const p = pos.get(nd.id);
              if (!p) return null;
              return (
                <g key={nd.id}>
                  <circle cx={p.x} cy={p.y} r={15}
                    fill={modeInfo.color + '25'} stroke={modeInfo.color} strokeWidth="1.5" />
                  <text x={p.x} y={p.y + 4} textAnchor="middle" fontSize="11" fontWeight="bold"
                    fill={modeInfo.color}>{nd.key}</text>
                </g>
              );
            })}
          </svg>
        </div>

        {/* Comparison table */}
        <div className="bg-white dark:bg-zinc-900 border border-slate-200 dark:border-zinc-700 rounded-xl overflow-hidden">
          <div className="px-5 py-3 bg-slate-50 dark:bg-zinc-800/50 border-b border-slate-200 dark:border-zinc-700">
            <span className="text-sm font-semibold text-slate-700 dark:text-zinc-200">四种插入方式高度对比（n = {N}）</span>
          </div>
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-200 dark:border-zinc-700">
                {['方式', '高度 h', '操作复杂度', '特点'].map(h => (
                  <th key={h} className="px-5 py-3 text-left text-xs font-semibold text-slate-500 dark:text-zinc-400 uppercase tracking-wide">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {[
                { label: '升序', h: N-1, op: `O(n) = O(${N})`, note: '退化为链表（最坏）', color: '#ef4444' },
                { label: '降序', h: N-1, op: `O(n) = O(${N})`, note: '退化为链表（最坏）', color: '#f97316' },
                { label: '随机', h: Math.round(2.88*Math.log2(N)), op: `E[O(log n)]≈O(${Math.round(Math.log2(N))})`, note: '期望 O(log n)', color: '#10b981' },
                { label: '中值', h: Math.floor(Math.log2(N)), op: `O(log n)=O(${Math.floor(Math.log2(N))})`, note: '高度最小（完美平衡）', color: '#3b82f6' },
              ].map(r => (
                <tr key={r.label} className="border-b border-slate-100 dark:border-zinc-800 last:border-0">
                  <td className="px-5 py-3 font-semibold text-slate-700 dark:text-zinc-300">{r.label}</td>
                  <td className="px-5 py-3 font-bold font-mono" style={{ color: r.color }}>{r.h}</td>
                  <td className="px-5 py-3 font-mono text-xs" style={{ color: r.color }}>{r.op}</td>
                  <td className="px-5 py-3 text-slate-500 dark:text-zinc-400">{r.note}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="text-sm text-slate-500 dark:text-zinc-500 text-center px-4">
          结论：普通 BST 对输入顺序敏感，需要平衡机制（AVL / 红黑树）保证 O(log n) 最坏复杂度。
        </div>
      </div>
    </div>
  );
}
