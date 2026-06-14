'use client';
import React, { useState } from 'react';

// ══════════════════════════════════════════════════════
//  TreeDefinitionAnimator  —  树的术语与性质可视化
// ══════════════════════════════════════════════════════

interface TNode { val: number; left: TNode | null; right: TNode | null; id: number; }
let _gid = 0;
function makeNode(val: number, l: TNode | null = null, r: TNode | null = null): TNode {
  return { val, left: l, right: r, id: _gid++ };
}

_gid = 0;
const DEMO_TREE: TNode = makeNode(8,
  makeNode(3,
    makeNode(1),
    makeNode(6, makeNode(4), makeNode(7))
  ),
  makeNode(10,
    null,
    makeNode(14, makeNode(13), null)
  )
);

// 计算每个节点的深度和高度
function computeDepths(root: TNode | null): Map<number, number> {
  const map = new Map<number, number>();
  function dfs(n: TNode | null, d: number) {
    if (!n) return;
    map.set(n.id, d);
    dfs(n.left, d + 1);
    dfs(n.right, d + 1);
  }
  dfs(root, 0);
  return map;
}

function computeHeights(root: TNode | null): Map<number, number> {
  const map = new Map<number, number>();
  function dfs(n: TNode | null): number {
    if (!n) return -1;
    const h = 1 + Math.max(dfs(n.left), dfs(n.right));
    map.set(n.id, h);
    return h;
  }
  dfs(root);
  return map;
}

interface Pos { x: number; y: number; }
function layout(root: TNode | null): Map<number, Pos> {
  const pos = new Map<number, Pos>();
  function assign(n: TNode | null, d: number, lo: number, hi: number) {
    if (!n) return;
    const x = (lo + hi) / 2;
    pos.set(n.id, { x, y: 32 + d * 60 });
    assign(n.left, d + 1, lo, x);
    assign(n.right, d + 1, x, hi);
  }
  assign(root, 0, 0, 440);
  return pos;
}

interface Edge { from: number; to: number; }
function collectAll(n: TNode | null): TNode[] {
  return n ? [n, ...collectAll(n.left), ...collectAll(n.right)] : [];
}
function collectEdges(n: TNode | null): Edge[] {
  if (!n) return [];
  const edges: Edge[] = [];
  if (n.left)  { edges.push({ from: n.id, to: n.left.id });  edges.push(...collectEdges(n.left)); }
  if (n.right) { edges.push({ from: n.id, to: n.right.id }); edges.push(...collectEdges(n.right)); }
  return edges;
}

// 节点属性信息
interface NodeInfo {
  id: number;
  val: number;
  depth: number;
  height: number;
  type: 'root' | 'leaf' | 'internal';
  parent: number | null;
  children: number[];
}

function buildInfoMap(root: TNode | null): Map<number, NodeInfo> {
  const depths  = computeDepths(root);
  const heights = computeHeights(root);
  const map     = new Map<number, NodeInfo>();
  const nodes   = collectAll(root);

  // build parent map
  const parentMap = new Map<number, number | null>();
  function assignParent(n: TNode | null, parentId: number | null) {
    if (!n) return;
    parentMap.set(n.id, parentId);
    assignParent(n.left, n.id);
    assignParent(n.right, n.id);
  }
  assignParent(root, null);

  nodes.forEach(n => {
    const children: number[] = [];
    if (n.left)  children.push(n.left.id);
    if (n.right) children.push(n.right.id);
    const isLeaf = children.length === 0;
    const isRoot = parentMap.get(n.id) === null;
    map.set(n.id, {
      id: n.id,
      val: n.val,
      depth: depths.get(n.id) ?? 0,
      height: heights.get(n.id) ?? 0,
      type: isRoot ? 'root' : isLeaf ? 'leaf' : 'internal',
      parent: parentMap.get(n.id) ?? null,
      children,
    });
  });
  return map;
}

const TYPE_COLOR: Record<NodeInfo['type'], string> = {
  root: '#f59e0b',
  leaf: '#10b981',
  internal: '#3b82f6',
};

const MODE_LIST = [
  { key: 'depth',   label: '深度（Depth）',   desc: '从根到该节点的边数，根深度=0' },
  { key: 'height',  label: '高度（Height）',  desc: '从该节点到最远叶节点的边数，叶节点高度=0' },
  { key: 'type',    label: '节点类型',         desc: '根节点/叶节点/内部节点' },
  { key: 'click',   label: '点击探索',         desc: '点击节点查看其所有属性详情' },
] as const;
type Mode = typeof MODE_LIST[number]['key'];

export default function TreeDefinitionAnimator() {
  const [mode, setMode]     = useState<Mode>('click');
  const [selected, setSelected] = useState<number | null>(null);

  const pos     = layout(DEMO_TREE);
  const nodes   = collectAll(DEMO_TREE);
  const edges   = collectEdges(DEMO_TREE);
  const infoMap = buildInfoMap(DEMO_TREE);
  const treeHeight = infoMap.get(DEMO_TREE.id)?.height ?? 0;

  const getNodeLabel = (n: TNode): string => {
    const info = infoMap.get(n.id)!;
    if (mode === 'depth')  return `d=${info.depth}`;
    if (mode === 'height') return `h=${info.height}`;
    if (mode === 'type')   return info.type === 'root' ? '根' : info.type === 'leaf' ? '叶' : '内';
    return String(n.val);
  };

  const getNodeColor = (n: TNode): string => {
    const info = infoMap.get(n.id)!;
    if (selected !== null && n.id === selected) return '#818cf8';
    if (mode === 'depth') {
      const d = info.depth;
      return ['#1e3a5f', '#1e4a6f', '#1e5a7f', '#1e6a8f'][d] ?? '#1e3a5f';
    }
    if (mode === 'height') {
      const h = info.height;
      const colors = ['#14532d', '#166534', '#15803d', '#16a34a', '#22c55e'];
      return colors[Math.min(h, colors.length - 1)];
    }
    if (mode === 'type') return TYPE_COLOR[info.type] + '44';
    return '#3f3f46';
  };

  const selInfo = selected !== null ? infoMap.get(selected) : null;

  return (
    <div className="rounded-2xl border border-slate-200 dark:border-zinc-800 bg-slate-50 dark:bg-zinc-950 overflow-hidden text-slate-900 dark:text-white">
      {/* 头部 */}
      <div className="px-6 py-5 bg-slate-100 dark:bg-zinc-900 border-b border-slate-200 dark:border-zinc-800">
        <h3 className="text-xl font-bold text-sky-600 dark:text-sky-400">树的术语与性质探索</h3>
        <p className="text-sm text-slate-500 dark:text-zinc-400 mt-1">点击树中节点，查看其深度、高度、类型等关键属性</p>
      </div>

      <div className="p-6 space-y-5">
        {/* 模式选择 —— 胶囊式 segmented control */}
        <div className="inline-flex gap-1 p-1 bg-slate-200 dark:bg-zinc-800 rounded-xl">
          {MODE_LIST.map(m => (
            <button key={m.key} onClick={() => { setMode(m.key as Mode); setSelected(null); }}
              className={`px-3 py-2 text-xs rounded-lg font-medium transition-all ${mode === m.key
                ? 'bg-white dark:bg-zinc-700 text-sky-600 dark:text-sky-400 shadow-sm'
                : 'text-slate-500 dark:text-zinc-400 hover:text-slate-700 dark:hover:text-zinc-200'}`}>
              {m.label}
            </button>
          ))}
        </div>

        {/* 当前模式说明 */}
        <div className="flex items-center gap-2 px-4 py-3 bg-blue-50 dark:bg-blue-950/40 border border-blue-200 dark:border-blue-800 rounded-xl text-sm text-blue-700 dark:text-blue-300">
          <span className="text-base">💡</span>
          <span>{MODE_LIST.find(m => m.key === mode)?.desc}</span>
          {mode === 'click' && <span className="text-sky-600 dark:text-sky-400 font-medium ml-1">— 点击任意节点</span>}
        </div>

        {/* 主体：SVG + 侧面板 */}
        <div className={`flex gap-4 ${selInfo ? 'flex-col lg:flex-row' : 'flex-col'}`}>
          {/* SVG 树 */}
          <div className={`rounded-xl overflow-hidden border border-slate-200 dark:border-zinc-800 bg-zinc-950 ${selInfo ? 'lg:flex-1' : 'w-full'}`}>
            <svg width="100%" viewBox="0 0 440 290">
              {/* 深度层次背景条 */}
              {mode === 'depth' && Array.from({ length: treeHeight + 1 }, (_, d) => (
                <rect key={d} x="0" y={32 + d * 60 - 26} width="440" height="52"
                  fill={`rgba(255,255,255,${d % 2 === 0 ? '0.015' : '0.03'})`} />
              ))}

              {/* 边 */}
              {edges.map((e, i) => {
                const a = pos.get(e.from), b = pos.get(e.to);
                if (!a || !b) return null;
                const isSel = selected !== null && (e.from === selected || e.to === selected);
                return <line key={i} x1={a.x} y1={a.y} x2={b.x} y2={b.y}
                  stroke={isSel ? '#818cf8' : '#3f3f46'} strokeWidth={isSel ? 2.5 : 1.5}
                  strokeLinecap="round" />;
              })}

              {/* 节点 */}
              {nodes.map(n => {
                const p = pos.get(n.id);
                if (!p) return null;
                const info = infoMap.get(n.id)!;
                const isSel = n.id === selected;
                const fill = getNodeColor(n);
                const strokeC = mode === 'type' ? TYPE_COLOR[info.type] : isSel ? '#818cf8' : '#52525b';
                const strokeW = isSel ? 3 : 1.5;
                return (
                  <g key={n.id} style={{ cursor: 'pointer' }} onClick={() => setSelected(isSel ? null : n.id)}>
                    {isSel && <circle cx={p.x} cy={p.y} r={28} fill="#818cf820" />}
                    <circle cx={p.x} cy={p.y} r={20} fill={fill} stroke={strokeC} strokeWidth={strokeW} />
                    <text x={p.x} y={p.y - 3} textAnchor="middle" fontSize="13" fontWeight="bold"
                      fill={isSel ? '#c7d2fe' : '#d4d4d8'}>{n.val}</text>
                    <text x={p.x} y={p.y + 11} textAnchor="middle" fontSize="9"
                      fill={isSel ? '#818cf8' : '#71717a'}>{getNodeLabel(n)}</text>
                  </g>
                );
              })}

              {/* 深度层标注 */}
              {mode === 'depth' && Array.from({ length: treeHeight + 1 }, (_, d) => (
                <g key={d}>
                  <text x="12" y={32 + d * 60 + 5} fontSize="10" fill="#4b5563" fontWeight="bold">L{d}</text>
                  <line x1="28" y1={32 + d * 60} x2="40" y2={32 + d * 60} stroke="#374151" strokeWidth="1" />
                </g>
              ))}

              {/* 类型图例 */}
              {mode === 'type' && (
                <g>
                  {[['root','根'], ['internal','内部'], ['leaf','叶']].map(([t, label], i) => (
                    <g key={t} transform={`translate(${340 + i * 0}, ${20 + i * 18})`}>
                      <circle cx="8" cy="0" r="6" fill={TYPE_COLOR[t as NodeInfo['type']] + '44'}
                        stroke={TYPE_COLOR[t as NodeInfo['type']]} strokeWidth="1.5" />
                      <text x="18" y="4" fontSize="9" fill={TYPE_COLOR[t as NodeInfo['type']]}>{label}</text>
                    </g>
                  ))}
                </g>
              )}
            </svg>
          </div>

          {/* 节点详情面板 */}
          {selInfo && (
            <div className="lg:w-60 bg-white dark:bg-zinc-900 border border-slate-200 dark:border-zinc-700 rounded-xl p-5 space-y-3">
              <div className="flex items-center justify-between">
                <div className="text-base font-bold text-indigo-600 dark:text-indigo-400">节点 {selInfo.val}</div>
                <span className="text-xs px-2 py-0.5 rounded-md font-medium"
                  style={{ backgroundColor: TYPE_COLOR[selInfo.type] + '25', color: TYPE_COLOR[selInfo.type] }}>
                  {selInfo.type === 'root' ? '根节点' : selInfo.type === 'leaf' ? '叶节点' : '内部节点'}
                </span>
              </div>
              <hr className="border-slate-100 dark:border-zinc-800" />
              {[
                { label: '键值', value: String(selInfo.val), color: 'text-slate-800 dark:text-white' },
                { label: '深度 (Depth)', value: `${selInfo.depth}`, color: 'text-blue-600 dark:text-blue-400' },
                { label: '高度 (Height)', value: `${selInfo.height}`, color: 'text-emerald-600 dark:text-emerald-400' },
                { label: '子节点数', value: String(selInfo.children.length), color: 'text-slate-700 dark:text-zinc-200' },
                { label: '父节点', value: selInfo.parent === null ? '无（根）' : `节点 ${nodes.find(n => n.id === selInfo.parent)?.val}`, color: 'text-slate-600 dark:text-zinc-300' },
              ].map(row => (
                <div key={row.label} className="flex items-baseline justify-between gap-2">
                  <span className="text-xs text-slate-400 dark:text-zinc-500 shrink-0">{row.label}</span>
                  <span className={`text-sm font-mono font-bold ${row.color}`}>{row.value}</span>
                </div>
              ))}
              {selInfo.children.length > 0 && (
                <div className="flex items-baseline justify-between gap-2">
                  <span className="text-xs text-slate-400 dark:text-zinc-500">子节点值</span>
                  <span className="text-sm font-mono font-bold text-slate-700 dark:text-zinc-200">
                    {selInfo.children.map(cid => nodes.find(n => n.id === cid)?.val).join(', ')}
                  </span>
                </div>
              )}
              <button onClick={() => setSelected(null)}
                className="w-full mt-2 py-1.5 text-xs rounded-lg bg-slate-100 dark:bg-zinc-800 hover:bg-slate-200 dark:hover:bg-zinc-700 text-slate-500 dark:text-zinc-400 transition-colors">
                取消选择
              </button>
            </div>
          )}
        </div>

        {/* 全树统计 */}
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
          {[
            { label: '总节点数 n', value: nodes.length, color: 'text-slate-800 dark:text-white', bg: 'bg-white dark:bg-zinc-900' },
            { label: '树的高度 h', value: treeHeight, color: 'text-sky-600 dark:text-sky-400', bg: 'bg-sky-50 dark:bg-sky-950/40' },
            { label: '叶节点数 n₀', value: nodes.filter(n => infoMap.get(n.id)!.type === 'leaf').length, color: 'text-emerald-600 dark:text-emerald-400', bg: 'bg-emerald-50 dark:bg-emerald-950/40' },
            { label: '内部节点 n₂ (度=2)', value: nodes.filter(n => infoMap.get(n.id)!.children.length === 2).length, color: 'text-blue-600 dark:text-blue-400', bg: 'bg-blue-50 dark:bg-blue-950/40' },
            { label: '总边数', value: `${nodes.length - 1} = n − 1`, color: 'text-violet-600 dark:text-violet-400', bg: 'bg-violet-50 dark:bg-violet-950/40' },
            {
              label: '定理 n₀ = n₂ + 1',
              value: nodes.filter(n => infoMap.get(n.id)!.type === 'leaf').length ===
                     nodes.filter(n => infoMap.get(n.id)!.children.length === 2).length + 1 ? '✓ 成立' : '✗ 不成立',
              color: 'text-amber-600 dark:text-amber-400',
              bg: 'bg-amber-50 dark:bg-amber-950/40',
            },
          ].map(item => (
            <div key={item.label} className={`${item.bg} border border-slate-200 dark:border-zinc-800 rounded-xl p-4`}>
              <div className="text-xs text-slate-400 dark:text-zinc-500 mb-1">{item.label}</div>
              <div className={`text-xl font-bold font-mono ${item.color}`}>{item.value}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}