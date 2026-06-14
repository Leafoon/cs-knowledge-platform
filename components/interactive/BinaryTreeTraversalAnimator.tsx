'use client';
import React, { useState, useCallback, useEffect } from 'react';

// ══════════════════════════════════════════════════════
//  BinaryTreeTraversalAnimator — 四种遍历可视化
// ══════════════════════════════════════════════════════

interface TNode {
  val: number;
  left: TNode | null;
  right: TNode | null;
  id: number;
}

// ── 树构建辅助 ────────────────────────────────────────
let _id = 0;
function makeNode(val: number, left: TNode | null = null, right: TNode | null = null): TNode {
  return { val, left, right, id: _id++ };
}

function buildPreset(name: string): TNode {
  _id = 0;
  switch (name) {
    case 'expr_tree':
      // 表达式树: (3+4)×(5-2)
      return makeNode(
        -1, // ×
        makeNode(-2, makeNode(3), makeNode(4)), // +
        makeNode(-3, makeNode(5), makeNode(2))  // -
      );
    case 'bst':
      // BST 示例
      return makeNode(8,
        makeNode(3, makeNode(1), makeNode(6, makeNode(4), makeNode(7))),
        makeNode(10, null, makeNode(14, makeNode(13), null))
      );
    default: // 'basic'
      return makeNode(1,
        makeNode(2, makeNode(4), makeNode(5)),
        makeNode(3, makeNode(6), makeNode(7))
      );
  }
}

const VAL_LABELS: Record<number, string> = { '-1': '×', '-2': '+', '-3': '-' };
function displayVal(v: number) { return VAL_LABELS[v] ?? String(v); }

// ── 遍历生成步骤 ───────────────────────────────────────
interface Step { nodeId: number; phase: 'visit' | 'call' | 'return'; order: number; }

function preorderSteps(root: TNode | null): Step[] {
  const steps: Step[] = [];
  let order = 0;
  function dfs(n: TNode | null) {
    if (!n) return;
    steps.push({ nodeId: n.id, phase: 'visit', order: order++ });
    dfs(n.left);
    dfs(n.right);
  }
  dfs(root);
  return steps;
}
function inorderSteps(root: TNode | null): Step[] {
  const steps: Step[] = [];
  let order = 0;
  function dfs(n: TNode | null) {
    if (!n) return;
    dfs(n.left);
    steps.push({ nodeId: n.id, phase: 'visit', order: order++ });
    dfs(n.right);
  }
  dfs(root);
  return steps;
}
function postorderSteps(root: TNode | null): Step[] {
  const steps: Step[] = [];
  let order = 0;
  function dfs(n: TNode | null) {
    if (!n) return;
    dfs(n.left);
    dfs(n.right);
    steps.push({ nodeId: n.id, phase: 'visit', order: order++ });
  }
  dfs(root);
  return steps;
}
function levelorderSteps(root: TNode | null): Step[] {
  if (!root) return [];
  const steps: Step[] = [];
  let order = 0;
  const queue: TNode[] = [root];
  while (queue.length) {
    const n = queue.shift()!;
    steps.push({ nodeId: n.id, phase: 'visit', order: order++ });
    if (n.left)  queue.push(n.left);
    if (n.right) queue.push(n.right);
  }
  return steps;
}

// ── 树布局计算 ────────────────────────────────────────
interface Pos { x: number; y: number; }
function computeLayout(root: TNode | null): Map<number, Pos> {
  const pos = new Map<number, Pos>();
  const X_GAP = 44, Y_GAP = 56;
  function assign(n: TNode | null, depth: number, left: number, right: number) {
    if (!n) return;
    const x = (left + right) / 2;
    pos.set(n.id, { x, y: 30 + depth * Y_GAP });
    assign(n.left,  depth + 1, left,  x);
    assign(n.right, depth + 1, x,     right);
  }
  assign(root, 0, 0, 440);
  return pos;
}

// ── 边收集 ────────────────────────────────────────────
interface Edge { from: number; to: number; }
function collectEdges(n: TNode | null): Edge[] {
  if (!n) return [];
  const edges: Edge[] = [];
  if (n.left)  { edges.push({ from: n.id, to: n.left.id });  edges.push(...collectEdges(n.left)); }
  if (n.right) { edges.push({ from: n.id, to: n.right.id }); edges.push(...collectEdges(n.right)); }
  return edges;
}
function collectNodes(n: TNode | null): TNode[] {
  if (!n) return [];
  return [n, ...collectNodes(n.left), ...collectNodes(n.right)];
}

// ════════════════════════════════════════════════════════
//  组件主体
// ════════════════════════════════════════════════════════
const TRAVERSALS = ['preorder', 'inorder', 'postorder', 'levelorder'] as const;
type Traversal = typeof TRAVERSALS[number];

const TRAV_META: Record<Traversal, { label: string; desc: string; color: string; appColor: string; app: string }> = {
  preorder:   { label: '前序（根-左-右）', desc: 'NLR', color: '#3b82f6', appColor: 'bg-blue-900/50 border-blue-700',   app: '树的序列化 / 复制 / 目录打印' },
  inorder:    { label: '中序（左-根-右）', desc: 'LNR', color: '#10b981', appColor: 'bg-emerald-900/50 border-emerald-700', app: 'BST 有序输出 / 中缀表达式' },
  postorder:  { label: '后序（左-右-根）', desc: 'LRN', color: '#f59e0b', appColor: 'bg-amber-900/50 border-amber-700',  app: '树的释放 / 目录大小计算' },
  levelorder: { label: '层序（BFS）',      desc: 'BFS', color: '#a855f7', appColor: 'bg-purple-900/50 border-purple-700', app: '最短路 / 逐层处理 / 序列化' },
};

const PRESETS = [
  { name: 'basic',     label: '基础示例树' },
  { name: 'bst',       label: 'BST 样例' },
  { name: 'expr_tree', label: '表达式树' },
];

export default function BinaryTreeTraversalAnimator() {
  const [preset, setPreset]       = useState('basic');
  const [traversal, setTraversal] = useState<Traversal>('preorder');
  const [stepIdx, setStepIdx]     = useState(-1); // -1 = 未开始
  const [playing, setPlaying]     = useState(false);

  const root = buildPreset(preset);
  const allNodes = collectNodes(root);
  const edges    = collectEdges(root);
  const pos      = computeLayout(root);

  const stepsMap: Record<Traversal, Step[]> = {
    preorder:   preorderSteps(root),
    inorder:    inorderSteps(root),
    postorder:  postorderSteps(root),
    levelorder: levelorderSteps(root),
  };
  const steps = stepsMap[traversal];

  // 当前已访问节点 set
  const visitedIds = new Set(
    steps.slice(0, Math.max(0, stepIdx + 1)).map(s => s.nodeId)
  );
  const currentId = stepIdx >= 0 && stepIdx < steps.length ? steps[stepIdx].nodeId : null;
  const outputSeq = steps.slice(0, Math.max(0, stepIdx + 1)).map(s => {
    const n = allNodes.find(n => n.id === s.nodeId);
    return n ? displayVal(n.val) : '?';
  });

  // 自动播放
  useEffect(() => {
    if (!playing) return;
    if (stepIdx >= steps.length - 1) { setPlaying(false); return; }
    const t = setTimeout(() => setStepIdx(i => i + 1), 700);
    return () => clearTimeout(t);
  }, [playing, stepIdx, steps.length]);

  const reset = useCallback(() => { setStepIdx(-1); setPlaying(false); }, []);
  const changeTraversal = (t: Traversal) => { setTraversal(t); reset(); };
  const changePreset = (p: string) => { setPreset(p); reset(); };

  const meta = TRAV_META[traversal];
  const SVG_W = 440, SVG_H = 290;
  const nodeR = 20;

  return (
    <div className="rounded-2xl border border-slate-200 dark:border-zinc-800 bg-slate-50 dark:bg-zinc-950 overflow-hidden text-slate-900 dark:text-white">
      {/* 头部 */}
      <div className="px-6 py-5 bg-slate-100 dark:bg-zinc-900 border-b border-slate-200 dark:border-zinc-800 flex flex-wrap items-center justify-between gap-3">
        <div>
          <h3 className="text-xl font-bold text-sky-600 dark:text-sky-400">二叉树四种遍历</h3>
          <p className="text-sm text-slate-500 dark:text-zinc-400 mt-0.5">Binary Tree Traversals — 逐步动画演示</p>
        </div>
        <div className="flex gap-1.5 flex-wrap">
          {PRESETS.map(p => (
            <button key={p.name} onClick={() => changePreset(p.name)}
              className={`px-3 py-1.5 text-xs rounded-lg font-medium transition-colors ${preset === p.name ? 'bg-sky-600 text-white' : 'bg-slate-200 dark:bg-zinc-700 hover:bg-slate-300 dark:hover:bg-zinc-600 text-slate-700 dark:text-zinc-200'}`}>
              {p.label}
            </button>
          ))}
        </div>
      </div>

      <div className="p-6 space-y-5">
        {/* 遍历类型选择 */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
          {TRAVERSALS.map(t => {
            const m = TRAV_META[t];
            const active = traversal === t;
            return (
              <button key={t} onClick={() => changeTraversal(t)}
                className={`px-3 py-3 rounded-xl text-sm font-semibold transition-all border-2 ${active ? 'border-transparent text-white shadow-md' : 'border-slate-200 dark:border-zinc-700 text-slate-600 dark:text-zinc-400 hover:border-slate-300 dark:hover:border-zinc-600 bg-white dark:bg-zinc-900'}`}
                style={active ? { backgroundColor: m.color, boxShadow: `0 4px 14px ${m.color}55` } : {}}>
                <div className="font-bold text-base">{m.desc}</div>
                <div className={`text-xs mt-0.5 ${active ? 'text-white/80' : 'text-slate-400 dark:text-zinc-500'}`}>{m.label.split('（')[0]}</div>
              </button>
            );
          })}
        </div>

        {/* 应用场景 */}
        <div className="flex items-center gap-3 px-4 py-3 rounded-xl"
          style={{ backgroundColor: meta.color + '15', border: `1px solid ${meta.color}44` }}>
          <span className="text-xs font-bold shrink-0" style={{ color: meta.color }}>典型应用</span>
          <span className="text-sm" style={{ color: meta.color }}>{meta.app}</span>
        </div>

        {/* SVG 树 —— 深色 canvas 风格 */}
        <div className="rounded-xl overflow-hidden bg-zinc-950 border border-zinc-800">
          <svg width="100%" viewBox={`0 0 ${SVG_W} ${SVG_H}`}>
            {/* 层次网格线 */}
            {[0,1,2,3].map(d => (
              <line key={d} x1="0" y1={30 + d * 56} x2={SVG_W} y2={30 + d * 56}
                stroke="#27272a" strokeWidth="1" strokeDasharray="3,5" />
            ))}
            {/* 边 */}
            {edges.map((e, i) => {
              const a = pos.get(e.from), b = pos.get(e.to);
              if (!a || !b) return null;
              return <line key={i} x1={a.x} y1={a.y} x2={b.x} y2={b.y}
                stroke="#3f3f46" strokeWidth="1.5" strokeLinecap="round" />;
            })}
            {/* 节点 */}
            {allNodes.map(n => {
              const p = pos.get(n.id);
              if (!p) return null;
              const isCurrent = n.id === currentId;
              const isVisited = visitedIds.has(n.id) && !isCurrent;
              const fill = isCurrent ? meta.color : isVisited ? meta.color + '3a' : '#27272a';
              const stroke = isCurrent ? '#fff' : isVisited ? meta.color + 'cc' : '#52525b';
              const orderStep = steps.find(s => s.nodeId === n.id);
              const orderNum = orderStep && visitedIds.has(n.id) ? orderStep.order + 1 : null;
              return (
                <g key={n.id}>
                  {isCurrent && <circle cx={p.x} cy={p.y} r={nodeR + 7} fill={meta.color + '25'} />}
                  <circle cx={p.x} cy={p.y} r={nodeR} fill={fill} stroke={stroke} strokeWidth={isCurrent ? 2.5 : 1.5} />
                  <text x={p.x} y={p.y + 5} textAnchor="middle" fontSize="13" fontWeight="bold"
                    fill={isCurrent ? '#fff' : isVisited ? '#d4d4d8' : '#71717a'}>
                    {displayVal(n.val)}
                  </text>
                  {orderNum !== null && (
                    <g>
                      <circle cx={p.x + nodeR - 1} cy={p.y - nodeR + 1} r={9} fill={meta.color} />
                      <text x={p.x + nodeR - 1} y={p.y - nodeR + 5} textAnchor="middle"
                        fontSize="8" fill="#fff" fontWeight="bold">{orderNum}</text>
                    </g>
                  )}
                </g>
              );
            })}
          </svg>
        </div>

        {/* 控制栏 + 进度 */}
        <div className="flex gap-2 flex-wrap items-center">
          <button onClick={() => setStepIdx(i => Math.max(-1, i - 1))} disabled={stepIdx < 0}
            className="px-4 py-2 text-sm bg-slate-200 dark:bg-zinc-700 hover:bg-slate-300 dark:hover:bg-zinc-600 disabled:opacity-40 rounded-lg text-slate-700 dark:text-zinc-200 transition-colors">← 上一步</button>
          <button onClick={() => { if (stepIdx >= steps.length - 1) return; setPlaying(p => !p); }}
            disabled={stepIdx >= steps.length - 1}
            className={`px-4 py-2 text-sm rounded-lg font-medium transition-colors disabled:opacity-40 ${playing ? 'bg-amber-500 hover:bg-amber-400 text-white' : 'bg-sky-600 hover:bg-sky-500 text-white'}`}>
            {playing ? '⏸ 暂停' : stepIdx < 0 ? '▶ 开始' : '▶ 继续'}
          </button>
          <button onClick={() => setStepIdx(i => Math.min(steps.length - 1, i + 1))} disabled={stepIdx >= steps.length - 1}
            className="px-4 py-2 text-sm bg-slate-200 dark:bg-zinc-700 hover:bg-slate-300 dark:hover:bg-zinc-600 disabled:opacity-40 rounded-lg text-slate-700 dark:text-zinc-200 transition-colors">下一步 →</button>
          <button onClick={reset} className="px-4 py-2 text-sm bg-slate-200 dark:bg-zinc-700 hover:bg-slate-300 dark:hover:bg-zinc-600 rounded-lg text-slate-700 dark:text-zinc-200 transition-colors">↺ 重置</button>
          <div className="flex-1 min-w-[120px]">
            <div className="flex justify-between text-xs text-slate-400 dark:text-zinc-500 mb-1">
              <span>进度</span>
              <span>{stepIdx < 0 ? 0 : stepIdx + 1}/{steps.length}</span>
            </div>
            <div className="h-1.5 bg-slate-200 dark:bg-zinc-800 rounded-full overflow-hidden">
              <div className="h-full rounded-full transition-all duration-300"
                style={{ width: `${stepIdx < 0 ? 0 : ((stepIdx + 1) / steps.length) * 100}%`, backgroundColor: meta.color }} />
            </div>
          </div>
        </div>

        {/* 访问序列 */}
        <div className="bg-white dark:bg-zinc-900 border border-slate-200 dark:border-zinc-800 rounded-xl p-4">
          <div className="text-xs font-semibold text-slate-400 dark:text-zinc-500 uppercase tracking-wider mb-3">访问序列</div>
          <div className="flex flex-wrap gap-2 min-h-[36px]">
            {outputSeq.map((v, i) => (
              <span key={i} className="inline-flex items-center gap-1 px-3 py-1 rounded-lg text-sm font-mono font-bold transition-all"
                style={{
                  backgroundColor: i === stepIdx ? meta.color : meta.color + '20',
                  color: i === stepIdx ? '#fff' : meta.color,
                  border: `1.5px solid ${meta.color}55`,
                  transform: i === stepIdx ? 'scale(1.1)' : 'scale(1)',
                  boxShadow: i === stepIdx ? `0 2px 8px ${meta.color}55` : 'none',
                }}>
                <span style={{ fontSize: 10, opacity: 0.6 }}>{i + 1}</span> {v}
              </span>
            ))}
            {outputSeq.length === 0 && (
              <span className="text-slate-400 dark:text-zinc-500 text-sm">点击 ▶ 开始，观察遍历过程…</span>
            )}
          </div>
        </div>

        {/* 遍历对比表 */}
        <div className="bg-white dark:bg-zinc-900 border border-slate-200 dark:border-zinc-800 rounded-xl overflow-hidden">
          <div className="px-4 py-3 border-b border-slate-100 dark:border-zinc-800">
            <span className="text-xs font-semibold text-slate-400 dark:text-zinc-500 uppercase tracking-wider">四种遍历对比</span>
          </div>
          <table className="w-full text-sm">
            <thead>
              <tr className="text-xs text-slate-400 dark:text-zinc-500 border-b border-slate-100 dark:border-zinc-800">
                {['遍历方式', '顺序', '底层结构', '典型应用'].map(h => (
                  <th key={h} className="px-4 py-2.5 text-left font-medium">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {TRAVERSALS.map(t => {
                const m = TRAV_META[t];
                const active = traversal === t;
                return (
                  <tr key={t} onClick={() => changeTraversal(t)} style={{ cursor: 'pointer' }}
                    className={`border-b border-slate-50 dark:border-zinc-800/50 transition-colors last:border-0 ${active ? 'bg-slate-50 dark:bg-zinc-800/50' : 'hover:bg-slate-50/70 dark:hover:bg-zinc-800/30'}`}>
                    <td className="px-4 py-3">
                      <span className="inline-flex items-center gap-2">
                        <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: m.color }} />
                        <span className="font-bold font-mono text-xs" style={{ color: m.color }}>{m.desc}</span>
                      </span>
                    </td>
                    <td className="px-4 py-3 text-slate-700 dark:text-zinc-300 text-sm">{m.label}</td>
                    <td className="px-4 py-3 text-slate-500 dark:text-zinc-400 text-xs">{t === 'levelorder' ? '队列 Queue' : '调用栈（隐式）'}</td>
                    <td className="px-4 py-3 text-slate-500 dark:text-zinc-400 text-xs">{m.app.split(' / ')[0]}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
