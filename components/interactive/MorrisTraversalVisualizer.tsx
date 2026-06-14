'use client';
import React, { useState } from 'react';

// ══════════════════════════════════════════════════════
//  MorrisTraversalVisualizer — Morris 中序遍历步进可视化
// ══════════════════════════════════════════════════════

interface TNode { val: number; id: number; left: TNode | null; right: TNode | null; }

let _gid = 0;
function mk(val: number, l: TNode | null = null, r: TNode | null = null): TNode {
  return { val, left: l, right: r, id: _gid++ };
}

// 构建示例树（带 id 的稳定结构）
function buildTree(): TNode {
  _gid = 0;
  return mk(4,
    mk(2, mk(1), mk(3)),
    mk(6, mk(5), mk(7))
  );
}

// ── Morris 步骤生成 ────────────────────────────────────
type MorrisAction =
  | { type: 'no_left';    currId: number; output: number }
  | { type: 'find_pred';  currId: number; predId: number }
  | { type: 'thread';     currId: number; predId: number }
  | { type: 'restore';    currId: number; predId: number; output: number };

interface MStep {
  action: MorrisAction;
  curr: number;
  threads: [number, number][];   // [predId → currId] 当前所有线索
  visited: number[];
  desc: string;
}

function buildMorrisSteps(root: TNode): MStep[] {
  const steps: MStep[] = [];
  const threads: [number, number][] = [];
  const visited: number[] = [];

  // 我们在生成步骤时，需要模拟树的线索状态
  // 为了安全，使用额外 map 表示线索：predId → currId
  const threadMap = new Map<number, number>(); // pred.right → curr（临时线索）
  // 为了找 pred, 需要在每次迭代直接操作
  const allNodes = new Map<number, TNode>();

  function gatherNodes(n: TNode | null) {
    if (!n) return;
    allNodes.set(n.id, n);
    gatherNodes(n.left);
    gatherNodes(n.right);
  }
  gatherNodes(root);

  // Morris simulation using id-based virtual right pointers
  // virtualRight: 临时覆盖 right 指针（仅在 simulation 中）
  const virtualRight = new Map<number, number | null>(); // nodeId -> rightId (override)

  function getRight(n: TNode): TNode | null {
    if (virtualRight.has(n.id)) {
      const id = virtualRight.get(n.id);
      return id == null ? null : (allNodes.get(id) ?? null);
    }
    return n.right;
  }

  let curr: TNode | null = root;
  const MAX_STEPS = 50;
  let safety = 0;

  while (curr && safety < MAX_STEPS) {
    safety++;
    if (!curr.left) {
      // 无左子树：访问并向右
      visited.push(curr.val);
      steps.push({
        action: { type: 'no_left', currId: curr.id, output: curr.val },
        curr: curr.id,
        threads: [...threadMap].map(([a, b]) => [a, b] as [number, number]),
        visited: [...visited],
        desc: `curr=${curr.val} 无左子树 → 输出 ${curr.val}，curr 移到右子（${getRight(curr)?.val ?? 'null'}）`,
      });
      curr = getRight(curr);
    } else {
      // 找中序前驱
      let pred = curr.left;
      while (true) {
        const predRight = getRight(pred);
        if (predRight === null || predRight?.id === curr.id) break;
        pred = predRight;
      }

      const predRight = getRight(pred);
      if (predRight === null) {
        // 建立线索
        threadMap.set(pred.id, curr.id);
        virtualRight.set(pred.id, curr.id);
        steps.push({
          action: { type: 'thread', currId: curr.id, predId: pred.id },
          curr: curr.id,
          threads: [...threadMap].map(([a, b]) => [a, b] as [number, number]),
          visited: [...visited],
          desc: `建立线索：${pred.val}.right → ${curr.val}（前驱指向后继），curr 进入左子树 ${curr.left?.val}`,
        });
        curr = curr.left;
      } else {
        // pred.right === curr：还原线索，访问 curr
        threadMap.delete(pred.id);
        virtualRight.set(pred.id, null);
        visited.push(curr.val);
        steps.push({
          action: { type: 'restore', currId: curr.id, predId: pred.id, output: curr.val },
          curr: curr.id,
          threads: [...threadMap].map(([a, b]) => [a, b] as [number, number]),
          visited: [...visited],
          desc: `还原线索：${pred.val}.right → null，输出 ${curr.val}，curr 移到右子（${getRight(curr)?.val ?? 'null'}）`,
        });
        curr = getRight(curr);
      }
    }
  }

  return steps;
}

// ── 布局 ──────────────────────────────────────────────
interface Pos { x: number; y: number; }
function layoutTree(root: TNode): Map<number, Pos> {
  const pos = new Map<number, Pos>();
  function assign(n: TNode | null, d: number, lo: number, hi: number) {
    if (!n) return;
    const x = (lo + hi) / 2;
    pos.set(n.id, { x, y: 30 + d * 65 });
    assign(n.left, d + 1, lo, x);
    assign(n.right, d + 1, x, hi);
  }
  assign(root, 0, 0, 440);
  return pos;
}

function allNodes(n: TNode | null): TNode[] {
  return n ? [n, ...allNodes(n.left), ...allNodes(n.right)] : [];
}
function allEdges(n: TNode | null): [number, number][] {
  if (!n) return [];
  const edges: [number, number][] = [];
  if (n.left)  { edges.push([n.id, n.left.id]);  edges.push(...allEdges(n.left)); }
  if (n.right) { edges.push([n.id, n.right.id]); edges.push(...allEdges(n.right)); }
  return edges;
}

const ACTION_COLOR: Record<MorrisAction['type'], string> = {
  no_left:    '#10b981',
  find_pred:  '#94a3b8',
  thread:     '#f59e0b',
  restore:    '#a855f7',
};

export default function MorrisTraversalVisualizer() {
  const [stepIdx, setStepIdx] = useState(-1);

  const root  = buildTree();
  const steps = buildMorrisSteps(root);
  const nodes = allNodes(root);
  const edges = allEdges(root);
  const pos   = layoutTree(root);

  const step = stepIdx >= 0 && stepIdx < steps.length ? steps[stepIdx] : null;
  const currId    = step?.curr ?? null;
  const threadSet = new Set((step?.threads ?? []).map(([a, b]) => `${a}-${b}`));
  const visitedSet = new Set(step?.visited ?? []);

  const SVG_W = 440, SVG_H = 290;

  return (
    <div className="rounded-2xl overflow-hidden bg-zinc-950 border border-zinc-800 text-zinc-100">
      {/* IDE 风格头部 */}
      <div className="flex items-center gap-3 px-5 py-4 bg-zinc-900 border-b border-zinc-800">
        <div className="flex gap-1.5">
          <span className="w-3 h-3 rounded-full bg-red-500/80" />
          <span className="w-3 h-3 rounded-full bg-amber-500/80" />
          <span className="w-3 h-3 rounded-full bg-emerald-500/80" />
        </div>
        <span className="text-sm font-mono text-zinc-400 ml-1">morris_traversal.py</span>
        <div className="ml-auto">
          <span className="px-2 py-0.5 text-xs rounded bg-emerald-900/50 border border-emerald-700 text-emerald-400 font-mono">O(1) space</span>
        </div>
      </div>

      <div className="p-5 space-y-4">
        {/* 说明 + 图例 */}
        <div className="bg-zinc-900 border border-zinc-800 rounded-xl px-4 py-3 text-sm text-zinc-400 space-y-2">
          <p>Morris 遍历利用叶节点的空 <code className="text-amber-400 font-mono bg-amber-900/20 px-1 rounded">right</code> 指针建立临时"线索"——无需栈即可完成中序遍历，空间复杂度 O(1)。</p>
          <div className="flex flex-wrap gap-4 pt-1">
            <span className="flex items-center gap-2">
              <span className="w-8 h-0.5 rounded border-t-2 border-dashed border-amber-400" />
              <span className="text-xs text-amber-400">建立线索</span>
            </span>
            <span className="flex items-center gap-2">
              <span className="w-3.5 h-3.5 rounded" style={{ background: '#92400e' }} />
              <span className="text-xs text-amber-300">前驱节点</span>
            </span>
            <span className="flex items-center gap-2">
              <span className="w-3.5 h-3.5 rounded" style={{ background: '#581c87' }} />
              <span className="text-xs text-purple-300">还原+访问</span>
            </span>
            <span className="flex items-center gap-2">
              <span className="w-3.5 h-3.5 rounded bg-blue-700" />
              <span className="text-xs text-blue-300">当前节点</span>
            </span>
            <span className="flex items-center gap-2">
              <span className="w-3.5 h-3.5 rounded bg-emerald-900" />
              <span className="text-xs text-emerald-400">已访问</span>
            </span>
          </div>
        </div>

        {/* 控制栏 */}
        <div className="flex gap-2 flex-wrap items-center">
          <button onClick={() => setStepIdx(i => Math.max(-1, i - 1))} disabled={stepIdx < 0}
            className="px-4 py-2 text-sm bg-zinc-800 hover:bg-zinc-700 disabled:opacity-40 rounded-lg text-zinc-300 border border-zinc-700 transition-colors">← 上一步</button>
          <button onClick={() => setStepIdx(i => Math.min(steps.length - 1, i + 1))} disabled={stepIdx >= steps.length - 1}
            className="px-4 py-2 text-sm bg-sky-700 hover:bg-sky-600 disabled:opacity-40 rounded-lg text-white transition-colors">下一步 →</button>
          <button onClick={() => setStepIdx(steps.length - 1)}
            className="px-4 py-2 text-sm bg-emerald-800 hover:bg-emerald-700 rounded-lg text-emerald-200 transition-colors">⏭ 最终结果</button>
          <button onClick={() => setStepIdx(-1)}
            className="px-4 py-2 text-sm bg-zinc-800 hover:bg-zinc-700 rounded-lg text-zinc-300 border border-zinc-700 transition-colors">↺ 重置</button>
          <span className="ml-auto text-xs text-zinc-500 font-mono">
            step {stepIdx < 0 ? 0 : stepIdx + 1} / {steps.length}
          </span>
        </div>

        {/* 步骤描述 —— 控制台风格 */}
        <div className="bg-zinc-900 border border-zinc-800 rounded-xl overflow-hidden">
          <div className="px-4 py-2 border-b border-zinc-800 flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-zinc-600" />
            <span className="text-xs text-zinc-500 font-mono">stdout</span>
          </div>
          <div className="px-4 py-3 font-mono text-sm min-h-[52px]">
            {step ? (
              <span style={{ color: ACTION_COLOR[step.action.type] }}>
                <span className="text-zinc-600 mr-2">&gt;&gt;&gt;</span>
                <span className="font-bold mr-2">
                  {step.action.type === 'no_left' ? '[直接访问]' :
                   step.action.type === 'thread'  ? '[建立线索]' :
                   step.action.type === 'restore' ? '[还原+访问]' : '[查找前驱]'}
                </span>
                <span className="text-zinc-300">{step.desc}</span>
              </span>
            ) : (
              <span className="text-zinc-600">点击「下一步」开始 Morris 算法…</span>
            )}
          </div>
        </div>

        {/* SVG 树可视化 */}
        <div className="rounded-xl overflow-hidden border border-zinc-800">
          <svg width="100%" viewBox={`0 0 ${SVG_W} ${SVG_H}`} style={{ background: '#09090b' }}>
            <defs>
              <marker id="arrow-thread-m" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
                <path d="M0,0 L0,7 L7,3.5 z" fill="#f59e0b" />
              </marker>
            </defs>

            {/* 普通边 */}
            {edges.map(([a, b], i) => {
              const pa = pos.get(a), pb = pos.get(b);
              if (!pa || !pb) return null;
              return <line key={i} x1={pa.x} y1={pa.y} x2={pb.x} y2={pb.y}
                stroke="#3f3f46" strokeWidth="1.5" strokeLinecap="round" />;
            })}

            {/* 线索边（虚线弧） */}
            {(step?.threads ?? []).map(([predId, currId_], i) => {
              const pa = pos.get(predId), pb = pos.get(currId_);
              if (!pa || !pb) return null;
              const cx = (pa.x + pb.x) / 2 + 50;
              const cy = (pa.y + pb.y) / 2 - 30;
              return (
                <path key={`thread-${i}`}
                  d={`M${pa.x},${pa.y} Q${cx},${cy} ${pb.x},${pb.y}`}
                  fill="none" stroke="#f59e0b" strokeWidth="2"
                  strokeDasharray="6,3" markerEnd="url(#arrow-thread-m)" />
              );
            })}

            {/* 节点 */}
            {nodes.map(n => {
              const p = pos.get(n.id);
              if (!p) return null;
              const isCurr = n.id === currId;
              const isVisited = visitedSet.has(n.val);
              const isThread = step?.action.type === 'thread' && n.id === (step.action as {predId:number}).predId;
              const isRestore = step?.action.type === 'restore' && n.id === (step.action as {predId:number}).predId;
              let fill = '#27272a';
              let strokeC = '#52525b';
              let strokeW = 1.5;
              if (isCurr)      { fill = '#1d4ed8'; strokeC = '#60a5fa'; strokeW = 2.5; }
              else if (isThread)  { fill = '#78350f'; strokeC = '#f59e0b'; strokeW = 2; }
              else if (isRestore) { fill = '#4c1d95'; strokeC = '#c084fc'; strokeW = 2; }
              else if (isVisited) { fill = '#14532d'; strokeC = '#4ade80'; }
              return (
                <g key={n.id}>
                  {isCurr && <circle cx={p.x} cy={p.y} r={27} fill="#1d4ed820" />}
                  <circle cx={p.x} cy={p.y} r={20} fill={fill} stroke={strokeC} strokeWidth={strokeW} />
                  <text x={p.x} y={p.y + 5} textAnchor="middle" fontSize="13" fontWeight="bold"
                    fill={isCurr || isVisited ? '#f4f4f5' : '#71717a'}>{n.val}</text>
                </g>
              );
            })}
          </svg>
        </div>

        {/* 中序输出序列 */}
        <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-4">
          <div className="text-xs font-mono text-zinc-500 mb-3">inorder_output = [</div>
          <div className="flex flex-wrap gap-2 min-h-[36px] pl-4">
            {(step?.visited ?? []).map((v, i) => (
              <span key={i}
                className="px-3 py-1 rounded-lg text-sm font-mono font-bold bg-emerald-900/50 text-emerald-300 border border-emerald-800">
                {v}
              </span>
            ))}
            {(!step || step.visited.length === 0) && (
              <span className="text-zinc-600 font-mono text-sm">…</span>
            )}
          </div>
          <div className="text-xs font-mono text-zinc-500 mt-2">]</div>
        </div>

        {/* 复杂度对比 */}
        <div className="grid grid-cols-3 gap-3">
          {[
            { name: '递归中序', time: 'Θ(n)', space: 'O(h)', highlight: false },
            { name: '迭代中序', time: 'Θ(n)', space: 'O(h)', highlight: false },
            { name: 'Morris', time: 'Θ(n)', space: 'O(1) ✓', highlight: true },
          ].map(r => (
            <div key={r.name}
              className={`rounded-xl p-4 text-center border ${r.highlight ? 'bg-emerald-950/60 border-emerald-800' : 'bg-zinc-900 border-zinc-800'}`}>
              <div className={`font-bold text-sm mb-2 ${r.highlight ? 'text-emerald-400' : 'text-zinc-300'}`}>{r.name}</div>
              <div className="text-zinc-500 text-xs">时间</div>
              <div className="text-zinc-200 font-mono font-bold mb-1">{r.time}</div>
              <div className="text-zinc-500 text-xs">空间</div>
              <div className={`font-mono font-bold ${r.highlight ? 'text-emerald-400 text-base' : 'text-zinc-400 text-sm'}`}>{r.space}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}