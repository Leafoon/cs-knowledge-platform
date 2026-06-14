"use client";
import React, { useState, useMemo } from "react";

/** MorrisTraversalStep — Morris 中序遍历线索指针修改步进图 */

// ─────────────── 数据结构 ───────────────────────────
interface TNode { val: number; left: TNode | null; right: TNode | null; id: number; origRight?: number | null; }
interface LayoutNode extends TNode { x: number; y: number; left: LayoutNode | null; right: LayoutNode | null; }

let _uid = 0;
function n(val: number, l: TNode | null = null, r: TNode | null = null): TNode {
  return { val, left: l, right: r, id: _uid++ };
}

const PRESETS: Record<string, TNode> = (() => {
  _uid = 0; const t1 = n(4, n(2, n(1), n(3)), n(6, n(5), n(7)));
  _uid = 0; const t2 = n(1, n(2, n(4), n(5)), n(3));
  _uid = 0; const t3 = n(3, n(1, null, n(2)), n(5, n(4), null));
  return { "标准平衡": t1, "简单树": t2, "非对称": t3 };
})();

function layout(node: TNode | null, depth: number, xMin: number, xMax: number): LayoutNode | null {
  if (!node) return null;
  const x = (xMin + xMax) / 2, y = depth * 70 + 40;
  return { ...node, x, y, left: layout(node.left, depth + 1, xMin, (xMin + xMax) / 2), right: layout(node.right, depth + 1, (xMin + xMax) / 2, xMax) };
}

// ─────────────── 步骤生成 ───────────────────────────
interface Thread { fromId: number; toId: number; }
interface MorrisStep {
  currId: number | null;
  predId: number | null;
  threads: Thread[];           // 当前所有活跃线索指针
  visited: number[];           // 已访问节点 id（有序）
  phase: 'find_pred' | 'build_thread' | 'visit' | 'remove_thread' | 'done';
  description: string;
}

function buildMorrisSteps(root: TNode | null): MorrisStep[] {
  if (!root) return [];

  // 深拷贝树（避免修改 preset）
  function cloneTree(node: TNode | null): TNode | null {
    if (!node) return null;
    return { val: node.val, left: cloneTree(node.left), right: cloneTree(node.right), id: node.id };
  }
  const cloned = cloneTree(root) as TNode;

  // 建立 id → 节点映射
  const nodeMap = new Map<number, TNode>();
  function buildMap(node: TNode | null) {
    if (!node) return;
    nodeMap.set(node.id, node);
    buildMap(node.left);
    buildMap(node.right);
  }
  buildMap(cloned);

  const steps: MorrisStep[] = [];
  const threads: Thread[] = [];
  const visited: number[] = [];
  let curr: TNode | null = cloned;

  while (curr) {
    if (!curr.left) {
      // 直接访问
      visited.push(curr.id);
      steps.push({
        currId: curr.id, predId: null, threads: [...threads], visited: [...visited],
        phase: 'visit',
        description: `节点 ${curr.val} 无左子树，直接访问（中序位置），移动到右子/线索`,
      });
      curr = curr.right;
    } else {
      // 找前驱
      steps.push({
        currId: curr.id, predId: null, threads: [...threads], visited: [...visited],
        phase: 'find_pred',
        description: `节点 ${curr.val} 有左子树，开始寻找中序前驱（左子树最右节点）`,
      });

      let pred: TNode = curr.left;
      while (pred.right && pred.right.id !== curr.id) {
        pred = pred.right;
      }

      steps.push({
        currId: curr.id, predId: pred.id, threads: [...threads], visited: [...visited],
        phase: 'find_pred',
        description: `找到中序前驱：节点 ${pred.val}，检查是否已有线索…`,
      });

      if (!pred.right) {
        // 建立线索
        pred.right = curr;
        threads.push({ fromId: pred.id, toId: curr.id });
        steps.push({
          currId: curr.id, predId: pred.id, threads: [...threads], visited: [...visited],
          phase: 'build_thread',
          description: `建立线索：${pred.val} → ${curr.val}（暂时将 ${pred.val}.right 指向 ${curr.val}），向左移动`,
        });
        curr = curr.left;
      } else {
        // 断开线索，访问当前节点
        pred.right = null;
        const tIdx = threads.findIndex(t => t.fromId === pred.id && t.toId === curr!.id);
        if (tIdx !== -1) threads.splice(tIdx, 1);
        visited.push(curr.id);
        steps.push({
          currId: curr.id, predId: pred.id, threads: [...threads], visited: [...visited],
          phase: 'remove_thread',
          description: `发现线索已存在（左子树已遍历完）！断开线索 ${pred.val}→${curr.val}，访问节点 ${curr.val}，移动到右子树`,
        });
        curr = curr.right;
      }
    }
  }

  steps.push({
    currId: null, predId: null, threads: [], visited: [...visited],
    phase: 'done',
    description: `✅ Morris 中序遍历完成！访问序列：[${visited.map(id => nodeMap.get(id)?.val).join(', ')}]（有序 ✓）`,
  });

  return steps;
}

// ─────────────── SVG 树渲染 ─────────────────────────
function TreeSVG({ tree, step, nodeMap }: { tree: LayoutNode | null; step: MorrisStep; nodeMap: Map<number, LayoutNode> }) {
  const W = 300, H = 220;

  function edges(node: LayoutNode | null): React.ReactNode[] {
    if (!node) return [];
    const res: React.ReactNode[] = [];
    if (node.left) res.push(<line key={`el${node.id}`} x1={node.x} y1={node.y} x2={node.left.x} y2={node.left.y} stroke="#374151" strokeWidth={1.5} />);
    if (node.right) res.push(<line key={`er${node.id}`} x1={node.x} y1={node.y} x2={node.right.x} y2={node.right.y} stroke="#374151" strokeWidth={1.5} />);
    return [...res, ...edges(node.left), ...edges(node.right)];
  }

  function nodes(node: LayoutNode | null): React.ReactNode[] {
    if (!node) return [];
    const isCurr = node.id === step.currId;
    const isPred = node.id === step.predId;
    const isVisited = step.visited.includes(node.id);
    const fill = isCurr ? '#2563eb' : isPred ? '#d97706' : isVisited ? '#166534' : '#1e293b';
    const stroke = isCurr ? '#93c5fd' : isPred ? '#fbbf24' : isVisited ? '#4ade80' : '#4b5563';
    const visitedOrder = step.visited.indexOf(node.id);
    return [
      <circle key={`c${node.id}`} cx={node.x} cy={node.y} r={18} fill={fill} stroke={stroke} strokeWidth={(isCurr || isPred) ? 2.5 : 1.5} />,
      <text key={`v${node.id}`} x={node.x} y={node.y + 1} textAnchor="middle" dominantBaseline="middle" fill="white" fontSize={12} fontWeight="bold">{node.val}</text>,
      isVisited && <text key={`ord${node.id}`} x={node.x + 20} y={node.y - 14} fill="#4ade80" fontSize={9}>{visitedOrder + 1}</text>,
      ...nodes(node.left), ...nodes(node.right),
    ].filter(Boolean);
  }

  // 线索箭头（用弯曲路径）
  function threadArrows(): React.ReactNode[] {
    if (!step.threads?.length) return [];
    return step.threads.map(({ fromId, toId }) => {
      const from = nodeMap.get(fromId);
      const to = nodeMap.get(toId);
      if (!from || !to) return null;
      // 弯曲箭头（贝塞尔）
      const dx = to.x - from.x, dy = to.y - from.y;
      const cx1 = from.x + dx * 0.3 + 30, cy1 = from.y + 20;
      const cx2 = to.x - dx * 0.1 + 30, cy2 = to.y - 30;
      return (
        <g key={`thread-${fromId}-${toId}`}>
          <path
            d={`M ${from.x} ${from.y + 18} C ${cx1} ${cy1}, ${cx2} ${cy2}, ${to.x} ${to.y - 18}`}
            fill="none" stroke="#fbbf24" strokeWidth={1.5} strokeDasharray="4,3" opacity={0.8}
          />
          <polygon
            points={`${to.x},${to.y - 17} ${to.x - 4},${to.y - 26} ${to.x + 4},${to.y - 26}`}
            fill="#fbbf24"
          />
        </g>
      );
    });
  }

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ maxHeight: 220 }}>
      {edges(tree)}
      {threadArrows()}
      {nodes(tree)}
    </svg>
  );
}

// ─────────────── 主组件 ─────────────────────────────
export default function MorrisTraversalStep() {
  const [preset, setPreset] = useState("标准平衡");
  const [stepIdx, setStepIdx] = useState(0);

  const tree = useMemo(() => { _uid = 0; return layout(PRESETS[preset], 0, 0, 300); }, [preset]);
  const steps = useMemo(() => buildMorrisSteps(PRESETS[preset]), [preset]);
  const step = steps[Math.min(stepIdx, steps.length - 1)];

  // 建立 id → LayoutNode 映射（供线索箭头用）
  const nodeMap = useMemo(() => {
    const m = new Map<number, LayoutNode>();
    function collect(n: LayoutNode | null) {
      if (!n) return;
      m.set(n.id, n);
      collect(n.left); collect(n.right);
    }
    if (tree) collect(tree);
    return m;
  }, [tree]);

  const reset = () => setStepIdx(0);

  const phaseColor: Record<string, string> = {
    find_pred: 'bg-amber-500/10 border-amber-500/40',
    build_thread: 'bg-yellow-500/10 border-yellow-500/40',
    visit: 'bg-green-500/10 border-green-500/40',
    remove_thread: 'bg-red-500/10 border-red-500/40',
    done: 'bg-blue-500/10 border-blue-500/40',
  };

  const phaseTextColor: Record<string, string> = {
    find_pred: 'text-amber-300', build_thread: 'text-yellow-300',
    visit: 'text-green-300', remove_thread: 'text-red-300', done: 'text-blue-300',
  };

  return (
    <div className="rounded-xl border border-border-subtle bg-bg-secondary p-5 space-y-4 text-sm font-mono">
      <div className="flex items-start justify-between gap-3 flex-wrap">
        <div>
          <h3 className="text-base font-bold text-text-primary">🧵 Morris 中序遍历步进图</h3>
          <p className="text-xs text-text-tertiary mt-0.5">线索二叉树原理：临时修改空指针为"线索"，实现 O(1) 空间遍历</p>
        </div>
        <div className="flex items-center gap-2 text-xs">
          <span className="px-2 py-1 bg-green-500/10 border border-green-500/30 rounded text-green-300">
            已访问：[{step?.visited.map(id => nodeMap.get(id)?.val ?? '?').join(', ')}]
          </span>
        </div>
      </div>

      {/* 预设 */}
      <div className="flex gap-2 flex-wrap">
        {Object.keys(PRESETS).map(p => (
          <button key={p} onClick={() => { setPreset(p); reset(); }}
            className={`px-2 py-1 rounded text-xs border transition-colors ${preset === p ? "bg-blue-600 text-white border-blue-600" : "bg-bg-tertiary text-text-secondary border-border-subtle hover:border-blue-400"}`}>
            {p}
          </button>
        ))}
      </div>

      {/* 树可视化 */}
      <div className="bg-bg-tertiary rounded-lg border border-border-subtle p-3">
        {tree && <TreeSVG tree={tree} step={step} nodeMap={nodeMap} />}
        <div className="flex gap-4 text-xs text-text-secondary mt-2 flex-wrap">
          <span><span className="inline-block w-3 h-3 rounded-full bg-blue-600 mr-1" />curr（当前）</span>
          <span><span className="inline-block w-3 h-3 rounded-full bg-amber-600 mr-1" />pred（前驱）</span>
          <span><span className="inline-block w-3 h-3 rounded-full bg-green-800 mr-1" />已访问</span>
          <span><span className="inline-block w-2 h-0.5 bg-yellow-400 mr-1 inline-block align-middle border-dashed border-b-2 border-yellow-400" />线索指针</span>
        </div>
      </div>

      {/* 当前步骤 */}
      <div className={`rounded-lg p-3 border min-h-[56px] transition-all ${phaseColor[step?.phase] ?? "bg-bg-tertiary border-border-subtle"}`}>
        <div className="flex justify-between text-xs mb-1">
          <span className="text-text-tertiary">步骤 {stepIdx + 1} / {steps.length}</span>
          <span className="text-xs font-bold text-text-tertiary capitalize">
            {step?.phase === 'find_pred' ? '寻找前驱' : step?.phase === 'build_thread' ? '建立线索' : step?.phase === 'visit' ? '访问节点' : step?.phase === 'remove_thread' ? '断开线索' : '完成'}
          </span>
        </div>
        <p className={`text-sm ${phaseTextColor[step?.phase] ?? "text-text-primary"}`}>
          {step?.description}
        </p>
      </div>

      {/* 算法摘要 */}
      <div className="bg-bg-tertiary rounded-lg border border-border-subtle p-3 text-xs text-text-secondary space-y-2">
        <div className="font-bold text-text-primary">Morris 遍历核心逻辑</div>
        <div className="space-y-0.5 font-mono">
          <div><span className="text-blue-400">while</span> curr:</div>
          <div className="pl-3"><span className="text-blue-400">if</span> curr.left <span className="text-blue-400">is None</span>: <span className="text-green-400">访问 curr</span>; curr = curr.right</div>
          <div className="pl-3"><span className="text-blue-400">else</span>: pred = 左子树最右节点</div>
          <div className="pl-6"><span className="text-blue-400">if</span> pred.right <span className="text-blue-400">is None</span>: <span className="text-amber-400">建线索</span>; curr = curr.left</div>
          <div className="pl-6"><span className="text-blue-400">else</span>: <span className="text-rose-400">断线索</span>; <span className="text-green-400">访问 curr</span>; curr = curr.right</div>
        </div>
        <div className="flex gap-4 pt-1 border-t border-border-subtle">
          <span>时间：<span className="text-green-400">O(n)</span></span>
          <span>空间：<span className="text-green-400">O(1)</span>（不计输出）</span>
          <span>每条边遍历：<span className="text-text-primary">≤ 2 次</span></span>
        </div>
      </div>

      {/* 控制 */}
      <div className="flex gap-2 justify-center">
        <button onClick={reset} className="px-4 py-2 rounded-lg bg-bg-tertiary text-text-secondary border border-border-subtle text-xs hover:border-blue-400">↩ 重置</button>
        <button onClick={() => setStepIdx(i => Math.max(0, i - 1))} disabled={stepIdx === 0} className="px-4 py-2 rounded-lg bg-bg-tertiary text-text-secondary border border-border-subtle text-xs disabled:opacity-40 hover:border-blue-400">← 上一步</button>
        <button onClick={() => setStepIdx(i => Math.min(steps.length - 1, i + 1))} disabled={stepIdx >= steps.length - 1} className="px-4 py-2 rounded-lg bg-blue-600 text-white text-xs disabled:opacity-40 hover:bg-blue-700">下一步 →</button>
      </div>
    </div>
  );
}
