"use client";
import React, { useState, useMemo } from "react";

/** TreeHeightCalculator — 递归高度计算可视化：递归调用栈展开动画 */

// ─────────────── 树节点与布局 ───────────────────────
interface TNode { val: number; left: TNode | null; right: TNode | null; id: number; }
interface LayoutNode extends TNode { x: number; y: number; left: LayoutNode | null; right: LayoutNode | null; }

let _uid = 0;
function n(val: number, l: TNode | null = null, r: TNode | null = null): TNode {
  return { val, left: l, right: r, id: _uid++ };
}
function reset() { _uid = 0; }

const PRESETS: Record<string, TNode> = (() => {
  reset(); const t1 = n(1, n(2, n(4), n(5)), n(3));
  reset(); const t2 = n(1, n(2, n(3, n(4), null), null), null);
  reset(); const t3 = n(1, n(2), n(3, n(4), n(5, n(6), n(7))));
  reset(); const t4 = n(1);
  return { "平衡树": t1, "退化链（左偏）": t2, "不平衡右深": t3, "单节点": t4 };
})();

function layout(node: TNode | null, depth: number, xMin: number, xMax: number): LayoutNode | null {
  if (!node) return null;
  const x = (xMin + xMax) / 2, y = depth * 70 + 40;
  const mid = (xMin + xMax) / 2;
  return { ...node, x, y, left: layout(node.left, depth + 1, xMin, mid), right: layout(node.right, depth + 1, mid, xMax) };
}

// ─────────────── 步骤生成 ───────────────────────────
interface Step {
  nodeId: number | null;   // 当前焦点节点
  phase: 'enter' | 'waitLeft' | 'waitRight' | 'return';
  depth: number;           // 递归深度（0 = root）
  callStack: Array<{ nodeId: number | null; label: string; phase: string }>;
  returned: Record<number, number>; // nodeId → 已返回的高度
  description: string;
  returnVal?: number;
}

function buildHeightSteps(root: TNode | null): Step[] {
  const steps: Step[] = [];
  const returned: Record<number, number> = {};

  function generateSteps(node: TNode | null, depth: number, stack: Step['callStack']): number {
    if (!node) {
      steps.push({
        nodeId: null, phase: 'return', depth,
        callStack: [...stack, { nodeId: null, label: 'height(null)', phase: 'return' }],
        returned: { ...returned },
        description: `height(null) → 空树，返回 -1`,
        returnVal: -1,
      });
      return -1;
    }

    const enterStack = [...stack, { nodeId: node.id, label: `height(${node.val})`, phase: 'enter' }];
    steps.push({
      nodeId: node.id, phase: 'enter', depth,
      callStack: enterStack,
      returned: { ...returned },
      description: `进入 height(${node.val})，准备递归左子树`,
    });

    const leftH = generateSteps(node.left, depth + 1, enterStack);

    steps.push({
      nodeId: node.id, phase: 'waitRight', depth,
      callStack: enterStack,
      returned: { ...returned },
      description: `height(${node.val}) 左子树已返回 ${leftH}，准备递归右子树`,
    });

    const rightH = generateSteps(node.right, depth + 1, enterStack);
    const result = 1 + Math.max(leftH, rightH);
    returned[node.id] = result;

    steps.push({
      nodeId: node.id, phase: 'return', depth,
      callStack: enterStack,
      returned: { ...returned },
      description: `height(${node.val}) = 1 + max(${leftH}, ${rightH}) = ${result}，返回父节点`,
      returnVal: result,
    });

    return result;
  }

  generateSteps(root, 0, []);
  return steps;
}

// ─────────────── SVG 树渲染 ─────────────────────────
function TreeSVG({ tree, step }: { tree: LayoutNode | null; step: Step }) {
  const W = 300, H = 200;
  function renderEdges(node: LayoutNode | null): React.ReactNode[] {
    if (!node) return [];
    const edges: React.ReactNode[] = [];
    if (node.left) edges.push(<line key={`e-l-${node.id}`} x1={node.x} y1={node.y} x2={node.left.x} y2={node.left.y} stroke="#4b5563" strokeWidth={1.5} />);
    if (node.right) edges.push(<line key={`e-r-${node.id}`} x1={node.x} y1={node.y} x2={node.right.x} y2={node.right.y} stroke="#4b5563" strokeWidth={1.5} />);
    return [...edges, ...renderEdges(node.left), ...renderEdges(node.right)];
  }

  function renderNodes(node: LayoutNode | null): React.ReactNode[] {
    if (!node) return [];
    const isFocus = step.nodeId === node.id;
    const hasReturned = step.returned[node.id] !== undefined;
    const fill = isFocus
      ? step.phase === 'return' ? '#22c55e' : '#3b82f6'
      : hasReturned ? '#166534' : '#1e293b';
    const stroke = isFocus ? (step.phase === 'return' ? '#86efac' : '#93c5fd') : '#4b5563';

    return [
      <circle key={`c-${node.id}`} cx={node.x} cy={node.y} r={18} fill={fill} stroke={stroke} strokeWidth={isFocus ? 2.5 : 1.5} />,
      <text key={`v-${node.id}`} x={node.x} y={node.y + 1} textAnchor="middle" dominantBaseline="middle" fill="white" fontSize={12} fontWeight="bold">{node.val}</text>,
      hasReturned && (
        <text key={`r-${node.id}`} x={node.x} y={node.y - 24} textAnchor="middle" fill="#4ade80" fontSize={10}>{`h=${step.returned[node.id]}`}</text>
      ),
      ...renderNodes(node.left), ...renderNodes(node.right),
    ].filter(Boolean);
  }

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ maxHeight: 200 }}>
      {renderEdges(tree)}
      {renderNodes(tree)}
    </svg>
  );
}

// ─────────────── 主组件 ─────────────────────────────
export default function TreeHeightCalculator() {
  const [preset, setPreset] = useState("平衡树");
  const [stepIdx, setStepIdx] = useState(0);

  const tree = useMemo(() => { reset(); return layout(PRESETS[preset], 0, 0, 300); }, [preset]);
  const steps = useMemo(() => buildHeightSteps(PRESETS[preset]), [preset]);
  const step = steps[Math.min(stepIdx, steps.length - 1)];

  const reset2 = () => setStepIdx(0);
  const prev = () => setStepIdx(i => Math.max(0, i - 1));
  const next = () => setStepIdx(i => Math.min(steps.length - 1, i + 1));

  return (
    <div className="rounded-xl border border-border-subtle bg-bg-secondary p-5 space-y-4 text-sm font-mono">
      <div className="flex items-start justify-between gap-3 flex-wrap">
        <div>
          <h3 className="text-base font-bold text-text-primary">🌲 递归高度计算可视化</h3>
          <p className="text-xs text-text-tertiary mt-0.5">高度 = 1 + max(左子树高度, 右子树高度)，空树高度 = -1</p>
        </div>
        <div className="text-xs px-3 py-1 bg-bg-tertiary border border-border-subtle rounded text-text-secondary">
          步骤 {stepIdx + 1} / {steps.length}
        </div>
      </div>

      {/* 预设 */}
      <div className="flex gap-2 flex-wrap">
        {Object.keys(PRESETS).map(p => (
          <button key={p} onClick={() => { setPreset(p); reset2(); }}
            className={`px-2 py-1 rounded text-xs border transition-colors ${preset === p ? "bg-blue-600 text-white border-blue-600" : "bg-bg-tertiary text-text-secondary border-border-subtle hover:border-blue-400"}`}>
            {p}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* 树可视化 */}
        <div className="bg-bg-tertiary rounded-lg border border-border-subtle p-3">
          <div className="text-xs text-text-tertiary mb-2">树结构（🔵 当前节点 | 🟢 已返回）</div>
          <TreeSVG tree={tree} step={step} />
          {/* 图例 */}
          <div className="flex gap-3 text-xs text-text-secondary mt-2">
            <span><span className="inline-block w-3 h-3 rounded-full bg-blue-500 mr-1" />当前处理</span>
            <span><span className="inline-block w-3 h-3 rounded-full bg-green-500 mr-1" />已返回</span>
            <span><span className="inline-block w-3 h-3 rounded-full bg-green-800 mr-1" />已完成</span>
          </div>
        </div>

        {/* 调用栈 */}
        <div className="bg-bg-tertiary rounded-lg border border-border-subtle p-3">
          <div className="text-xs text-text-tertiary mb-2">递归调用栈（栈顶 = 当前执行帧）</div>
          <div className="space-y-1 min-h-[120px]">
            {[...step.callStack].reverse().map((frame, i) => (
              <div key={i} className={`rounded px-2 py-1.5 border text-xs flex justify-between transition-colors ${
                i === 0 // 栈顶（显示时反转后的第一个）
                  ? "bg-blue-500/20 border-blue-500/50 text-blue-300"
                  : "bg-bg-secondary border-border-subtle text-text-secondary"
              }`}>
                <span style={{ paddingLeft: `${(step.callStack.length - 1 - i) * 8}px` }}>
                  {frame.label}
                </span>
                {frame.phase === 'return' && <span className="text-green-400 text-[10px]">→ 返回</span>}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* 当前步骤说明 */}
      <div className={`rounded-lg p-3 border min-h-[56px] transition-colors ${
        step.phase === 'return' && step.returnVal !== undefined
          ? "bg-green-500/10 border-green-500/40"
          : "bg-bg-tertiary border-border-subtle"
      }`}>
        <p className={`text-sm ${step.phase === 'return' ? "text-green-300" : "text-text-primary"}`}>
          {step.description}
        </p>
        {step.returnVal !== undefined && (
          <div className="mt-1 text-xs font-mono">
            <span className="text-text-tertiary">返回值：</span>
            <span className="text-green-400 font-bold">{step.returnVal}</span>
          </div>
        )}
      </div>

      {/* 递归公式 */}
      <div className="bg-bg-tertiary rounded p-2 border border-border-subtle text-xs text-text-secondary font-mono">
        <span className="text-blue-400">height</span>(node) = <span className="text-text-tertiary">node == null ?</span>
        <span className="text-rose-400"> -1</span>
        <span className="text-text-tertiary"> : </span>
        <span className="text-amber-400">1</span>
        <span className="text-text-tertiary"> + </span>
        <span className="text-blue-400">max</span>(<span className="text-blue-400">height</span>(node.left), <span className="text-blue-400">height</span>(node.right))
      </div>

      {/* 控制按钮 */}
      <div className="flex gap-2 justify-center">
        <button onClick={reset2} className="px-4 py-2 rounded-lg bg-bg-tertiary text-text-secondary border border-border-subtle text-xs hover:border-blue-400 transition-colors">↩ 重置</button>
        <button onClick={prev} disabled={stepIdx === 0} className="px-4 py-2 rounded-lg bg-bg-tertiary text-text-secondary border border-border-subtle text-xs disabled:opacity-40 hover:border-blue-400 transition-colors">← 上一步</button>
        <button onClick={next} disabled={stepIdx >= steps.length - 1} className="px-4 py-2 rounded-lg bg-blue-600 text-white text-xs disabled:opacity-40 hover:bg-blue-700 transition-colors">下一步 →</button>
      </div>
    </div>
  );
}
