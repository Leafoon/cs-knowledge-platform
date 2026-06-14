"use client";
import React, { useState, useMemo } from "react";

// ─── Types ────────────────────────────────────────────────────────────────────
interface OSTNode {
  key: number;
  size: number;   // subtree node count
  left: OSTNode | null;
  right: OSTNode | null;
  id: number;
}

interface Step {
  visitedIds: number[];
  currentId: number | null;
  resultId: number | null;
  message: string;
  detail: string;
  leftSize: number;    // size of left subtree at current node
  rank: number | null; // computed so far
}

// ─── OST utilities ────────────────────────────────────────────────────────────
let _id = 0;

function sz(node: OSTNode | null) { return node?.size ?? 0; }

function insert(root: OSTNode | null, key: number): OSTNode {
  if (!root) return { key, size: 1, left: null, right: null, id: _id++ };
  if (key < root.key) {
    const left = insert(root.left, key);
    return { ...root, left, size: sz(left) + sz(root.right) + 1 };
  } else if (key > root.key) {
    const right = insert(root.right, key);
    return { ...root, right, size: sz(root.left) + sz(right) + 1 };
  }
  return root;
}

function buildOST(keys: number[]): OSTNode | null {
  _id = 0;
  let root: OSTNode | null = null;
  for (const k of keys) root = insert(root, k);
  return root;
}

// ─── Layout ───────────────────────────────────────────────────────────────────
interface LayoutNode { node: OSTNode; x: number; y: number; }

function layoutOST(root: OSTNode | null, xMin = 0, xMax = 540, depth = 0): LayoutNode[] {
  if (!root) return [];
  const x = (xMin + xMax) / 2, y = depth * 72 + 36;
  return [{ node: root, x, y }, ...layoutOST(root.left, xMin, x, depth + 1), ...layoutOST(root.right, x, xMax, depth + 1)];
}

function treeHeight(n: OSTNode | null): number {
  if (!n) return 0;
  return 1 + Math.max(treeHeight(n.left), treeHeight(n.right));
}

// ─── Step generators ──────────────────────────────────────────────────────────
function generateSelectSteps(root: OSTNode | null, k: number): Step[] {
  const steps: Step[] = [];
  const path: number[] = [];

  function dfs(node: OSTNode | null, kk: number): number | null {
    if (!node) {
      steps.push({ visitedIds: [...path], currentId: null, resultId: null,
        message: `越界：k=${kk} 超出范围`, detail: "", leftSize: 0, rank: null });
      return null;
    }
    path.push(node.id);
    const ls = sz(node.left);
    const r = ls + 1;
    steps.push({
      visitedIds: [...path], currentId: node.id, resultId: null,
      message: `节点 ${node.key}：左子树大小 ls=${ls}，当前排名 r=${r}`,
      detail: kk === r ? `k=${kk} = r=${r} → 命中！` : kk < r ? `k=${kk} < r=${r} → 去左子树找第 ${kk} 小` : `k=${kk} > r=${r} → 去右子树找第 ${kk - r} 小`,
      leftSize: ls, rank: r,
    });
    if (kk === r) {
      steps.push({ visitedIds: [...path], currentId: node.id, resultId: node.id,
        message: `✅ 第 ${k} 小 = ${node.key}`, detail: `ls+1=${r} = k=${kk}，命中节点 ${node.key}`, leftSize: ls, rank: r });
      return node.id;
    } else if (kk < r) {
      return dfs(node.left, kk);
    } else {
      return dfs(node.right, kk - r);
    }
  }

  dfs(root, k);
  return steps;
}

function generateRankSteps(root: OSTNode | null, targetKey: number): Step[] {
  const steps: Step[] = [];
  const path: number[] = [];
  let accumulated = 0;

  function dfs(node: OSTNode | null): boolean {
    if (!node) {
      steps.push({ visitedIds: [...path], currentId: null, resultId: null,
        message: `未找到 ${targetKey}`, detail: "", leftSize: 0, rank: null });
      return false;
    }
    path.push(node.id);
    const ls = sz(node.left);
    if (node.key === targetKey) {
      accumulated += ls + 1;
      steps.push({ visitedIds: [...path], currentId: node.id, resultId: node.id,
        message: `✅ 找到 ${targetKey}，排名 = 累计 + 左子树大小 + 1 = ${accumulated}`,
        detail: `左子树大小=${ls}，累计=${accumulated}`, leftSize: ls, rank: accumulated });
      return true;
    } else if (targetKey < node.key) {
      steps.push({ visitedIds: [...path], currentId: node.id, resultId: null,
        message: `${targetKey} < ${node.key}：向左走（当前累计=${accumulated}，暂不计入）`,
        detail: `去左子树，累计不变`, leftSize: ls, rank: null });
      return dfs(node.left);
    } else {
      accumulated += ls + 1; // left subtree + root
      steps.push({ visitedIds: [...path], currentId: node.id, resultId: null,
        message: `${targetKey} > ${node.key}：向右走，累计 += ls+1 = ${accumulated}`,
        detail: `去右子树，累计 += ${ls}+1`, leftSize: ls, rank: null });
      return dfs(node.right);
    }
  }

  dfs(root, );
  return steps;
}

// ─── SVG Node ─────────────────────────────────────────────────────────────────
function TreeSVG({ layoutNodes, step, showSize }: { layoutNodes: LayoutNode[]; step: Step | null; showSize: boolean }) {
  const nodeMap = useMemo(() => {
    const m = new Map<number, LayoutNode>();
    for (const ln of layoutNodes) m.set(ln.node.id, ln);
    return m;
  }, [layoutNodes]);

  const h = Math.max(...layoutNodes.map(l => l.y), 0);
  const svgH = Math.max(200, h + 60);

  function getColors(id: number) {
    if (!step) return { fill: "var(--color-bg-tertiary)", stroke: "var(--color-border-subtle)", text: "var(--color-text-primary)" };
    if (step.resultId === id) return { fill: "#f59e0b", stroke: "#f59e0b", text: "#fff" };
    if (step.currentId === id) return { fill: "#3b82f6", stroke: "#3b82f6", text: "#fff" };
    if (step.visitedIds.includes(id)) return { fill: "#6b728022", stroke: "#6b7280", text: "var(--color-text-secondary)" };
    return { fill: "var(--color-bg-tertiary)", stroke: "var(--color-border-subtle)", text: "var(--color-text-primary)" };
  }

  return (
    <svg width="100%" viewBox={`0 0 540 ${svgH}`} className="overflow-visible">
      {layoutNodes.map(({ node, x, y }) => {
        const edges: React.ReactNode[] = [];
        const lc = node.left ? nodeMap.get(node.left.id) : null;
        const rc = node.right ? nodeMap.get(node.right.id) : null;
        if (lc) edges.push(<line key={`l${node.id}`} x1={x} y1={y} x2={lc.x} y2={lc.y} stroke="var(--color-border-subtle)" strokeWidth={1.5} />);
        if (rc) edges.push(<line key={`r${node.id}`} x1={x} y1={y} x2={rc.x} y2={rc.y} stroke="var(--color-border-subtle)" strokeWidth={1.5} />);
        return edges;
      })}
      {layoutNodes.map(({ node, x, y }) => {
        const { fill, stroke, text } = getColors(node.id);
        return (
          <g key={node.id}>
            <circle cx={x} cy={y} r={22} fill={fill} stroke={stroke} strokeWidth={2} />
            <text x={x} y={y + 4} textAnchor="middle" fontSize={12} fill={text} fontWeight="700">{node.key}</text>
            {showSize && (
              <>
                <rect x={x - 12} y={y + 25} width={24} height={14} rx={7} fill="#8b5cf622" stroke="#8b5cf6" strokeWidth={1} />
                <text x={x} y={y + 35} textAnchor="middle" fontSize={9} fill="#8b5cf6" fontWeight="600">s={node.size}</text>
              </>
            )}
          </g>
        );
      })}
    </svg>
  );
}

// ─── Main Component ───────────────────────────────────────────────────────────
const PRESETS = [
  { label: "7节点完全BST", keys: [8, 4, 12, 2, 6, 10, 14] },
  { label: "9节点综合", keys: [10, 5, 15, 3, 7, 12, 20, 1, 8] },
  { label: "5节点简单", keys: [5, 2, 8, 1, 6] },
];

type QueryMode = "select" | "rank";

export default function AugmentedBSTDemo() {
  const [presetIdx, setPresetIdx] = useState(0);
  const root = useMemo(() => buildOST(PRESETS[presetIdx].keys), [presetIdx]);
  const layoutNodes = useMemo(() => layoutOST(root), [root]);
  const [queryMode, setQueryMode] = useState<QueryMode>("select");
  const [kInput, setKInput] = useState(3);
  const [keyInput, setKeyInput] = useState(PRESETS[0].keys[2]);
  const [showSize, setShowSize] = useState(true);
  const [steps, setSteps] = useState<Step[]>([]);
  const [stepIdx, setStepIdx] = useState(-1);

  const n = root?.size ?? 0;
  const currentStep = stepIdx >= 0 && stepIdx < steps.length ? steps[stepIdx] : null;

  function handleRun() {
    let s: Step[] = [];
    if (queryMode === "select") s = generateSelectSteps(root, kInput);
    else s = generateRankSteps(root, keyInput);
    setSteps(s);
    setStepIdx(0);
  }

  function handlePreset(idx: number) {
    setPresetIdx(idx);
    setSteps([]);
    setStepIdx(-1);
    setKInput(2);
    setKeyInput(PRESETS[idx].keys[1]);
  }

  return (
    <div className="rounded-xl border border-border-subtle bg-bg-secondary p-5 space-y-4 font-mono text-sm">
      <h3 className="text-base font-semibold text-text-primary">顺序统计树（Order-Statistics Tree）</h3>

      {/* Presets */}
      <div className="flex flex-wrap gap-2">
        {PRESETS.map((p, i) => (
          <button key={i} onClick={() => handlePreset(i)}
            className={`px-3 py-1 rounded-lg text-xs border transition-colors ${i === presetIdx
              ? "bg-purple-500/20 border-purple-500 text-purple-400"
              : "border-border-subtle text-text-secondary hover:border-purple-400"}`}>
            {p.label}
          </button>
        ))}
        <button onClick={() => setShowSize(s => !s)}
          className={`ml-auto px-3 py-1 rounded-lg text-xs border transition-colors ${showSize
            ? "bg-purple-500/20 border-purple-500 text-purple-400" : "border-border-subtle text-text-secondary"}`}>
          {showSize ? "隐藏 size" : "显示 size"}
        </button>
      </div>

      {/* Explanation banner */}
      <div className="rounded-lg bg-purple-500/10 border border-purple-500/30 px-4 py-2 text-xs text-purple-300 space-y-1">
        <div className="font-semibold text-purple-200">顺序统计树 = BST + 每节点额外存 size（子树节点数）</div>
        <div>• <span className="text-purple-200">OS-SELECT(k)</span>：O(log n) 找第 k 小元素，利用 size 决策左/右走向</div>
        <div>• <span className="text-purple-200">OS-RANK(x)</span>：O(log n) 求键 x 的排名（中序第几位），沿路累加左子树大小</div>
        <div>• 每次插入/删除后，沿路径回溯更新 <span className="text-purple-200">size(x) = size(left)+size(right)+1</span></div>
      </div>

      {/* Query controls */}
      <div className="flex flex-wrap gap-3 items-center">
        <div className="flex gap-2">
          {(["select", "rank"] as QueryMode[]).map(m => (
            <button key={m} onClick={() => { setQueryMode(m); setSteps([]); setStepIdx(-1); }}
              className={`px-3 py-1 rounded-lg border text-xs transition-colors ${queryMode === m
                ? "bg-bg-tertiary border-border-strong text-text-primary font-semibold"
                : "border-border-subtle text-text-secondary hover:border-border-strong"}`}>
              {m === "select" ? "OS-SELECT(k)" : "OS-RANK(key)"}
            </button>
          ))}
        </div>
        {queryMode === "select" ? (
          <div className="flex items-center gap-2">
            <span className="text-text-secondary text-xs">k =</span>
            <input type="number" min={1} max={n} value={kInput} onChange={e => setKInput(Number(e.target.value))}
              className="w-16 px-2 py-1 rounded border border-border-subtle bg-bg-tertiary text-text-primary text-center text-xs" />
            <span className="text-text-tertiary text-xs">（1 ≤ k ≤ {n}）</span>
          </div>
        ) : (
          <div className="flex items-center gap-2">
            <span className="text-text-secondary text-xs">key =</span>
            <input type="number" value={keyInput} onChange={e => setKeyInput(Number(e.target.value))}
              className="w-20 px-2 py-1 rounded border border-border-subtle bg-bg-tertiary text-text-primary text-center text-xs" />
          </div>
        )}
        <button onClick={handleRun}
          className="px-3 py-1 rounded-lg bg-purple-500/20 border border-purple-500 text-purple-400 text-xs hover:bg-purple-500/30">
          分步演示
        </button>
      </div>

      {/* Tree SVG */}
      <div className="bg-bg-tertiary rounded-lg p-3 min-h-[240px]">
        <TreeSVG layoutNodes={layoutNodes} step={currentStep} showSize={showSize} />
      </div>

      {/* Legend */}
      <div className="flex flex-wrap gap-3 text-xs text-text-secondary">
        <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-blue-500 inline-block" /> 当前节点</span>
        <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-amber-400 inline-block" /> 结果节点</span>
        <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-gray-500 inline-block" /> 已访问</span>
        <span className="flex items-center gap-1"><span className="px-2 py-0.5 rounded-full bg-purple-500/20 border border-purple-500 text-purple-400">s=N</span> 子树大小</span>
      </div>

      {/* Step message */}
      {currentStep && (
        <div className="rounded-lg px-4 py-3 border border-border-subtle bg-bg-tertiary space-y-1">
          <div className="text-sm text-blue-400">步骤 {stepIdx + 1}/{steps.length}：{currentStep.message}</div>
          {currentStep.detail && <div className="text-xs text-text-tertiary">{currentStep.detail}</div>}
          {currentStep.leftSize !== undefined && currentStep.rank !== null && (
            <div className="text-xs font-mono bg-bg-secondary rounded px-3 py-1 text-purple-300">
              ls = {currentStep.leftSize}，排名 r = ls+1 = {currentStep.rank}
            </div>
          )}
        </div>
      )}

      {/* Step controls */}
      {steps.length > 0 && (
        <div className="flex items-center gap-3">
          <button onClick={() => setStepIdx(i => Math.max(0, i - 1))} disabled={stepIdx === 0}
            className="px-3 py-1 rounded border border-border-subtle text-text-secondary text-xs disabled:opacity-40">← 上一步</button>
          <span className="text-xs text-text-tertiary">{stepIdx + 1} / {steps.length}</span>
          <button onClick={() => setStepIdx(i => Math.min(steps.length - 1, i + 1))} disabled={stepIdx === steps.length - 1}
            className="px-3 py-1 rounded border border-border-subtle text-text-secondary text-xs disabled:opacity-40">下一步 →</button>
        </div>
      )}

      {/* Algorithm pseudocode */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
        <div className="bg-bg-tertiary border border-border-subtle rounded-lg p-3 space-y-2">
          <div className="font-semibold text-text-primary">OS-SELECT(x, k) — 第k小</div>
          <pre className="text-text-tertiary text-xs whitespace-pre-wrap leading-relaxed">{`r = x.left.size + 1
if k == r: return x     # 命中
if k < r:  return OS-SELECT(x.left, k)
else:      return OS-SELECT(x.right, k-r)`}</pre>
          <div className="text-text-secondary">时间：O(log n)（树高）</div>
        </div>
        <div className="bg-bg-tertiary border border-border-subtle rounded-lg p-3 space-y-2">
          <div className="font-semibold text-text-primary">OS-RANK(T, x) — 求排名</div>
          <pre className="text-text-tertiary text-xs whitespace-pre-wrap leading-relaxed">{`rank = x.left.size + 1
y = x
while y != T.root:
  if y == y.parent.right:
    rank += y.parent.left.size + 1
  y = y.parent
return rank`}</pre>
          <div className="text-text-secondary">时间：O(log n)（树高）</div>
        </div>
      </div>

      {/* Size update rule */}
      <div className="bg-bg-tertiary border border-border-subtle rounded-lg p-3 text-xs space-y-1">
        <div className="font-semibold text-text-primary">size 字段维护规则</div>
        <div className="text-text-secondary">
          每次插入后，沿插入路径从叶到根逐一更新：
          <span className="font-mono text-purple-300 ml-2">size(x) = size(x.left) + size(x.right) + 1</span>
        </div>
        <div className="text-text-secondary">
          额外开销：O(h) 时间 / O(1) 额外空间（每节点 1 个整数字段），不影响整体复杂度。
        </div>
      </div>
    </div>
  );
}
