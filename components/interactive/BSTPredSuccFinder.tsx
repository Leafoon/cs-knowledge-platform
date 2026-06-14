"use client";
import React, { useState, useMemo } from "react";

// ─── Types ────────────────────────────────────────────────────────────────────
interface BSTNode {
  key: number;
  left: BSTNode | null;
  right: BSTNode | null;
  id: number;
}

interface Step {
  pathIds: number[];          // nodes visited so far
  currentId: number | null;  // currently highlighted node
  targetId: number | null;   // target node (p)
  resultId: number | null;   // found predecessor/successor
  message: string;
  phase: "search-target" | "go-right-subtree" | "go-up" | "found" | "none";
}

// ─── BST utilities ─────────────────────────────────────────────────────────────
let idCounter = 0;

function ins(root: BSTNode | null, key: number): BSTNode {
  if (!root) return { key, left: null, right: null, id: idCounter++ };
  if (key < root.key) return { ...root, left: ins(root.left, key) };
  if (key > root.key) return { ...root, right: ins(root.right, key) };
  return root;
}

function buildBST(keys: number[]): BSTNode | null {
  idCounter = 0;
  let root: BSTNode | null = null;
  for (const k of keys) root = ins(root, k);
  return root;
}

function allKeys(node: BSTNode | null): number[] {
  if (!node) return [];
  return [...allKeys(node.left), node.key, ...allKeys(node.right)];
}

// ─── Layout ───────────────────────────────────────────────────────────────────
interface LayoutNode { node: BSTNode; x: number; y: number; }

function layoutTree(root: BSTNode | null, xMin = 0, xMax = 520, depth = 0): LayoutNode[] {
  if (!root) return [];
  const x = (xMin + xMax) / 2, y = depth * 62 + 32;
  return [{ node: root, x, y }, ...layoutTree(root.left, xMin, x, depth + 1), ...layoutTree(root.right, x, xMax, depth + 1)];
}

function treeHeight(n: BSTNode | null): number {
  if (!n) return 0;
  return 1 + Math.max(treeHeight(n.left), treeHeight(n.right));
}

// Build parent map
function buildParentMap(root: BSTNode | null): Map<number, BSTNode | null> {
  const map = new Map<number, BSTNode | null>();
  function dfs(node: BSTNode | null, parent: BSTNode | null) {
    if (!node) return;
    map.set(node.id, parent);
    dfs(node.left, node);
    dfs(node.right, node);
  }
  dfs(root, null);
  return map;
}

// ─── Step generation ─────────────────────────────────────────────────────────
type Mode = "successor" | "predecessor";

function generateSteps(root: BSTNode | null, targetKey: number, mode: Mode): Step[] {
  const steps: Step[] = [];
  const parentMap = buildParentMap(root);

  // Phase 1: find target node
  const path: number[] = [];

  function findTarget(node: BSTNode | null): BSTNode | null {
    if (!node) return null;
    path.push(node.id);
    steps.push({
      pathIds: [...path],
      currentId: node.id,
      targetId: null,
      resultId: null,
      message: `寻找节点 ${targetKey}：当前 ${targetKey} ${targetKey < node.key ? "< " + node.key + "，向左" : targetKey > node.key ? "> " + node.key + "，向右" : "= " + node.key + "，找到！"}`,
      phase: "search-target",
    });
    if (node.key === targetKey) return node;
    if (targetKey < node.key) return findTarget(node.left);
    return findTarget(node.right);
  }
  const targetNode = findTarget(root);
  if (!targetNode) {
    steps.push({ pathIds: path, currentId: null, targetId: null, resultId: null, message: `节点 ${targetKey} 不存在`, phase: "none" });
    return steps;
  }
  const targetId = targetNode.id;

  if (mode === "successor") {
    // Case 1: has right subtree → go to minimum of right
    if (targetNode.right) {
      steps.push({
        pathIds: [...path],
        currentId: targetNode.id,
        targetId,
        resultId: null,
        message: `${targetKey} 有右子树 → 后继 = 右子树最小值（向右子树走）`,
        phase: "go-right-subtree",
      });
      let cur = targetNode.right;
      while (cur) {
        path.push(cur.id);
        const isMin = !cur.left;
        steps.push({
          pathIds: [...path],
          currentId: cur.id,
          targetId,
          resultId: isMin ? cur.id : null,
          message: isMin
            ? `✅ 后继 = ${cur.key}（右子树最小值，无左子）`
            : `${cur.key} 有左子 → 继续向左找最小`,
          phase: isMin ? "found" : "go-right-subtree",
        });
        if (isMin) break;
        cur = cur.left!;
      }
    } else {
      // Case 2: no right subtree → go up until we come from left
      steps.push({
        pathIds: [...path],
        currentId: targetNode.id,
        targetId,
        resultId: null,
        message: `${targetKey} 无右子树 → 后继 = 向上走，直到"从左侧进入祖先"`,
        phase: "go-up",
      });
      let x: BSTNode = targetNode;
      let y = parentMap.get(x.id) ?? null;
      while (y !== null && y !== undefined && x.id === y.right?.id) {
        path.push(y.id);
        steps.push({
          pathIds: [...path],
          currentId: y.id,
          targetId,
          resultId: null,
          message: `从${x.key}的父节点${y.key}的右侧进入，继续向上`,
          phase: "go-up",
        });
        x = y;
        y = parentMap.get(y.id) ?? null;
      }
      if (y) {
        path.push(y.id);
        steps.push({
          pathIds: [...path],
          currentId: y.id,
          targetId,
          resultId: y.id,
          message: `✅ 后继 = ${y.key}（从左侧进入的第一个祖先）`,
          phase: "found",
        });
      } else {
        steps.push({
          pathIds: [...path],
          currentId: null,
          targetId,
          resultId: null,
          message: `${targetKey} 是树中最大值，无后继`,
          phase: "none",
        });
      }
    }
  } else {
    // predecessor
    if (targetNode.left) {
      steps.push({
        pathIds: [...path],
        currentId: targetNode.id,
        targetId,
        resultId: null,
        message: `${targetKey} 有左子树 → 前驱 = 左子树最大值（向左子树走）`,
        phase: "go-right-subtree",
      });
      let cur = targetNode.left;
      while (cur) {
        path.push(cur.id);
        const isMax = !cur.right;
        steps.push({
          pathIds: [...path],
          currentId: cur.id,
          targetId,
          resultId: isMax ? cur.id : null,
          message: isMax
            ? `✅ 前驱 = ${cur.key}（左子树最大值，无右子）`
            : `${cur.key} 有右子 → 继续向右找最大`,
          phase: isMax ? "found" : "go-right-subtree",
        });
        if (isMax) break;
        cur = cur.right!;
      }
    } else {
      steps.push({
        pathIds: [...path],
        currentId: targetNode.id,
        targetId,
        resultId: null,
        message: `${targetKey} 无左子树 → 前驱 = 向上走，直到"从右侧进入祖先"`,
        phase: "go-up",
      });
      let x: BSTNode = targetNode;
      let y = parentMap.get(x.id) ?? null;
      while (y !== null && y !== undefined && x.id === y.left?.id) {
        path.push(y.id);
        steps.push({
          pathIds: [...path],
          currentId: y.id,
          targetId,
          resultId: null,
          message: `从${x.key}的父节点${y.key}的左侧进入，继续向上`,
          phase: "go-up",
        });
        x = y;
        y = parentMap.get(y.id) ?? null;
      }
      if (y) {
        path.push(y.id);
        steps.push({
          pathIds: [...path],
          currentId: y.id,
          targetId,
          resultId: y.id,
          message: `✅ 前驱 = ${y.key}（从右侧进入的第一个祖先）`,
          phase: "found",
        });
      } else {
        steps.push({
          pathIds: [...path],
          currentId: null,
          targetId,
          resultId: null,
          message: `${targetKey} 是树中最小值，无前驱`,
          phase: "none",
        });
      }
    }
  }
  return steps;
}

// ─── SVG Tree ─────────────────────────────────────────────────────────────────
function TreeSVG({ root, layoutNodes, step }: { root: BSTNode | null; layoutNodes: LayoutNode[]; step: Step | null }) {
  const nodeMap = useMemo(() => {
    const m = new Map<number, LayoutNode>();
    for (const ln of layoutNodes) m.set(ln.node.id, ln);
    return m;
  }, [layoutNodes]);

  const h = treeHeight(root);
  const svgH = Math.max(180, h * 62 + 50);

  function getColor(id: number): { fill: string; stroke: string; textFill: string } {
    if (!step) return { fill: "var(--color-bg-tertiary)", stroke: "var(--color-border-subtle)", textFill: "var(--color-text-primary)" };
    if (step.resultId === id) return { fill: "#f59e0b", stroke: "#f59e0b", textFill: "#fff" };
    if (step.currentId === id) return { fill: "#3b82f6", stroke: "#3b82f6", textFill: "#fff" };
    if (step.targetId === id) return { fill: "#8b5cf6", stroke: "#8b5cf6", textFill: "#fff" };
    if (step.pathIds.includes(id)) return { fill: "#6b728033", stroke: "#6b7280", textFill: "var(--color-text-secondary)" };
    return { fill: "var(--color-bg-tertiary)", stroke: "var(--color-border-subtle)", textFill: "var(--color-text-primary)" };
  }

  return (
    <svg width="100%" viewBox={`0 0 520 ${svgH}`} className="overflow-visible">
      {layoutNodes.map(({ node, x, y }) => (
        [
          node.left && nodeMap.get(node.left.id) && (
            <line key={`el${node.id}`} x1={x} y1={y} x2={nodeMap.get(node.left.id)!.x} y2={nodeMap.get(node.left.id)!.y}
              stroke="var(--color-border-subtle)" strokeWidth={1.5} />),
          node.right && nodeMap.get(node.right.id) && (
            <line key={`er${node.id}`} x1={x} y1={y} x2={nodeMap.get(node.right.id)!.x} y2={nodeMap.get(node.right.id)!.y}
              stroke="var(--color-border-subtle)" strokeWidth={1.5} />),
        ]
      ))}
      {layoutNodes.map(({ node, x, y }) => {
        const { fill, stroke, textFill } = getColor(node.id);
        return (
          <g key={node.id}>
            <circle cx={x} cy={y} r={20} fill={fill} stroke={stroke} strokeWidth={2} />
            <text x={x} y={y + 5} textAnchor="middle" fontSize={12} fill={textFill} fontWeight="600">{node.key}</text>
          </g>
        );
      })}
    </svg>
  );
}

// ─── Main Component ───────────────────────────────────────────────────────────
const PRESETS = [
  { label: "标准BST", keys: [10, 5, 15, 3, 7, 12, 20] },
  { label: "完整BST", keys: [8, 4, 12, 2, 6, 10, 14, 1, 3, 5, 7] },
  { label: "右链形状", keys: [5, 8, 12, 15, 20] },
];

export default function BSTPredSuccFinder() {
  const [presetIdx, setPresetIdx] = useState(0);
  const root = useMemo(() => buildBST(PRESETS[presetIdx].keys), [presetIdx]);
  const keys = useMemo(() => allKeys(root), [root]);
  const layoutNodes = useMemo(() => layoutTree(root), [root]);
  const [mode, setMode] = useState<Mode>("successor");
  const [targetKey, setTargetKey] = useState<number>(keys[2] ?? 5);
  const [steps, setSteps] = useState<Step[]>([]);
  const [stepIdx, setStepIdx] = useState(-1);

  const currentStep = stepIdx >= 0 && stepIdx < steps.length ? steps[stepIdx] : null;

  function handleRun() {
    const s = generateSteps(root, targetKey, mode);
    setSteps(s);
    setStepIdx(0);
  }

  function handlePreset(idx: number) {
    setPresetIdx(idx);
    setSteps([]);
    setStepIdx(-1);
    const newRoot = buildBST(PRESETS[idx].keys);
    const ks = allKeys(newRoot);
    setTargetKey(ks[Math.floor(ks.length / 2)] ?? ks[0]);
  }

  // Compute actual pred/succ for reference
  const inorder = useMemo(() => allKeys(root), [root]);
  const targetIdx = inorder.indexOf(targetKey);
  const actualSucc = targetIdx >= 0 && targetIdx < inorder.length - 1 ? inorder[targetIdx + 1] : null;
  const actualPred = targetIdx > 0 ? inorder[targetIdx - 1] : null;

  const phaseColors: Record<string, string> = {
    "search-target": "text-blue-400",
    "go-right-subtree": "text-yellow-400",
    "go-up": "text-orange-400",
    "found": "text-green-400",
    "none": "text-red-400",
  };

  return (
    <div className="rounded-xl border border-border-subtle bg-bg-secondary p-5 space-y-4 font-mono text-sm">
      <h3 className="text-base font-semibold text-text-primary">前驱 / 后继查找可视化</h3>

      {/* Controls */}
      <div className="flex flex-wrap gap-2">
        {PRESETS.map((p, i) => (
          <button key={i} onClick={() => handlePreset(i)}
            className={`px-3 py-1 rounded-lg text-xs border transition-colors ${i === presetIdx
              ? "bg-blue-500/20 border-blue-500 text-blue-400"
              : "border-border-subtle text-text-secondary hover:border-blue-400"}`}>
            {p.label}
          </button>
        ))}
      </div>

      <div className="flex flex-wrap gap-3 items-center">
        <div className="flex gap-2">
          {(["successor", "predecessor"] as Mode[]).map(m => (
            <button key={m} onClick={() => { setMode(m); setSteps([]); setStepIdx(-1); }}
              className={`px-3 py-1 rounded-lg border text-xs transition-colors ${mode === m
                ? "bg-bg-tertiary border-border-strong text-text-primary font-semibold"
                : "border-border-subtle text-text-secondary hover:border-border-strong"}`}>
              {m === "successor" ? "后继 Successor" : "前驱 Predecessor"}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-2">
          <span className="text-text-secondary text-xs">目标节点：</span>
          <select value={targetKey} onChange={e => { setTargetKey(Number(e.target.value)); setSteps([]); setStepIdx(-1); }}
            className="px-2 py-1 rounded border border-border-subtle bg-bg-tertiary text-text-primary text-xs">
            {keys.map(k => <option key={k} value={k}>{k}</option>)}
          </select>
          <button onClick={handleRun}
            className="px-3 py-1 rounded-lg bg-blue-500/20 border border-blue-500 text-blue-400 text-xs hover:bg-blue-500/30">
            开始演示
          </button>
        </div>
      </div>

      {/* Reference info */}
      <div className="flex gap-3 text-xs">
        <span className="bg-bg-tertiary border border-border-subtle rounded px-3 py-1 text-text-secondary">
          中序序列：{inorder.join(" < ")}
        </span>
        <span className="bg-purple-500/10 border border-purple-500/30 rounded px-3 py-1 text-purple-400">
          目标 {targetKey} → 后继：{actualSucc ?? "无"} | 前驱：{actualPred ?? "无"}
        </span>
      </div>

      {/* Legend */}
      <div className="flex flex-wrap gap-3 text-xs text-text-secondary">
        <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-purple-500 inline-block" /> 目标节点</span>
        <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-blue-500 inline-block" /> 当前位置</span>
        <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-amber-400 inline-block" /> 结果（前驱/后继）</span>
        <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-gray-500 inline-block" /> 已访问路径</span>
      </div>

      {/* Tree SVG */}
      <div className="bg-bg-tertiary rounded-lg p-3 min-h-[200px]">
        <TreeSVG root={root} layoutNodes={layoutNodes} step={currentStep} />
      </div>

      {/* Step message */}
      {currentStep && (
        <div className={`rounded-lg px-4 py-2 border text-sm bg-bg-tertiary border-border-subtle ${phaseColors[currentStep.phase]}`}>
          步骤 {stepIdx + 1}/{steps.length}：{currentStep.message}
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

      {/* Two-case explanation */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
        <div className="bg-bg-tertiary border border-border-subtle rounded-lg p-3 space-y-1">
          <div className="font-semibold text-text-primary">后继（Successor）算法</div>
          <div className="text-text-secondary space-y-1">
            <div><span className="text-yellow-400">情形1</span>：有右子树 → 右子树的最小值</div>
            <div><span className="text-orange-400">情形2</span>：无右子树 → 向上找，直到从左侧进入祖先</div>
          </div>
          <div className="font-mono text-text-tertiary text-xs mt-2">
            {`if node.right:\n  return min(node.right)\nelse:\n  go up while from right child`}
          </div>
        </div>
        <div className="bg-bg-tertiary border border-border-subtle rounded-lg p-3 space-y-1">
          <div className="font-semibold text-text-primary">前驱（Predecessor）算法</div>
          <div className="text-text-secondary space-y-1">
            <div><span className="text-yellow-400">情形1</span>：有左子树 → 左子树的最大值</div>
            <div><span className="text-orange-400">情形2</span>：无左子树 → 向上找，直到从右侧进入祖先</div>
          </div>
          <div className="font-mono text-text-tertiary text-xs mt-2">
            {`if node.left:\n  return max(node.left)\nelse:\n  go up while from left child`}
          </div>
        </div>
      </div>
    </div>
  );
}
