"use client";
import React, { useState, useMemo, useCallback } from "react";

// ─── Types ────────────────────────────────────────────────────────────────────
interface BSTNode {
  key: number;
  left: BSTNode | null;
  right: BSTNode | null;
  id: number;
}

type Op = "search" | "insert" | "delete";

interface Step {
  visitedIds: Set<number>;
  currentId: number | null;
  foundId: number | null;
  highlightType: "search" | "insert" | "delete-found" | "delete-successor" | null;
  message: string;
  insertPos?: { parentId: number | null; side: "left" | "right" | "root" };
  successorId?: number | null;
}

// ─── BST utilities (pure functions, immutable style) ─────────────────────────
let nodeIdCounter = 0;
function makeNode(key: number): BSTNode {
  return { key, left: null, right: null, id: nodeIdCounter++ };
}

function cloneTree(node: BSTNode | null): BSTNode | null {
  if (!node) return null;
  return { key: node.key, id: node.id, left: cloneTree(node.left), right: cloneTree(node.right) };
}

function insertBST(root: BSTNode | null, key: number): BSTNode {
  if (!root) return makeNode(key);
  if (key < root.key) return { ...root, left: insertBST(root.left, key) };
  if (key > root.key) return { ...root, right: insertBST(root.right, key) };
  return root; // duplicate: ignore
}

function minNode(node: BSTNode): BSTNode {
  while (node.left) node = node.left;
  return node;
}

function deleteBST(root: BSTNode | null, key: number): BSTNode | null {
  if (!root) return null;
  if (key < root.key) return { ...root, left: deleteBST(root.left, key) };
  if (key > root.key) return { ...root, right: deleteBST(root.right, key) };
  if (!root.left) return root.right;
  if (!root.right) return root.left;
  const succ = minNode(root.right);
  return { ...root, key: succ.key, id: root.id, right: deleteBST(root.right, succ.key) };
}

function buildTree(keys: number[]): BSTNode | null {
  nodeIdCounter = 0;
  let root: BSTNode | null = null;
  for (const k of keys) root = insertBST(root, k);
  return root;
}

// ─── Layout ───────────────────────────────────────────────────────────────────
interface LayoutNode {
  node: BSTNode;
  x: number;
  y: number;
}

function layoutTree(root: BSTNode | null, xMin = 0, xMax = 560, depth = 0): LayoutNode[] {
  if (!root) return [];
  const x = (xMin + xMax) / 2;
  const y = depth * 64 + 36;
  return [
    { node: root, x, y },
    ...layoutTree(root.left, xMin, x, depth + 1),
    ...layoutTree(root.right, x, xMax, depth + 1),
  ];
}

function treeHeight(node: BSTNode | null): number {
  if (!node) return 0;
  return 1 + Math.max(treeHeight(node.left), treeHeight(node.right));
}

// ─── Step generation ─────────────────────────────────────────────────────────
function generateSearchSteps(root: BSTNode | null, key: number): Step[] {
  const steps: Step[] = [];
  const visited = new Set<number>();

  function dfs(node: BSTNode | null) {
    if (!node) {
      steps.push({
        visitedIds: new Set(visited),
        currentId: null,
        foundId: null,
        highlightType: "search",
        message: `未找到 ${key}，到达 null`,
      });
      return;
    }
    visited.add(node.id);
    if (node.key === key) {
      steps.push({
        visitedIds: new Set(visited),
        currentId: node.id,
        foundId: node.id,
        highlightType: "search",
        message: `✅ 找到 ${key}！`,
      });
    } else if (key < node.key) {
      steps.push({
        visitedIds: new Set(visited),
        currentId: node.id,
        foundId: null,
        highlightType: "search",
        message: `${key} < ${node.key}，向左走`,
      });
      dfs(node.left);
    } else {
      steps.push({
        visitedIds: new Set(visited),
        currentId: node.id,
        foundId: null,
        highlightType: "search",
        message: `${key} > ${node.key}，向右走`,
      });
      dfs(node.right);
    }
  }

  dfs(root);
  return steps;
}

function generateInsertSteps(root: BSTNode | null, key: number): Step[] {
  const steps: Step[] = [];
  const visited = new Set<number>();

  if (!root) {
    steps.push({
      visitedIds: new Set(),
      currentId: null,
      foundId: null,
      highlightType: "insert",
      message: `树为空，将 ${key} 插入为根`,
      insertPos: { parentId: null, side: "root" },
    });
    return steps;
  }

  function dfs(node: BSTNode | null, parentId: number | null, side: "left" | "right" | "root"): void {
    if (!node) {
      steps.push({
        visitedIds: new Set(visited),
        currentId: null,
        foundId: null,
        highlightType: "insert",
        message: `到达空位，在 ${side === "root" ? "根" : side === "left" ? "左" : "右"}侧插入 ${key}`,
        insertPos: { parentId, side },
      });
      return;
    }
    visited.add(node.id);
    if (key === node.key) {
      steps.push({
        visitedIds: new Set(visited),
        currentId: node.id,
        foundId: node.id,
        highlightType: "insert",
        message: `${key} 已存在，忽略重复插入`,
      });
      return;
    } else if (key < node.key) {
      steps.push({
        visitedIds: new Set(visited),
        currentId: node.id,
        foundId: null,
        highlightType: "insert",
        message: `${key} < ${node.key}，向左走`,
      });
      dfs(node.left, node.id, "left");
    } else {
      steps.push({
        visitedIds: new Set(visited),
        currentId: node.id,
        foundId: null,
        highlightType: "insert",
        message: `${key} > ${node.key}，向右走`,
      });
      dfs(node.right, node.id, "right");
    }
  }

  dfs(root, null, "root");
  return steps;
}

function generateDeleteSteps(root: BSTNode | null, key: number): Step[] {
  const steps: Step[] = [];
  const visited = new Set<number>();

  function dfs(node: BSTNode | null): void {
    if (!node) {
      steps.push({
        visitedIds: new Set(visited),
        currentId: null,
        foundId: null,
        highlightType: "delete-found",
        message: `未找到 ${key}，无需删除`,
      });
      return;
    }
    visited.add(node.id);
    if (key < node.key) {
      steps.push({
        visitedIds: new Set(visited),
        currentId: node.id,
        foundId: null,
        highlightType: null,
        message: `${key} < ${node.key}，向左找`,
      });
      dfs(node.left);
    } else if (key > node.key) {
      steps.push({
        visitedIds: new Set(visited),
        currentId: node.id,
        foundId: null,
        highlightType: null,
        message: `${key} > ${node.key}，向右找`,
      });
      dfs(node.right);
    } else {
      // found
      if (!node.left && !node.right) {
        steps.push({
          visitedIds: new Set(visited),
          currentId: node.id,
          foundId: node.id,
          highlightType: "delete-found",
          message: `情形1：${key} 是叶节点，直接删除`,
        });
      } else if (!node.left || !node.right) {
        const child = node.left ?? node.right!;
        steps.push({
          visitedIds: new Set(visited),
          currentId: node.id,
          foundId: node.id,
          highlightType: "delete-found",
          message: `情形2：${key} 只有${!node.left ? "右" : "左"}子树，用子树替换`,
        });
      } else {
        // two children
        let succ = node.right!;
        while (succ.left) succ = succ.left;
        steps.push({
          visitedIds: new Set(visited),
          currentId: node.id,
          foundId: node.id,
          highlightType: "delete-found",
          successorId: succ.id,
          message: `情形3：${key} 有两个子节点，找后继节点（右子树最小值=${succ.key}）`,
        });
        steps.push({
          visitedIds: new Set(visited),
          currentId: succ.id,
          foundId: node.id,
          highlightType: "delete-successor",
          successorId: succ.id,
          message: `用后继 ${succ.key} 替换 ${key}，再从右子树删除 ${succ.key}`,
        });
      }
    }
  }

  dfs(root);
  return steps;
}

// ─── SVG Tree Component ───────────────────────────────────────────────────────
function TreeSVG({
  root,
  layoutNodes,
  step,
  insertGhost,
}: {
  root: BSTNode | null;
  layoutNodes: LayoutNode[];
  step: Step | null;
  insertGhost: { x: number; y: number; parentX: number; parentY: number } | null;
}) {
  const nodeMap = useMemo(() => {
    const m = new Map<number, LayoutNode>();
    for (const ln of layoutNodes) m.set(ln.node.id, ln);
    return m;
  }, [layoutNodes]);

  const h = treeHeight(root);
  const svgH = Math.max(180, h * 64 + 60);

  function getNodeColor(id: number) {
    if (!step) return "bg-bg-tertiary";
    if (step.highlightType === "delete-successor" && step.successorId === id) return "#f59e0b";
    if (step.foundId === id) {
      if (step.highlightType === "delete-found") return "#ef4444";
      if (step.highlightType === "insert") return "#8b5cf6";
      return "#22c55e";
    }
    if (step.currentId === id) return "#3b82f6";
    if (step.visitedIds.has(id)) return "#6b7280";
    return "";
  }

  return (
    <svg width="100%" viewBox={`0 0 560 ${svgH}`} className="overflow-visible">
      {/* Edges */}
      {layoutNodes.map(({ node, x, y }) => {
        const links: React.ReactNode[] = [];
        if (node.left) {
          const child = nodeMap.get(node.left.id);
          if (child)
            links.push(
              <line key={`e-l-${node.id}`} x1={x} y1={y} x2={child.x} y2={child.y}
                stroke="var(--color-border-subtle)" strokeWidth={1.5} />
            );
        }
        if (node.right) {
          const child = nodeMap.get(node.right.id);
          if (child)
            links.push(
              <line key={`e-r-${node.id}`} x1={x} y1={y} x2={child.x} y2={child.y}
                stroke="var(--color-border-subtle)" strokeWidth={1.5} />
            );
        }
        return links;
      })}

      {/* Ghost insert edge */}
      {insertGhost && (
        <line x1={insertGhost.parentX} y1={insertGhost.parentY}
          x2={insertGhost.x} y2={insertGhost.y}
          stroke="#8b5cf6" strokeWidth={1.5} strokeDasharray="4 3" />
      )}

      {/* Nodes */}
      {layoutNodes.map(({ node, x, y }) => {
        const color = getNodeColor(node.id);
        const fill = color || "var(--color-bg-tertiary)";
        const stroke = color ? color : "var(--color-border-subtle)";
        return (
          <g key={node.id}>
            <circle cx={x} cy={y} r={20} fill={fill} stroke={stroke} strokeWidth={2} />
            <text x={x} y={y + 5} textAnchor="middle" fontSize={12}
              fill={color ? "#fff" : "var(--color-text-primary)"} fontWeight="600">
              {node.key}
            </text>
          </g>
        );
      })}

      {/* Ghost insert node */}
      {insertGhost && (
        <g>
          <circle cx={insertGhost.x} cy={insertGhost.y} r={20}
            fill="#8b5cf6" stroke="#8b5cf6" strokeWidth={2} opacity={0.7} strokeDasharray="4 3" />
          <text x={insertGhost.x} y={insertGhost.y + 5} textAnchor="middle"
            fontSize={11} fill="#fff">new</text>
        </g>
      )}
    </svg>
  );
}

// ─── Main Component ───────────────────────────────────────────────────────────
const PRESETS = [
  { label: "标准BST", keys: [10, 5, 15, 3, 7, 12, 20] },
  { label: "左偏树", keys: [20, 15, 10, 7, 5] },
  { label: "右偏树", keys: [5, 7, 10, 15, 20] },
  { label: "综合示例", keys: [8, 4, 12, 2, 6, 10, 14, 1, 3] },
];

export default function BSTOperationsVisualizer() {
  const [presetIdx, setPresetIdx] = useState(0);
  const [root, setRoot] = useState<BSTNode | null>(() => buildTree(PRESETS[0].keys));
  const [op, setOp] = useState<Op>("search");
  const [inputKey, setInputKey] = useState("7");
  const [steps, setSteps] = useState<Step[]>([]);
  const [stepIdx, setStepIdx] = useState(-1);
  const [postOpRoot, setPostOpRoot] = useState<BSTNode | null>(null);
  const [executed, setExecuted] = useState(false);

  const layoutNodes = useMemo(() => layoutTree(root), [root]);

  const currentStep = stepIdx >= 0 && stepIdx < steps.length ? steps[stepIdx] : null;

  // Calculate ghost insert position
  const insertGhost = useMemo((): { x: number; y: number; parentX: number; parentY: number } | null => {
    if (!currentStep || currentStep.highlightType !== "insert" || !currentStep.insertPos) return null;
    const pos = currentStep.insertPos;
    if (pos.side === "root") return { x: 280, y: 36, parentX: 280, parentY: 36 };
    const parent = layoutNodes.find(ln => ln.node.id === pos.parentId);
    if (!parent) return null;
    const parentX = parent.x, parentY = parent.y;
    const dx = pos.side === "left" ? -40 : 40;
    return { x: parentX + dx, y: parentY + 64, parentX, parentY };
  }, [currentStep, layoutNodes]);

  const handleRun = useCallback(() => {
    const key = parseInt(inputKey);
    if (isNaN(key)) return;
    let newSteps: Step[] = [];
    if (op === "search") newSteps = generateSearchSteps(root, key);
    else if (op === "insert") newSteps = generateInsertSteps(root, key);
    else newSteps = generateDeleteSteps(root, key);
    setSteps(newSteps);
    setStepIdx(0);
    setExecuted(false);
    setPostOpRoot(null);
  }, [root, op, inputKey]);

  const handleApply = useCallback(() => {
    const key = parseInt(inputKey);
    if (isNaN(key)) return;
    let newRoot = root;
    if (op === "insert") newRoot = insertBST(root, key);
    else if (op === "delete") newRoot = deleteBST(root, key);
    setRoot(newRoot);
    setPostOpRoot(newRoot);
    setExecuted(true);
    setSteps([]);
    setStepIdx(-1);
  }, [root, op, inputKey]);

  const handlePreset = (idx: number) => {
    setPresetIdx(idx);
    setRoot(buildTree(PRESETS[idx].keys));
    setSteps([]);
    setStepIdx(-1);
    setExecuted(false);
    setPostOpRoot(null);
  };

  const opColors: Record<Op, string> = {
    search: "text-blue-400",
    insert: "text-purple-400",
    delete: "text-red-400",
  };
  const opLabels: Record<Op, string> = {
    search: "查找 (Search)",
    insert: "插入 (Insert)",
    delete: "删除 (Delete)",
  };

  return (
    <div className="rounded-xl border border-border-subtle bg-bg-secondary p-5 space-y-4 font-mono text-sm">
      <h3 className="text-base font-semibold text-text-primary">BST 基本操作可视化</h3>

      {/* Preset + Operation controls */}
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
          {(["search", "insert", "delete"] as Op[]).map(o => (
            <button key={o} onClick={() => { setOp(o); setSteps([]); setStepIdx(-1); }}
              className={`px-3 py-1 rounded-lg border text-xs transition-colors ${op === o
                ? "bg-bg-tertiary border-border-strong text-text-primary font-semibold"
                : "border-border-subtle text-text-secondary hover:border-border-strong"}`}>
              {opLabels[o]}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-2">
          <input type="number" value={inputKey} onChange={e => setInputKey(e.target.value)}
            className="w-20 px-2 py-1 rounded border border-border-subtle bg-bg-tertiary text-text-primary text-center"
            placeholder="键值" />
          <button onClick={handleRun}
            className="px-3 py-1 rounded-lg bg-blue-500/20 border border-blue-500 text-blue-400 text-xs hover:bg-blue-500/30">
            分步演示
          </button>
          {op !== "search" && (
            <button onClick={handleApply}
              className="px-3 py-1 rounded-lg bg-green-500/20 border border-green-500 text-green-400 text-xs hover:bg-green-500/30">
              直接执行
            </button>
          )}
        </div>
      </div>

      {/* Legend */}
      <div className="flex flex-wrap gap-3 text-xs text-text-secondary">
        <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-blue-500 inline-block" /> 当前节点</span>
        <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-gray-500 inline-block" /> 已访问</span>
        <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-green-500 inline-block" /> 查找命中</span>
        <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-red-500 inline-block" /> 待删除</span>
        <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-amber-400 inline-block" /> 后继节点</span>
        <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-purple-500 inline-block" /> 新插入</span>
      </div>

      {/* SVG Tree */}
      <div className="bg-bg-tertiary rounded-lg p-3 min-h-[220px]">
        {root ? (
          <TreeSVG root={root} layoutNodes={layoutNodes} step={currentStep} insertGhost={insertGhost} />
        ) : (
          <div className="h-32 flex items-center justify-center text-text-tertiary">（空树）</div>
        )}
      </div>

      {/* Step message */}
      {currentStep && (
        <div className={`rounded-lg px-4 py-2 border text-sm ${opColors[op]} bg-bg-tertiary border-border-subtle`}>
          步骤 {stepIdx + 1}/{steps.length}：{currentStep.message}
        </div>
      )}

      {/* Step controls */}
      {steps.length > 0 && (
        <div className="flex items-center gap-3">
          <button onClick={() => setStepIdx(i => Math.max(0, i - 1))} disabled={stepIdx === 0}
            className="px-3 py-1 rounded border border-border-subtle text-text-secondary text-xs disabled:opacity-40 hover:border-border-strong">
            ← 上一步
          </button>
          <span className="text-xs text-text-tertiary">{stepIdx + 1} / {steps.length}</span>
          <button onClick={() => setStepIdx(i => Math.min(steps.length - 1, i + 1))} disabled={stepIdx === steps.length - 1}
            className="px-3 py-1 rounded border border-border-subtle text-text-secondary text-xs disabled:opacity-40 hover:border-border-strong">
            下一步 →
          </button>
          {!executed && op !== "search" && stepIdx === steps.length - 1 && (
            <button onClick={handleApply}
              className="ml-2 px-3 py-1 rounded-lg bg-green-500/20 border border-green-500 text-green-400 text-xs hover:bg-green-500/30">
              应用到树
            </button>
          )}
        </div>
      )}

      {executed && (
        <div className="text-xs text-green-400 bg-green-500/10 border border-green-500/30 rounded-lg px-3 py-2">
          ✅ 操作已应用。当前树：{PRESETS[presetIdx].keys.join(", ")} → 执行 {opLabels[op]}({inputKey})
        </div>
      )}

      {/* Info: three delete cases */}
      {op === "delete" && (
        <div className="grid grid-cols-3 gap-2 text-xs">
          {[
            { case: "情形 1", desc: "叶节点", detail: "直接删除，无需调整" },
            { case: "情形 2", desc: "单子树", detail: "用唯一子树替换该节点" },
            { case: "情形 3", desc: "双子树", detail: "用中序后继替换键值，再删后继" },
          ].map((c) => (
            <div key={c.case} className="bg-bg-tertiary border border-border-subtle rounded-lg p-2">
              <div className="font-semibold text-text-primary">{c.case}：{c.desc}</div>
              <div className="text-text-secondary mt-1">{c.detail}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
